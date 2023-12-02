import argparse
import os
import pickle
from datetime import datetime
import random

import numpy as np
import sas.subset_dataset
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb

from configs import SupportedDatasets, get_datasets
from projection_heads.critic import LinearCritic
from resnet import *
from trainer import Trainer
from util import Random
import logging
from logger import get_logger, set_logger

def main(rank: int, world_size: int, args):

    # Determine Device 
    device = rank
    if args.distributed:
        device = args.device_ids[rank]
        torch.cuda.set_device(args.device_ids[rank])
        args.lr *= world_size
    
    # Create logger
    logger, listener = get_logger(args.log_file_path)
    
    listener.start()
    set_logger(rank=0, logger=logger, distributed=args.distributed)
    logger.info(args)

    # WandB Logging
    if not args.distributed or rank == 0:
        logging.debug("Starting wandb")
        wandb.init(
            project="data-efficient-contrastive-learning",
            config=args
        )
        wandb.run.name = args.name
        wandb.save(os.path.join(args.log_dir_path, "params.txt"))

    if args.distributed:
        args.batch_size = int(args.batch_size / world_size)

    # Set all seeds
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Random(args.seed)

    logging.info('==> Preparing data..')
    datasets = get_datasets(args.dataset)

    ##############################################################
    # Load Subset Indices
    ##############################################################

    if args.random_subset:
        trainset = sas.subset_dataset.RandomSubsetDataset(
            dataset=datasets.trainset,
            subset_fraction=args.subset_fraction
        )
    elif args.subset_indices != "":
        with open(args.subset_indices, "rb") as f:
            subset_indices = pickle.load(f)
        trainset = sas.subset_dataset.CustomSubsetDataset(
            dataset=datasets.trainset,
            subset_indices=subset_indices
        )
    else:
        trainset = datasets.trainset
    logging.info(f"subset_size: {len(trainset)}")

    # Model
    logging.info('==> Building model..')

    ##############################################################
    # Encoder
    ##############################################################

    if args.arch == 'resnet10':
        net = ResNet10(stem=datasets.stem)
    elif args.arch == 'resnet18':
        net = ResNet18(stem=datasets.stem)
    elif args.arch == 'resnet34':
        net = ResNet34(stem=datasets.stem)
    elif args.arch == 'resnet50':
        net = ResNet50(stem=datasets.stem)
    else:
        raise ValueError("Bad architecture specification")

    ##############################################################
    # Critic
    ##############################################################

    critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

    # DCL Setup
    optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6)
    if args.dataset == SupportedDatasets.TINY_IMAGENET.value:
        optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=2 * args.lr, weight_decay=1e-6)
        

    ##############################################################
    # Data Loaders
    ##############################################################

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=(not args.distributed),
        sampler=DistributedSampler(trainset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True) if args.distributed else None,
        num_workers=4,
        pin_memory=True,
    )

    clftrainloader = torch.utils.data.DataLoader(
        dataset=datasets.clftrainset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        dataset=datasets.testset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
    )
    
    ##############################################################
    # Main Loop (Train, Test)
    ##############################################################

    if args.distributed:
        ddp_setup(rank, world_size, str(args.port))

    net = net.to(device)
    critic = critic.to(device)
    if args.distributed:
        net = DDP(net, device_ids=[device])

    trainer = Trainer(
        device=device,
        distributed=args.distributed,
        rank=rank if args.distributed else 0,
        world_size=world_size,
        net=net,
        critic=critic,
        trainloader=trainloader,
        clftrainloader=clftrainloader,
        testloader=testloader,
        num_classes=datasets.num_classes,
        optimizer=optimizer,
    )

    for epoch in range(0, args.num_epochs):
        logging.info(f"step: {epoch}")

        train_loss = trainer.train()
        logging.info(f"train_loss: {train_loss}")
        if not args.distributed or rank == 0:
            wandb.log(
                data={"train": {
                    "loss": train_loss,
                    "lr": optimizer.param_groups[0]["lr"]
                }},
                step=epoch
            )

        if (args.test_freq > 0) and (not args.distributed or rank == 0) and ((epoch + 1) % args.test_freq == 0):
            test_acc = trainer.test()
            logging.info(f"test_acc: {test_acc}")
            wandb.log(
                data={"test": {
                    "acc": test_acc
                }},
                step=epoch
            )

        # Checkpoint Model
        if (args.checkpoint_freq > 0) and ((not args.distributed or rank == 0) and (epoch + 1) % args.checkpoint_freq == 0):
            trainer.save_checkpoint(prefix=f"{args.name}-{epoch}")

    if not args.distributed or rank == 0:
        logging.info(f"best_test_acc: {trainer.best_acc}")
        wandb.log(
            data={"test": {
                "best_acc": trainer.best_acc,
            }}
        )
        wandb.finish(quiet=True)

    listener.stop()
    
    if args.distributed:
        destroy_process_group()


##############################################################
# Distributed Training Setup
##############################################################
def ddp_setup(rank: int, world_size: int, port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning.')
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--num-epochs", type=int, default=400, help='Number of training epochs')
    parser.add_argument("--arch", type=str, default='resnet18', help='Encoder architecture',
                        choices=['resnet10', 'resnet18', 'resnet34', 'resnet50'])
    parser.add_argument("--test-freq", type=int, default=10, help='Frequency to fit a linear clf with L-BFGS for testing'
                                                                'Not appropriate for large datasets. Set 0 to avoid '
                                                                'classifier only training here.')
    parser.add_argument("--checkpoint-freq", type=int, default=400, help="How often to checkpoint model")
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR100.value), help='dataset',
                        choices=[x.value for x in SupportedDatasets])
    parser.add_argument('--subset-indices', type=str, default="", help="Path to subset indices")
    parser.add_argument('--random-subset', action="store_true", help="Random subset")
    parser.add_argument('--subset-fraction', type=float, help="Size of Subset as fraction (only needed for random subset)")
    parser.add_argument('--device', type=int, default=-1, help="GPU number to use")
    parser.add_argument("--device-ids", nargs = "+", default = None, help = "Specify device ids if using multiple gpus")
    parser.add_argument('--port', type=int, default=random.randint(49152, 65535), help="free port to use")
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument("--logs", type = str, default = "logs/", help = "Logs directory path")

    # Parse arguments
    args = parser.parse_args()

    # Arguments check and initialize global variables
    device = "cpu"
    device_ids = None
    distributed = False
    if torch.cuda.is_available():
        if args.device_ids is None:
            if args.device >= 0:
                device = args.device
            else:
                device = 0
        else:
            distributed = True
            device_ids = [int(id) for id in args.device_ids]
    args.device = device
    args.device_ids = device_ids
    args.distributed = distributed
    
    # Date Time String
    DT_STRING = str(datetime.now())[:-3]
    args.name = f"{DT_STRING}-{args.dataset}-{args.arch}-seed{args.seed}"
    
    # Create directory to save results
    os.makedirs(args.logs, exist_ok = True)
    args.log_dir_path = os.path.join(args.logs, args.name)
    os.makedirs(args.log_dir_path, exist_ok = True)
    args.log_file_path = os.path.join(args.log_dir_path, "training.log")
    
    if distributed:
        mp.spawn(
            fn=main, 
            args=(len(device_ids), args),
            nprocs=len(device_ids)
        )
    else:
        main(device, 1, args)