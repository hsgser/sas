import os
import argparse
import random 
from util import Random
from configs import SupportedDatasets
from sas.approx_latent_classes import clip_approx
from sas.subset_dataset import SASSubsetDataset
from data_proc.dataset import *
import torch
from torch import nn 
import torchvision
from torchvision import transforms


class ProxyModel(nn.Module):
    def __init__(self, net, critic):
        super().__init__()
        self.net = net
        self.critic = critic
    def forward(self, x):
        return self.critic.project(self.net(x))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR100.value), help='dataset',
                        choices=[x.value for x in SupportedDatasets])
    parser.add_argument('--device', type=int, default=0, help="GPU number to use")
    parser.add_argument('--subset-fraction', type=float, help="Size of Subset as fraction (only needed for random subset)")
    parser.add_argument('--net-path', type=str, default="", help="Path to net")
    parser.add_argument('--critic-path', type=str, default="", help="Path to critic")
    parser.add_argument('--subset-path', type=str, default="", help="Path to save subset indices")
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    
    return parser.parse_args()


def get_dataset(dataset_name):
    if dataset_name == SupportedDatasets.CIFAR10.value:
        dataset = torchvision.datasets.CIFAR10("/data/cifar10/", transform=transforms.ToTensor())
        num_classes = 10
    elif dataset_name == SupportedDatasets.CIFAR100.value:
        dataset = torchvision.datasets.CIFAR100("/data/cifar100/", transform=transforms.ToTensor())
        num_classes = 100
    elif dataset_name == SupportedDatasets.STL10.value:
        dataset = torchvision.datasets.STL10("/data/", transform=transforms.ToTensor())
        num_classes = 10
    elif dataset_name == SupportedDatasets.TINY_IMAGENET.value:
        dataset = ImageFolder(root=f"/data/tiny_imagenet/train/", transform=transforms.ToTensor())
        num_classes = 200
    elif dataset_name == SupportedDatasets.IMAGENET.value:
        dataset = ImageNet(root=f"/data/ILSVRC/train_full/", transform=transforms.ToTensor())
        num_classes = 1000
    
    return dataset, num_classes

if __name__ == "__main__":
    args = get_args()
    device = f"cuda:{args.device}"
    
    # Set all seeds
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Random(args.seed)
    
    # Load data
    dataset, num_classes = get_dataset(args.dataset)
    
    # Partition into approximate latent classes
    rand_labeled_examples_indices = random.sample(range(len(dataset)), 500)
    rand_labeled_examples_labels = [dataset[i][1] for i in rand_labeled_examples_indices]

    partition = clip_approx(
        img_trainset=dataset,
        labeled_example_indices=rand_labeled_examples_indices, 
        labeled_examples_labels=rand_labeled_examples_labels,
        num_classes=num_classes,
        device=device
    )
    
    # Load proxy model
    net = torch.load(args.net_path)
    critic = torch.load(args.critic_path)
    proxy_model = ProxyModel(net, critic)
        
    subset_dataset = SASSubsetDataset(
        dataset=dataset,
        subset_fraction=args.subset_fraction,
        num_downstream_classes=num_classes,
        device=device,
        proxy_model=proxy_model,
        approx_latent_class_partition=partition,
        verbose=True
    )
    
    # Save subset to file
    os.makedirs(args.subset_path, exist_ok=True)
    subset_dataset.save_to_file(f"{args.subset_path}/{args.dataset}-{args.subset_fraction}-sas-indices.pkl")