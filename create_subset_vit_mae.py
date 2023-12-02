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
from transformers import ViTMAEForPreTraining


class ProxyModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.encoder = net.vit
    def forward(self, inputs):
        # Take the embedding of the [class] token
        return self.encoder(inputs).last_hidden_state[:,0,:].squeeze(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR100.value), help='dataset',
                        choices=[x.value for x in SupportedDatasets])
    parser.add_argument('--device', type=int, default=0, help="GPU number to use")
    parser.add_argument('--subset-fractions', nargs='+', type=float, help='List of subset sizes')
    parser.add_argument('--net-path', type=str, default="", help="Path to net")
    parser.add_argument('--critic-path', type=str, default="", help="Path to critic")
    parser.add_argument('--subset-path', type=str, default="", help="Path to save subset indices")
    parser.add_argument('--proxy-img-size', type=int, default=224, help="Input image size for proxy model")
    parser.add_argument('--proxy-dataset', type=str, default="cifar100", help="Dataset to train proxy model")
    parser.add_argument('--proxy-arch', type=str, default="resnet10", help="Proxy model architecture")
    parser.add_argument('--proxy-pretrain', type=str, default="", help="Pretrained model name to be loaded from huggingface")
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    
    return parser.parse_args()


def get_dataset(dataset_name, img_size):
    transform_inference = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    if dataset_name == SupportedDatasets.CIFAR10.value:
        dataset = torchvision.datasets.CIFAR10("/data/cifar10/", transform=transform_inference)
        num_classes = 10
    elif dataset_name == SupportedDatasets.CIFAR100.value:
        dataset = torchvision.datasets.CIFAR100("/data/cifar100/", transform=transform_inference)
        num_classes = 100
    elif dataset_name == SupportedDatasets.STL10.value:
        dataset = torchvision.datasets.STL10("/data/", transform=transform_inference)
        num_classes = 10
    elif dataset_name == SupportedDatasets.TINY_IMAGENET.value:
        dataset = ImageFolder(root=f"/data/tiny_imagenet/train/", transform=transform_inference)
        num_classes = 200
    elif dataset_name == SupportedDatasets.IMAGENET.value:
        dataset = ImageNet(root=f"/data/ILSVRC/train_full/", transform=transform_inference)
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
    dataset, num_classes = get_dataset(args.dataset, args.proxy_img_size)
    
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
    model = ViTMAEForPreTraining.from_pretrained(args.proxy_pretrain)
    proxy_model = ProxyModel(model)
    
    for frac in args.subset_fractions:
        print(f"Subset fraction = {frac}")
        subset_dataset = SASSubsetDataset(
            dataset=dataset,
            subset_fraction=frac,
            num_downstream_classes=num_classes,
            device=device,
            proxy_model=proxy_model,
            approx_latent_class_partition=partition,
            pairwise_distance_block_size=1024,
            verbose=True
        )
        
        # Save subset to file
        os.makedirs(args.subset_path, exist_ok=True)
        subset_dataset.save_to_file(f"{args.subset_path}/{args.dataset}-{frac}-sas-indices-{args.proxy_arch}-{args.proxy_dataset}.pkl")