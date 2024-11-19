import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

from ign_common import *
from ign_celeba import *

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image_size = 64
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.CelebA(root='./data_celeb', split="train", transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = datasets.CelebA(root='./data_celeb', split="valid", transform=transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    model = IGN()
    losses, val_losses = train_ign(model, IGN, train_loader, val_loader, device=device, total_iterations=1000)
    
    # Plot final loss curves
    plt.figure(figsize=(10, 5))
    for loss_name, values in losses.items():
        if loss_name != 'total':  # Skip total loss for clarity
            plt.plot(values, label=f'Train {loss_name}')
    for loss_name, values in val_losses.items():
        if loss_name != 'total':  # Skip total loss for clarity
            plt.plot(values, label=f'Val {loss_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('losses.png')
    plt.close()

if __name__ == '__main__':
    main()