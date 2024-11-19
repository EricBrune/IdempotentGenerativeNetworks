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
from ign_mnist import *

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # For MNIST, keep original 28x28 size as per paper
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    
    model = IGN()
    losses, val_losses = train_ign(model, IGN, train_loader, val_loader, device=device, total_iterations=3000)
    
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
