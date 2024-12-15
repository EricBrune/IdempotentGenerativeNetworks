import os
import typing
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

#####################
# Gaussian Blur Helper
#####################
def gaussian_kernel_1d(kernel_size: int, sigma: float):
    k = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1)/2
    kernel_1d = torch.exp(-0.5 * (k/sigma)**2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d

def gaussian_blur(img: torch.Tensor, kernel_size: int = 21):
    # σ as specified
    sigma = 0.3*((kernel_size - 1)*0.5 - 1)+0.8
    kernel_1d = gaussian_kernel_1d(kernel_size, sigma).to(img.device)
    
    C = img.size(1)  # number of channels
    # Reshape and repeat the kernel for each channel
    kernel_x = kernel_1d.view(1,1,-1,1).repeat(C,1,1,1)  # [C,1,K,1]
    kernel_y = kernel_1d.view(1,1,1,-1).repeat(C,1,1,1)  # [C,1,1,K]
    
    # Apply depthwise conv in x-direction
    img = F.conv2d(img, kernel_x, padding=(kernel_size//2,0), groups=C)
    # Apply depthwise conv in y-direction
    img = F.conv2d(img, kernel_y, padding=(0,kernel_size//2), groups=C)
    
    return img


#####################
# Data and Model Loading
#####################
image_size = 64
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.CenterCrop(image_size),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # This puts data in [-1,1]
])

local_data_path = '/home/e/b/ebrune/final_project/Idempotent-Generative-Network/celeba/img_align_celeba'

orig_set = datasets.ImageFolder(
    root=os.path.join(local_data_path, ''),
    transform=transform
)

n = len(orig_set)
n_test = int(0.05 * n)
val_dataset = torch.utils.data.Subset(orig_set, range(n_test))
train_dataset = torch.utils.data.Subset(orig_set, range(n_test, n))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

#####################
# Load IGN Model
#####################
from ign_celeba import IGN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = 'checkpoints/_L1/ign_iter_92000.pt'
model = IGN().to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
state_dict = ckpt['model_state_dict']
new_state_dict = {k[len('module.'):]: v if k.startswith('module.') else v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

#####################
# Helper Functions
#####################
to_01 = lambda img: (img * 0.5) + 0.5
to_m11 = lambda img: img * 2 - 1

def add_noise(img, std=0.15):
    # img in [-1,1], noise N(0, 0.15)
    return img + torch.randn_like(img)*std

def make_gray(img):
    # g(x) = mean over channels, repeated for 3 channels
    g = img.mean(dim=1, keepdim=True)
    g = g.repeat(1,3,1,1)
    return g

def make_sketch(img):
    # s(x) = ( g(x+1) / (gaussian_blur(g(x+1),21)+1e-10) ) - 1
    g = make_gray(img)
    g_plus = g + 1.0
    blurred = gaussian_blur(g_plus, kernel_size=21)
    s = g_plus / (blurred + 1e-10) - 1.0
    return s

#####################
# Prepare Images
#####################
val_iter = iter(val_loader)
x = next(val_iter)[0].to(device)  # [B, 3, H, W]
x = x[4:8]  # Take first 4 images

x_noisy = add_noise(x)  # noisy images
x_gray = make_gray(x)    # grayscale
x_sketch = make_sketch(x) # pencil sketch as per text

with torch.no_grad():
    f_noisy = model(x_noisy)
    ff_noisy = model(f_noisy)
    f_gray = model(x_gray)
    ff_gray = model(f_gray)
    f_sketch = model(x_sketch)
    ff_sketch = model(f_sketch)

#####################
# Arrange Images for Plotting
#####################
def blank_image(batch_size, height, width, device):
    return torch.zeros((batch_size,3,height,width), device=device)

rows = []
for i in range(4):
    row_images = torch.cat([
        x[i:i+1], x_noisy[i:i+1], f_noisy[i:i+1], ff_noisy[i:i+1],
        blank_image(1, x.shape[2], x.shape[3], device),
        x[i:i+1], x_gray[i:i+1], f_gray[i:i+1], ff_gray[i:i+1],
        blank_image(1, x.shape[2], x.shape[3], device),
        x[i:i+1], x_sketch[i:i+1], f_sketch[i:i+1], ff_sketch[i:i+1]
    ], dim=0)
    rows.append(row_images)

all_images = torch.cat(rows, dim=0)  # shape [56,3,H,W]

grid = make_grid(all_images, nrow=14, normalize=True, value_range=(-1,1), padding=2)

plt.figure(figsize=(24, 6))
plt.imshow(grid.permute(1,2,0).cpu().numpy())
plt.axis('off')

# Labeling groups: 
# Noisy: columns 0-3
# Blank at col 4
# Grayscale: columns 5-8
# Blank at col 9
# Sketch: columns 10-13
plt.text((0+3)/2/14, 1.12, 'Noisy', transform=plt.gca().transAxes, ha='center', va='top', fontsize=14)
plt.text((5+8)/2/14, 1.12, 'Grayscale', transform=plt.gca().transAxes, ha='center', va='top', fontsize=14)
plt.text((10+13)/2/14, 1.12, 'Sketch', transform=plt.gca().transAxes, ha='center', va='top', fontsize=14)

col_labels = ['Original', 'Degraded', 'f(Degraded)', 'f(f(Degraded))']

# Noisy group (cols 0-3)
for gi, label in enumerate(col_labels):
    x_pos = (gi + 0.5)/14.0
    plt.text(x_pos, 1.02, label, transform=plt.gca().transAxes, ha='center', va='bottom', fontsize=10)

# Grayscale group (cols 5-8)
for gi, label in enumerate(col_labels):
    x_pos = (gi + 5 + 0.5)/14.0
    plt.text(x_pos, 1.02, label, transform=plt.gca().transAxes, ha='center', va='bottom', fontsize=10)

# Sketch group (cols 10-13)
for gi, label in enumerate(col_labels):
    x_pos = (gi + 10 + 0.5)/14.0
    plt.text(x_pos, 1.02, label, transform=plt.gca().transAxes, ha='center', va='bottom', fontsize=10)

plt.subplots_adjust(top=0.85)
plt.savefig("figure5.png", dpi=300)
plt.show()
