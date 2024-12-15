import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from ign_common_mods import *

# Changes based on user's new request:
# For each dataset-loss cell (4x3 grid):
# - We will have reconstructions on the left (3 rows x 2 columns: input on the left, output on the right)
# - Samples on the right (3 rows x 2 columns: noise on the left, sample on the right)
#
# So each cell now has 3 rows and 4 columns total:
#   Row example: [recon_input, recon_output, sample_noise, sample_output]
#
# We'll label the left block as "Reconstructions" and the right block as "Samples".
# We'll keep the same dataset/loss layout: 4 rows (CelebA, FashionMNIST, MNIST, S2) and 3 columns (L1, L2, LPIPS).
#
# Note: Adjust paths and imports as needed. Ensure ign_mnist.py and ign_celeba.py are accessible.

def get_celeba_val_loader(image_size=64):
    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    local_data_path = '/home/e/b/ebrune/final_project/Idempotent-Generative-Network/celeba/img_align_celeba'
    orig_set = datasets.ImageFolder(root=os.path.join(local_data_path, ''), transform=transform)
    n = len(orig_set)
    n_test = int(0.05 * n)
    val_dataset = Subset(orig_set, range(n_test))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    return val_loader

def get_s2_val_loader(image_size=64):
    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    local_data_path = '/home/e/b/ebrune/FireSR_pngs/dataset/S2/'
    orig_set = datasets.ImageFolder(root=os.path.join(local_data_path, ''), transform=transform)
    n = len(orig_set)
    n_test = int(0.05 * n)
    val_dataset = Subset(orig_set, range(n_test))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    return val_loader

def get_mnist_val_loader():
    from torchvision import datasets
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    return val_loader

def get_fashionmnist_val_loader():
    from torchvision import datasets
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_dataset = datasets.FashionMNIST(root='./data_fmnist', train=False, transform=transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    return val_loader

def load_val_loader(dataset):
    if dataset == 'CelebA':
        return get_celeba_val_loader(64)
    elif dataset == 'S2':
        return get_s2_val_loader(64)
    elif dataset == 'MNIST':
        return get_mnist_val_loader()
    elif dataset == 'FashionMNIST':
        return get_fashionmnist_val_loader()
    else:
        raise ValueError(f"Unknown dataset {dataset}")

def load_model(ckpt_path, device, dataset):
    if dataset in ['MNIST', 'FashionMNIST']:
        from ign_mnist import IGN as IGN_Model
        model = IGN_Model().to(device)
    elif dataset in ['CelebA', 'S2']:
        from ign_celeba import IGN as IGN_Model
        model = IGN_Model().to(device)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' not in ckpt:
        raise ValueError(f"'model_state_dict' not found in {ckpt_path}")
    state_dict = ckpt['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

def compute_tightness_loss(model, device, dataset, rec_loss_type='L1', 
                           lambda_rec=1.0, lambda_idem=1.0, lambda_tight=1.0, loss_fn_vgg=None):
    """
    Compute the average tightness loss over a batch of validation data for selecting the checkpoint
    with the lowest tightness loss.
    """
    # Load some validation data
    val_loader = load_val_loader(dataset)
    model_copy = type(model)().to(device)
    model_copy.eval()

    # Take one batch from val_loader
    batch = next(iter(val_loader))
    x = batch[0].to(device)
    # Generate noise z from x
    z = sample_frequency_noise(x, device=device)

    # Compute losses
    with torch.no_grad():
        loss_dict = compute_losses(model, model_copy, x, z, device, rec_loss_type,
                                   lambda_rec, lambda_idem, lambda_tight, loss_fn_vgg)
    # Return the tightness loss
    return loss_dict['total']


def to_img(x):
    x = (x * 0.5) + 0.5
    x = x.clamp(0,1)
    img = x.permute(1,2,0).cpu().numpy()
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:,:,0]
    return img

def visualize_noise(z):
    z = z.clone()
    z -= z.min()
    z /= (z.max() + 1e-7)
    img = z.permute(1,2,0).cpu().numpy()
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:,:,0]
    return img

def get_all_val_images(dataset, device='cuda'):
    val_loader = load_val_loader(dataset)
    all_val_images = []
    for batch in val_loader:
        all_val_images.append(batch[0])
    all_val_images = torch.cat(all_val_images, dim=0)
    return all_val_images.to(device)

def get_random_val_batch(dataset, batch_size=32, device='cuda'):
    all_val_images = get_all_val_images(dataset, device=device)
    if all_val_images.size(0) < batch_size:
        reps = (batch_size // all_val_images.size(0)) + 1
        all_val_images = all_val_images.repeat(reps,1,1,1)
    idxs = np.random.choice(all_val_images.size(0), batch_size, replace=False)
    chosen = all_val_images[idxs]
    return chosen

def generate_samples(model, dataset, num_samples=32, device='cuda'):
    # Get a template batch for frequency noise from val set
    template_batch = get_random_val_batch(dataset, batch_size=num_samples, device=device)
    noise = sample_frequency_noise(template_batch, device=device)
    with torch.no_grad():
        samples = model(noise)
    return noise, samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoints_dir = 'checkpoints'
datasets = ['CelebA', 'FashionMNIST', 'MNIST', 'S2']
losses = ['L1', 'L2', 'LPIPS']
dataset_loss_combos = [(d, l) for d in datasets for l in losses]

# Select checkpoint with most variation
chosen_checkpoints = {}
for (dataset, loss_type) in dataset_loss_combos:
    folder_name = f"{dataset}_{loss_type}"
    folder_path = os.path.join(checkpoints_dir, folder_name)

    candidates = []
    for fname in os.listdir(folder_path):
        if fname.startswith("ign_iter_") and fname.endswith(".pt"):
            iter_str = fname[len("ign_iter_"):-3]
            iteration = int(iter_str)
            if iteration > 14000:
                candidates.append((iteration, os.path.join(folder_path, fname)))

    if not candidates:
        all_ckpts = [(int(f[len("ign_iter_"):-3]), os.path.join(folder_path, f)) 
                     for f in os.listdir(folder_path) if f.startswith("ign_iter_")]
        if not all_ckpts:
            raise ValueError(f"No checkpoints found in {folder_path}")
        candidates = all_ckpts

    best_tightness_loss = None
    best_ckpt = None
    for iteration, ckpt_path in candidates:
        model = load_model(ckpt_path, device, dataset)
        # Compute tightness loss instead of variation
        tightness_val = compute_tightness_loss(
            model, 
            device, 
            dataset, 
            rec_loss_type='L1', 
            lambda_rec=10.34, 
            lambda_idem=11.51, 
            lambda_tight=3.42, 
            loss_fn_vgg=None
        )
        # Now we want to select the checkpoint with the *lowest* tightness loss
        if (best_tightness_loss is None) or (tightness_val < best_tightness_loss):
            best_tightness_loss = tightness_val
            best_ckpt = ckpt_path


    chosen_checkpoints[(dataset, loss_type)] = best_ckpt

all_images = {}

# For each chosen checkpoint:
# - random val batch -> reconstruct
# - pick 3 random images for reconstruction pairs (orig, rec)
# - generate samples -> pick 3 random samples (noise, sample)
# Layout in each cell (3 rows x 4 columns):
# Each row: [recon_orig, recon_out, sample_noise, sample_out]

for (dataset, loss_type), ckpt_path in chosen_checkpoints.items():
    model = load_model(ckpt_path, device, dataset)

    val_batch = get_random_val_batch(dataset, batch_size=32, device=device)
    with torch.no_grad():
        recon_batch = model(val_batch)
    recon_idxs = np.random.choice(32, 3, replace=False)

    noise, samples = generate_samples(model, dataset, num_samples=32, device=device)
    sample_idxs = np.random.choice(32, 3, replace=False)

    # Build the image grid for this cell
    rows = []
    for r_i, s_i in zip(recon_idxs, sample_idxs):
        orig_img = to_img(val_batch[r_i])
        rec_img = to_img(recon_batch[r_i])
        noise_img = visualize_noise(noise[s_i])
        samp_img = to_img(samples[s_i])

        # Convert all to (H,W,3) if needed
        def ensure_3d(im):
            if im.ndim == 2:
                im = np.expand_dims(im, axis=2)
            return im
        orig_img = ensure_3d(orig_img)
        rec_img = ensure_3d(rec_img)
        noise_img = ensure_3d(noise_img)
        samp_img = ensure_3d(samp_img)

        row_im = np.concatenate([orig_img, rec_img, noise_img, samp_img], axis=1)
        rows.append(row_im)
    full_cell = np.concatenate(rows, axis=0)

    all_images[(dataset, loss_type)] = full_cell

# Create final figure: 4 rows (datasets), 3 columns (losses)
fig, axs = plt.subplots(len(datasets), len(losses), figsize=(12,16))

for i, dataset in enumerate(datasets):
    for j, loss_type in enumerate(losses):
        ax = axs[i,j]
        cell_img = all_images[(dataset, loss_type)]
        ax.imshow(cell_img.squeeze(), cmap='gray' if cell_img.shape[2]==1 else None)
        ax.axis('off')
        ax.set_title(f"{dataset}_{loss_type}", fontsize=10)

        # Add labels for "Reconstructions" and "Samples"
        # Recon pairs occupy columns [0:2], samples occupy columns [2:4]
        # Put text above first row of those columns
        h = cell_img.shape[0]
        w = cell_img.shape[1]
        # Each row has 4 images horizontally. Let's assume they have same width each.
        # If images are uniform size from that dataset: 
        # width per image = w/4
        img_w = w/4
        # Place "Reconstructions" over columns 0 and 1 at a small offset
        # ax.text(img_w*0.5, 15, "Reconstructions", color='white', fontsize=10,
        #         ha='center', bbox=dict(facecolor='black', alpha=0.5, pad=2))
        # # Place "Samples" over columns 2 and 3
        # ax.text(img_w*2.5, 15, "Samples", color='white', fontsize=10,
        #         ha='center', bbox=dict(facecolor='black', alpha=0.5, pad=2))

plt.tight_layout()
plt.savefig("final_plot.png", dpi=300)
plt.close()
