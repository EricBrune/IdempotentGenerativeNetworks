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

    
# def sample_frequency_noise(real_batch):
#     """Sample noise with frequency statistics of real data."""
#     shape = real_batch.shape
#     x_flat = real_batch.view(shape[0], -1)

#     # Compute FFT
#     fft = torch.fft.fft(x_flat.float())

#     # Calculate statistics in frequency domain
#     mean = torch.mean(fft, dim=0)
#     std = torch.std(fft, dim=0)

#     # Generate noise in frequency domain
#     noise = torch.randn_like(fft) * std + mean

#     # Convert back to spatial domain
#     noise = torch.fft.ifft(noise).real

#     # Reshape and normalize properly
#     noise = noise.view(*shape)

#     # Scale to match input range [-1, 1]
#     min_val = noise.min()
#     max_val = noise.max()
#     noise = 2 * (noise - min_val) / (max_val - min_val) - 1

#     return noise

def get_mean_std_from_batch(batch):
    # get mean and std of real and imaginary part of batch, used for generate noise
    f = torch.fft.fft2(batch)
    real_mean = f.real.mean(dim=0)
    real_std = f.real.std(dim=0)
    imag_mean = f.imag.mean(dim=0)
    imag_std = f.imag.std(dim=0)
    return real_mean, real_std, imag_mean, imag_std


def get_noise_from_mean_std(batch_size, real_mean, real_std, imag_mean, imag_std):
    # generate noise from mean and std of real and imaginary part
    freq_real = [torch.normal(real_mean, real_std) for _ in range(batch_size)]
    freq_real = torch.stack(freq_real, dim=0)
    freq_imag = [torch.normal(imag_mean, imag_std) for _ in range(batch_size)]
    freq_imag = torch.stack(freq_imag, dim=0)
    freq = torch.complex(freq_real, freq_imag)
    noise = torch.fft.ifft2(freq)
    return noise.real

def sample_frequency_noise(real_batch):
    batch_size = real_batch.shape[0]
    mean_std = get_mean_std_from_batch(real_batch)
    z = get_noise_from_mean_std(batch_size, *mean_std)
    z = z.to(device, memory_format=torch.contiguous_format)

    return z

def save_image_grid(tensor, filename, nrow=8):
    """Save a grid of images."""
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, filename)

def save_reconstruction_samples(model, val_loader, device, epoch):
    """Save reconstruction samples during training."""
    model.eval()
    with torch.no_grad():
        x = next(iter(val_loader))[0][:8].to(device)
        fx = model(x)

        # Save real vs reconstructed
        samples = torch.cat([x, fx], dim=0)
        save_image_grid(samples, f'reconstructions/epoch_{epoch}_recon.png', nrow=8)

def generate_samples_with_iterations(model, val_loader, num_samples=8, num_iterations=4, device='cuda'):
    """Generate samples showing multiple applications of f."""
    model.eval()
    with torch.no_grad():
        # Get real batch for frequency statistics
        x = next(iter(val_loader))[0].to(device)

        # Generate frequency-matched noise
        z = sample_frequency_noise(x[:num_samples])

        # Store all iterations including original z
        samples = [z]

        # Generate f(z), f(f(z)), etc.
        current = z
        for _ in range(num_iterations):
            current = model(current)
            samples.append(current)

        return samples



def compute_losses(model, model_copy, x, z, device):
    """Compute all loss terms and return individually for monitoring."""
    batch_size = x.shape[0]
    
    model_copy.load_state_dict(model.state_dict()) # Update model copy after optimization step
    fx = model(x)
    fz = model(z) # First application of f
    f_z = fz.detach()  # Detach to stop gradients
    ff_z = model(f_z)  # Use original model, not copy
    f_fz = model_copy(fz)

    lambda_rec = 20
    lambda_idem = 20
    lambda_tight = 2.5
    a = 1.5

    loss_idem = F.l1_loss(f_fz, fz, reduction='mean') # Idempotence loss uses model_copy for second application
    loss_rec = F.l1_loss(fx, x, reduction='none').reshape(batch_size, -1).mean(dim=1) # Reconstruction loss
    raw_tight_loss = -F.l1_loss(ff_z, f_z, reduction='none').reshape(batch_size, -1).mean(dim=1) # Tightness loss with frozen first application
    loss_tight = F.tanh(raw_tight_loss / (a * loss_rec.detach())) * a * loss_rec.detach() # Smooth loss

    loss_rec = loss_rec.mean()
    loss_tight = loss_tight.mean()

    total_loss = lambda_rec*loss_rec + lambda_idem*loss_idem + lambda_tight*loss_tight

    return {
        'total': total_loss,
        'reconstruction': loss_rec,
        'idempotence': loss_idem,
        'tightness': loss_tight
    }

def compute_validation_losses(model, model_copy, val_loader, device):
    """Compute validation losses over the validation dataset."""
    model.eval()
    val_losses = {'reconstruction': 0.0, 'idempotence': 0.0, 'tightness': 0.0, 'total': 0.0}
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            batch_size = x.size(0)
            z = sample_frequency_noise(x)
            loss_dict = compute_losses(model, model_copy, x, z, device)
            for k in val_losses.keys():
                val_losses[k] += loss_dict[k].item() * batch_size
            total_samples += batch_size
    for k in val_losses.keys():
        val_losses[k] /= total_samples
    return val_losses

def train_ign(model, train_loader, val_loader, device='cuda', total_iterations=1000):
    """Train the IGN model."""
    model = model.to(device)
    model_copy = IGN().to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model_copy = nn.DataParallel(model_copy)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Track losses
    losses = {'reconstruction': [], 'idempotence': [], 'tightness': [], 'total': []}
    val_losses_history = {'reconstruction': [], 'idempotence': [], 'tightness': [], 'total': []}

    iteration = 0
    total_iterations = int(total_iterations)
    pbar = tqdm(total=total_iterations, desc="Training", leave=True)
    while iteration < total_iterations:
        model.train()

        for batch in train_loader:
            if iteration >= total_iterations:
                break

            x = batch[0].to(device)
            z = sample_frequency_noise(x)

            # Compute losses
            loss_dict = compute_losses(model, model_copy, x, z, device)

            # Optimize
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

            

            # Log losses
            for k, v in loss_dict.items():
                losses[k].append(v.item())

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Rec': f"{loss_dict['reconstruction'].item():.4f}",
                'Idem': f"{loss_dict['idempotence'].item():.4f}",
                'Tight': f"{loss_dict['tightness'].item():.4f}",
            })

            # Compute and print validation losses every 100 iterations
            if iteration % 100 == 0 and iteration != 0:
                val_losses = compute_validation_losses(model, model_copy, val_loader, device)

                # Log validation losses
                for k, v in val_losses.items():
                    val_losses_history[k].append(v)

                print(f"\nIteration {iteration}/{total_iterations}")
                print(f"Training Losses:")
                print(f"  Rec Loss: {loss_dict['reconstruction'].item():.4f}")
                print(f"  Idem Loss: {loss_dict['idempotence'].item():.4f}")
                print(f"  Tight Loss: {loss_dict['tightness'].item():.4f}")
                print(f"Validation Losses:")
                print(f"  Rec Loss: {val_losses['reconstruction']:.4f}")
                print(f"  Idem Loss: {val_losses['idempotence']:.4f}")
                print(f"  Tight Loss: {val_losses['tightness']:.4f}")

                # Save samples
                samples = generate_samples_with_iterations(model, val_loader, device=device)
                samples_grid = torch.cat([s[:8] for s in samples], dim=0)
                save_image_grid(samples_grid, f'samples/iter_{iteration}.png')

                # Save reconstructions
                save_reconstruction_samples(model, val_loader, device, iteration)

                # Save checkpoint
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'val_losses_history': val_losses_history,
                }, f'checkpoints/ign_iter_{iteration}.pt')

            iteration += 1

    pbar.close()
    return losses, val_losses_history

