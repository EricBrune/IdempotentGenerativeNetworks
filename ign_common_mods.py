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
import lpips

def get_mean_std_from_batch(batch):
    f = torch.fft.fft2(batch)
    real_mean = f.real.mean(dim=0)
    real_std = f.real.std(dim=0)
    imag_mean = f.imag.mean(dim=0)
    imag_std = f.imag.std(dim=0)

    # Clamp std values to a small positive number
    epsilon = 1e-8
    real_std = torch.clamp(real_std, min=epsilon)
    imag_std = torch.clamp(imag_std, min=epsilon)

    return real_mean, real_std, imag_mean, imag_std


def get_noise_from_mean_std(batch_size, real_mean, real_std, imag_mean, imag_std):
    freq_real = [torch.normal(real_mean, real_std) for _ in range(batch_size)]
    freq_real = torch.stack(freq_real, dim=0)
    freq_imag = [torch.normal(imag_mean, imag_std) for _ in range(batch_size)]
    freq_imag = torch.stack(freq_imag, dim=0)
    freq = torch.complex(freq_real, freq_imag)
    noise = torch.fft.ifft2(freq)
    return noise.real

def sample_frequency_noise(real_batch, device = "cuda"):
    batch_size = real_batch.shape[0]
    mean_std = get_mean_std_from_batch(real_batch)
    z = get_noise_from_mean_std(batch_size, *mean_std)
    z = z.to(device, memory_format=torch.contiguous_format)
    return z

def save_image_grid(tensor, filename, nrow=8):
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, filename)

def generate_samples_with_iterations(model, val_loader, num_samples=8, num_iterations=4, device='cuda'):
    model.eval()
    with torch.no_grad():
        x = next(iter(val_loader))[0].to(device)
        z = sample_frequency_noise(x[:num_samples])
        samples = [z]
        current = z
        for _ in range(num_iterations):
            current = model(current)
            samples.append(current)
        return samples

def compute_losses(model, model_copy, x, z, device, rec_loss_type, 
                   lambda_rec, lambda_idem, lambda_tight, loss_fn_vgg=None):
    batch_size = x.shape[0]

    model_copy.load_state_dict(model.state_dict()) 
    fx = model(x)
    fz = model(z)  
    f_z = fz.detach()  
    ff_z = model(f_z)  
    f_fz = model_copy(fz)

    # Idempotence loss
    loss_idem = F.l1_loss(f_fz, fz, reduction='mean')

    # Reconstruction loss
    if rec_loss_type == 'L1':
        rec_vals = F.l1_loss(fx, x, reduction='none')
        loss_rec = rec_vals.view(batch_size, -1).mean(dim=1)
    elif rec_loss_type == 'L2':
        rec_vals = F.mse_loss(fx, x, reduction='none')
        loss_rec = rec_vals.view(batch_size, -1).mean(dim=1)
    elif rec_loss_type == 'LPIPS':
        lpips_vals = loss_fn_vgg(x, fx)
        loss_rec = lpips_vals.view(batch_size)
    else:
        raise ValueError("Invalid rec_loss_type. Choose from 'L1', 'L2', or 'LPIPS'.")

    # Tightness loss
    raw_tight_loss = -F.l1_loss(ff_z, f_z, reduction='none').reshape(batch_size, -1).mean(dim=1)
    # Using a as a scaling factor or other means can be reintroduced if needed:
    loss_tight = F.tanh(raw_tight_loss / loss_rec.detach()) * loss_rec.detach()

    loss_rec = loss_rec.mean()
    loss_tight = loss_tight.mean()

    # Weighted sum of losses
    total_loss = lambda_rec * loss_rec + lambda_idem * loss_idem + lambda_tight * loss_tight

    return {
        'total': total_loss,
        'reconstruction': loss_rec,
        'idempotence': loss_idem,
        'tightness': loss_tight
    }

def compute_validation_losses(model, model_copy, val_loader, device, rec_loss_type, 
                              lambda_rec, lambda_idem, lambda_tight, loss_fn_vgg):
    model.eval()
    val_losses = {'reconstruction': 0.0, 'idempotence': 0.0, 'tightness': 0.0, 'total': 0.0}
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            batch_size = x.size(0)
            z = sample_frequency_noise(x, device=device)
            loss_dict = compute_losses(model, model_copy, x, z, device, rec_loss_type, 
                                       lambda_rec, lambda_idem, lambda_tight, loss_fn_vgg)
            for k in val_losses.keys():
                val_losses[k] += loss_dict[k].item() * batch_size
            total_samples += batch_size
    for k in val_losses.keys():
        val_losses[k] /= total_samples
    return val_losses

def train_ign(
    model, train_loader, val_loader, device='cuda', total_iterations=1000, 
    rec_loss_type='L1', loss_fn_vgg=None, dataset='', lr=1e-4, 
    lambda_rec=10.34, lambda_idem=11.51, lambda_tight=3.42, tightness_clamp_ratio=1.31
):
    model = model.to(device)
    model_copy = type(model)().to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model_copy = nn.DataParallel(model_copy)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    # Create directories
    base_dir = f"{dataset}_{rec_loss_type}"
    os.makedirs(os.path.join('samples', base_dir), exist_ok=True)
    os.makedirs(os.path.join('reconstructions', base_dir), exist_ok=True)
    os.makedirs(os.path.join('checkpoints', base_dir), exist_ok=True)

    losses = {'reconstruction': [], 'idempotence': [], 'tightness': [], 'total': []}
    val_losses_history = {'reconstruction': [], 'idempotence': [], 'tightness': [], 'total': []}

    iteration = 0
    pbar = tqdm(total=total_iterations, desc="Training", leave=True)
    while iteration <= total_iterations:
        model.train()
        for batch in train_loader:
            if iteration > total_iterations:
                break

            x = batch[0].to(device)
            z = sample_frequency_noise(x, device=device)
            loss_dict = compute_losses(
                model, model_copy, x, z, device, rec_loss_type, 
                lambda_rec, lambda_idem, lambda_tight, loss_fn_vgg
            )

            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

            for k, v in loss_dict.items():
                losses[k].append(v.item())

            pbar.update(1)
            pbar.set_postfix({
                'Rec': f"{loss_dict['reconstruction']:.4f}",
                'Idem': f"{loss_dict['idempotence']:.4f}",
                'Tight': f"{loss_dict['tightness']:.4f}",
            })

            # Compute validation, save checkpoints & images every 500 iterations
            if iteration % 1000 == 0 and iteration != 0:
                val_losses = compute_validation_losses(
                    model, model_copy, val_loader, device, rec_loss_type,
                    lambda_rec, lambda_idem, lambda_tight, loss_fn_vgg
                )
                for k, v in val_losses.items():
                    val_losses_history[k].append(v)

                print(f"\nIteration {iteration}/{total_iterations}")
                print("Training Losses:")
                print(f"  Rec Loss: {loss_dict['reconstruction']:.4f}")
                print(f"  Idem Loss: {loss_dict['idempotence']:.4f}")
                print(f"  Tight Loss: {loss_dict['tightness']:.4f}")
                print("Validation Losses:")
                print(f"  Rec Loss: {val_losses['reconstruction']:.4f}")
                print(f"  Idem Loss: {val_losses['idempotence']:.4f}")
                print(f"  Tight Loss: {val_losses['tightness']:.4f}")

                # Save samples
                samples = generate_samples_with_iterations(model, val_loader, device=device)
                samples_grid = torch.cat([s[:8] for s in samples], dim=0)
                sample_path = os.path.join('samples', base_dir, f'iter_{iteration}.png')
                save_image_grid(samples_grid, sample_path)

                # Save reconstructions
                model.eval()
                with torch.no_grad():
                    val_x = next(iter(val_loader))[0][:8].to(device)
                    fx = model(val_x)
                    recon_samples = torch.cat([val_x, fx], dim=0)
                    recon_path = os.path.join('reconstructions', base_dir, f'iter_{iteration}_recon.png')
                    save_image_grid(recon_samples, recon_path, nrow=8)

                # Save checkpoint
                ckpt_path = os.path.join('checkpoints', base_dir, f'ign_iter_{iteration}.pt')
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'val_losses_history': val_losses_history,
                }, ckpt_path)

            iteration += 1

    pbar.close()
    return losses, val_losses_history
