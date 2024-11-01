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

# Create output directories
os.makedirs('samples', exist_ok=True)
os.makedirs('reconstructions', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)  # 28->14
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)          # 14->7
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)         # 7->4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0)         # 4->1
        
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0)  # 1->4
        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)  # 4->7
        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 7->14
        self.convt4 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)  # 14->28
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.convt2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.convt3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.convt4(x)
        x = self.tanh(x)
        return x

class IGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
def sample_frequency_noise(real_batch):
    """Sample noise with frequency statistics of real data."""
    shape = real_batch.shape
    x_flat = real_batch.view(shape[0], -1)
    
    # Compute FFT
    fft = torch.fft.fft(x_flat.float())
    
    # Calculate statistics in frequency domain
    mean = torch.mean(fft, dim=0)
    std = torch.std(fft, dim=0)
    
    # Generate noise in frequency domain
    noise = torch.randn_like(fft) * std + mean
    
    # Convert back to spatial domain
    noise = torch.fft.ifft(noise).real
    
    # Reshape and normalize properly
    noise = noise.view(*shape)
    
    # Scale to match input range [-1, 1]
    min_val = noise.min()
    max_val = noise.max()
    noise = 2 * (noise - min_val) / (max_val - min_val) - 1
    
    return noise

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
    # Reconstruction loss
    fx = model(x)
    loss_rec = F.l1_loss(fx, x) * 20
    
    # First application of f
    fz = model(z)
    
    # Idempotence loss uses model_copy for second application
    f_fz = model_copy(fz)
    loss_idem = F.l1_loss(f_fz, fz) * 20
    
    # Tightness loss with frozen first application
    f_z = fz.detach()  # Detach to stop gradients
    ff_z = model(f_z)  # Use original model, not copy
    raw_tight_loss = -F.l1_loss(ff_z, f_z)
    loss_tight = torch.tanh(raw_tight_loss / (1.5 * loss_rec.detach())) * (1.5 * loss_rec.detach()) * 2.5
    
    total_loss = loss_rec + loss_idem + loss_tight
    
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
            
            # Update model copy after optimization step
            model_copy.load_state_dict(model.state_dict())
            
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
    losses, val_losses = train_ign(model, train_loader, val_loader, device=device, total_iterations=3000)
    
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
