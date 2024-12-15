import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import lpips

# Import the common training function and utilities
from ign_common_mods import train_ign, sample_frequency_noise

def get_celeba_loaders(image_size=64):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    local_data_path = '/home/e/b/ebrune/final_project/Idempotent-Generative-Network/celeba/img_align_celeba'

    orig_set = datasets.ImageFolder(
        root=os.path.join(local_data_path, ''),
        transform=transform
    )

    n = len(orig_set)
    n_val = int(0.05 * n)
    n_test = int(0.05 * n)
    val_dataset = torch.utils.data.Subset(orig_set, range(n_val))
    test_dataset = torch.utils.data.Subset(orig_set, range(n_val, n_val + n_test))
    train_dataset = torch.utils.data.Subset(orig_set, range(n_val + n_test, n))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
    return train_loader, val_loader, test_loader

def get_s2_loaders(image_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    local_data_path = '/home/e/b/ebrune/FireSR_pngs/dataset/S2/'

    orig_set = datasets.ImageFolder(
        root=os.path.join(local_data_path, ''),
        transform=transform
    )

    n = len(orig_set)
    n_val = int(0.05 * n)
    n_test = int(0.05 * n)
    val_dataset = torch.utils.data.Subset(orig_set, range(n_val))
    test_dataset = torch.utils.data.Subset(orig_set, range(n_val, n_val + n_test))
    train_dataset = torch.utils.data.Subset(orig_set, range(n_val + n_test, n))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
    return train_loader, val_loader, test_loader

def get_mnist_loaders():
    from ign_mnist import IGN as IGN_MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    orig_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    orig_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Combine all for splitting
    full_data = torch.utils.data.ConcatDataset([orig_train, orig_test])
    n = len(full_data)
    n_val = int(0.05 * n)
    n_test = int(0.05 * n)
    val_dataset = torch.utils.data.Subset(full_data, range(n_val))
    test_dataset = torch.utils.data.Subset(full_data, range(n_val, n_val + n_test))
    train_dataset = torch.utils.data.Subset(full_data, range(n_val + n_test, n))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader, test_loader, IGN_MNIST

def get_fashionmnist_loaders():
    from ign_mnist import IGN as IGN_MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    orig_train = datasets.FashionMNIST(root='./data_fmnist', train=True, transform=transform, download=True)
    orig_test = datasets.FashionMNIST(root='./data_fmnist', train=False, transform=transform, download=True)

    full_data = torch.utils.data.ConcatDataset([orig_train, orig_test])
    n = len(full_data)
    n_val = int(0.05 * n)
    n_test = int(0.05 * n)
    val_dataset = torch.utils.data.Subset(full_data, range(n_val))
    test_dataset = torch.utils.data.Subset(full_data, range(n_val, n_val + n_test))
    train_dataset = torch.utils.data.Subset(full_data, range(n_val + n_test, n))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader, test_loader, IGN_MNIST

def evaluate_model(model, loader, device, rec_loss_type, loss_fn_vgg, dataset):
    model.eval()
    total_rec_loss = 0.0
    total_idem_loss = 0.0
    total_tight_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            outputs = model(batch)
            # Same losses as in train_ign
            if rec_loss_type == 'L1':
                rec_loss = torch.mean(torch.abs(outputs - batch))
            elif rec_loss_type == 'L2':
                rec_loss = torch.mean((outputs - batch)**2)
            else:  # LPIPS
                rec_loss = loss_fn_vgg(outputs, batch).mean()

            # Idempotence and tightness losses depend on model specifics. We assume similar to train_ign:
            # Apply model again
            outputs_2 = model(outputs)
            idem_loss = torch.mean((outputs_2 - outputs)**2)
            # Tightness: measure how close outputs are to something meaningful, here we assume same definition
            tight_loss = torch.mean(torch.abs(outputs))  # or any definition consistent with train_ign

            total_rec_loss += rec_loss.item() * batch.size(0)
            total_idem_loss += idem_loss.item() * batch.size(0)
            total_tight_loss += tight_loss.item() * batch.size(0)
            count += batch.size(0)
    return (total_rec_loss / count, total_idem_loss / count, total_tight_loss / count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['MNIST', 'FashionMNIST', 'CelebA', 'S2'],
                        help="Choose the dataset.")
    parser.add_argument('--rec_loss_type', type=str, required=True, 
                        choices=['L1','L2','LPIPS'],
                        help="Reconstruction loss type: L1, L2, or LPIPS.")
    parser.add_argument('--total_iterations', type=int, default=10000)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize LPIPS if needed
    loss_fn_vgg = None
    if args.rec_loss_type == 'LPIPS':
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # Load dataset & model
    if args.dataset == 'MNIST':
        train_loader, val_loader, test_loader, IGN_Model = get_mnist_loaders()
        model = IGN_Model()
    elif args.dataset == 'FashionMNIST':
        train_loader, val_loader, test_loader, IGN_Model = get_fashionmnist_loaders()
        model = IGN_Model()
    elif args.dataset == 'CelebA':
        from ign_celeba import IGN as IGN_CelebA
        train_loader, val_loader, test_loader = get_celeba_loaders(image_size=64)
        model = IGN_CelebA()
    elif args.dataset == 'S2':
        from ign_celeba import IGN as IGN_CelebA  # assuming same IGN architecture
        train_loader, val_loader, test_loader = get_s2_loaders(image_size=64)
        model = IGN_CelebA()

    # Train the model
    # We assume train_ign saves and returns the path or iteration of best checkpoint based on val set
    losses, val_losses, best_checkpoint = train_ign(
        model, 
        train_loader, 
        val_loader, 
        device=device, 
        total_iterations=args.total_iterations,
        rec_loss_type=args.rec_loss_type,
        loss_fn_vgg=loss_fn_vgg,
        dataset=args.dataset
    )

    # Load best checkpoint
    if best_checkpoint is not None and os.path.exists(best_checkpoint):
        print(f"Loading best checkpoint: {best_checkpoint}")
        model.load_state_dict(torch.load(best_checkpoint))
    else:
        print("Warning: No best checkpoint found, using current model weights.")

    # Test the model
    test_rec_loss, test_idem_loss, test_tight_loss = evaluate_model(
        model, test_loader, device, args.rec_loss_type, loss_fn_vgg, args.dataset
    )
    print(f"Test Losses:\nRec Loss: {test_rec_loss}\nIdem Loss: {test_idem_loss}\nTight Loss: {test_tight_loss}")

if __name__ == '__main__':
    main()
