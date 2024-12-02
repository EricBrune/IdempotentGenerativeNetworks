import optuna

from ign_common import *
from ign_mnist import *

def objective(trial):
    # Hyperparameters to search
    # Training parameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    
    # Loss weights
    lambda_rec = trial.suggest_float('lambda_rec', 10.0, 30.0)
    lambda_idem = trial.suggest_float('lambda_idem', 10.0, 30.0)
    lambda_tight = trial.suggest_float('lambda_tight', 1.0, 5.0)
    
    # Tightness loss clamp ratio
    tightness_clamp_ratio = trial.suggest_float('tightness_clamp_ratio', 1.0, 3.0)
    
    # MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # # Celeb A
    # image_size = 64
    # transform = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.CenterCrop(image_size),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])

    # train_dataset = datasets.CelebA(root='./data_celeb', split="train", transform=transform, download=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # val_dataset = datasets.CelebA(root='./data_celeb', split="valid", transform=transform, download=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = IGN()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses, val_losses = train_ign(
        model,
        IGN,
        train_loader,
        val_loader,
        device=device,
        total_iterations=500,
        lr=lr,
        lambda_rec=lambda_rec,
        lambda_idem=lambda_idem,
        lambda_tight=lambda_tight,
        tightness_clamp_ratio=tightness_clamp_ratio
    )

    total_val_loss = val_losses['total']  # Total validation losses
    
    final_val_loss = total_val_loss[-1]  # Last recorded loss
    # final_val_loss = sum(total_val_loss) / len(total_val_loss)  # Average validation loss over all recorded iterations

    return final_val_loss

# Start the hyperparameter optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Output the best hyperparameters
print("Best hyperparameters:")
print(study.best_params)