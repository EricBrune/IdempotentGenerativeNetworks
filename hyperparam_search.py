import optuna

from ign_common_mods import *
from ign_celeba import *
#from ign_celeba_mods import *

def objective(trial):
    # Hyperparameters to search
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    lambda_rec = trial.suggest_float('lambda_rec', 1.0, 30.0)
    lambda_idem = trial.suggest_float('lambda_idem', 1.0, 30.0)
    lambda_tight = trial.suggest_float('lambda_tight', 1.0, 30.0)
    tightness_clamp_ratio = trial.suggest_float('tightness_clamp_ratio', 1.0, 3.0)

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.CenterCrop(64),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    local_data_path = '/home/e/b/ebrune/final_project/Idempotent-Generative-Network/celeba'
    orig_set = datasets.ImageFolder(
        root=os.path.join(local_data_path, 'img_align_celeba'),
        transform=transform
    )
    n = len(orig_set)
    n_test = int(0.05 * n)
    val_dataset = torch.utils.data.Subset(orig_set, range(n_test))
    train_dataset = torch.utils.data.Subset(orig_set, range(n_test, n))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    model = IGN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    losses, val_losses = train_ign(
        model, train_loader, val_loader, device=device, total_iterations=1000,
        rec_loss_type="L1", lr=lr, lambda_rec=lambda_rec, lambda_idem=lambda_idem,
        lambda_tight=lambda_tight, tightness_clamp_ratio=tightness_clamp_ratio
    )

    return val_losses['total'][-1]


# Start the hyperparameter optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Output the best hyperparameters
print("Best hyperparameters:")
print(study.best_params)