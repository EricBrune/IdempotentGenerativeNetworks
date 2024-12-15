import re
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

# Usage: python plot_losses.py run_logs/28095_output.txt

log_file = sys.argv[1]

# Data structure:
# data[dataset][rec_loss_type]['iterations'] = []
# data[dataset][rec_loss_type]['train_reconstruction'] = []
# data[dataset][rec_loss_type]['train_idempotence'] = []
# data[dataset][rec_loss_type]['train_tightness'] = []
# data[dataset][rec_loss_type]['val_reconstruction'] = []
# data[dataset][rec_loss_type]['val_idempotence'] = []
# data[dataset][rec_loss_type]['val_tightness'] = []
# best_epoch[dataset][rec_loss_type] = (iteration, total_loss)

data = defaultdict(lambda: defaultdict(lambda: {
    'iterations': [],
    'train_reconstruction': [],
    'train_idempotence': [],
    'train_tightness': [],
    'val_reconstruction': [],
    'val_idempotence': [],
    'val_tightness': []
}))

best_epoch = defaultdict(lambda: defaultdict(lambda: (None, float('inf'))))

current_dataset = None
current_loss = None
current_iteration = None

with open(log_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("Running"):
            # Example: "Running MNIST with L1 loss"
            m = re.match(r"Running\s+(\S+)\s+with\s+(\S+)\s+loss", line)
            if m:
                current_dataset = m.group(1)
                current_loss = m.group(2)
        elif line.startswith("Iteration"):
            # Example: "Iteration 500/20000"
            m = re.match(r"Iteration\s+(\d+)/(\d+)", line)
            if m:
                current_iteration = int(m.group(1))
        elif line.startswith("Training Losses:"):
            rec_line = next(f).strip()
            idem_line = next(f).strip()
            tight_line = next(f).strip()

            tm = re.match(r"Rec Loss:\s+([\-\d\.]+)", rec_line)
            im = re.match(r"Idem Loss:\s+([\-\d\.]+)", idem_line)
            rm = re.match(r"Tight Loss:\s+([\-\d\.]+)", tight_line)

            train_rec_loss = float(tm.group(1)) if tm else None
            train_idem_loss = float(im.group(1)) if im else None
            train_tight_loss = float(rm.group(1)) if rm else None

            d = data[current_dataset][current_loss]
            d['iterations'].append(current_iteration)
            d['train_reconstruction'].append(train_rec_loss)
            d['train_idempotence'].append(train_idem_loss)
            d['train_tightness'].append(train_tight_loss)

        elif line.startswith("Validation Losses:"):
            val_rec_line = next(f).strip()
            val_idem_line = next(f).strip()
            val_tight_line = next(f).strip()

            vm = re.match(r"Rec Loss:\s+([\-\d\.]+)", val_rec_line)
            vim = re.match(r"Idem Loss:\s+([\-\d\.]+)", val_idem_line)
            vtm = re.match(r"Tight Loss:\s+([\-\d\.]+)", val_tight_line)

            val_rec_loss = float(vm.group(1)) if vm else None
            val_idem_loss = float(vim.group(1)) if vim else None
            val_tight_loss = float(vtm.group(1)) if vtm else None

            d = data[current_dataset][current_loss]
            d['val_reconstruction'].append(val_rec_loss)
            d['val_idempotence'].append(val_idem_loss)
            d['val_tightness'].append(val_tight_loss)

            # Calculate total validation loss
            if val_rec_loss is not None and val_idem_loss is not None and val_tight_loss is not None:
                total_loss = val_rec_loss + val_idem_loss + val_tight_loss
                # Update best epoch
                best_iter, best_loss = best_epoch[current_dataset][current_loss]
                if total_loss < best_loss:
                    best_epoch[current_dataset][current_loss] = (current_iteration, total_loss)

# Print the best epochs and total losses
print("Best epochs and total validation losses for each dataset and loss type:")
for dataset, losses in best_epoch.items():
    for loss_type, (iteration, total_loss) in losses.items():
        print(f"{dataset} ({loss_type} loss): Best epoch = {iteration}, Total loss = {total_loss:.4f}")

# Plotting
datasets = ["MNIST", "FashionMNIST", "CelebA", "S2"]
loss_types = ["L1", "L2", "LPIPS"]

fig, axes = plt.subplots(len(datasets), len(loss_types), figsize=(15, 20), sharex=True, sharey=False)

for i, ds in enumerate(datasets):
    for j, lt in enumerate(loss_types):
        ax = axes[i][j]
        if ds in data and lt in data[ds]:
            d = data[ds][lt]

            if len(d['iterations']) > 0:
                # Plot training and validation curves
                ax.plot(d['iterations'], d['train_reconstruction'], label='Train Rec', color='blue')
                if len(d['val_reconstruction']) == len(d['iterations']):
                    ax.plot(d['iterations'], d['val_reconstruction'], label='Val Rec', linestyle='--', color='blue')

                ax.plot(d['iterations'], d['train_idempotence'], label='Train Idem', color='red')
                if len(d['val_idempotence']) == len(d['iterations']):
                    ax.plot(d['iterations'], d['val_idempotence'], label='Val Idem', linestyle='--', color='red')

                ax.plot(d['iterations'], d['train_tightness'], label='Train Tight', color='green')
                if len(d['val_tightness']) == len(d['iterations']):
                    ax.plot(d['iterations'], d['val_tightness'], label='Val Tight', linestyle='--', color='green')

            ax.set_title(f"{ds} - {lt}")
            if i == len(datasets)-1:
                ax.set_xlabel("Iteration")
            if j == 0:
                ax.set_ylabel("Loss")
            ax.legend(fontsize='small')
        else:
            ax.set_title(f"{ds} - {lt}\nNo Data")

plt.tight_layout()
plt.savefig("combined_losses_plot.png", dpi=300)
plt.show()

