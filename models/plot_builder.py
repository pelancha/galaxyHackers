import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_losses(results, val_results):
    os.makedirs("plots", exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.figure(figsize=(10, 6))
    for model_name, data in results.items():
        # available flags for customizing: linestyle="--", linewidth=2, marker,
        plt.plot(data['epochs'], data['losses'], label=model_name, marker=".")
    for model_name, data in val_results.items():
        plt.plot(data['val_epochs'], data['val_losses'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join("plots", f'losses_plot_{now}.png'))
    plt.show()

def plot_accuracies(results, val_results):
    os.makedirs("plots", exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.figure(figsize=(10, 6))
    for model_name, data in results.items():
        plt.plot(data['epochs'], data['accuracies'], label=model_name)
    for model_name, data in val_results.items():
        plt.plot(data['val_epochs'], data['val_accuracies'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join("plots", f'accuracies_plot_{now}.png'))
    plt.show()
