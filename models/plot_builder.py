import matplotlib.pyplot as plt

# results = {
#     'ResNet18': {'losses': [1.6998418815188938, 0.42324265739652844], 'epochs': [1, 2]},
#     'ResNet50': {'losses': [1.689236854256524, 0.3228870271788703], 'epochs': [1, 2]},
#     'EfficientNet': {'losses': [0.656188617808289, 0.11389627428981992], 'epochs': [1, 2]},
#     'ViT': {'losses': [0.23428887585136626, 0.10356608468770152], 'epochs': [1, 2]},
#     'VGGNet': {'losses': [0.5293657882160611, 0.16392043011718327], 'epochs': [1, 2]},
#     'DenseNet': {'losses': [0.6854921842310163, 0.11557560107575522], 'epochs': [1, 2]}
# }

def plot_losses(results):
    plt.figure(figsize=(10, 6))
    for model_name, data in results.items():
        # available flags for customizing: linestyle="--", linewidth=2, marker,
        plt.plot(data['epochs'], data['losses'], label=model_name, marker=".")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_accuracies(results):
    plt.figure(figsize=(10, 6))
    for model_name, data in results.items():
        plt.plot(data['epochs'], data['accuracies'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

plot_losses(results)
