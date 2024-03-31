import torch
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
import os
import timm

from train import train
from plot_builder import plot_losses, plot_accuracies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataset, _ = random_split(dataset, [len(dataset) - 5000, 5000])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

models = [
    ('ResNet18', timm.create_model('resnet18', pretrained=True)),
    ('ResNet50', timm.create_model('resnet50', pretrained=True)),
    ('EfficientNet', timm.create_model('efficientnet_b0', pretrained=True)),
    ('ViT', timm.create_model('vit_base_patch16_224', pretrained=True)),
    ('VGGNet', timm.create_model('vgg11', pretrained=True)),
    ('DenseNet', timm.create_model('densenet121', pretrained=True))
]

criterion = nn.CrossEntropyLoss()
num_epochs = 5
lr = 0.0001

results = {}

for model_name, model in models:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses, epochs, accuracies = train(model, train_loader, criterion, optimizer, device, num_epochs)
    results[model_name] = {'losses': losses, 'epochs': epochs, 'accuracies': accuracies}

os.makedirs('results', exist_ok=True)

for model_name, data in results.items():
    np.savez(f'results/{model_name}_results.npz', losses=data['losses'], epochs=data['epochs'], accuracies=data['accuracies'])

plot_losses(results)
plot_accuracies(results)

