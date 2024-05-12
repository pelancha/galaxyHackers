import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import os
import timm
import numpy as np
import data.data as data
import data.segmentation as segmentation
import metrics.metrics as metrics
import argparse
import torch_optimizer as optimizer

from models.train import train, validate, continue_training

import models.spinalnet_resnet as spinalnet_resnet
import models.spinalnet_vgg as spinalnet_vgg
import models.vitL16 as vitL16
import models.alexnet_vgg as alexnet_vgg
import models.resnet18 as resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = data.create_dataloaders()
train_loader = dataloader['train']
val_loader = dataloader['val']

models = [
    ('ResNet18', resnet18.load_model()),
    ('ResNet50', timm.create_model('resnet50', pretrained=True, num_classes=1)),
    ('EfficientNet', timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)),
    ('ViT', timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)),
    ('VGGNet', timm.create_model('vgg11', pretrained=True, num_classes=1)),
    ('DenseNet', timm.create_model('densenet121', pretrained=True, num_classes=1)),
    ('SpinalNet_ResNet', spinalnet_resnet.load_model()),
    ('SpinalNet_VGG', spinalnet_vgg.load_model()),
    ('ViTL16', vitL16.load_model()),
    ('AlexNet_VGG', alexnet_vgg.load_model())
]

optimizers = [
    ('SGD', optim.SGD),
    ('Adam', optim.Adam),
    ('Adagrad', optim.Adagrad),
    ('RMSprop', optim.RMSprop),
    ('Adadelta', optim.Adadelta),
    ('AdamW', optim.AdamW),
    ('DiffGrad', optimizer.DiffGrad),
    ('LBFGS', optim.LBFGS)
]

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--models', nargs='+', default=['ResNet18', 'ResNet50', 'EfficientNet', 'ViT', 'VGGNet', 'DenseNet', 'SpinalNet_ResNet', 'SpinalNet_VGG', 'ViTL16', 'AlexNet_VGG'],
                    help='List of models to train (default: all)')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimizer (default: 0.0001)')
parser.add_argument('--mm', type=float, default=0.9, help='Momentum for optimizer (default: 0.9)')
parser.add_argument('--optimizer', choices=[name for name, _ in optimizers], default='Adam', help='Optimizer to use (default: Adam)')

args = parser.parse_args()

selected_models = [(model_name, model) for model_name, model in models if model_name in args.models]

num_epochs = args.epochs
lr = args.lr
momentum = args.mm
optimizer_name = args.optimizer

# criterion = nn.CrossEntropyLoss()

criterion = nn.BCELoss()

results = {}
val_results = {}
test_loader = dataloader["test_dr5"]
classes = ('random', 'clusters')

for model_name, model in selected_models:
    optimizer_class = dict(optimizers)[optimizer_name]
    optimizer = optimizer_class(model.parameters(), lr=lr, momentum=momentum) if optimizer_name in ['SGD', 'RMSprop'] else optimizer_class(model.parameters(), lr=lr)

    losses, epochs, accuracies, val_losses, val_accuracies, model_X = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
    results[model_name] = {'losses': losses, 'epochs': epochs, 'accuracies': accuracies}
    val_results[model_name] = {'val_losses': val_losses, 'val_epochs': epochs, 'val_accuracies': val_accuracies}

# filepath = "/content/trained_models/ResNet_epoch_3.pth"
#
# for model_name, model in models:
#     losses, epochs, accuracies = continue_training(model, train_loader, criterion, optimizer, device, num_epochs, filepath)
#     results[model_name].update({'losses': losses, 'epochs': epochs, 'accuracies': accuracies})
#

    os.makedirs('results', exist_ok=True)
    for model_name, data in results.items():
        np.savez(f'results/{model_name}_{optimizer_name}_results.npz', losses=data['losses'], epochs=data['epochs'], accuracies=data['accuracies'])

    for model_name, data in val_results.items():
        np.savez(f'results/{model_name}_{optimizer_name}_val_results.npz', losses=data['val_losses'], epochs=data['val_epochs'], accuracies=data['val_accuracies'])

    y_pred, y_probs, y_true = [], [], []

    #device = torch.device("cpu")
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model_X(inputs) # Feed Network
        y_probs.extend(output.data.cpu().numpy().ravel())
        output = [1 if (i > 0.9) else 0 for i in output]
        y_pred.extend(output) # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth


    metrics.modelPerformance(model_name, optimizer_name, y_true, y_pred, y_probs, classes, results[model_name], val_results[model_name])

segmentation.saveSegMaps(selected_models, optimizer_name)
segmentation.saveBigSegMap(selected_models, optimizer_name)
