import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import os
import copy
import time

# Config
working_path = "./"
models_epoch = f'{working_path}trained_models/'
models_state_dict = f'{working_path}state_dict/'

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, start_epoch=0):
    model.to(device)
    model.train()

    best_loss, best_accuracy = float("inf"), 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    losses, epochs, accuracies = [], [], []

    since = time.time()
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        cool_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}. Training {model.__class__.__name__} with {optimizer.__class__.__name__} optimizer', unit='batch')
        for inputs, labels in cool_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1).float()).sum().item()
            cool_progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset), acc=correct / total)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        epochs.append(epoch + 1)

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if val_loss < best_loss:
            os.makedirs(models_state_dict, exist_ok=True)
            best_loss, best_accuracy = val_loss, val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, f'{models_state_dict}/{model.__class__.__name__}_{optimizer.__class__.__name__}_weights.pth')

        os.makedirs(models_epoch, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': epoch_loss,
            'best_accuracy': epoch_acc
        }, f'{models_epoch}/{model.__class__.__name__}_{optimizer.__class__.__name__}_epoch_{epoch + 1}.pth')

    os.makedirs(models_state_dict, exist_ok=True)
    model.load_state_dict(best_model_weights)
    torch.save(best_model_weights, f'{models_state_dict}/best_{model.__class__.__name__}_{optimizer.__class__.__name__}_weights.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Loss: {:.4f}, Best Validation Accuracy: {:.4f}'.format(best_loss, best_accuracy))

    return losses, epochs, accuracies

def validate(model, val_loader, criterion, device):
    model.to(device)
    model.eval()
    val_losses, val_accuracies = [], []
    correct, total = 0, 0
    since = time.time()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_losses.append(loss.item())
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1).float()).sum().item()
    val_loss = sum(val_losses) / len(val_loader.dataset)
    val_accuracy = correct / total

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(val_loss, val_accuracy))

    return val_loss, val_accuracy

def continue_training(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, filepath):
    loaded_model = torch.load(filepath)
    model.load_state_dict(loaded_model["model_state_dict"])
    optimizer.load_state_dict(loaded_model["optimizer_state_dict"])
    start_epoch = loaded_model['epoch'] + 1

    losses, epochs, accuracies = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, start_epoch)

    return losses, epochs, accuracies
