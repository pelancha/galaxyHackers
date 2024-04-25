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

def train(model, train_loader, criterion, optimizer, device, num_epochs, start_epoch=0):
    model.to(device)
    model.train()

    best_loss, best_accuracy = float("inf"), 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    losses, epochs, accuracies = [], [], []

    since = time.time()
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        cool_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}. Training {model.__class__.__name__}', unit='batch')
        for inputs, labels in cool_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            cool_progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset), acc=correct / total)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        epochs.append(epoch + 1)
        
        os.makedirs(models_epoch, exist_ok=True)

        epoch_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
        
        torch.save(epoch_data, f'{models_epoch}{model.__class__.__name__}_epoch_{epoch + 1}.pth')

        if epoch_loss < best_loss:
            best_loss, best_accuracy = epoch_loss, epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, f'{models_state_dict}/best_{model.__class__.__name__}_weights.pth')

    os.makedirs(models_state_dict, exist_ok=True)
    model.load_state_dict(best_model_weights)
    torch.save(epoch_data, f'{models_state_dict}/{model.__class__.__name__}_weights.pth')
    torch.save(model, f'{models_state_dict}/{model.__class__.__name__}.pth')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Loss: {:.4f}, Best Validation Accuracy: {:.4f}'.format(best_loss, best_accuracy))

    return losses, epochs, accuracies

def continue_training(model, train_loader, criterion, optimizer, device, num_epochs, filepath):
    loaded_model = torch.load(filepath)
    model.load_state_dict(loaded_model["model_state_dict"])
    optimizer.load_state_dict(loaded_model["optimizer_state_dict"])
    start_epoch = loaded_model['epoch'] + 1

    losses, epochs, accuracies = train(model, train_loader, criterion, optimizer, device, num_epochs, start_epoch)

    return losses, epochs, accuracies

def validate(model, val_loader, criterion, device, num_epochs, start_epoch=0):
    model.to(device)
    model.eval()
    val_losses, val_epochs, val_accuracies = [], [], []
    best_val_loss, best_val_accuracy = float("inf"), 0.0
    since = time.time()
    with torch.no_grad():
      for epoch in range(start_epoch, num_epochs):
          running_loss = 0.0
          correct, total = 0, 0
          cool_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs}. Validating {model.__class__.__name__}', unit='batch')
          for inputs, labels in cool_progress_bar:
              inputs, labels = inputs.to(device), labels.to(device)
              outputs = model(inputs)
              loss = criterion(outputs, labels.unsqueeze(1).float())
              running_loss += loss.item() * inputs.size(0)
              _, predicted = torch.max(outputs, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              cool_progress_bar.set_postfix(loss=running_loss / len(val_loader.dataset), acc=correct / total)
          val_epoch_loss = running_loss / len(val_loader.dataset)
          val_epoch_acc = correct / total
          val_losses.append(val_epoch_loss)
          val_accuracies.append(val_epoch_acc)
          val_epochs.append(1)
          
          if val_epoch_loss < best_val_loss:
              best_val_loss = val_epoch_loss
              best_val_accuracy = val_epoch_acc
    
    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Loss: {:.4f}, Best Validation Accuracy: {:.4f}'.format(best_val_loss, best_val_accuracy))


    return val_losses, val_epochs, val_accuracies


working_path = "./"
models_epoch = f'{working_path}trained_models/'
models_state_dict = f'{working_path}state_dict/'
