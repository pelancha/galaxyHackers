import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    model.train()
    losses, epochs, accuracies = [], [], []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        cool_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}. Training {model.__class__.__name__}', unit='batch')
        for inputs, labels in cool_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
    return losses, epochs, accuracies
