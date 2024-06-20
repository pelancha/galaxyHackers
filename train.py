import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import os
import copy
import time
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tqdm import trange
from copy import deepcopy
from collections import defaultdict
import settings


bet_models_folder = os.path.join(settings.STORAGE_PATH, "best_models")

class Trainer:
    def __init__(self, model: nn.Module,
                 optimizer: Optimizer,
                 criterion, 
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 lr_scheduler: LRScheduler | None = None,
                 lr_scheduler_type: str | None = None,
                 batch_size: int = 128):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size

        self.history = defaultdict(list)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.model = self.model.to(self.device)

        self.global_step = 0
        self.cache = self.cache_states()

    def post_train_batch(self):
        # called after every train batch
        if self.lr_scheduler is not None and self.lr_scheduler_type == 'per_batch':
            self.lr_scheduler.step()

    def post_val_batch(self):
        pass

    def post_train_stage(self):
        pass

    def post_val_stage(self, val_loss):
        # called after every end of val stage (equals to epoch end)
        if self.lr_scheduler is not None and self.lr_scheduler_type == 'per_epoch':
            self.lr_scheduler.step(val_loss)



    def save_checkpoint(self):
        
        filename = f'best_weights_{self.model.__class__.__name__}_{self.optimizer.__class__.__name__}_weights.pth'
        path = os.path.join(bet_models_folder, filename)
        
        torch.save(self.model.state_dict(), path)

    def train(self, num_epochs: int):
        model = self.model
        optimizer = self.optimizer

        best_loss = float('inf') # +inf


        for epoch in trange(num_epochs, unit="epoch", leave=False, desc=f'Training {model.__class__.__name__} with {optimizer.__class__.__name__} optimizer'):

            
            model.train()
            train_losses = []
            for batch in tqdm(self.train_dataloader, unit="batch"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, acc = self.compute_all(batch)

                train_losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.post_train_batch()

                self.history["train_loss"].append(loss.item())
                self.history["train_acc"].append(acc)
                self.global_step += 1

            model.eval()
            val_losses = []
            val_accs = []
            for batch in tqdm(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, acc = self.compute_all(batch)
                val_losses.append(loss.item())
                val_accs.append(acc)

      
            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)
            self.post_val_stage(val_loss)

            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)


            if val_loss < best_loss:
                self.save_checkpoint()
                best_loss = val_loss



    def compute_all(self, batch):  # удобно сделать функцию, в которой вычисляется лосс по пришедшему батчу
        x = batch['image']
        y = batch['label']
        logits = self.model(x)

        loss = self.criterion(logits[:, 1], y.float())

        assert logits.shape[1] == 2, logits.shape
        acc = (logits.argmax(axis=1) == y).float().mean().cpu().numpy()
        

        return loss, acc
    

    def cache_states(self):
        cache_dict = {'model_state': deepcopy(self.model.state_dict()),
                      'optimizer_state': deepcopy(self.optimizer.state_dict())}

        return cache_dict

    def rollback_states(self):
        self.model.load_state_dict(self.cache['model_state'])
        self.optimizer.load_state_dict(self.cache['optimizer_state'])

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
    val_losses, val_accuracies = [], []

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
            predicted = (outputs > 0.9).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1).float()).sum().item()
            cool_progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset), acc=correct / total)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        epochs.append(epoch + 1)

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
          
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
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Loss: {:.4f}, Best Validation Accuracy: {:.4f}'.format(best_loss, best_accuracy))

    return losses, epochs, accuracies, val_losses, val_accuracies, model

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
            predicted = (outputs > 0.9).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1).float()).sum().item()
    val_loss = sum(val_losses) / len(val_loader.dataset)
    val_accuracy = correct / total

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(val_loss, val_accuracy))

    return val_loss, val_accuracy

def continue_training(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, filepath):
    loaded_model = torch.load(filepath)
    model.load_state_dict(loaded_model["model_state_dict"])
    optimizer.load_state_dict(loaded_model["optimizer_state_dict"])
    start_epoch = loaded_model['epoch'] + 1

    losses, epochs, accuracies, val_losses, val_accuracies, model = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, start_epoch)

    return losses, epochs, accuracies, val_losses, val_accuracies, model
