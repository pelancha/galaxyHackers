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
from typing import Any

from tqdm import trange
from copy import deepcopy
from collections import defaultdict
from config import settings
import pandas as pd

class Trainer:
    def __init__(self, model: nn.Module,
                 optimizer: Optimizer,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 criterion: Any | None = None, 
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
        path = os.path.join(settings.BEST_MODELS_PATH, filename)
        
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
                *_, loss, acc = self.compute_all(batch)

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
                *_, loss, acc = self.compute_all(batch)
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

    def test(self, test_dataloader: DataLoader):

        test_losses = []
        test_accs = []

        y_pred, y_probs, y_true = [], [], []

        for batch in tqdm(test_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits, outputs, labels, loss, acc = self.compute_all(batch)
            test_losses.append(loss.item())
            test_accs.append(acc)

            y_probs.extend(logits[:, 1].data.cpu().numpy().ravel())
            y_pred.extend(outputs)
            y_true.extend(labels)

        return y_pred, y_probs, y_true, test_losses, test_accs


  

    def compute_all(self, batch):  # удобно сделать функцию, в которой вычисляется лосс по пришедшему батчу
        x = batch['image']
        y = batch['label']
        logits = self.model(x)

        assert self.criterion is not None 


        loss = self.criterion(logits[:, 1], y.float())

        assert logits.shape[1] == 2, logits.shape

        outputs = logits.argmax(axis=1)
        acc = (outputs == y).float().mean().cpu().numpy()
        


        return logits, outputs, y, loss, acc
    

    def cache_states(self):
        cache_dict = {'model_state': deepcopy(self.model.state_dict()),
                      'optimizer_state': deepcopy(self.optimizer.state_dict())}

        return cache_dict

    def rollback_states(self):
        self.model.load_state_dict(self.cache['model_state'])
        self.optimizer.load_state_dict(self.cache['optimizer_state'])


class Predictor():

    def __init__(self, model:nn.Module, device):

        
        self.model = model
        self.model.eval()

        self.device = device


    def predict(self, dataloader: DataLoader):

        y_pred, y_prob, y_true, y_names = [], [], [], []


        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs, logits, labels, idxs = self.compute_all(batch)

            y_pred.extend(outputs)
            y_prob.extend(logits[:, 1].data.cpu().numpy().ravel())
            y_true.extend(labels)
            y_names.extend(idxs)


        predictions = pd.DataFrame([
            np.array(y_pred), 
            np.array(y_prob), 
            np.array(y_true), 
            np.array(y_names)
        ], columns=["y_pred", "y_prob", "y_true", "idx"])

        predictions = predictions.set_index("idx").sort_index()
        
        return predictions
    

    def compute_all(self, batch): 
        x = batch['image']
        y = batch['label']
        idx = batch["idx"]

        logits = self.model(x)

        outputs = logits.argmax(axis=1)


        return logits, outputs, y, idx
