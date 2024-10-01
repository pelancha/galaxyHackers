import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Any

from scipy.signal import savgol_filter

from tqdm import trange
from copy import deepcopy
from collections import defaultdict
from config import settings
import pandas as pd
import matplotlib.pyplot as plt

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
        
        filename = f'best_weights_{self.model.__class__.__name__}_{self.optimizer.__class__.__name__}.pth'
        path = os.path.join(settings.BEST_MODELS_PATH, filename)
        
        torch.save(self.model.state_dict(), path)

    def train(self, num_epochs: int):
        model = self.model
        optimizer = self.optimizer

        best_loss = float('inf') # +inf


        for _ in trange(
            num_epochs, 
            unit="epoch", 
            leave=False, 
            desc=f'Training {model.__class__.__name__} with {optimizer.__class__.__name__} optimizer'
            ):

            
            model.train()
            train_losses = []
            for batch in tqdm(self.train_dataloader, unit="batch"):

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

            for batch in tqdm(self.val_dataloader):
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

        y_pred, y_probs, y_true, descriptions = [], [], [], []
        y_negative_target_probs = []

        for batch in tqdm(test_dataloader):

            logits, outputs, labels, loss, acc = self.compute_all(batch)
            test_losses.append(loss.item())
            test_accs.append(acc)

            y_probs.extend(logits[:, 1].data.cpu().numpy().ravel())
            y_negative_target_probs.extend(logits[:, 0].data.cpu().numpy().ravel())
            y_pred.extend(outputs.data.cpu().numpy().ravel())
            y_true.extend(labels.data.cpu().numpy().ravel())

            descriptions.append(pd.DataFrame(batch['description']))


        predictions = pd.concat(descriptions).reset_index(drop=True)
        predictions['y_pred'] = y_pred
        predictions['y_probs'] = y_probs
        predictions['y_negative_probs'] = y_negative_target_probs
        predictions['y_true'] = y_true

        return predictions, test_losses, test_accs,


  

    def compute_all(self, batch):  # удобно сделать функцию, в которой вычисляется лосс по пришедшему батчу
        x = batch['image'].to(self.device)
        y = batch['label'].to(self.device)
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


    def find_lr(self, min_lr: float = 1e-6,
                max_lr: float = 1e-1,
                num_lrs: int = 20,
                smoothing_window=30,
                smooth_beta: float = 0.8) -> dict:
        lrs = np.geomspace(start=min_lr, stop=max_lr, num=num_lrs)
        logs = {'lr': [], 'loss': [], 'avg_loss': []}
        avg_loss = None
        model, optimizer = self.model, self.optimizer

        model.train()
        for lr, batch in tqdm(zip(lrs, self.train_dataloader), desc='finding LR', total=num_lrs):
            # apply new lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # train step
            *_, loss, _  = self.compute_all(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.cpu().detach().numpy()
            # calculate smoothed loss
            if avg_loss is None:
                avg_loss = loss
            else:
                avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss

            # store values into logs
            logs['lr'].append(lr)
            logs['avg_loss'].append(avg_loss)
            logs['loss'].append(loss)


        # Compute the logarithm of learning rates
        log_lrs = np.log10(logs['lr'])

        smoothed_losses =  savgol_filter(logs['loss'], window_length=smoothing_window, polyorder=2)

        # Compute the derivative of the smoothed loss with respect to log_lr
        loss_derivatives = np.gradient(smoothed_losses, log_lrs)

        # Find the index where the derivative is minimum (most negative)
        optimal_idx = np.argmin(loss_derivatives)
        optimal_lr = logs['lr'][optimal_idx]

            

        logs.update({key: np.array(val) for key, val in logs.items()})

        plt.figure(figsize=(10, 6))

        plt.plot(logs['lr'], logs['loss'], label='Loss')
        plt.plot(logs['lr'], smoothed_losses, label='Smoothed loss')
        plt.axvline(x=optimal_lr, color='r', linestyle='--', label=f'Optimal LR: {optimal_lr:.2E}')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)

        # Mark the optimal learning rate
        # plt.axvline(x=optimal_lr, color='r', linestyle='--', label=f'Optimal LR: {optimal_lr:.2E}')
        plt.legend()
        plt.show()
        self.rollback_states()

        return logs


    def lr_find(self, start_lr=5, end_lr=1e-8, num_iter=79):

        initial_state = self.model.state_dict()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr

        self.model.train()
        # lr_lambda = lambda x: (end_lr / start_lr) ** (x / num_iter)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lr_lambda)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=start_lr, end_factor=end_lr, total_iters=num_iter) 


        losses = []
        lrs = []

        t = 0
        for batch in tqdm(self.train_dataloader, unit="batch"):

            *_, loss, _ = self.compute_all(batch)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            scheduler.step()


            # Record the learning rate and corresponding loss
            lr = self.optimizer.param_groups[0]['lr']
            lrs.append(lr)
            losses.append(loss.item())



        # Convert lists to numpy arrays for analysis
        losses = np.array(losses)
        lrs = np.array(lrs)

        # Smooth the loss curve to reduce noise
        from scipy.ndimage import gaussian_filter1d
        smoothed_losses = gaussian_filter1d(losses, sigma=2)

        # Compute the gradients (derivative) of the smoothed losses
        gradients = np.gradient(smoothed_losses)


        # Find index where loss starts increasing
        increasing_loss_idxs = np.where(gradients > 0)[0]
        if len(increasing_loss_idxs) > 0:
            first_increase_idx = increasing_loss_idxs[0]
            optimal_lr = lrs[first_increase_idx - 1]  # Choose LR before loss increases
            print(f"Suggested Optimal Learning Rate: {optimal_lr}")

            plt.figure(figsize=(10, 6))
            plt.plot(lrs, losses, label='Loss')
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            plt.grid(True)

            # Mark the optimal learning rate
            plt.axvline(x=optimal_lr, color='r', linestyle='--', label=f'Optimal LR: {optimal_lr:.2E}')
            plt.legend()
            plt.show()

            self.model.load_state_dict(initial_state)


            for param_group in self.optimizer.param_groups:
                param_group['lr'] = optimal_lr
        else:
            raise AttributeError("Could not find an optimal learning rate. Try adjusting the range.")


class Predictor():

    def __init__(self, model:nn.Module, device):

        
        self.model = model
        self.model.eval()

        self.device = device


    def predict(self, dataloader: DataLoader):

        y_pred, y_prob, descriptions = [], [], []


        for batch in tqdm(dataloader):

            logits, outputs = self.compute_all(batch)

            y_pred.extend(outputs.data.cpu().numpy().ravel())
            y_prob.extend(logits[:, 1].data.cpu().numpy().ravel())
            descriptions.append(pd.DataFrame(batch['description']))


        predictions = pd.DataFrame(
        np.array([
            np.array(y_pred), 
            np.array(y_prob)
        ]).T, columns=["y_pred", "y_prob"]).reset_index(drop=True)

        description_frame = pd.concat(descriptions).reset_index(drop=True)

        predictions = pd.concat([predictions, description_frame], axis=1)
        predictions = predictions.set_index("idx")
        predictions.index = predictions.index.astype(int)
        
        return predictions
    

    def compute_all(self, batch): 

        x = batch['image'].to(self.device)

        logits = self.model(x)

        outputs = logits.argmax(axis=1)


        return logits, outputs
