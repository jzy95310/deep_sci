# trainers.py: a file containing classes for training the nonlinear spatial causal inference model
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch

import os, sys
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from models import NonlinearSCI

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
OPTIMIZERS = {
    'adam': torch.optim.Adam, 
    'sgd': torch.optim.SGD, 
    'rmsprop': torch.optim.RMSprop, 
    'adagrad': torch.optim.Adagrad, 
    'adadelta': torch.optim.Adadelta, 
    'adamw': torch.optim.AdamW,
    'sparseadam': torch.optim.SparseAdam,
    'adamax': torch.optim.Adamax,
    'asgd': torch.optim.ASGD,
    'lbfgs': torch.optim.LBFGS,
    'rprop': torch.optim.Rprop,
}

class BaseTrainer(ABC):
    """
    Template for trainers

    Arguments
    --------------
    model: torch.nn.Module, the model to train
    data_generators: Dict, a dict of dataloaders where keys must be 'train', 'val', and 'test'
    optim: str, the name of the optimizer for training
    optim_params: Dict, a dict of parameters for the optimizer
    lr_scheduler: torch.optim.lr_scheduler, the learning rate scheduler for training
    model_save_dir: str, the directory to save the trained model. If None, the model will not be saved
    model_name: str, the name of the trained model
    loss_fn: torch.nn.modules.loss._Loss, the loss function for optimizing the model
    device: torch.device, the device to train the model on
    epochs: int, the number of epochs to train the model for
    patience: int, the number of epochs to wait before early stopping
    logger: logging.Logger, an instance of logging.Logger for logging messages, errors, exceptions
    """
    @abstractmethod
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, model_save_dir: str = None, model_name: str = 'model.pt', 
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), device: torch.device = torch.device('cpu'), 
                 epochs: int = 100, patience: int = 10, logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.model: torch.nn.Module = model
        self.data_generators: Dict = data_generators
        self.optim: str = optim
        self.optim_params: Dict = optim_params
        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler = lr_scheduler
        self.model_save_dir: str = model_save_dir
        self.model_name: str = model_name
        self.loss_fn: torch.nn.modules.loss._Loss = loss_fn
        self.device: torch.device = device
        self.epochs: int = epochs
        self.patience: int = patience
        self.logger: logging.Logger = logger
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        if self.model_save_dir is not None and not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self._validate_inputs()
        self._set_optimizer()
    
    def _assign_device_to_data(self, features: List, spatial_features: torch.Tensor, target: torch.Tensor) -> Tuple:
        """
        Assign the device to the features and the target
        """
        features = list(map(lambda x: x.to(self.device), features))
        spatial_features = spatial_features.to(self.device)
        target = target.to(self.device)
        return features, spatial_features, target
    
    @abstractmethod
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the trainer
        """
        if not isinstance(self.data_generators, Dict):
            raise TypeError("data_generators must be a dictionary.")
        if not set(self.data_generators.keys()).issubset({TRAIN, VAL, TEST}):
            raise ValueError("The keys of data_generators must be a subset of {\'train\', \'val\', and \'test\'}")
        if self.optim not in OPTIMIZERS:
            raise TypeError("The optimizer must be one of the following: {}".format(OPTIMIZERS.keys()))
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            raise TypeError("lr_scheduler must be an instance of torch.optim.lr_scheduler._LRScheduler")
        if not isinstance(self.loss_fn, torch.nn.modules.loss._Loss):
            raise TypeError("loss_fn must be an instance of torch.nn.modules.loss._Loss")
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be an instance of torch.device")
        if not isinstance(self.logger, logging.Logger):
            raise TypeError("logger must be an instance of logging.Logger")
    
    def _set_optimizer(self) -> None:
        """
        Set the optimizer for the trainer
        """
        self.optimizer = OPTIMIZERS[self.optim](self.model.parameters(), **self.optim_params)
    
    @abstractmethod
    def train(self) -> None:
        """
        Train the ICK model
        """
        pass

    @abstractmethod
    def predict(self) -> None:
        """
        Evaluate the ICK model on the test data
        """
        pass

class Trainer(BaseTrainer):
    """
    Trainer for the nonlinear spatial causal inference model
    """
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, model_save_dir: str = None, model_name: str = 'model.pt', 
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), device: torch.device = torch.device('cpu'), 
                 epochs: int = 100, patience: int = 10, logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        super(Trainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, model_save_dir, model_name, 
                                      loss_fn, device, epochs, patience, logger)
        self._validate_inputs()
        self._set_optimizer()
    
    def _validate_inputs(self) -> None:
        if not isinstance(self.model, NonlinearSCI):
            raise TypeError("model must be an instance of NonlinearSCI class.")
        super(Trainer, self)._validate_inputs()
    
    def _train_step(self) -> Tuple:
        """
        Perform a single training step
        """
        self.model.to(self.device)
        self.model.train()
        y_train_pred = torch.empty(0).to(self.device)
        y_train_true = torch.empty(0).to(self.device)
        for step, batch in enumerate(self.data_generators[TRAIN]):
            features, spatial_features, target = self._assign_device_to_data(*batch)
            # Zero the gradients
            self.optimizer.zero_grad()
            # Forward pass
            if not self.model.unobserved_confounder:
                y_pred = self.model(features).float()
            else:
                y_pred = self.model(features, spatial_features).float()
            loss = self.loss_fn(y_pred, target)
            # Backward pass
            loss.backward()
            self.optimizer.step()
            # Record the predictions
            y_train_pred = torch.cat((y_train_pred, y_pred), dim=0)
            y_train_true = torch.cat((y_train_true, target), dim=0)
        train_loss = self.loss_fn(y_train_pred, y_train_true).item()
        return train_loss, step
    
    def train(self) -> None:
        best_loss = 1e9
        best_model_state_dict = None
        trigger_times = 0
        
        self.logger.info("Training started:\n")
        for epoch in range(self.epochs):
            # Training
            train_start = time.time()
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            self.logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:7.6f}")
            # Log the training time and loss
            train_loss, step = self._train_step()
            train_time = time.time() - train_start
            self.logger.info("{:.0f}s for {} steps - {:.0f}ms/step - loss {:.4f}" \
                  .format(train_time, step + 1, train_time * 1000 // (step + 1), train_loss))
            # Validation
            val_start = time.time()
            self.logger.info("Validation:")
            val_loss = self.validate()
            val_time = time.time() - val_start
            self.logger.info("{:.0f}s - loss {:.4f}\n".format(val_time, val_loss))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Early stopping
            if val_loss > best_loss:
                trigger_times += 1
                if trigger_times >= self.patience:
                    # Trigger early stopping and save the best model
                    self.logger.info("Early stopping - patience reached")
                    if best_model_state_dict is not None:
                        self.logger.info("Restoring the best model")
                        self.model.load_state_dict(best_model_state_dict)
                    if self.model_save_dir is not None:
                        self.logger.info("Saving the best model")
                        torch.save(best_model_state_dict, os.path.join(self.model_save_dir, self.model_name))
                    break
            else:
                trigger_times = 0
                best_loss = val_loss
                best_model_state_dict = self.model.state_dict()

        if trigger_times < self.patience:
            self.logger.info("Training completed without early stopping.")
    
    def validate(self) -> float:
        """
        Evaluate the model on the validation data
        """
        y_val_pred = torch.empty(0).to(self.device)
        y_val_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[VAL]:
                features, spatial_features, target = self._assign_device_to_data(*batch)
                if not self.model.unobserved_confounder:
                    y_pred = self.model(features).float()
                else:
                    y_pred = self.model(features, spatial_features).float()
                y_val_pred = torch.cat((y_val_pred, y_pred), dim=0)
                y_val_true = torch.cat((y_val_true, target), dim=0)
        val_loss = self.loss_fn(y_val_pred, y_val_true).item()
        return val_loss
    
    def predict(self) -> None:
        """
        Evaluate the model on the test data
        """
        y_test_pred = torch.empty(0).to(self.device)
        y_test_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[TEST]:
                features, spatial_features, target = self._assign_device_to_data(*batch)
                if not self.model.unobserved_confounder:
                    y_pred = self.model(features).float()
                else:
                    y_pred = self.model(features, spatial_features).float()
                y_test_pred = torch.cat((y_test_pred, y_pred), dim=0)
                y_test_true = torch.cat((y_test_true, target), dim=0)
        return y_test_pred.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()

# ########################################################################################
# MIT License

# Copyright (c) 2023 Ziyang Jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:

# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
# OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################################