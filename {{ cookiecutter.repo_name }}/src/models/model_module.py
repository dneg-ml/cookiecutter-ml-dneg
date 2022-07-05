import torch
from torch.optim import AdamW

import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanMetric

from src.models.model import SimpleCNN

class ModelModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        self.model = SimpleCNN()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.train_metrics = {
            'acc': Accuracy(),
            'loss': MeanMetric()
        }
        self.val_metrics = {
                'acc': Accuracy(),
                'loss': MeanMetric()
        }

    def on_fit_start(self):
        self.logger.log_hyperparams(self.params)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.params['start_lr'])

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        return y_hat, loss

    def training_step(self, batch, batch_idx):
        _, y = batch
        y_hat, batch_loss = self._step(batch)
        
        batch_acc = self.train_metrics['acc'](
            y_hat.detach().cpu().softmax(-1), y.cpu())
        
        self.train_metrics['loss'].update(batch_loss.detach().cpu())

        logs = {
            'batch/loss': batch_loss,
            'batch/acc': batch_acc
        }

        self.log_dict(logs, on_step=True, logger=True)

        return batch_loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y_hat, batch_loss = self._step(batch)
        
        _ = self.val_metrics['acc'](
            y_hat.detach().cpu().softmax(-1), y.cpu())
        
        self.val_metrics['loss'].update(batch_loss.detach().cpu())

    def _log_epoch_metrics(self, metrics, prefix='train/'):
        logs = {f'{prefix}{k}': m.compute()
                for k, m in metrics.items()}
        _ = [m.reset() for m in metrics.values()]
        self.log_dict(logs, on_epoch=True, logger=True)

    def training_epoch_end(self, outputs):
        self._log_epoch_metrics(self.train_metrics, prefix='train/')

    def validation_epoch_end(self, outputs):
        self._log_epoch_metrics(self.val_metrics, prefix='val/')
