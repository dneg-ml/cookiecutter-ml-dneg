from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra

from src.models.model_module import ModelModule
from src.data.data_module import DataModule

@hydra.main(
    config_path='configs',
    config_name='config',
    version_base='1.1')
def train(cfg):
    # Get the training config
    cfg = cfg['train']

    model = ModelModule(cfg.model)
    data = DataModule(cfg.data)
    logger = TensorBoardLogger('logs')
    checkpointer = ModelCheckpoint(monitor='val/loss', mode='min')

    trainer = Trainer(
        callbacks=[checkpointer],
        accelerator=cfg.trainer.accelerator,
        logger=logger,
        gpus=cfg.trainer.gpus,
        max_epochs=cfg.trainer.nepochs,
        log_every_n_steps=10
    )

    trainer.fit(model, data)

if __name__ == "__main__":
    train()
