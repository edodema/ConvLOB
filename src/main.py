from pathlib import Path
from re import X
import numpy as np
import torch
from src.dataset import LOBDataModule
from src.model import LitModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

if __name__ == "__main__":
    pl.seed_everything(42)

    ROOT_DIR = Path(".")

    datamodule = LOBDataModule(data_dir=ROOT_DIR / "data", batch_size=16)
    model = LitModel(lr=1e-3, decay=False)

    logger = NeptuneLogger(
        project="edodema/LimitOrderBook",
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4MTIxYzBhMy1iNTAwLTQ0MzktOTlkNy1iMmVhNWMyZjM5MGIifQ==",
        tags=["try"],
    )

    # ! Need to choose for which metrics we want to monitor.

    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=1,
        logger=logger,
        val_check_interval=1.0,
        num_sanity_val_steps=1,
    )

    # * Train and val.
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
