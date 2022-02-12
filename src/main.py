from pathlib import Path
import torch
from src.parser import args
from src.dataset import LOBDataModule
from src.model import LitModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

if __name__ == "__main__":
    pl.seed_everything(42)

    datamodule = LOBDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = LitModel(lr=args.lr, decay=args.decay)

    if args.logger:
        logger = NeptuneLogger(
            project="edodema/LimitOrderBook",
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4MTIxYzBhMy1iNTAwLTQ0MzktOTlkNy1iMmVhNWMyZjM5MGIifQ==",
            tags=["try"],
        )
    else:
        logger = None

    # ! Need to choose for which metrics we want to monitor.

    trainer = pl.Trainer(
        gpus=-1 if args.device == "cuda" else 0,
        max_epochs=args.epochs,
        logger=logger,
        val_check_interval=1.0,
        num_sanity_val_steps=1,
    )

    # * Train and val.
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
