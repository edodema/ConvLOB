from typing import *
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics


class Flatten(nn.Module):
    def __init__(self):
        """Flatten module."""
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Unflatten(nn.Module):
    def __init__(self, channels: int = 4):
        """Unflatten module.

        Args:
            channels (int, optional): Number of channels we want. Defaults to 4.
        """
        super(Unflatten, self).__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        b, h, w = x.shape
        return x.view(b, self.channels, h, w // self.channels)


class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        """Squeeze module.

        Args:
            dim (int, optional): Dimension on which we squeeze. Defaults to -1.
        """
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x.squeeze(dim=self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim: int = -1):
        """Unsqueeze module.

        Args:
            dim (int, optional): Dimension on which we unsqueeze. Defaults to -1.
        """
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x.unsqueeze(dim=self.dim)


class Conv2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        kernel: int = 3,
        stride: int = 1,
    ):
        super(Conv2D, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        kernel: int = 3,
        stride: int = 1,
    ):
        super(Conv1D, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=0,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Attention(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, embed_dim: int, num_heads: int
    ):
        super(Attention, self).__init__()

        self.q = Conv1D(
            in_channels=in_channels, out_channels=embed_dim, kernel=3, stride=1
        )
        self.k = Conv1D(
            in_channels=in_channels, out_channels=embed_dim, kernel=3, stride=1
        )
        self.v = Conv1D(
            in_channels=in_channels, out_channels=embed_dim, kernel=3, stride=1
        )

        self.att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0
        )

        self.conv = Conv1D(
            in_channels=embed_dim, out_channels=out_channels, kernel=1, stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x).permute(2, 0, 1)
        k = self.k(x).permute(2, 0, 1)
        v = self.v(x).permute(2, 0, 1)

        out = self.att(query=q, key=k, value=v, need_weights=False)[0].permute(1, 2, 0)
        out = self.conv(out)

        return out


class Model(nn.Module):
    def __init__(self, num_classes: int = 3):
        """Neural network.

        Args:
            num_classes (int, optional): Number of classes. Defaults to 3.
        """
        super(Model, self).__init__()

        self.net = nn.Sequential(
            # Get (pb,vb,pa,ba) as channels.
            Unflatten(channels=4),
            # Feature extraction.
            Conv2D(in_channels=4, out_channels=16, kernel=3, stride=2),
            Conv2D(in_channels=16, out_channels=8, kernel=3, stride=1),
            Conv2D(in_channels=8, out_channels=64, kernel=1, stride=2),
            # We now work on three dimensional data.
            Squeeze(dim=-1),
            Attention(in_channels=64, out_channels=16, embed_dim=32, num_heads=8),
            # We flatten everything.
            Flatten(),
            nn.Linear(in_features=176, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.net(x)


class LitModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        decay: bool = False,
        num_classes: int = 3,
    ):
        """Neural network Lightning module.

        Args:
            num_classes (int, optional): Number of classes. Defaults to 3.
        """
        super(LitModel, self).__init__()
        self.save_hyperparameters()

        self.model = Model()
        self.criterion = nn.CrossEntropyLoss()

        # * Metrics.
        self.train_recall = torchmetrics.Recall()
        self.val_recall = torchmetrics.Recall()
        self.test_recall = torchmetrics.Recall()

        self.train_precision = torchmetrics.Precision()
        self.val_precision = torchmetrics.Precision()
        self.test_precision = torchmetrics.Precision()

        self.train_f1 = torchmetrics.F1Score()
        self.val_f1 = torchmetrics.F1Score()
        self.test_f1 = torchmetrics.F1Score()

        self.train_cohen = torchmetrics.CohenKappa(num_classes=num_classes)
        self.val_cohen = torchmetrics.CohenKappa(num_classes=num_classes)
        self.test_cohen = torchmetrics.CohenKappa(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)

    def step(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Base step.

        Args:
            batch (List[torch.Tensor]): Input batch.

        Returns:
            List[torch.Tensor]: Loss and predictions' indices.
        """
        x, y = batch
        x = x.to(torch.float)
        y = y.to(torch.long)

        logits = self(x)

        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)

        return [loss, preds]

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: Optional[int]
    ) -> torch.Tensor:
        """Base trining step.

        Args:
            batch (List[torch.Tensor]): Input batch.
            batch_idx (Optional[int]): Input batch's index.

        Returns:
            [torch.Tensor]: Loss.
        """
        y = batch[1].to(torch.long)
        loss, preds = self.step(batch)

        self.train_recall(preds, y)
        self.train_precision(preds, y)
        self.train_f1(preds, y)
        self.train_cohen(preds, y)

        self.log_dict(
            {
                "train/loss": loss,
                "train/recall": self.train_recall.compute(),
                "train/precision": self.train_precision.compute(),
                "train/f1": self.train_f1.compute(),
                "train/cohen": self.train_cohen.compute(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: Optional[int]):
        """Base validation step.

        Args:
            batch (List[torch.Tensor]): Input batch.
            batch_idx (Optional[int]): Input batch's index.
        """
        y = batch[1].to(torch.long)
        loss, preds = self.step(batch)

        self.val_recall(preds, y)
        self.val_precision(preds, y)
        self.val_f1(preds, y)
        self.val_cohen(preds, y)

        self.log_dict(
            {
                "val/loss": loss,
                "val/recall": self.val_recall.compute(),
                "val/precision": self.val_precision.compute(),
                "val/f1": self.val_f1.compute(),
                "val/cohen": self.val_cohen.compute(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

    def test_step(self, batch: List[torch.Tensor], batch_idx: Optional[int]):
        """Base test step.

        Args:
            batch (List[torch.Tensor]): Input batch.
            batch_idx (Optional[int]): Input batch's index.
        """
        y = batch[1].to(torch.long)
        loss, preds = self.step(batch)

        self.test_recall(preds, y)
        self.test_precision(preds, y)
        self.test_f1(preds, y)
        self.test_cohen(preds, y)

        self.log_dict(
            {
                "test/loss": loss,
                "test/recall": self.test_recall.compute(),
                "test/precision": self.test_precision.compute(),
                "test/f1": self.test_f1.compute(),
                "test/cohen": self.test_cohen.compute(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure optimizer.

        Returns:
            List[torch.optim.Optimizer]: List of optimizers.
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        if not self.hparams.decay:
            return [opt]
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt)
            return [opt, scheduler]


if __name__ == "__main__":
    t = torch.randn(16, 100, 40)

    model = Model()

    out = model(t)
    print(out.shape)
