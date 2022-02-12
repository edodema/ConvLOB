from typing import *
import requests
import zipfile
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class LOBDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        input_idx: int = 40,
        label_idx: int = -5,
        window: int = 100,
        pred_horizon_idx: int = -1,
    ):
        """Dataset object for FI-2010 dataset.

        Args:
            data (np.ndarray): Input data array.
            input_idx (int, optional): Last column for input data. Defaults to 40.
            label_idx (int, optional): First column for labels. Defaults to -5.
            window (int, optional): Window size. Defaults to 100.
            pred_horizon_idx (int, optional): Prediction horizon index. Defaults to -1.
        """
        super(LOBDataset, self).__init__()
        x, y = self._init_data(
            data=data,
            in_idx=input_idx,
            gt_idx=label_idx,
            win=window,
            ph_idx=pred_horizon_idx,
        )

        self.x = torch.from_numpy(x.copy())
        self.y = torch.from_numpy(y)

    def _init_data(
        self, data: np.ndarray, in_idx: int, gt_idx: int, win: int, ph_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess data.

        Args:
            data (np.ndarray): Input data.
            in_idx (int): Last column of input data we aim to consider.
            gt_idx (int): First column of ground truth we aim to consider.
            win (int): Window size.
            ph_idx (int): Prediction horizon index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Preprocessed input and ground truth data.
        """
        # Input data are the first `in_idx`` (40 by default) columns i.e. the first 10 levels
        # due to each level being defined by a 4-tuple (price_bid, volume_bid, price_ask, volume_ask).
        x = data[:, :in_idx]

        # Labels are the last `gt_idx`` (5 by default) columns of the LOB. Possible values are:
        # - 1: Positive percentage change.
        # - 2: Stationary behavior.
        # - 3: Negative percentage change.
        # We also want them to start from 0.
        y = data[:, -gt_idx:] - 1

        # Each of the `gt_idx` columns represents a different projection horizon, for simplicity we keep one only.
        y = y[:, ph_idx]

        # We split the input data in windows of length `win`, then trim the first `win` elements of the labels.
        x_win, y_trim = self._slide_window(x=x, y=y, win=win)

        return x_win, y_trim

    def _slide_window(
        self, x: np.ndarray, y: np.ndarray, win: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split data in windows.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): Ground truth.
            win (int): Window size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Obtained windows with their ground truth.
        """
        x_win = sliding_window_view(x=x, window_shape=win, axis=0).transpose(0, 2, 1)
        y_trim = y[win - 1 :]
        return x_win, y_trim

    def __len__(self) -> int:
        """Data length.

        Returns:
            int: Length.
        """
        return self.x.shape[0]

    def __getitem__(self, item: int) -> List[torch.Tensor]:
        """Get item by index.

        Args:
            item (int): Index.

        Returns:
            List[torch.Tensor]: List with input data and label corresponding to the specified index.
        """
        return [self.x[item], self.y[item]]


class LOBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        input_idx: int = 40,
        label_idx: int = -5,
        window: int = 100,
        pred_horizon_idx: int = -1,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.input_idx = input_idx
        self.label_idx = label_idx
        self.window = window
        self.pred_horizon_idx = pred_horizon_idx

    def prepare_data(self):
        """Prepare data."""
        data_file = self.data_dir / "data.zip"

        # * Download data.zip if necessary.
        if not data_file.exists():
            url = "https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip"

            # Download.
            print(f"Downloading data from {url}...")
            r = requests.get(url)
            open(data_file, "wb").write(r.content)

            # Extract.
            print(f"Inflating {data_file}...")
            with zipfile.ZipFile(data_file, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

        # # * Data preprocessing.
        train_path = self.data_dir / "train.gz"
        val_path = self.data_dir / "val.gz"
        test_path = self.data_dir / "test.gz"

        # If data is not preprocessed, do it.
        if not all(f.exists() for f in [train_path, val_path, test_path]):
            self._prepare_data()
            print()

    def _prepare_data(self):
        """Load and split data according to a 80-20 rate, then save the new splits.

        Args:
            path (Path): Path of the FI-2010 `data.zip` file.
        """

        train_file = "Train_Dst_NoAuction_DecPre_CF_7.txt"
        test_files = [
            "Test_Dst_NoAuction_DecPre_CF_7.txt",
            "Test_Dst_NoAuction_DecPre_CF_8.txt",
            "Test_Dst_NoAuction_DecPre_CF_9.txt",
        ]

        # * Prepare data.
        # Load as NumPy arrays.

        print(f"Loading {train_file}...")
        train_val = np.loadtxt(self.data_dir / train_file)

        # Split into train and val according to a 80-20 ratio.
        train = train_val[:, : int(np.floor(train_val.shape[1] * 0.8))]
        val = train_val[:, int(np.floor(train_val.shape[1] * 0.8)) :]

        test = []
        for f in test_files:
            print(f"Loading {train_file}...")
            test.append(np.loadtxt(self.data_dir / f))

        test = np.hstack(test)

        # * Save data.
        print(f"Saving {self.data_dir / 'train.gz'}...")
        np.savetxt(self.data_dir / "train.gz", train.T)

        print(f"Saving {self.data_dir / 'val.gz'}...")
        np.savetxt(self.data_dir / "val.gz", val.T)

        print(f"Saving {self.data_dir / 'test.gz'}...")
        np.savetxt(self.data_dir / "test.gz", test.T)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets.

        Args:
            stage (Optional[str], optional): Stage in which we are e.g. "fit", "test". Defaults to None.
        """
        # Assign train/val splits.
        if stage in (None, "fit"):
            train_file = self.data_dir / "train.gz"
            print(f"Loading {train_file}...")
            self.train = LOBDataset(
                data=np.loadtxt(train_file),
                input_idx=self.input_idx,
                label_idx=self.label_idx,
                window=self.window,
                pred_horizon_idx=self.pred_horizon_idx,
            )

            val_file = self.data_dir / "val.gz"
            print(f"Loading {val_file}...")
            self.val = LOBDataset(
                data=np.loadtxt(val_file),
                input_idx=self.input_idx,
                label_idx=self.label_idx,
                window=self.window,
                pred_horizon_idx=self.pred_horizon_idx,
            )
        # Assign test split.
        if stage in (None, "test"):
            test_file = self.data_dir / "test.gz"
            print(f"Loading {test_file}...")
            self.test = LOBDataset(
                data=np.loadtxt(test_file),
                input_idx=self.input_idx,
                label_idx=self.label_idx,
                window=self.window,
                pred_horizon_idx=self.pred_horizon_idx,
            )

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader.

        Returns:
            DataLoader: Train dataloader.
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader.

        Returns:
            DataLoader: Val dataloader.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader.

        Returns:
            DataLoader: Test dataloader.
        """
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
        )
