import argparse
import torch

parser = argparse.ArgumentParser(
    prog="Add name",  # TODO
    description="Add description",  # TODO
    add_help=True,
)

# * Directories.
parser.add_argument(
    "--data_dir",
    type=str,
    default="./data",
    help="Directory in which data are stored.",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="./checkpoints",
    help="Directory in which checkpoints are stored.",
)

parser.add_argument("--num_classes", type=int, default=3, help="Number of classes.")

# * Training.
parser.add_argument("--train", dest="train", action="store_true", help="Train model.")
parser.set_defaults(train=False)

parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")

parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")

parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")

parser.add_argument(
    "--lr_decay",
    dest="decay",
    action="store_true",
    help="Whether to use learning rate scheduling or not.",
)
parser.set_defaults(decay=False)

parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")

parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)

# ! Currently unused.
parser.add_argument(
    "--early-stopping",
    dest="monitor",
    action="store_true",
    help="Enable early stopping by picking best model only.",
)
parser.set_defaults(monitor=False)

# ! Currently unused.
# * Logging support. I use WandB since it allows to log images,
parser.add_argument(
    "--neptune",
    dest="logger",
    action="store_true",
    help="Logging using Neptune.",
)
parser.set_defaults(logger=False)

# Arguments.
args = parser.parse_args()
