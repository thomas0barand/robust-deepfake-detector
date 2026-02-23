import argparse

import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger

from src.models.detector import RobustDetector
from src.data.dataset import FakeprintDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test RobustDetector")

    parser.add_argument("--data_dir", type=str, default="src/checkpoints/fp/")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results")

    # Convolution
    parser.add_argument("--use_convolution", action=argparse.BooleanOptionalAction, default=True, help="Whether to use convolution during testing")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def test(args):
    L.seed_everything(args.seed)

    mode = "cqt" if "cqt" in args.ckpt_path else "stft"
    train_conv = True if "conv=True" in args.ckpt_path else False
    print(f"Testing {mode} model (trained with convolution={train_conv}) on {mode} data with convolution={args.use_convolution}")

    dataset = FakeprintDataset(args.data_dir, mode=mode)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    print(f"Test samples: {len(dataset)}")

    model = RobustDetector.load_from_checkpoint(args.ckpt_path)
    model.use_convolution = args.use_convolution

    trainer = L.Trainer(deterministic=True, logger=CSVLogger(args.output_dir, name=f"{mode}-conv={train_conv},{args.use_convolution}"))
    trainer.test(model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    test(args)