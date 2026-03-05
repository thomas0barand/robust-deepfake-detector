import argparse

import lightning as L

from torch.utils.data import DataLoader

from src.models import RobustDetector, MetricsCallback
from src.data import FakeprintDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test RobustDetector")

    parser.add_argument("--data_dir", type=str, default="data/test/attack/", help="Directory containing test fakeprint data")
    parser.add_argument("--output_dir", type=str, default="results/attack/", help="Directory to save test results")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--use_convolution", action=argparse.BooleanOptionalAction, default=True, help="Whether to use convolution during testing")
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def test(args):
    L.seed_everything(args.seed)

    model = RobustDetector.load_from_checkpoint(args.ckpt_path)
    model.eval()

    train_conv = model.use_convolution
    model.use_convolution = args.use_convolution
    mode = model.transform_type

    print(f"Testing {mode} model (trained with convolution={train_conv}) on {mode} data with convolution={args.use_convolution}")
    
    dataset = FakeprintDataset(
        args.data_dir,
        mode=mode,
        freq_range=model.freq_range,
        n_fft=model.n_fft,
        sampling_rate=model.sampling_rate,
        bins_per_octave=model.bins_per_octave,
    )
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    print(f"Test samples: {len(dataset)}")

    filename = f"{mode}-train_conv={train_conv}-test_conv={args.use_convolution}-no-sigmoid"
    callback = MetricsCallback(output_dir=args.output_dir, filename=filename, threshold_metric="f1")
    
    trainer = L.Trainer(deterministic=True, callbacks=[callback], logger=False)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    test(args)