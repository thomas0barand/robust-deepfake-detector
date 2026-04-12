import argparse

import os
import lightning as L

from torch.utils.data import DataLoader

from src.models import RobustDetector, MetricsCallback
from src.data import FakeprintDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test RobustDetector")

    parser.add_argument("--data_dir", type=str, default="data/", help="Directory containing test fakeprint data")
    parser.add_argument("--music_generator", type=str, default="suno_v5", choices=["udio_v120", "suno_v3.5", "suno_v5"], help="Directory containing AI-generated fakeprint data")
    parser.add_argument("--attack", type=str, choices=["resample", "pitch_shift", "noattack"], default="noattack", help="Whether to test on attacked samples (resampling, pitch_shift) for robustness")
    parser.add_argument("--out_dir", type=str, default="results/", help="Directory to save test results")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def test(args):
    L.seed_everything(args.seed)

    model = RobustDetector.load_from_checkpoint(args.ckpt_path)
    model.eval()

    ai_dir = os.path.join(args.data_dir, args.music_generator, "test", args.attack)
    human_dir = os.path.join(args.data_dir, "human", "test", args.attack)
    
    dataset = FakeprintDataset(
        ai_dir=ai_dir,
        human_dir=human_dir,
        mode=model.transform_type,
        freq_range=model.freq_range,
        n_fft=model.n_fft,
        sampling_rate=model.sampling_rate,
        bins_per_octave=model.bins_per_octave,
    )
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    print(f"Test samples: {len(dataset)}")

    filename = f"{"log_stft" if model.log_stft else model.transform_type}"
    filename += "-use_conv" if model.use_convolution else ""
    filename += f"-lamb={model.lamb}" if model.use_convolution else ""
    output_dir = os.path.join(args.out_dir, args.music_generator, args.attack)
    os.makedirs(output_dir, exist_ok=True)
    callback = MetricsCallback(output_dir=output_dir, filename=filename, threshold_metric="f1")
    
    trainer = L.Trainer(deterministic=True, callbacks=[callback], logger=False)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    test(args)