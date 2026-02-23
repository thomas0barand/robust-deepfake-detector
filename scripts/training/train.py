import argparse
import torch
import lightning as L

from torch.utils.data import DataLoader, random_split

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from src.models.detector import RobustDetector
from src.data.dataset import FakeprintDataset
from src.models.utils import get_feature_dim


def parse_args():
    parser = argparse.ArgumentParser(description="Train RobustDetector")

    parser.add_argument("--data_dir", type=str, default="src/checkpoints/fp/")
    parser.add_argument("--mode", type=str, default="stft", choices=["cqt", "stft"])

    # Dataset
    parser.add_argument("--val_split", type=float, default=0.2)

    # Model
    parser.add_argument("--use_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_convolution", action=argparse.BooleanOptionalAction, default=False)

    # CQT
    parser.add_argument("--n_fft", type=int, default=16384)
    parser.add_argument("--sampling_rate", type=int, default=48000)
    parser.add_argument("--bins_per_octave", type=int, default=96)
    parser.add_argument("--freq_range", type=int, nargs=2, default=[200, 6000], metavar=("F_MIN", "F_MAX"))
    parser.add_argument("--f_min", type=float, default=32.7)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ckpt_dir", type=str, default="src/checkpoints/models/")

    return parser.parse_args()


def train(args):
    L.seed_everything(args.seed)

    dataset = FakeprintDataset(args.data_dir, mode=args.mode)

    n_val = int(args.val_split * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"Train: {len(train_set)} — Val: {len(val_set)}")

    # Compute pos_weight from train set to handle class imbalance
    train_labels = torch.tensor([dataset.samples[i][1] for i in train_set.indices])
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / (n_pos + 1e-6)
    print(f"Class balance — AI: {n_pos}, Human: {n_neg}, pos_weight: {pos_weight:.2f}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    feature_dim = get_feature_dim(args.n_fft, args.sampling_rate, transform=args.mode, bins_per_octave=args.bins_per_octave, freq_range=args.freq_range, f_min=args.f_min)
    use_cqt = (args.mode == "cqt")
    print(f"Feature dimension: {feature_dim}")

    model = RobustDetector(
        feature_dim=feature_dim,
        use_cqt=use_cqt,
        use_bias=args.use_bias,
        use_norm=args.use_norm,
        use_convolution=args.use_convolution,
        n_fft=args.n_fft,
        sampling_rate=args.sampling_rate,
        bins_per_octave=args.bins_per_octave,
        freq_range=args.freq_range,
        f_min=args.f_min,
        pos_weight=pos_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename=f"robustdetector-{args.mode}-use_conv={args.use_convolution}",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_auroc",
        mode="max",
        patience=args.patience,
        verbose=True,
    )

    logger = TensorBoardLogger(args.log_dir, name="robustdetector")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"Best model: {checkpoint_cb.best_model_path} (val_auroc={checkpoint_cb.best_model_score:.4f})")
    return checkpoint_cb.best_model_path


if __name__ == "__main__":
    args = parse_args()
    train(args)