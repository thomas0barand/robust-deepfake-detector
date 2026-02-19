import torch
import lightning as L

from torch.utils.data import DataLoader, random_split

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from src.models.detector import RobustDetector
from src.data.dataset import FakeprintDataset
from src.models.utils import get_feature_dim


def train(
    data_dir: str,
    # Dataset
    val_split: float = 0.2,
    # Model
    use_bias: bool = True,
    use_norm: bool = True,
    use_convolution: bool = False,
    # CQT
    n_fft: int = 16384,
    sampling_rate: int = 48000,
    bins_per_octave: int = 96,
    freq_range: list = [200, 6000],
    f_min: float = 32.7,
    # Training
    batch_size: int = 64,
    num_workers: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 50,
    patience: int = 10,
    # Misc
    seed: int = 42,
    log_dir: str = "logs",
    ckpt_dir: str = "checkpoints",
):
    L.seed_everything(seed)

    dataset = FakeprintDataset(data_dir)

    n_val = int(val_split * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"Train: {len(train_set)} — Val: {len(val_set)}")

    # Compute pos_weight from train set to handle class imbalance
    train_labels = torch.tensor([dataset.samples[i][1] for i in train_set.indices])
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / (n_pos + 1e-6)
    print(f"Class balance — AI: {n_pos}, Human: {n_neg}, pos_weight: {pos_weight:.2f}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    feature_dim = get_feature_dim(sampling_rate, bins_per_octave, freq_range, f_min)
    print(f"Feature dimension: {feature_dim}")

    model = RobustDetector(
        feature_dim=feature_dim,
        use_bias=use_bias,
        use_norm=use_norm,
        use_convolution=use_convolution,
        n_fft=n_fft,
        sampling_rate=sampling_rate,
        bins_per_octave=bins_per_octave,
        freq_range=freq_range,
        f_min=f_min,
        pos_weight=pos_weight,
        lr=lr,
        weight_decay=weight_decay,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"robustdetector-use_conv={use_convolution}",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_auroc",
        mode="max",
        patience=patience,
        verbose=True,
    )

    logger = TensorBoardLogger(log_dir, name="robustdetector")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"Best model: {checkpoint_cb.best_model_path} (val_auroc={checkpoint_cb.best_model_score:.4f})")
    return checkpoint_cb.best_model_path


if __name__ == "__main__":
    best_ckpt = train(
        data_dir="src/checkpoints/fp/",
        freq_range=[200, 6000],
        batch_size=64,
        max_epochs=50,
        patience=5,
        log_dir="logs",
        ckpt_dir="src/checkpoints/models/",
    )