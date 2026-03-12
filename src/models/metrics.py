import numpy as np
import matplotlib.pyplot as plt
import torch
import lightning as L
from sklearn.metrics import roc_auc_score, f1_score, precision_score, accuracy_score, roc_curve, precision_recall_curve
from pathlib import Path


class MetricsCallback(L.Callback):
    def __init__(self, output_dir="results/", filename="test", threshold_metric="f1"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.threshold_metric = threshold_metric
        self._preds, self._labels = [], []

    def _collect(self, pl_module, batch):
        x, y, lag = batch
        with torch.inference_mode():
            logits, cross_corr = pl_module(x.to(pl_module.device), convolve=pl_module.use_convolution)
            preds = torch.sigmoid(logits).squeeze(-1)
        self._preds.append(preds.cpu())
        self._labels.append(y.cpu())

    def _flush(self):
        preds = torch.cat(self._preds).numpy()
        labels = torch.cat(self._labels).numpy().astype(int)
        self._preds, self._labels = [], []

        thresholds = np.linspace(0.01, 0.99, 199)
        f1_scores = [f1_score(labels, (preds >= t).astype(int), zero_division=0) for t in thresholds]
        best_t = thresholds[np.argmax(f1_scores)]
        pred_bin = (preds >= best_t).astype(int)

        metrics = {
            "auroc":     roc_auc_score(labels, preds),
            "f1":        f1_score(labels, pred_bin, zero_division=0),
            "precision": precision_score(labels, pred_bin, zero_division=0),
            "accuracy":  accuracy_score(labels, pred_bin),
        }

        self._plot(preds, labels, metrics, best_t, f1_scores, thresholds)
        print(f"\n[TEST] threshold={best_t:.3f} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics, best_t

    def _plot(self, preds, labels, metrics, best_t, f1_scores, thresholds):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        fig.suptitle(f"TEST | " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()), fontsize=10)

        bins = np.linspace(0, 1, 51)
        axes[0].hist(preds[labels == 0], bins=bins, alpha=0.7, density=True, label="Real")
        axes[0].hist(preds[labels == 1], bins=bins, alpha=0.7, density=True, label="Fake")
        axes[0].axvline(best_t, color="yellow", linestyle="--", label=f"t={best_t:.2f}")
        axes[0].set(title="Score Distribution", xlabel="P(fake)")
        axes[0].legend(fontsize=8)

        fpr, tpr, _ = roc_curve(labels, preds)
        axes[1].plot(fpr, tpr, label=f"AUC={metrics['auroc']:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[1].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
        axes[1].legend(fontsize=8)

        prec, rec, _ = precision_recall_curve(labels, preds)
        axes[2].plot(rec, prec)
        axes[2].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")

        axes[3].plot(thresholds, f1_scores)
        axes[3].axvline(best_t, color="yellow", linestyle="--", label=f"best={best_t:.2f}")
        axes[3].set(title="F1 vs Threshold", xlabel="Threshold", ylabel="F1")
        axes[3].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(self.output_dir / f"{self.filename}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._collect(pl_module, batch)

    def on_test_epoch_end(self, trainer, pl_module):
        if self._preds:
            self._flush()