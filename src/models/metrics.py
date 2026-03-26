import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import lightning as L
from sklearn.metrics import roc_auc_score, f1_score, precision_score, accuracy_score, recall_score, roc_curve, precision_recall_curve
from pathlib import Path


class MetricsCallback(L.Callback):
    def __init__(self, output_dir="results/", filename="test", threshold_metric="f1"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.threshold_metric = threshold_metric
        self._preds, self._labels = [], []
        # Latency tracking (per-batch, in milliseconds)
        self._batch_latencies: list[float] = []
        self._batch_sizes: list[int] = []

    def _collect(self, pl_module, batch):
        x, y, lag = batch
        t_start = time.perf_counter()
        with torch.inference_mode():
            if pl_module.transform_type == "stft" and pl_module.log_stft:
                x = pl_module.stft_to_log(x)
            logits, cross_corr = pl_module(x.to(pl_module.device), convolve=pl_module.use_convolution)
            preds = torch.sigmoid(logits).squeeze(-1)
        # Sync CUDA ops before measuring elapsed time
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t_start) * 1_000

        self._preds.append(preds.cpu())
        self._labels.append(y.cpu())
        self._batch_latencies.append(elapsed_ms)
        self._batch_sizes.append(x.shape[0])

    def _compute_latency_stats(self) -> dict:
        """Return latency statistics (ms) across all collected batches."""
        latencies = np.array(self._batch_latencies)
        sizes = np.array(self._batch_sizes)
        total_samples = sizes.sum()
        total_ms = latencies.sum()
        per_sample_ms = latencies / sizes  # per-sample latency for every batch

        return {
            "latency_per_sample_mean_ms":   float(per_sample_ms.mean()),
            "throughput_samples_per_sec": float(total_samples / (total_ms / 1_000)),
        }

    def _flush(self):
        preds = torch.cat(self._preds).numpy()
        labels = torch.cat(self._labels).numpy().astype(int)

        # Compute latency stats before resetting
        latency_stats = self._compute_latency_stats()

        # Reset state
        self._preds, self._labels = [], []
        self._batch_latencies, self._batch_sizes = [], []

        thresholds = np.linspace(0.01, 0.99, 199)
        f1_scores = [f1_score(labels, (preds >= t).astype(int), zero_division=0) for t in thresholds]
        best_t = thresholds[np.argmax(f1_scores)]
        pred_bin = (preds >= best_t).astype(int)

        metrics = {
            "auroc":     roc_auc_score(labels, preds),
            "f1":        f1_score(labels, pred_bin, zero_division=0),
            "precision": precision_score(labels, pred_bin, zero_division=0),
            "accuracy":  accuracy_score(labels, pred_bin),
            "recall":    recall_score(labels, pred_bin, zero_division=0),
            "threshold": best_t,
        }

        all_stats = {**metrics, **latency_stats}
        self._save_metrics(all_stats)
        self._plot(preds, labels, metrics, best_t, f1_scores, thresholds, latency_stats)

        print(
            f"\n[TEST] threshold={best_t:.3f} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "threshold")
            + f"\n[LATENCY] mean={latency_stats['latency_per_sample_mean_ms']:.1f}ms"
            + f"  throughput={latency_stats['throughput_samples_per_sec']:.1f} samples/s"
        )
        return metrics, best_t

    # Persistence helpers
    def _save_metrics(self, stats: dict):
        """Save all metrics + latency stats to JSON and append a row to CSV."""
        # JSON (one file per run)
        json_path = self.output_dir / f"{self.filename}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

        # CSV (append so successive runs accumulate)
        csv_path = self.output_dir / "all_runs_metrics.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run", *stats.keys()])
            if write_header:
                writer.writeheader()
            writer.writerow({"run": self.filename, **stats})

        print(f"[METRICS] Saved → {json_path}  (CSV → {csv_path})")

    # Plotting
    def _plot(self, preds, labels, metrics, best_t, f1_scores, thresholds, latency_stats):
        fig, axes = plt.subplots(1, 4, figsize=(24, 4))
        fig.suptitle(
            "TEST | " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "threshold"),
            fontsize=10,
        )

        # Score distribution
        bins = np.linspace(0, 1, 51)
        axes[0].hist(preds[labels == 0], bins=bins, alpha=0.7, density=True, label="Real")
        axes[0].hist(preds[labels == 1], bins=bins, alpha=0.7, density=True, label="Fake")
        axes[0].axvline(best_t, color="yellow", linestyle="--", label=f"t={best_t:.2f}")
        axes[0].set(title="Score Distribution", xlabel="P(fake)")
        axes[0].legend(fontsize=8)

        # ROC
        fpr, tpr, _ = roc_curve(labels, preds)
        axes[1].plot(fpr, tpr, label=f"AUC={metrics['auroc']:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[1].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
        axes[1].legend(fontsize=8)

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(labels, preds)
        axes[2].plot(rec, prec)
        axes[2].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")

        # F1 vs Threshold
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