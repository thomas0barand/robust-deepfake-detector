import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import lightning as L
from sklearn.metrics import roc_auc_score, f1_score, precision_score, accuracy_score, recall_score, roc_curve, confusion_matrix
from pathlib import Path

METRIC_RANGE = (0.7, 1.4)
AF_PLOT_POINTS = np.linspace(0.5, 2.0, 16)

class MetricsCallback(L.Callback):
    def __init__(self, output_dir="results/", filename="test", threshold_metric="f1"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.threshold_metric = threshold_metric
        self._preds, self._labels, self._afs = [], [], []
        self._batch_latencies, self._batch_sizes = [], []

    def _collect(self, pl_module, batch):
        x, y, af = batch
        t0 = time.perf_counter()
        with torch.inference_mode():
            if pl_module.transform_type == "stft" and pl_module.log_stft:
                x = pl_module.stft_to_log(x)
            logits, _ = pl_module(x.to(pl_module.device), convolve=pl_module.use_convolution)
            preds = torch.sigmoid(logits).squeeze(-1)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        self._preds.append(preds.cpu())
        self._labels.append(y.cpu())
        self._afs.append(af.cpu() if isinstance(af, torch.Tensor) else torch.tensor(af, dtype=torch.float32))
        self._batch_latencies.append((time.perf_counter() - t0) * 1000)
        self._batch_sizes.append(x.shape[0])

    def _latency_stats(self):
        lat, sz = np.array(self._batch_latencies), np.array(self._batch_sizes)
        return {
            "latency_per_sample_mean_ms": float((lat / sz).mean()),
            "throughput_samples_per_sec": float(sz.sum() / (lat.sum() / 1000)),
        }

    def _flush(self):
        all_preds = torch.cat(self._preds).numpy()
        all_labels = torch.cat(self._labels).numpy().astype(int)
        all_afs = torch.cat(self._afs).numpy()
        latency_stats = self._latency_stats()
        self._preds, self._labels, self._afs = [], [], []
        self._batch_latencies, self._batch_sizes = [], []

        lo, hi = METRIC_RANGE
        mask = (all_afs > lo) & (all_afs < hi)
        preds, labels = all_preds[mask], all_labels[mask]

        thresholds = np.linspace(0.01, 0.99, 199)
        f1s = [f1_score(labels, (preds >= t).astype(int), zero_division=0) for t in thresholds]
        best_t = thresholds[np.argmax(f1s)]
        pred_bin = (preds >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pred_bin).ravel()

        metrics = {
            "auroc":     roc_auc_score(labels, preds),
            "f1":        f1_score(labels, pred_bin, zero_division=0),
            "precision": precision_score(labels, pred_bin, zero_division=0),
            "accuracy":  accuracy_score(labels, pred_bin),
            "recall":    recall_score(labels, pred_bin, zero_division=0),
            "fpr":       float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "threshold": best_t,
        }

        stats = {**metrics, **latency_stats}
        json_path = self.output_dir / f"{self.filename}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

        csv_path = self.output_dir / "all_runs_metrics.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", *stats.keys()])
            if write_header:
                w.writeheader()
            w.writerow({"run": self.filename, **stats})

        self._plot(preds, labels, metrics, best_t, f1s, thresholds, all_preds, all_labels, all_afs)
        print(
            f"\n[TEST] Attacks in range {METRIC_RANGE}: ({mask.sum()}/{len(mask)}) | threshold={best_t:.3f} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "threshold")
            + f"\n[LATENCY] mean={latency_stats['latency_per_sample_mean_ms']:.1f}ms  "
            + f"throughput={latency_stats['throughput_samples_per_sec']:.1f} samples/s"
        )

    def _f1_vs_af(self, all_preds, all_labels, all_afs, best_t, hw=0.1):
        f1s, counts = np.full(len(AF_PLOT_POINTS), np.nan), np.zeros(len(AF_PLOT_POINTS), int)
        for i, c in enumerate(AF_PLOT_POINTS):
            m = (all_afs >= c - hw) & (all_afs < c + hw)
            if m.sum() >= 2 and len(np.unique(all_labels[m])) == 2:
                f1s[i] = f1_score(all_labels[m], (all_preds[m] >= best_t).astype(int), zero_division=0)
                counts[i] = m.sum()
        return f1s, counts

    def _plot(self, preds, labels, metrics, best_t, f1s, thresholds, all_preds, all_labels, all_afs):
        fig, axes = plt.subplots(1, 4, figsize=(25, 4))
        fig.suptitle("TEST | " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "threshold"), fontsize=10)

        bins = np.linspace(0, 1, 51)
        axes[0].hist(preds[labels == 0], bins=bins, alpha=0.7, density=True, label="Real")
        axes[0].hist(preds[labels == 1], bins=bins, alpha=0.7, density=True, label="Fake")
        axes[0].axvline(best_t, color="yellow", linestyle="--", label=f"t={best_t:.2f}")
        axes[0].set(title="Score Distribution", xlabel="P(fake)"); axes[0].legend(fontsize=8)

        fpr, tpr, _ = roc_curve(labels, preds)
        axes[1].plot(fpr, tpr, label=f"AUC={metrics['auroc']:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[1].set(title="ROC Curve", xlabel="FPR", ylabel="TPR"); axes[1].legend(fontsize=8)

        axes[2].plot(thresholds, f1s)
        axes[2].axvline(best_t, color="yellow", linestyle="--", label=f"best={best_t:.2f}")
        axes[2].set(title="F1 vs Threshold", xlabel="Threshold", ylabel="F1"); axes[2].legend(fontsize=8)

        f1_af, counts = self._f1_vs_af(all_preds, all_labels, all_afs, best_t)
        valid = ~np.isnan(f1_af)
        axes[3].plot(AF_PLOT_POINTS[valid], f1_af[valid], marker="o", linewidth=1.5, markersize=5)
        axes[3].axvspan(*METRIC_RANGE, alpha=0.12, color="green", label=f"metric range {METRIC_RANGE}")
        axes[3].set(title="F1 vs Attack Factor", xlabel="Attack Factor", ylabel="F1", xlim=(0.4, 2.1), ylim=(0, 1.05))
        axes[3].grid(alpha=0.3)
        axes[3].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(self.output_dir / f"{self.filename}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._collect(pl_module, batch)

    def on_test_epoch_end(self, trainer, pl_module):
        if self._preds:
            self._flush()