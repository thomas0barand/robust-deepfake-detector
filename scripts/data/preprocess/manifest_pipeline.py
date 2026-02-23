import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_REPO_ID = "aurelvague/audio-genai"

class ManifestPipeline:
    def __init__(self, default_label: str = "ai", schema_version: str = "v1") -> None:
        if default_label not in {"ai", "human"}:
            raise ValueError("default_label must be 'ai' or 'human'")
        self.default_label = default_label
        self.schema_version = schema_version

    @staticmethod
    def read_csv(path: Path) -> pd.DataFrame:
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Only CSV input is supported, got: {path}")
        return pd.read_csv(path)

    @staticmethod
    def write_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Only CSV output is supported, got: {path}")
        df.to_csv(path, index=False)

    @staticmethod
    def _infer_generator(source: str) -> tuple[str, str]:
        s = source.lower()
        if "suno" in s and "50" in s:
            return "suno", "5.0"
        if "suno" in s and "35" in s:
            return "suno", "3.5"
        if "udio" in s:
            return "udio", "unknown"
        return "unknown", "unknown"

    @staticmethod
    def _pick_text(row: pd.Series, key: str):
        value = row.get(key)
        if pd.isna(value):
            return None
        value = str(value).strip()
        return value if value else None

    @staticmethod
    def _make_sample_id(source: str, source_track_id: str, audio_uri: str) -> str:
        seed = f"{source}|{source_track_id}|{audio_uri}".encode("utf-8")
        return hashlib.sha1(seed).hexdigest()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for _, row in df.iterrows():
            source = str(row.get("source") or "unknown")
            source_track_id = str(row.get("source_track_id") or "")
            audio_uri = row.get("audio_uri")
            if pd.isna(audio_uri) or not str(audio_uri).strip():
                continue
            audio_uri = str(audio_uri).strip()
            inferred_generator, inferred_version = self._infer_generator(source)
            generator = self._pick_text(row, "generator") or inferred_generator
            version = self._pick_text(row, "generator_version") or inferred_version
            label = self._pick_text(row, "label") or self.default_label
            if label not in {"ai", "human"}:
                label = self.default_label

            styles = row.get("styles")
            if isinstance(styles, str):
                try:
                    styles = json.loads(styles)
                except json.JSONDecodeError:
                    styles = [styles]
            if not isinstance(styles, list):
                styles = None

            metadata_raw = row.get("metadata")
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    metadata = {"raw": metadata_raw}
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw
            else:
                metadata = None

            rows.append(
                {
                    "sample_id": self._make_sample_id(source, source_track_id, audio_uri),
                    "source": source,
                    "source_track_id": source_track_id,
                    "generator": generator,
                    "generator_version": version,
                    "label": label,
                    "label_confidence": 1.0,
                    "split": None,
                    "audio_uri": audio_uri,
                    "audio_local_path": None,
                    "title": row.get("title"),
                    "artist": row.get("artist"),
                    "username": row.get("username"),
                    "styles": json.dumps(styles, ensure_ascii=False) if styles is not None else None,
                    "metadata": json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
                    "duration_sec": None,
                    "sample_rate": None,
                    "channels": None,
                    "bitrate": None,
                    "license": None,
                    "usage_rights": None,
                    "sha256": None,
                    "ingested_at": now,
                    "schema_version": self.schema_version,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop_duplicates(subset=["audio_uri"], keep="first")
        out = out.drop_duplicates(subset=["source", "source_track_id"], keep="first")
        return out

    def run(self, input_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        frames = [self.read_csv(path) for path in input_paths]
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        canonical = self.normalize(merged)
        deduped = self.deduplicate(canonical)
        return canonical, deduped

    @staticmethod
    def push_to_hub(
        repo_id: str,
        raw_paths: list[Path],
        processed_paths: list[Path],
        private: bool = True,
        token: str | None = None,
        commit_message: str = "Upload raw and processed manifests",
    ) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError("huggingface_hub is required for push_to_hub().") from exc

        resolved_token = token or os.getenv("HF_TOKEN")
        if not resolved_token:
            raise ValueError("Missing HF token. Pass --hub-token or set HF_TOKEN.")

        api = HfApi(token=resolved_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

        for path in raw_paths:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"raw/{path.name}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )

        for path in processed_paths:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"processed/{path.name}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical manifest from raw CSV source exports.")
    parser.add_argument(
        "--raw-input",
        action="append",
        default=[],
        help="Path to raw source CSV produced by notebooks (repeatable)",
    )
    parser.add_argument("--workdir", default="data/interim", help="Intermediate output folder")
    parser.add_argument("--output", default="data/processed/manifest_v1.csv", help="Final manifest output")
    parser.add_argument("--default-label", default="ai", choices=["ai", "human"], help="Default label")
    parser.add_argument("--hub-repo-id", default=DEFAULT_REPO_ID, help="Optional HF dataset repo ID (e.g. org/dataset-name)")
    parser.add_argument("--hub-token", help="Optional HF token. Falls back to HF_TOKEN env var")
    parser.add_argument("--hub-private", action="store_true", help="Create/push repo as private")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    raw_tables: list[Path] = [Path(p) for p in args.raw_input]

    if not raw_tables:
        raise ValueError("No inputs provided. Use --raw-input at least once.")

    pipeline = ManifestPipeline(default_label=args.default_label)
    canonical, deduped = pipeline.run(raw_tables)

    canonical_path = workdir / "canonical_manifest.csv"
    output_path = Path(args.output)
    pipeline.write_csv(canonical, canonical_path)
    pipeline.write_csv(deduped, output_path)

    print(f"Wrote {len(canonical)} canonical rows to {canonical_path}")
    print(f"Deduplicated manifest from {len(canonical)} to {len(deduped)} rows")
    print(f"Final manifest written to {output_path}")

    if args.hub_repo_id:
        pipeline.push_to_hub(
            repo_id=args.hub_repo_id,
            raw_paths=raw_tables,
            processed_paths=[output_path],
            private=args.hub_private,
            token=args.hub_token,
        )
        print(f"Pushed raw and processed files to HF dataset repo: {args.hub_repo_id}")


if __name__ == "__main__":
    main()
