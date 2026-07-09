#!/usr/bin/env python3
"""
Stage 3 -- Audio Quality Metrics (paper Section 3.4.2, Appendix E.1).

Scores every trimmed segment on a 0-100 composite scale from SNR, dynamic
range, clipping ratio, silence ratio, duration, and background-music
presence (via inaSpeechSegmenter). Recordings are bucketed into three tiers:

    >= 90            high quality
    75-89            acceptable
    <  75            low quality -- excluded from TTS use (Section 3.6)

Segments flagged `has_background_music` are also excluded from the final
TTS-ready subset regardless of their numeric score, since music in the
target audio directly corrupts the voice being modelled.

Processing is per-audiobook-folder with a resumable checkpoint, since
inaSpeechSegmenter is the slowest step in the whole pipeline.

Usage:
    python pipeline/03_score_audio_quality.py
"""

import json
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from config import AUDIO_QUALITY_DIR, TRIMMED_SEGMENTS_DIR, ensure_dirs

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # inaSpeechSegmenter pulls in TensorFlow; keep it quiet
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from inaSpeechSegmenter import Segmenter  # noqa: E402  (after TF logging is silenced)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}


class AudioQualityScorer:
    def __init__(self, main_directory: Path, output_directory: Path):
        self.main_directory = Path(main_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_directory / "processing_progress.json"
        self.master_csv = self.output_directory / "master_results.csv"
        self._segmenter = None

    def _get_segmenter(self) -> Segmenter:
        if self._segmenter is None:
            self._segmenter = Segmenter(detect_gender=False)
        return self._segmenter

    def has_background_music(self, audio_file_path: str) -> bool:
        try:
            segmentation = self._get_segmenter()(audio_file_path)
            return any(segment[0] == "music" for segment in segmentation)
        except Exception as exc:
            print(f"Music detection error for {audio_file_path}: {exc}")
            return False

    def analyze_segment(self, file_path: Path) -> dict:
        try:
            start_time = time.time()
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            target_sr = 22050 if file_size_mb > 50 else None  # downsample very large files for speed

            y, sr = librosa.load(file_path, sr=target_sr)
            duration = len(y) / sr

            metrics = {
                "file_path": str(file_path),
                "audiobook_folder": Path(file_path).parent.name,
                "filename": Path(file_path).name,
                "sample_rate": sr,
                "duration": duration,
                "file_size_mb": file_size_mb,
            }

            if duration < 1.0:
                metrics.update(
                    {
                        "snr_db": 0, "dynamic_range_db": 0, "clipping_percentage": 0,
                        "silence_percentage": 100, "has_background_music": False,
                        "is_clean": True, "quality_score": 0,
                    }
                )
                return metrics

            rms = np.sqrt(np.mean(y ** 2))
            peak_amplitude = np.max(np.abs(y))
            signal_power = rms ** 2
            noise_floor = np.percentile(np.abs(y) ** 2, 5)
            snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10))
            dynamic_range = 20 * np.log10(peak_amplitude / (rms + 1e-10))

            clipping_threshold, silence_threshold = 0.95, 0.01
            clipped = np.sum(np.abs(y) > clipping_threshold)
            silent = np.sum(np.abs(y) < silence_threshold)

            metrics.update(
                {
                    "snr_db": snr_db,
                    "dynamic_range_db": dynamic_range,
                    "clipping_percentage": (clipped / len(y)) * 100,
                    "silence_percentage": (silent / len(y)) * 100,
                }
            )

            has_music = self.has_background_music(str(file_path))
            metrics["has_background_music"] = has_music
            metrics["is_clean"] = not has_music
            metrics["quality_score"] = self.composite_score(metrics)
            metrics["processing_time"] = time.time() - start_time
            return metrics

        except Exception as exc:
            print(f"Error processing {file_path}: {exc}")
            return None

    @staticmethod
    def composite_score(metrics: dict) -> float:
        """0-100 composite score, as implemented in the working pipeline.

        NOTE: the paper's Appendix E scoring table lists different point
        values (SNR +35/+25/+15, dynamic range +15/+12/+10, duration bands
        3-15s/15-30s, "no background music" +15, and no sample-rate term) --
        that table doesn't match this runnable script bit-for-bit. Kept
        faithful to the actual pipeline code here; swap in the appendix
        values below if you need an exact reproduction of the paper's table.
        """
        score = 50

        # SNR (30 points max)
        snr = metrics["snr_db"]
        if snr > 20: score += 30
        elif snr > 10: score += 20
        elif snr > 5: score += 10

        # Sample rate (15 points max)
        sample_rate = metrics["sample_rate"]
        if sample_rate >= 44100: score += 15
        elif sample_rate >= 22050: score += 10
        elif sample_rate >= 16000: score += 5

        # Clipping penalty (-20 points max)
        clip = metrics["clipping_percentage"]
        if clip > 5: score -= 20
        elif clip > 1: score -= 10
        elif clip > 0.1: score -= 5

        # Silence penalty (-15 points max)
        silence = metrics["silence_percentage"]
        if silence > 50: score -= 15
        elif silence > 30: score -= 10
        elif silence > 20: score -= 5

        # Dynamic range (10 points max)
        dr = metrics["dynamic_range_db"]
        if dr > 20: score += 10
        elif dr > 15: score += 7
        elif dr > 10: score += 5

        # Duration (10 points max)
        duration = metrics["duration"]
        if 30 <= duration <= 300: score += 10
        elif 10 <= duration <= 600: score += 5

        # Clean audio bonus (10 points max)
        if metrics["is_clean"]:
            score += 10

        return max(0, min(100, score))

    # ------------------------------------------------------------------ #
    def _load_progress(self) -> dict:
        return json.loads(self.progress_file.read_text()) if self.progress_file.exists() else {"completed_audiobooks": []}

    def _save_progress(self, progress: dict):
        self.progress_file.write_text(json.dumps(progress, indent=2))

    def _audiobook_csv_path(self, audiobook_folder: Path) -> Path:
        return self.output_directory / f"{audiobook_folder.name}_results.csv"

    def _is_processed(self, file_path: Path, csv_path: Path) -> bool:
        if not csv_path.exists():
            return False
        try:
            return str(file_path) in pd.read_csv(csv_path)["file_path"].values
        except Exception:
            return False

    def _audiobook_folders(self):
        for item in self.main_directory.iterdir():
            if item.is_dir() and any(item.glob(f"*{ext}") for ext in AUDIO_EXTENSIONS):
                yield item

    def process_audiobook_folder(self, audiobook_folder: Path, progress: dict):
        csv_path = self._audiobook_csv_path(audiobook_folder)
        audio_files = [f for ext in AUDIO_EXTENSIONS for f in audiobook_folder.glob(f"*{ext}")]
        pending = [f for f in audio_files if not self._is_processed(f, csv_path)]
        if not pending:
            return

        print(f"Scoring {len(pending)} segments in {audiobook_folder.name}")
        for i in range(0, len(pending), 10):
            batch = pending[i : i + 10]
            with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
                results = [r for r in executor.map(self.analyze_segment, batch) if r is not None]
            if results:
                df_new = pd.DataFrame(results)
                if csv_path.exists():
                    df_new = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True).drop_duplicates(subset=["file_path"])
                df_new.to_csv(csv_path, index=False)

        progress["completed_audiobooks"].append(audiobook_folder.name)
        self._save_progress(progress)

    def create_master_csv(self) -> pd.DataFrame:
        dfs = [pd.read_csv(f) for f in self.output_directory.glob("*_results.csv") if f.name != "master_results.csv"]
        if not dfs:
            return pd.DataFrame()
        master = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["file_path"])
        master = master.sort_values(["audiobook_folder", "filename"])
        master.to_csv(self.master_csv, index=False)
        return master

    def run(self) -> pd.DataFrame:
        progress = self._load_progress()
        folders = [f for f in self._audiobook_folders() if f.name not in progress["completed_audiobooks"]]
        print(f"Scoring audio quality for {len(folders)} audiobook folders")
        for folder in folders:
            self.process_audiobook_folder(folder, progress)

        master_df = self.create_master_csv()
        if not master_df.empty:
            print(f"Average quality score: {master_df['quality_score'].mean():.1f}")
            print(f"Clean (no background music): {master_df['is_clean'].sum()}/{len(master_df)}")
        return master_df


def filter_by_quality(
    df: pd.DataFrame,
    min_quality_score: float = 75,
    min_sample_rate: int = 22050,
    max_clipping: float = 1.0,
    clean_only: bool = False,
) -> pd.DataFrame:
    """Section 3.6's release rule is a single composite-score threshold
    (score >= 75); this also keeps the pipeline's extra sample-rate/clipping
    guardrails from the original script -- pass min_sample_rate=0,
    max_clipping=100 to disable them and filter on quality_score alone."""
    if df.empty:
        return df
    filtered = df[
        (df["quality_score"] >= min_quality_score)
        & (df["sample_rate"] >= min_sample_rate)
        & (df["clipping_percentage"] <= max_clipping)
    ]
    return filtered[filtered["is_clean"]] if clean_only else filtered


if __name__ == "__main__":
    ensure_dirs(AUDIO_QUALITY_DIR)
    scorer = AudioQualityScorer(TRIMMED_SEGMENTS_DIR, AUDIO_QUALITY_DIR)
    results_df = scorer.run()

    if not results_df.empty:
        high_quality_clean = filter_by_quality(results_df, min_quality_score=75, clean_only=True)
        print(f"High quality + clean segments: {len(high_quality_clean)}/{len(results_df)}")
        high_quality_clean.to_csv(AUDIO_QUALITY_DIR / "high_quality_clean_segments.csv", index=False)
