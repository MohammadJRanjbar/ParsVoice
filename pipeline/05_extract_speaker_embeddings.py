#!/usr/bin/env python3
"""
Stage 5 -- Speaker Embedding Extraction (paper Section 3.5, Appendix D).

Extracts a fixed-size ECAPA-TDNN speaker embedding (Desplanques et al., 2020)
for every boundary-optimized segment, using the pretrained SpeechBrain
checkpoint `speechbrain/spkrec-ecapa-voxceleb`. These per-segment embeddings
are the input to Stage 6 (local clustering within a book, then global
merging across books).

Embeddings are cached to disk as one `.npy` file per segment, plus one CSV
manifest per audiobook folder (`embedding_path` + narrator/title/completeness
metadata carried over from Stages 2-3) so Stage 6 never has to recompute
them. Processing is GPU-batched and checkpointed at (audiobook, file-index)
granularity so a run over the full corpus is safe to kill and resume.

NOTE: `completion_status` and `final_duration` are carried through here
beyond what the original embedding-extraction script recorded, because
Stage 7's Section-3.6 release filter (exclude segments flagged incomplete)
and hour totals need them and nothing later in the pipeline re-reads the
Stage 2 metadata directly.

Usage:
    python pipeline/05_extract_speaker_embeddings.py
    python pipeline/05_extract_speaker_embeddings.py --test   # 3 books, 10 files each
"""

import argparse
import gc
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from config import EMBEDDINGS_DIR, SPEAKER_EMBEDDING_MODEL, TRIMMED_SEGMENTS_DIR, ensure_dirs

warnings.filterwarnings("ignore")

CHECKPOINT_FILE_NAME = "processing_checkpoint.json"


class SpeakerEmbeddingExtractor:
    def __init__(
        self,
        main_folder_path: Path,
        output_folder_path: Path,
        test_mode: bool = False,
        max_audiobooks: int = 3,
        max_files_per_book: int = 10,
        batch_size: Optional[int] = None,
        use_gpu: bool = True,
    ):
        self.main_folder_path = Path(main_folder_path)
        self.output_folder_path = Path(output_folder_path)
        self.output_folder_path.mkdir(parents=True, exist_ok=True)

        self.test_mode = test_mode
        self.max_audiobooks = max_audiobooks
        self.max_files_per_book = max_files_per_book
        self.use_gpu = use_gpu and torch.cuda.is_available()
        # GPU: batch_size=8 was fastest in benchmarking (40+ files/sec); CPU: 2 (2.7+ files/sec).
        self.batch_size = batch_size or (8 if self.use_gpu else 2)

        self.checkpoint_file = self.output_folder_path / CHECKPOINT_FILE_NAME
        self.checkpoint = self._load_checkpoint()
        self.model = None
        self.performance_stats = {"total_files_processed": 0, "total_processing_time": 0.0, "files_per_second": 0.0}

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    def _load_model(self):
        if self.model is None:
            from speechbrain.inference.speaker import SpeakerRecognition

            device = "cuda" if self.use_gpu else "cpu"
            print(f"Loading {SPEAKER_EMBEDDING_MODEL} on {device}...")
            self.model = SpeakerRecognition.from_hparams(
                source=SPEAKER_EMBEDDING_MODEL,
                savedir=str(self.output_folder_path / ".model_cache" / "spkrec-ecapa-voxceleb"),
                run_opts={"device": device},
            )
        return self.model

    # ------------------------------------------------------------------ #
    # Checkpointing: resume at (audiobook, file-index) granularity
    # ------------------------------------------------------------------ #
    def _load_checkpoint(self) -> Dict:
        if self.checkpoint_file.exists():
            try:
                checkpoint = json.loads(self.checkpoint_file.read_text())
                print(f"Checkpoint loaded: last audiobook = {checkpoint.get('last_audiobook', 'None')}")
                return checkpoint
            except Exception as exc:
                print(f"Error loading checkpoint: {exc}")
        return {}

    def _save_checkpoint(self, audiobook_name: str, file_index: int, total_files: int):
        self.checkpoint_file.write_text(
            json.dumps(
                {
                    "last_audiobook": audiobook_name,
                    "last_file_index": file_index,
                    "total_files_in_current_book": total_files,
                    "timestamp": pd.Timestamp.now().isoformat(),
                },
                indent=2,
            )
        )

    def _clear_checkpoint(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("Checkpoint cleared -- processing complete!")

    def _starting_file_index(self, audiobook_name: str) -> int:
        if not self.checkpoint or self.checkpoint.get("last_audiobook") != audiobook_name:
            return 0
        return self.checkpoint.get("last_file_index", 0) + 1

    # ------------------------------------------------------------------ #
    # Batch audio loading + embedding extraction
    # ------------------------------------------------------------------ #
    def _load_audio_batch(self, file_paths: List[str]) -> Tuple[List[torch.Tensor], List[int]]:
        tensors, valid_indices = [], []
        for i, file_path in enumerate(file_paths):
            try:
                signal, _ = torchaudio.load(file_path, normalize=True)
                if self.use_gpu:
                    signal = signal.cuda()
                tensors.append(signal)
                valid_indices.append(i)
            except Exception as exc:
                print(f"  Error loading {file_path}: {exc}")
        return tensors, valid_indices

    def _extract_batch_embeddings(self, audio_tensors: List[torch.Tensor]) -> List[Optional[np.ndarray]]:
        embeddings = []
        for signal in audio_tensors:
            try:
                with torch.no_grad():
                    embedding = self.model.encode_batch(signal)
                embeddings.append(embedding.squeeze().detach().cpu().numpy())
            except Exception as exc:
                print(f"  Error extracting embedding: {exc}")
                embeddings.append(None)
        return embeddings

    # ------------------------------------------------------------------ #
    # Per-book processing
    # ------------------------------------------------------------------ #
    def _files_to_process(self, df: pd.DataFrame, start_index: int, output_dir: Path, existing_paths: set) -> List[Tuple]:
        pending = []
        for idx in range(start_index, len(df)):
            row = df.iloc[idx]
            segment_path = row["fixed_segment_path"]
            if segment_path in existing_paths or not os.path.exists(segment_path):
                continue

            embedding_filename = f"{Path(segment_path).stem}_embedding.npy"
            embedding_path = output_dir / embedding_filename
            if embedding_path.exists():
                continue

            metadata = {
                "narrator": row.get("narrator", ""),
                "title_index": row.get("title_index", ""),
                "original_audio_book_name": row.get("original_audio_book_name", row.get("audiobook_name", "")),
                # Carried forward beyond the original script's metadata set --
                # Stage 7 needs the transcript itself (it's the whole point of
                # the release manifest), completion_status to apply Section
                # 3.6's release filter, and the post-trim duration (Stage 2's
                # final_duration_ms, not Stage 1's pre-trim final_duration)
                # for accurate release hour totals.
                "final_transcript": row.get("final_transcript", ""),
                "completion_status": row.get("completion_status", ""),
                "final_duration_ms": row.get("final_duration_ms", ""),
            }
            pending.append((idx, segment_path, str(embedding_path), embedding_filename, metadata))
        return pending

    def _process_batch(self, file_batch: List[Tuple], output_dir: Path) -> List[Dict]:
        if not file_batch:
            return []
        self._load_model()

        file_paths = [item[1] for item in file_batch]
        batch_start = time.time()
        audio_tensors, valid_indices = self._load_audio_batch(file_paths)
        embeddings = self._extract_batch_embeddings(audio_tensors)

        results = []
        for i, (idx, segment_path, embedding_path, embedding_filename, metadata) in enumerate(file_batch):
            embedding = embeddings[valid_indices.index(i)] if i in valid_indices else None
            if embedding is not None:
                np.save(embedding_path, embedding)
                results.append(
                    {"idx": idx, "segment_path": segment_path, "embedding_path": embedding_path, "metadata": metadata, "success": True}
                )
            else:
                results.append(
                    {"idx": idx, "segment_path": segment_path, "embedding_path": None, "metadata": metadata, "success": False}
                )

        del audio_tensors, embeddings
        if self.use_gpu:
            torch.cuda.empty_cache()

        successful = sum(1 for r in results if r["success"])
        if successful:
            elapsed = time.time() - batch_start
            print(f"    batch: {successful} files in {elapsed:.2f}s ({successful / elapsed:.2f} files/sec)")
        return results

    def process_audiobook(self, audiobook_dir: Path):
        audiobook_name = audiobook_dir.name
        print(f"\nProcessing audiobook: {audiobook_name}")

        metadata_csv = audiobook_dir / "fixed_metadata.csv"
        if not metadata_csv.exists():
            print(f"  No fixed_metadata.csv in {audiobook_dir} -- run stage 2 first. Skipping.")
            return

        df = pd.read_csv(metadata_csv)
        if "fixed_segment_path" not in df.columns:
            print(f"  'fixed_segment_path' column missing in {metadata_csv}. Skipping.")
            return
        if self.test_mode:
            df = df.head(self.max_files_per_book)

        output_dir = self.output_folder_path / audiobook_name
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"{audiobook_name}_embeddings.csv"

        existing_paths = set()
        if manifest_path.exists():
            existing_paths = set(pd.read_csv(manifest_path)["segment_path"].tolist())
            print(f"  Found existing manifest with {len(existing_paths)} entries")

        start_index = self._starting_file_index(audiobook_name)
        pending = self._files_to_process(df, start_index, output_dir, existing_paths)
        if not pending:
            print(f"  Nothing to do for {audiobook_name}.")
            return

        print(f"  {len(pending)} segments to embed (batch size {self.batch_size}, {'GPU' if self.use_gpu else 'CPU'})")

        all_results = []
        audiobook_start = time.time()
        for i in tqdm(range(0, len(pending), self.batch_size), desc=audiobook_name):
            batch_results = self._process_batch(pending[i : i + self.batch_size], output_dir)
            all_results.extend(batch_results)

            successful = [r for r in batch_results if r["success"]]
            if successful:
                self._save_checkpoint(audiobook_name, max(r["idx"] for r in successful), len(df))
            if i % (self.batch_size * 4) == 0:
                gc.collect()
                if self.use_gpu:
                    torch.cuda.empty_cache()

        manifest_rows = [
            {
                "segment_path": r["segment_path"],
                "embedding_path": r["embedding_path"],
                "narrator": r["metadata"].get("narrator", ""),
                "title_index": r["metadata"].get("title_index", ""),
                "original_audio_book_name": r["metadata"].get("original_audio_book_name", ""),
                "final_transcript": r["metadata"].get("final_transcript", ""),
                "completion_status": r["metadata"].get("completion_status", ""),
                "final_duration_ms": r["metadata"].get("final_duration_ms", ""),
            }
            for r in all_results
            if r["success"]
        ]

        if manifest_rows:
            new_df = pd.DataFrame(manifest_rows)
            if manifest_path.exists():
                new_df = pd.concat([pd.read_csv(manifest_path), new_df], ignore_index=True).drop_duplicates(
                    subset=["segment_path"], keep="last"
                )
            new_df.to_csv(manifest_path, index=False)
            print(f"  Saved manifest with {len(new_df)} embeddings -> {manifest_path}")

        elapsed = time.time() - audiobook_start
        n_done = len(manifest_rows)
        if n_done:
            print(f"  {n_done} files in {elapsed:.1f}s ({n_done / elapsed:.2f} files/sec)")
            self.performance_stats["total_files_processed"] += n_done
            self.performance_stats["total_processing_time"] += elapsed

        self._save_checkpoint(audiobook_name, len(df) - 1, len(df))
        gc.collect()
        if self.use_gpu:
            torch.cuda.empty_cache()

    def run(self):
        print("Starting speaker embedding extraction...")
        print(f"Device: {'GPU (CUDA)' if self.use_gpu else 'CPU'}  Batch size: {self.batch_size}  Test mode: {self.test_mode}")

        audiobook_dirs = sorted(d for d in self.main_folder_path.iterdir() if d.is_dir())
        if self.test_mode:
            audiobook_dirs = audiobook_dirs[: self.max_audiobooks]
        print(f"Found {len(audiobook_dirs)} audiobook folders")

        start_index = 0
        if self.checkpoint.get("last_audiobook"):
            names = [d.name for d in audiobook_dirs]
            if self.checkpoint["last_audiobook"] in names:
                last_index = names.index(self.checkpoint["last_audiobook"])
                completed = self.checkpoint.get("last_file_index", 0) >= self.checkpoint.get("total_files_in_current_book", 0) - 1
                start_index = last_index + 1 if completed else last_index

        for audiobook_dir in audiobook_dirs[start_index:]:
            self.process_audiobook(audiobook_dir)

        self._clear_checkpoint()
        total_time = self.performance_stats["total_processing_time"]
        total_files = self.performance_stats["total_files_processed"]
        print(f"\nDone. {total_files} embeddings extracted in {total_time:.1f}s ({total_files / max(total_time, 1):.2f}/s)")
        print(f"Results saved in: {self.output_folder_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 5: extract ECAPA-TDNN speaker embeddings")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 audiobooks, 10 files each")
    args = parser.parse_args()

    ensure_dirs(EMBEDDINGS_DIR)
    extractor = SpeakerEmbeddingExtractor(
        main_folder_path=TRIMMED_SEGMENTS_DIR,
        output_folder_path=EMBEDDINGS_DIR,
        test_mode=args.test,
    )
    extractor.run()


if __name__ == "__main__":
    main()
