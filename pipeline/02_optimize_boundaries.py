#!/usr/bin/env python3
"""
Stage 2 -- Boundary Optimization Algorithm (paper Section 3.3).

Even after Stage 1, segment boundaries can carry leading/trailing silence,
breath noise, or acoustic artifacts that hurt downstream TTS training. This
stage trims each boundary independently using a hybrid search:

    1. Initial adjustment: remove up to 3s from the boundary and re-transcribe.
    2. Binary search: halve the trim interval until the re-transcription is
       character-for-character identical to the original transcript, finding
       the largest "safe" trim quickly.
    3. Linear fine-tuning (+-5 steps of 0.1s) around the binary-search result,
       verifying every candidate, to correct for ASR non-determinism near the
       boundary and land on the true maximal safe trim.

The end boundary is processed first, then the start boundary on the
already-trimmed audio. Processing is batched, checkpointed at the segment and
book level, and safe to resume.

Usage:
    python pipeline/02_optimize_boundaries.py
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment

from config import SEGMENTS_DIR, TRIMMED_SEGMENTS_DIR, ensure_dirs

BATCH_SIZE = 256
TRIM_STEP_MS = 100  # 0.1s, matching the paper's fine-tuning granularity
LANGUAGE = "fa-IR"
# NOTE: the original script ran with MAX_WORKERS=128, sized for the lab's
# infrastructure. Defaulting lower here since 128 concurrent requests against
# the free Google ASR endpoint is likely to get you rate-limited; raise it if
# you have your own ASR backend or higher rate limits.
MAX_WORKERS = 32
CHECKPOINT_FREQUENCY = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DETAILED_CHECKPOINT_FILE = TRIMMED_SEGMENTS_DIR / "detailed_checkpoint.json"
PROGRESS_CHECKPOINT_FILE = TRIMMED_SEGMENTS_DIR / "progress_checkpoint.json"
COMPLETED_BOOKS_CHECKPOINT_FILE = TRIMMED_SEGMENTS_DIR / "completed_books_checkpoint.json"


@dataclass
class ProcessingResult:
    success: bool
    file_info: Dict
    error_message: Optional[str] = None


class BoundaryOptimizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.total_processed = 0
        self.start_time = time.time()

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def load_checkpoints(self) -> Tuple[Dict, Dict, Dict]:
        detailed = json.loads(DETAILED_CHECKPOINT_FILE.read_text()) if DETAILED_CHECKPOINT_FILE.exists() else {"processed_segments": {}}
        progress = json.loads(PROGRESS_CHECKPOINT_FILE.read_text()) if PROGRESS_CHECKPOINT_FILE.exists() else {"total_processed_count": 0}
        completed = json.loads(COMPLETED_BOOKS_CHECKPOINT_FILE.read_text()) if COMPLETED_BOOKS_CHECKPOINT_FILE.exists() else {"completed_books": []}
        return detailed, progress, completed

    def save_detailed_checkpoint(self, detailed: Dict):
        DETAILED_CHECKPOINT_FILE.write_text(json.dumps(detailed, indent=2))

    def save_progress_checkpoint(self, total_count: int):
        PROGRESS_CHECKPOINT_FILE.write_text(
            json.dumps({"total_processed_count": total_count, "last_checkpoint_time": time.time()}, indent=2)
        )

    def mark_book_completed(self, book_name: str):
        _, _, completed = self.load_checkpoints()
        if book_name not in completed["completed_books"]:
            completed["completed_books"].append(book_name)
            COMPLETED_BOOKS_CHECKPOINT_FILE.write_text(json.dumps(completed, indent=2))

    # ------------------------------------------------------------------ #
    # Transcription helpers
    # ------------------------------------------------------------------ #
    async def _transcribe(self, audio_data: sr.AudioData) -> Optional[str]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                return await loop.run_in_executor(pool, lambda: self.recognizer.recognize_google(audio_data, language=LANGUAGE))
            except sr.UnknownValueError:
                return None
            except Exception as exc:
                logger.debug(f"ASR error: {exc}")
                return None

    async def _transcribe_batch(self, candidates: List[Tuple[str, sr.AudioData]]) -> List[Tuple[str, Optional[str]]]:
        async def one(seg_id, audio_data):
            return seg_id, await self._transcribe(audio_data)

        results = await asyncio.gather(*[one(sid, ad) for sid, ad in candidates], return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    # ------------------------------------------------------------------ #
    # Hybrid binary + linear boundary search (Section 3.3, steps 1-4)
    # ------------------------------------------------------------------ #
    async def hybrid_search_trim(self, audio: AudioSegment, target_transcript: str, max_trim_ms: int, trim_from_end: bool = True) -> int:
        max_steps = max_trim_ms // TRIM_STEP_MS

        def trimmed(step: int) -> AudioSegment:
            trim_ms = step * TRIM_STEP_MS
            if trim_ms == 0:
                return audio
            return audio[:-trim_ms] if trim_from_end else audio[trim_ms:]

        # Phase A: binary search for the largest trim that still matches exactly.
        left, right, binary_best = 0, max_steps, 0
        while left <= right:
            mid = (left + right) // 2
            candidate = trimmed(mid)
            if len(candidate) < 500:
                right = mid - 1
                continue
            audio_data = sr.AudioData(candidate.raw_data, candidate.frame_rate, candidate.sample_width)
            transcription = await self._transcribe(audio_data)
            if transcription and transcription.strip() == target_transcript.strip():
                binary_best = mid
                left = mid + 1
            else:
                right = mid - 1

        # Phase B: linear verification +-5 steps around the binary result,
        # batched, to correct for ASR non-determinism at the boundary.
        start_step = max(0, binary_best - 5)
        end_step = min(max_steps, binary_best + 5)
        candidates = []
        for step in range(start_step, end_step + 1):
            candidate = trimmed(step)
            if len(candidate) >= 500:
                audio_data = sr.AudioData(candidate.raw_data, candidate.frame_rate, candidate.sample_width)
                candidates.append((f"{'end' if trim_from_end else 'start'}_{step * TRIM_STEP_MS}", audio_data))

        if not candidates:
            return binary_best * TRIM_STEP_MS

        results = await self._transcribe_batch(candidates)
        best_trim_ms = 0
        for candidate_id, transcription in results:
            if transcription and transcription.strip() == target_transcript.strip():
                best_trim_ms = max(best_trim_ms, int(candidate_id.split("_")[1]))
        return best_trim_ms

    # ------------------------------------------------------------------ #
    # Per-segment processing
    # ------------------------------------------------------------------ #
    async def load_audio_async(self, audio_path: Path) -> Optional[AudioSegment]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                return await loop.run_in_executor(pool, AudioSegment.from_wav, audio_path)
            except Exception as exc:
                logger.error(f"Failed to load {audio_path}: {exc}")
                return None

    async def save_audio_async(self, audio: AudioSegment, output_path: Path) -> bool:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                await loop.run_in_executor(pool, lambda: audio.export(output_path, format="wav"))
                return True
            except Exception as exc:
                logger.error(f"Failed to save {output_path}: {exc}")
                return False

    async def optimize_segment_boundaries(self, file_info: pd.Series) -> ProcessingResult:
        original_path = Path(file_info["segment_path"])
        target_transcript = file_info["final_transcript"]
        book_name = file_info["audiobook_name"]

        output_dir = TRIMMED_SEGMENTS_DIR / book_name
        output_dir.mkdir(parents=True, exist_ok=True)
        fixed_path = output_dir / original_path.name

        if not original_path.exists():
            return ProcessingResult(False, {}, f"Missing audio: {original_path}")

        audio = await self.load_audio_async(original_path)
        if audio is None:
            return ProcessingResult(False, {}, f"Failed to load: {original_path}")

        try:
            # Ceiling of 3s (or half the clip) prevents over-trimming edge cases.
            max_search_ms = min(3000, len(audio) // 2)
            max_search_ms = (max_search_ms // TRIM_STEP_MS) * TRIM_STEP_MS

            end_trim_ms = await self.hybrid_search_trim(audio, target_transcript, max_search_ms, trim_from_end=True)
            end_trimmed = audio[:-end_trim_ms] if end_trim_ms > 0 else audio

            start_trim_ms = await self.hybrid_search_trim(end_trimmed, target_transcript, max_search_ms, trim_from_end=False)
            final_audio = end_trimmed[start_trim_ms:] if start_trim_ms > 0 else end_trimmed

            if not await self.save_audio_async(final_audio, fixed_path):
                return ProcessingResult(False, {}, f"Failed to save: {fixed_path}")

            result = file_info.to_dict()
            result.update(
                {
                    "fixed_segment_path": str(fixed_path),
                    "start_trimmed_ms": start_trim_ms,
                    "end_trimmed_ms": end_trim_ms,
                    "original_duration_ms": len(audio),
                    "final_duration_ms": len(final_audio),
                    "processing_timestamp": time.time(),
                }
            )
            return ProcessingResult(True, result)
        except Exception as exc:
            return ProcessingResult(False, {}, f"Processing error: {exc}")

    async def process_batch(self, batch_df: pd.DataFrame) -> List[ProcessingResult]:
        semaphore = asyncio.Semaphore(MAX_WORKERS)

        async def guarded(row):
            async with semaphore:
                return await self.optimize_segment_boundaries(row)

        results = await asyncio.gather(*[guarded(row) for _, row in batch_df.iterrows()], return_exceptions=True)
        return [r if not isinstance(r, Exception) else ProcessingResult(False, {}, str(r)) for r in results]

    async def save_batch_results(self, results: List[ProcessingResult], detailed_checkpoint: Dict):
        successful = [r.file_info for r in results if r.success]
        if not successful:
            return
        results_df = pd.DataFrame(successful)
        for book_name, group in results_df.groupby("audiobook_name"):
            output_csv = TRIMMED_SEGMENTS_DIR / book_name / "fixed_metadata.csv"
            group.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)

            detailed_checkpoint["processed_segments"].setdefault(book_name, [])
            for segment_path in group["segment_path"]:
                if segment_path not in detailed_checkpoint["processed_segments"][book_name]:
                    detailed_checkpoint["processed_segments"][book_name].append(segment_path)

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    async def load_books_metadata(self, completed_books: List[str]) -> pd.DataFrame:
        metadata_paths = list(SEGMENTS_DIR.glob("**/metadata.csv"))
        active = [p for p in metadata_paths if p.parent.name not in completed_books]
        if not active:
            return pd.DataFrame()

        dfs = []
        for path in active:
            try:
                df = pd.read_csv(path).dropna(subset=["segment_path"])
                df["audiobook_name"] = path.parent.name
                dfs.append(df)
            except Exception as exc:
                logger.error(f"Failed to load {path}: {exc}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def filter_unprocessed(self, master_df: pd.DataFrame, detailed_checkpoint: Dict) -> pd.DataFrame:
        if not detailed_checkpoint["processed_segments"]:
            return master_df
        mask = master_df.apply(
            lambda row: row["segment_path"] not in detailed_checkpoint["processed_segments"].get(row["audiobook_name"], []),
            axis=1,
        )
        return master_df[mask]

    async def run(self):
        detailed_checkpoint, progress_data, completed_data = self.load_checkpoints()
        self.total_processed = progress_data.get("total_processed_count", 0)
        completed_books = completed_data.get("completed_books", [])

        master_df = await self.load_books_metadata(completed_books)
        if master_df.empty:
            logger.info("Nothing to process.")
            return

        tasks_df = self.filter_unprocessed(master_df, detailed_checkpoint)
        if tasks_df.empty:
            logger.info("All segments already processed.")
            return

        logger.info(f"Optimizing boundaries for {len(tasks_df)} segments in batches of {BATCH_SIZE}")

        checkpoint_counter = 0
        for i in range(0, len(tasks_df), BATCH_SIZE):
            batch = tasks_df.iloc[i : i + BATCH_SIZE]
            results = await self.process_batch(batch)
            await self.save_batch_results(results, detailed_checkpoint)

            successful = sum(1 for r in results if r.success)
            self.total_processed += successful
            checkpoint_counter += successful
            if checkpoint_counter >= CHECKPOINT_FREQUENCY or i + BATCH_SIZE >= len(tasks_df):
                self.save_detailed_checkpoint(detailed_checkpoint)
                self.save_progress_checkpoint(self.total_processed)
                checkpoint_counter = 0

        for book_name in master_df["audiobook_name"].unique():
            if book_name in completed_books:
                continue
            original = set(master_df[master_df["audiobook_name"] == book_name]["segment_path"])
            processed = set(detailed_checkpoint["processed_segments"].get(book_name, []))
            if processed >= original:
                self.mark_book_completed(book_name)

        elapsed = time.time() - self.start_time
        logger.info(f"Done. {self.total_processed} segments processed in {elapsed:.1f}s ({self.total_processed / max(elapsed, 1):.2f}/s)")


def main():
    ensure_dirs(TRIMMED_SEGMENTS_DIR)
    asyncio.run(BoundaryOptimizer().run())


if __name__ == "__main__":
    main()
