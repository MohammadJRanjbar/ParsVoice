#!/usr/bin/env python3
"""
Stage 1 -- Intelligent Audio Segmentation (paper Section 3.2).

Input: a CSV of audiobooks you already have on disk (see AUDIOBOOKS_CSV in
pipeline/config.py) -- this stage does not fetch or download anything.

Three-phase pipeline that turns a raw, multi-hour audiobook recording into
short, sentence-aligned (audio, transcript) segments:

    Phase 1 (Acoustic Boundary Detection): WebRTC VAD proposes silence-based
        candidate boundaries (Appendix A compares aggressiveness levels --
        see the VAD_AGGRESSIVENESS note below).
    Phase 2 (Transcription): each candidate segment is transcribed with the
        Google Web Speech backend (best of the ASR systems compared in
        Appendix B), chosen for being free, fast, and the most accurate of
        the alternatives tested.
    Phase 3 (Completeness Validation and Boundary Extension): a ParsBERT
        sentence-completion classifier (Appendix C) flags incomplete
        transcripts; incomplete segments are extended in 0.1s steps (up to
        5s) and re-transcribed until the classifier accepts them or the
        extension budget runs out. Segments that remain incomplete are kept
        with an `Incomplete` flag rather than discarded, so downstream users
        can apply their own thresholds.

Processing is checkpointed per book (pending/processing/completed/failed) so
a run over thousands of books can be safely interrupted and resumed, and runs
several books in parallel via a process pool. Audio is only written to disk
for segments that produced a (possibly empty) transcript.

Usage:
    python pipeline/01_segment_and_transcribe.py
    python pipeline/01_segment_and_transcribe.py --test      # small subset
    python pipeline/01_segment_and_transcribe.py --retry-failed
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import webrtcvad
import speech_recognition as sr
import torch
from concurrent.futures import ProcessPoolExecutor
from filelock import FileLock, Timeout
from pydub import AudioSegment
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm

from config import AUDIOBOOKS_CSV, SEGMENTS_DIR, SENTENCE_COMPLETION_MODEL_PATH, ensure_dirs

warnings.filterwarnings("ignore")

# --- Audio processing parameters (Section 3.2 / Appendix A) -----------------
SAMPLE_RATE = 16000
# NOTE: Appendix A reports WebRTC level 0 as having the highest completion
# rate and states it was selected as the segmentation backend; the working
# pipeline script this stage is ported from actually runs at level 1. Kept
# faithful to the runnable code here -- set to 0 if you want to match the
# text of Appendix A exactly.
VAD_AGGRESSIVENESS = 1
NORMALIZATION_FACTOR = 0.95

# --- Boundary-extension parameters (Section 3.2, Phase 3) -------------------
EXTENSION_STEP = 0.1  # seconds
MAX_EXTENSION = 5.0  # seconds
COMPLETION_BATCH_SIZE = 32

MAX_PROCESSES = int(os.environ.get("PARSVOICE_MAX_PROCESSES", "4"))
SEGMENTS_PER_BATCH = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = None  # set in main() once SEGMENTS_DIR is known
CHECKPOINT_LOCK = None

DEVICE = None
TOKENIZER = None
MODEL = None


# ============================================================================
# Checkpoint management (per book: pending / processing / completed / failed)
# ============================================================================
def initialize_checkpoint(all_books_df: pd.DataFrame) -> None:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        new_books = set(str(i) for i in all_books_df.index) - set(data.get("books", {}).keys())
        for book_index in new_books:
            data["books"][book_index] = "pending"
        CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))
        return
    CHECKPOINT_FILE.write_text(json.dumps({"books": {str(i): "pending" for i in all_books_df.index}}, indent=2))
    logger.info(f"Initialized checkpoint for {len(all_books_df)} books")


def update_book_status(book_index: int, status: str) -> None:
    lock = FileLock(str(CHECKPOINT_LOCK), timeout=30)
    try:
        with lock:
            data = json.loads(CHECKPOINT_FILE.read_text())
            data["books"][str(book_index)] = status
            CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))
    except Timeout:
        logger.error(f"Could not acquire checkpoint lock for book {book_index}")


def get_books_to_process(all_books_df: pd.DataFrame) -> List[Tuple[int, pd.Series]]:
    if not CHECKPOINT_FILE.exists():
        return list(all_books_df.iterrows())
    data = json.loads(CHECKPOINT_FILE.read_text())
    books_status = data.get("books", {})
    # A previous run that was killed mid-book leaves books "processing"; reset them.
    for index_str, status in list(books_status.items()):
        if status == "processing":
            update_book_status(int(index_str), "pending")
            books_status[index_str] = "pending"
    pending = [int(i) for i, s in books_status.items() if s != "completed" and int(i) in all_books_df.index]
    logger.info(f"{len(pending)} books left to process")
    return list(all_books_df.loc[pending].iterrows())


def reset_checkpoint_status(all_books_df: pd.DataFrame, status_to_reset: str = "failed") -> None:
    if not CHECKPOINT_FILE.exists():
        return
    with FileLock(str(CHECKPOINT_LOCK), timeout=30):
        data = json.loads(CHECKPOINT_FILE.read_text())
        reset = sum(1 for k, v in data["books"].items() if v == status_to_reset)
        data["books"] = {k: ("pending" if v == status_to_reset else v) for k, v in data["books"].items()}
        CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))
        logger.info(f"Reset {reset} books from '{status_to_reset}' to 'pending'")


# ============================================================================
# Model, metadata, audio prep
# ============================================================================
def initialize_sentence_completion_model() -> bool:
    global DEVICE, TOKENIZER, MODEL
    if MODEL is not None:
        return True
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TOKENIZER = AutoTokenizer.from_pretrained(SENTENCE_COMPLETION_MODEL_PATH)
        MODEL = AutoModelForSequenceClassification.from_pretrained(SENTENCE_COMPLETION_MODEL_PATH)
        MODEL.to(DEVICE)
        MODEL.eval()
        return True
    except Exception as exc:
        logger.error(f"Failed to load sentence-completion model: {exc}")
        return False


def predict_completion_status_batch(sentences: List[str], batch_size: int = COMPLETION_BATCH_SIZE) -> List[str]:
    """Classify each transcript as Complete/Incomplete (Appendix C classifier)."""
    if MODEL is None or TOKENIZER is None:
        return ["Invalid"] * len(sentences)
    status_map = {0: "Incomplete", 1: "Complete"}
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        valid_idx = [j for j, s in enumerate(batch) if isinstance(s, str) and s.strip()]
        batch_results = ["Invalid"] * len(batch)
        if valid_idx:
            inputs = TOKENIZER([batch[j] for j in valid_idx], return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            with torch.no_grad():
                preds = torch.argmax(MODEL(**inputs).logits, dim=-1)
            for j, pred in zip(valid_idx, preds):
                batch_results[j] = status_map[pred.item()]
        results.extend(batch_results)
    return results


def load_and_prepare_metadata(test_subset: bool = False) -> pd.DataFrame:
    """Loads the user-provided audiobooks CSV (title, narrator, audio_path).

    audio_path may point to either a single audio file for the whole book,
    or a directory containing one audio file per chapter -- see
    get_audio_files_for_book().
    """
    if not AUDIOBOOKS_CSV.exists():
        raise FileNotFoundError(
            f"{AUDIOBOOKS_CSV} not found. Provide a CSV with columns "
            "'title', 'narrator', 'audio_path' (set PARSVOICE_AUDIOBOOKS_CSV to point elsewhere)."
        )
    df = pd.read_csv(AUDIOBOOKS_CSV)
    missing = {"title", "narrator", "audio_path"} - set(df.columns)
    if missing:
        raise ValueError(f"{AUDIOBOOKS_CSV} is missing required column(s): {sorted(missing)}")

    df = df[["title", "narrator", "audio_path"]].copy()
    df["narrator"] = df["narrator"].fillna("unknown")
    df["full_path"] = df["audio_path"]
    df["title_index"] = df["title"].map({t: i for i, t in enumerate(df["title"].unique())})
    df["narrator_index"] = df["narrator"].map({n: i for i, n in enumerate(df["narrator"].unique())})
    if test_subset:
        df = df.head(MAX_PROCESSES * 2)
    logger.info(f"Prepared metadata for {len(df)} books")
    return df


def get_audio_files_for_book(audio_path: str) -> List[str]:
    """audio_path is either a single audio file, or a directory of chapter files."""
    exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4"}
    path = Path(audio_path)
    if path.is_file():
        return [str(path)] if path.suffix.lower() in exts else []
    if path.is_dir():
        return sorted(str(f) for f in path.iterdir() if f.suffix.lower() in exts)
    return []


def preprocess_audio(audio_path: str) -> Tuple[Optional[np.ndarray], float]:
    try:
        audio, sr_orig = librosa.load(audio_path, sr=None)
        if sr_orig != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=SAMPLE_RATE)
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio *= NORMALIZATION_FACTOR / max_amp
        return audio, len(audio) / SAMPLE_RATE
    except Exception as exc:
        logger.error(f"Failed to preprocess {audio_path}: {exc}")
        return None, 0.0


def perform_vad(audio: np.ndarray) -> List[Tuple[float, float]]:
    """Phase 1: WebRTC VAD (level 0) candidate speech boundaries."""
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    audio_int16 = (audio * 32767).astype(np.int16)
    frame_length = int(SAMPLE_RATE * 30 / 1000)  # 30ms frames
    segments, current_start = [], None
    for i in range(0, len(audio_int16) - frame_length, frame_length):
        frame = audio_int16[i : i + frame_length].tobytes()
        timestamp = i / SAMPLE_RATE
        try:
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            continue
        if is_speech and current_start is None:
            current_start = timestamp
        elif not is_speech and current_start is not None:
            segments.append((current_start, timestamp))
            current_start = None
    if current_start is not None:
        segments.append((current_start, len(audio_int16) / SAMPLE_RATE))
    return segments


async def transcribe_segment_async(audio_segment: np.ndarray, loop) -> str:
    """Phase 2: transcribe via the Google Web Speech backend (Appendix B)."""
    try:
        audio_int16 = (audio_segment * 32767).astype(np.int16)
        recognizer = sr.Recognizer()
        audio_data = sr.AudioData(audio_int16.tobytes(), SAMPLE_RATE, 2)
        transcript = await loop.run_in_executor(None, lambda: recognizer.recognize_google(audio_data, language="fa-IR"))
        return transcript.strip()
    except (sr.UnknownValueError, Exception):
        return ""


# ============================================================================
# Three-pass segmentation: end-time discovery -> timeline correction -> final transcribe
# ============================================================================
async def _pass1_determine_end_time(segment: Tuple[float, float], audio: np.ndarray, loop) -> Tuple[float, float, int]:
    """Phase 3: extend segment end in 0.1s steps until the completion
    classifier accepts the transcript, or MAX_EXTENSION is reached."""
    start_time, end_time = segment
    start_sample = int(start_time * SAMPLE_RATE)
    extensions = 0
    try:
        transcript = await transcribe_segment_async(audio[start_sample : int(end_time * SAMPLE_RATE)], loop)
        current_end = end_time
        if transcript and predict_completion_status_batch([transcript])[0] == "Incomplete":
            while extensions * EXTENSION_STEP < MAX_EXTENSION:
                proposed_end = current_end + EXTENSION_STEP
                extended_transcript = await transcribe_segment_async(audio[start_sample : int(proposed_end * SAMPLE_RATE)], loop)
                if not extended_transcript:
                    break
                extensions += 1
                current_end = proposed_end
                if predict_completion_status_batch([extended_transcript])[0] == "Complete":
                    break
        return start_time, current_end, extensions
    except Exception:
        return start_time, end_time, 0


def _pass2_correct_timeline(pass1_results: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    """Extended segments can overlap their successor; clip each segment's
    start to the previous segment's (possibly extended) end."""
    if not pass1_results:
        return []
    sorted_segments = sorted(pass1_results, key=lambda x: x[0])
    corrected, last_end = [], sorted_segments[0][0]
    for start, end, extensions in sorted_segments:
        new_start = last_end
        if new_start >= end:
            continue
        corrected.append((new_start, end, extensions))
        last_end = end
    return corrected


async def _pass3_transcribe_final(segment_data: Tuple[float, float, int], segment_index: int, audio: np.ndarray, loop, book_info: Dict[str, Any]) -> Dict:
    start_time, end_time, extensions = segment_data
    result = {
        "segment_index": segment_index,
        "start_time": start_time,
        "end_time": end_time,
        "audio_data": np.array([]),
        "transcript": "",
        "extensions_applied": extensions,
        "completion_status": "Error",
        "duration": max(0, end_time - start_time),
        "book_info": book_info,
    }
    try:
        audio_slice = audio[int(start_time * SAMPLE_RATE) : int(end_time * SAMPLE_RATE)]
        transcript = await transcribe_segment_async(audio_slice, loop)
        result.update(
            {
                "audio_data": audio_slice,
                "transcript": transcript,
                "completion_status": predict_completion_status_batch([transcript])[0] if transcript else "Empty",
            }
        )
    except Exception:
        pass
    return result


async def process_audio_file(audio_path: str, book_info: Dict[str, Any]) -> List[Dict]:
    audio, _ = preprocess_audio(audio_path)
    if audio is None:
        return []
    initial_segments = perform_vad(audio)
    if not initial_segments:
        return []
    loop = asyncio.get_event_loop()
    pass1_results = await asyncio.gather(*[_pass1_determine_end_time(seg, audio, loop) for seg in initial_segments])
    corrected_timeline = _pass2_correct_timeline(pass1_results)
    return await asyncio.gather(
        *[_pass3_transcribe_final(seg, idx, audio, loop, book_info) for idx, seg in enumerate(corrected_timeline)]
    )


# ============================================================================
# Output: per-book folder, conditional audio save, metadata consolidation
# ============================================================================
def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "", name).replace(" ", "_")[:100]


def save_segment_audio(segment_data: np.ndarray, segment_path: Path) -> bool:
    try:
        audio_int16 = (segment_data * 32767).astype(np.int16)
        AudioSegment(audio_int16.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1).export(segment_path, format="wav")
        return True
    except Exception as exc:
        logger.warning(f"Failed to save segment {segment_path}: {exc}")
        return False


def save_batch_metadata(segments_batch: List[Dict], book_output_dir: Path, batch_index: int) -> None:
    """Save a batch of segment metadata; audio is only written for segments
    with a non-empty transcript."""
    rows = []
    for segment in segments_batch:
        segment_path = ""
        if segment.get("transcript", "").strip():
            filename = f"segment_{segment['segment_index']:06d}.wav"
            if save_segment_audio(segment["audio_data"], book_output_dir / filename):
                segment_path = str(book_output_dir / filename)

        info = segment["book_info"]
        rows.append(
            {
                "segment_path": segment_path,
                "original_audio_book_name": info["title"],
                "title_index": info["title_index"],
                "narrator": info["narrator"],
                "narrator_index": info["narrator_index"],
                "final_duration": segment["duration"],
                "number_of_extensions": segment["extensions_applied"],
                "completion_status": segment["completion_status"],
                "final_transcript": segment["transcript"],
                "audio_book_time": segment["start_time"],
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(book_output_dir / f"batch_{batch_index}_metadata.csv", index=False)


def consolidate_book_metadata(book_output_dir: Path) -> bool:
    batch_files = sorted(book_output_dir.glob("batch_*_metadata.csv"))
    if not batch_files:
        return False
    all_segments = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)
    all_segments.to_csv(book_output_dir / "metadata.csv", index=False)

    master_path = SEGMENTS_DIR / "master_segments_metadata.csv"
    all_segments.to_csv(master_path, mode="a", header=not master_path.exists(), index=False)

    for f in batch_files:
        f.unlink()
    return True


def process_book(book_data: Tuple[int, pd.Series]) -> Dict:
    book_index, book_row = book_data
    logger.info(f"Starting book {book_index}: {book_row.get('title', 'Unknown')}")
    update_book_status(book_index, "processing")

    book_output_dir = SEGMENTS_DIR / f"{sanitize_filename(str(book_row.get('title', 'Unknown_Book')))}_{book_index}"
    book_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not initialize_sentence_completion_model():
            raise RuntimeError("Sentence-completion model failed to load")

        book_info = book_row.to_dict()
        audio_files = get_audio_files_for_book(book_row["full_path"])
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {book_row['full_path']}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        total_segment_index = 0

        for audio_file in audio_files:
            segments = loop.run_until_complete(process_audio_file(audio_file, book_info))
            if not segments:
                continue
            for seg in segments:
                seg["segment_index"] = total_segment_index
                total_segment_index += 1

            for i in range(0, len(segments), SEGMENTS_PER_BATCH):
                batch = segments[i : i + SEGMENTS_PER_BATCH]
                save_batch_metadata(batch, book_output_dir, i // SEGMENTS_PER_BATCH)
                for seg in batch:
                    seg["audio_data"] = np.array([])
                gc.collect()

        if consolidate_book_metadata(book_output_dir):
            logger.info(f"Book {book_index}: consolidated metadata -> {book_output_dir}")

        update_book_status(book_index, "completed")
        return {"book_index": book_index, "status": "success", "title": book_info.get("title", "Unknown")}

    except Exception as exc:
        logger.error(f"Book {book_index} failed: {exc}")
        update_book_status(book_index, "failed")
        return {"book_index": book_index, "status": "failed", "message": str(exc), "title": book_row.get("title", "Unknown")}


def main():
    global CHECKPOINT_FILE, CHECKPOINT_LOCK

    parser = argparse.ArgumentParser(description="Stage 1: segment, transcribe, and validate completeness")
    parser.add_argument("--test", action="store_true", help="Run on a small test subset of books")
    parser.add_argument("--reset", action="store_true", help="Re-initialize checkpoint (all books -> pending)")
    parser.add_argument("--retry-failed", action="store_true", help="Reset only failed books to pending")
    args = parser.parse_args()

    ensure_dirs(SEGMENTS_DIR)
    CHECKPOINT_FILE = SEGMENTS_DIR / "checkpoint_status.json"
    CHECKPOINT_LOCK = SEGMENTS_DIR / "checkpoint_status.json.lock"

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    elif args.retry_failed:
        reset_checkpoint_status(load_and_prepare_metadata(args.test), "failed")

    all_books_df = load_and_prepare_metadata(args.test)
    initialize_checkpoint(all_books_df)

    books_to_process = get_books_to_process(all_books_df)
    if not books_to_process:
        logger.info("All books already completed.")
        return

    logger.info(f"Processing {len(books_to_process)} books with {MAX_PROCESSES} worker processes")
    results = []
    with tqdm(total=len(books_to_process), desc="Books", unit="book") as pbar:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            for result in executor.map(process_book, books_to_process):
                results.append(result)
                pbar.update(1)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    logger.info(f"Done. Successful: {len(successful)}  Failed: {len(failed)}")
    if failed:
        for f in failed:
            logger.warning(f"  {f['book_index']}: {f['title']} - {f['message']}")


if __name__ == "__main__":
    main()
