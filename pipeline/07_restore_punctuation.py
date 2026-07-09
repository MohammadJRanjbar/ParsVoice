#!/usr/bin/env python3
"""
Stage 7 -- Final Cleaning, Punctuation Restoration, and Release (paper
Section 3.6).

Google's ASR backend (used in Stage 1) returns unpunctuated text, but
punctuation materially affects prosody modelling in TTS. This final stage:

    1. Applies audio- and text-quality filtering (score >= 75/100 audio,
       >= 0.5 text -- see stages 4-5) to the boundary-optimized corpus.
    2. Further restricts to the TTS-ready subset by excluding segments with
       background music, low-confidence speaker clusters, or an
       "Incomplete" sentence-completion flag.
    3. Restores punctuation on every surviving transcript with the
       ParsBERT-based token-classification model from PersianPunc (citation
       omitted for anonymous review), predicting one of {none, '.', '،',
       '؟', ':'} per word.
    4. Writes the release-ready manifest (audio path, punctuated transcript,
       speaker IDs, audio/text quality scores).

Usage:
    python pipeline/07_restore_punctuation.py
"""

import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import (
    AUDIO_QUALITY_DIR,
    GLOBAL_SPEAKER_ID_DIR,
    PUNCTUATION_MODEL_PATH,
    RELEASE_DIR,
    TEXT_QUALITY_DIR,
    ensure_dirs,
)

# Section 3.6 filtering thresholds.
MIN_AUDIO_QUALITY_SCORE = 75  # out of 100
MIN_TEXT_QUALITY_SCORE = 0.5  # out of 1.0

TARGET_PUNCTUATION = {".", "،", "؟", ":"}
PUNCTUATION_BATCH_SIZE = 32


def normalize_punctuation(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r"[,،]", "،", text)
    text = re.sub(r"[.!。]", ".", text)
    text = re.sub(r"[?؟]", "؟", text)
    return text


def strip_punctuation(text: str) -> str:
    for punct in TARGET_PUNCTUATION:
        text = text.replace(punct, "")
    return text


def load_punctuation_model():
    tokenizer = AutoTokenizer.from_pretrained(PUNCTUATION_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(PUNCTUATION_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device


def restore_punctuation(text: str, model, tokenizer, device) -> str:
    """Predicts one punctuation label per word, preserving original spacing."""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""

    clean_text = strip_punctuation(normalize_punctuation(text))
    words = clean_text.split()
    if not words:
        return ""

    encoding = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=False, max_length=512).to(device)
    with torch.no_grad():
        predictions = torch.argmax(model(**encoding).logits, dim=-1)[0].cpu()

    word_ids = encoding.word_ids()
    id2label = model.config.id2label
    word_punctuation = [""] * len(words)

    for i, (word_idx, prediction) in enumerate(zip(word_ids, predictions)):
        if word_idx is None or word_idx >= len(words):
            continue
        # Only the last subtoken of a word carries its punctuation label.
        if i == len(word_ids) - 1 or word_ids[i + 1] != word_idx:
            label = id2label[prediction.item()]
            if label != "O":
                word_punctuation[word_idx] = label.replace("B-", "")

    return " ".join(word + punct for word, punct in zip(words, word_punctuation))


def restore_punctuation_batch(texts: pd.Series, model, tokenizer, device, batch_size: int = PUNCTUATION_BATCH_SIZE) -> list:
    results = []
    texts = texts.fillna("").astype(str).tolist()
    for i in tqdm(range(0, len(texts), batch_size), desc="Restoring punctuation"):
        for text in texts[i : i + batch_size]:
            try:
                results.append(restore_punctuation(text, model, tokenizer, device))
            except Exception as exc:
                print(f"  Punctuation restoration failed for '{text[:50]}...': {exc}")
                results.append(text)
    return results


def build_release_manifest() -> pd.DataFrame:
    """Joins global speaker IDs with audio/text quality scores and applies
    the Section 3.6 filtering rules.

    All three inputs (Stage 3 audio scores, Stage 4 text scores, Stage 6
    speaker labels) key on the Stage 2 boundary-optimized audio path
    ("segment_path" here, "file_path" in Stage 3's output) -- *not* the
    pre-trim path from Stage 1 -- so a plain merge on that path is safe.
    """
    speaker_summaries = list(GLOBAL_SPEAKER_ID_DIR.glob("*/*_final_speakers.csv"))
    if not speaker_summaries:
        raise FileNotFoundError(f"No speaker-labelled segments found under {GLOBAL_SPEAKER_ID_DIR} -- run stage 6 first.")
    segments = pd.concat([pd.read_csv(f) for f in speaker_summaries], ignore_index=True)

    audio_quality = pd.read_csv(AUDIO_QUALITY_DIR / "master_results.csv")[
        ["file_path", "quality_score", "has_background_music"]
    ].rename(columns={"file_path": "segment_path", "quality_score": "audio_quality_score"})

    text_quality = pd.read_csv(TEXT_QUALITY_DIR / "text_quality_scores.csv")[["segment_path", "total_score"]].rename(
        columns={"total_score": "text_quality_score"}
    )

    merged = segments.merge(audio_quality, on="segment_path", how="left")
    merged = merged.merge(text_quality, on="segment_path", how="left")

    filtered = merged[
        (merged["audio_quality_score"] >= MIN_AUDIO_QUALITY_SCORE)
        & (merged["text_quality_score"] >= MIN_TEXT_QUALITY_SCORE)
        & (~merged["has_background_music"].fillna(False))
        & (merged["global_speaker_id"] != -1)
        & (merged["completion_status"] != "Incomplete")
    ].copy()

    print(f"Release candidates after quality/completeness/speaker filtering: {len(filtered)}/{len(merged)}")
    return filtered


def main():
    ensure_dirs(RELEASE_DIR)

    manifest = build_release_manifest()
    model, tokenizer, device = load_punctuation_model()
    print(f"Restoring punctuation for {len(manifest)} segments on {device}...")

    manifest["final_transcript_punctuated"] = restore_punctuation_batch(manifest["final_transcript"], model, tokenizer, device)

    release_columns = [
        "segment_path", "final_transcript_punctuated", "final_duration_ms",
        "global_speaker_id", "speaker_confidence", "audio_quality_score", "text_quality_score",
    ]
    release_df = manifest[[c for c in release_columns if c in manifest.columns]]

    output_path = RELEASE_DIR / "parsvoice_release_manifest.csv"
    release_df.to_csv(output_path, index=False)

    total_hours = manifest["final_duration_ms"].sum() / 3_600_000
    print(f"Release manifest written -> {output_path}")
    print(f"{len(release_df)} segments, {total_hours:.1f} hours, {manifest['global_speaker_id'].nunique()} speakers")


if __name__ == "__main__":
    main()
