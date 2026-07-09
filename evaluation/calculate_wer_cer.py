#!/usr/bin/env python3
"""
Multi-ASR intelligibility evaluation: WER/CER of synthesized speech
(paper Section 5.4 "Objective Results", Table `tab:wer_multiASR`).

The paper scores synthesized samples with three independent ASR systems --
Google ASR (the same backend used during corpus construction, so treated
only as a diagnostic check), a Persian-finetuned Whisper model (the primary
non-circular intelligibility reference, since it is independent of the
corpus pipeline), and Qwen ASR -- then compares WER/CER against the same
systems run on real speech (Google FLEURS Persian) to get a TTS-vs-real gap.

This script provides two interchangeable ASR engines over the same input:

    google  -- free Google Web Speech API via `SpeechRecognition`
               (no model download, matches the paper's Google ASR row)
    hf      -- any Hugging Face `automatic-speech-recognition` checkpoint,
               e.g. a Whisper-family model (matches the paper's
               Whisper-finetuned row; swap in any other seq2seq ASR
               checkpoint including your own Qwen ASR wrapper)

Both engines apply the same Persian-specific text normalization before
computing WER/CER with `jiwer`, so scores are comparable across engines.

Input CSV (same file used by evaluation/goyar_mos_bot.py, e.g. the samples
CSV pointed to by GOYAR_SAMPLES_CSV): needs `generated_audio` (path to the
synthesized clip) and `transcript` (ground-truth reference text) columns.

Usage:
    python evaluation/calculate_wer_cer.py --engine google --samples-csv samples.csv
    python evaluation/calculate_wer_cer.py --engine hf --model MODEL_NAME_OR_PATH \
        --samples-csv samples.csv --output results_whisper.csv
"""

import argparse
import os
import re
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import jiwer
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Persian-specific text normalization, shared by both ASR engines -------
# Arabic-script variants normalized to their Persian equivalents, digits
# normalized to Persian-Indic, diacritics and zero-width marks stripped.
_CHAR_MAPPINGS = {
    "ي": "ی",  # Arabic yeh -> Persian yeh
    "ك": "ک",  # Arabic kaf -> Persian kaf
    "ة": "ه",  # teh marbuta -> heh
    "ء": "",        # hamza
    "أ": "ا",  # alef with hamza above -> alef
    "إ": "ا",  # alef with hamza below -> alef
    "آ": "ا",  # alef with madda -> alef
    "ؤ": "و",  # waw with hamza -> waw
    "ئ": "ی",  # yeh with hamza -> yeh
    "١": "۱", "٢": "۲", "٣": "۳", "٤": "۴", "٥": "۵",
    "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹", "٠": "۰",
    "1": "۱", "2": "۲", "3": "۳", "4": "۴", "5": "۵",
    "6": "۶", "7": "۷", "8": "۸", "9": "۹", "0": "۰",
    "«": '"', "»": '"', "؟": "?", "؛": ";", "،": ",",
    "٬": ",", "٫": ".", "﴾": "(", "﴿": ")",
    "‌": "", "‍": "",  # ZWNJ, ZWJ
    "ـ": "",  # kashida
}
_DIACRITICS = [
    "ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ",
    "ٓ", "ٔ", "ٕ", "ٖ", "ٗ", "٘", "ٙ", "ٚ",
    "ٛ", "ٜ", "ٝ", "ٞ", "ٟ", "ٰ",
]
# Persian/Arabic-script blocks + Persian-Indic digits, matching the source
# script's exact ranges (verified against the paper's evaluation code).
_KEEP_CHARS_RE = r"؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿۰-۹"
_EXTRA_PUNCT_RE = " .,?!:;()\\-/\"'"
_KEEP_CHARS_WITH_PUNCT_RE = _KEEP_CHARS_RE + "‌‍" + _EXTRA_PUNCT_RE


def normalize_persian_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    for d in _DIACRITICS:
        text = text.replace(d, "")
    for old, new in _CHAR_MAPPINGS.items():
        text = text.replace(old, new)
    text = re.sub(rf"[^{_KEEP_CHARS_WITH_PUNCT_RE}]", " ", text)
    return re.sub(r"\s+", " ", text.strip()).lower()


def normalize_for_word_comparison(text: str) -> str:
    text = normalize_persian_text(text)
    text = re.sub(rf"[^{_KEEP_CHARS_RE} ]", "", text)
    return re.sub(r"\s+", " ", text.strip())


def normalize_for_char_comparison(text: str) -> str:
    text = normalize_persian_text(text)
    return re.sub(rf"[^{_KEEP_CHARS_RE}]", "", text)


def compute_wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_for_word_comparison(reference), normalize_for_word_comparison(hypothesis)
    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0
    return jiwer.wer(ref, hyp)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_for_char_comparison(reference), normalize_for_char_comparison(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


# ============================================================================
# Engine 1: Google Web Speech (free, matches the paper's "Google ASR" row)
# ============================================================================
def transcribe_google(audio_path: str, language: str = "fa-IR") -> Tuple[Optional[str], Optional[str]]:
    import speech_recognition as sr

    if not os.path.exists(audio_path):
        return None, f"File not found: {audio_path}"
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language=language), None
    except sr.UnknownValueError:
        return "", "Could not understand audio"
    except Exception as exc:
        return None, f"Recognition error: {exc}"


def evaluate_with_google_asr(samples_csv: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(samples_csv)
    print(f"Transcribing {len(df)} samples with Google ASR...")

    rows = []
    for _, row in df.iterrows():
        audio_path, reference = row["generated_audio"], row["transcript"]
        hypothesis, error = transcribe_google(audio_path)
        if hypothesis is None:
            rows.append({"generated_audio": audio_path, "transcript": reference, "hypothesis": None, "wer": 1.0, "cer": 1.0, "status": "failed", "error": error})
            continue
        rows.append(
            {
                "generated_audio": audio_path,
                "transcript": reference,
                "hypothesis": hypothesis,
                "wer": compute_wer(reference, hypothesis),
                "cer": compute_cer(reference, hypothesis),
                "status": "success" if not error else "warning",
                "error": error,
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_path, index=False)
    _print_summary("Google ASR", results_df)
    return results_df


# ============================================================================
# Engine 2: any Hugging Face automatic-speech-recognition checkpoint
# (matches the paper's Whisper-finetuned row; swap `--model` for other
# seq2seq ASR checkpoints, e.g. Qwen ASR, if the model has HF pipeline support)
# ============================================================================
def build_hf_asr_pipeline(model_name_or_path: str, processor_source: Optional[str] = None, batch_size: int = 16):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    # A fine-tuned checkpoint's own directory may not include tokenizer/feature-extractor
    # files; processor_source lets you point those at the base model it was fine-tuned from.
    processor = AutoProcessor.from_pretrained(processor_source or model_name_or_path)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=batch_size,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )


def _load_audio_batch(paths: List[str], target_sr: int = 16000) -> List[np.ndarray]:
    import librosa

    def load_one(path):
        try:
            audio, _ = librosa.load(path, sr=target_sr)
            return audio
        except Exception as exc:
            print(f"  Error loading {path}: {exc}")
            return np.zeros(target_sr)

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(load_one, paths))


def evaluate_with_hf_asr(samples_csv: str, model_name_or_path: str, output_path: str, processor_source: Optional[str] = None, batch_size: int = 16) -> pd.DataFrame:
    df = pd.read_csv(samples_csv).dropna(subset=["generated_audio", "transcript"]).reset_index(drop=True)
    print(f"Loading ASR model {model_name_or_path}...")
    pipe = build_hf_asr_pipeline(model_name_or_path, processor_source, batch_size)

    all_rows = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        audio_batch = _load_audio_batch(batch["generated_audio"].tolist())

        results = pipe(audio_batch)
        hypotheses = [r["text"].strip() if isinstance(r, dict) else str(r).strip() for r in results]

        for (_, row), hypothesis in zip(batch.iterrows(), hypotheses):
            all_rows.append(
                {
                    "generated_audio": row["generated_audio"],
                    "transcript": row["transcript"],
                    "hypothesis": hypothesis,
                    "wer": compute_wer(row["transcript"], hypothesis),
                    "cer": compute_cer(row["transcript"], hypothesis),
                }
            )
        print(f"  {min(start + batch_size, len(df))}/{len(df)} done")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(output_path, index=False)
    _print_summary(model_name_or_path, results_df)
    return results_df


def _print_summary(engine_name: str, results_df: pd.DataFrame):
    successful = results_df[results_df["wer"].notna()]
    print(f"\n{engine_name} results ({len(successful)}/{len(results_df)} samples):")
    if len(successful):
        print(f"  Average WER: {successful['wer'].mean() * 100:.2f}%")
        print(f"  Average CER: {successful['cer'].mean() * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="WER/CER intelligibility evaluation of synthesized speech (paper Section 5.4)")
    parser.add_argument("--engine", choices=["google", "hf"], required=True)
    parser.add_argument("--samples-csv", default=os.environ.get("GOYAR_SAMPLES_CSV", "samples.csv"))
    parser.add_argument("--model", help="HF model id or local path (required for --engine hf)")
    parser.add_argument("--processor", help="HF processor source, if different from --model (e.g. the base model a checkpoint was fine-tuned from)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=None, help="Output CSV path (default: wer_cer_<engine>.csv)")
    args = parser.parse_args()

    output_path = args.output or f"wer_cer_{args.engine}.csv"

    if args.engine == "google":
        evaluate_with_google_asr(args.samples_csv, output_path)
    else:
        if not args.model:
            parser.error("--model is required for --engine hf")
        evaluate_with_hf_asr(args.samples_csv, args.model, output_path, args.processor, args.batch_size)


if __name__ == "__main__":
    main()
