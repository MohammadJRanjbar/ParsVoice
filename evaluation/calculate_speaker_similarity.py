#!/usr/bin/env python3
"""
Objective speaker-similarity evaluation (paper Section 5.4 "Objective
Results"): cosine similarity between ECAPA-TDNN embeddings of the reference
speaker's recording and the corresponding synthesized clip, via SpeechBrain's
`speechbrain/spkrec-ecapa-voxceleb` -- the same embedding model used for
corpus-side speaker identification in pipeline/05_extract_speaker_embeddings.py.
This is an objective measurement independent of the subjective SMOS ratings
collected by evaluation/goyar_mos_bot.py.

Input CSV (same file used by the Goyar bot, e.g. GOYAR_SAMPLES_CSV): needs
`reference_audio` and `generated_audio` columns; `speaker_gender`/`speaker_age`
are used for the breakdown in the report if present.

Usage:
    python evaluation/calculate_speaker_similarity.py --samples-csv samples.csv
    python evaluation/calculate_speaker_similarity.py --pair ref.wav gen.wav
"""

import argparse
import os
import warnings
from typing import Optional

import pandas as pd
import torch
import torchaudio

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


class SpeakerSimilarityCalculator:
    """Cosine similarity between ECAPA-TDNN embeddings of two audio files,
    rescaled from [-1, 1] to [0, 1] (higher = more similar)."""

    def __init__(self, model_source: str = DEFAULT_MODEL, device: Optional[str] = None):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_source} on {self.device}...")
        self.classifier = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=f"pretrained_models/{model_source.split('/')[-1]}",
            run_opts={"device": self.device},
        )

    def load_audio(self, audio_path: str, target_sr: int = 16000) -> Optional[torch.Tensor]:
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != target_sr:
                waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
            return waveform.squeeze(0)
        except Exception as exc:
            print(f"  Error loading {audio_path}: {exc}")
            return None

    def extract_embedding(self, audio_path: str) -> Optional[torch.Tensor]:
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        with torch.no_grad():
            embedding = self.classifier.encode_batch(waveform.unsqueeze(0))
        return embedding.squeeze()

    def calculate_similarity(self, reference_path: str, generated_path: str) -> Optional[float]:
        ref_emb = self.extract_embedding(reference_path)
        gen_emb = self.extract_embedding(generated_path)
        if ref_emb is None or gen_emb is None:
            return None
        similarity = torch.nn.functional.cosine_similarity(ref_emb.unsqueeze(0), gen_emb.unsqueeze(0)).item()
        return (similarity + 1) / 2  # rescale [-1, 1] -> [0, 1]

    def calculate_batch(self, df: pd.DataFrame, ref_col: str = "reference_audio", gen_col: str = "generated_audio") -> pd.DataFrame:
        print(f"Calculating speaker similarity for {len(df)} pairs...")
        rows = []
        for idx, row in df.iterrows():
            similarity = self.calculate_similarity(row[ref_col], row[gen_col])
            rows.append(
                {
                    "reference_audio": row[ref_col],
                    "generated_audio": row[gen_col],
                    "speaker_similarity": round(similarity, 4) if similarity is not None else None,
                    "transcript": row.get("transcript", ""),
                    "speaker_gender": row.get("speaker_gender", row.get("Gender", "")),
                    "speaker_age": row.get("speaker_age", row.get("age", "")),
                }
            )
            if (idx + 1) % 10 == 0:
                print(f"  {idx + 1}/{len(df)}")
        return pd.DataFrame(rows)


def print_summary(results_df: pd.DataFrame):
    valid = results_df.dropna(subset=["speaker_similarity"])
    print(f"\nProcessed: {len(valid)}/{len(results_df)} pairs")
    if valid.empty:
        return

    print(f"Mean similarity:   {valid['speaker_similarity'].mean():.4f}")
    print(f"Median similarity: {valid['speaker_similarity'].median():.4f}")
    print(f"Std deviation:     {valid['speaker_similarity'].std():.4f}")

    excellent = (valid["speaker_similarity"] >= 0.8).sum()
    good = ((valid["speaker_similarity"] >= 0.7) & (valid["speaker_similarity"] < 0.8)).sum()
    acceptable = ((valid["speaker_similarity"] >= 0.6) & (valid["speaker_similarity"] < 0.7)).sum()
    poor = (valid["speaker_similarity"] < 0.6).sum()
    n = len(valid)
    print(f"\nQuality tiers:")
    print(f"  Excellent (>=0.80): {excellent} ({excellent / n * 100:.1f}%)")
    print(f"  Good      (0.70-0.79): {good} ({good / n * 100:.1f}%)")
    print(f"  Acceptable(0.60-0.69): {acceptable} ({acceptable / n * 100:.1f}%)")
    print(f"  Poor      (<0.60): {poor} ({poor / n * 100:.1f}%)")

    for col in ("speaker_gender", "speaker_age"):
        if col in valid.columns and valid[col].notna().any():
            print(f"\nBy {col}:")
            print(valid.groupby(col)["speaker_similarity"].agg(["mean", "count"]).round(4))


def main():
    parser = argparse.ArgumentParser(description="Objective ECAPA-TDNN speaker-similarity evaluation (paper Section 5.4)")
    parser.add_argument("--samples-csv", default=os.environ.get("GOYAR_SAMPLES_CSV", "samples.csv"))
    parser.add_argument("--ref-col", default="reference_audio")
    parser.add_argument("--gen-col", default="generated_audio")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="speaker_similarity_scores.csv")
    parser.add_argument("--pair", nargs=2, metavar=("REFERENCE_AUDIO", "GENERATED_AUDIO"), help="Score a single pair instead of a CSV")
    args = parser.parse_args()

    calculator = SpeakerSimilarityCalculator(args.model)

    if args.pair:
        similarity = calculator.calculate_similarity(*args.pair)
        print(f"Speaker similarity: {similarity:.4f}" if similarity is not None else "Failed to compute similarity")
        return

    df = pd.read_csv(args.samples_csv)
    results_df = calculator.calculate_batch(df, args.ref_col, args.gen_col)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved -> {args.output}")
    print_summary(results_df)


if __name__ == "__main__":
    main()
