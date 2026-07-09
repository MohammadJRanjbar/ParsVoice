#!/usr/bin/env python3
"""
Stage 4 -- Persian Text Quality Metrics (paper Section 3.4.1, Appendix E.2).

Scores every transcript on a 0-1 scale across weighted Persian-specific
dimensions: character-set validity, length appropriateness for TTS,
sentence structure, lexical repetition, linguistic complexity, and phonetic
(character) coverage. Segments are bucketed into descriptive tiers:

    >= 0.78          high quality
    0.62-0.78        mid quality
    <  0.62          low quality

Tiers are for reporting only -- the threshold that actually excludes
segments from the TTS-ready subset is 0.5 (Section 3.6). The default weights
below are the ones used to score the released corpus; NOTE they were tuned
iteratively against manual review and don't split identically into the six
named categories printed in the paper's Appendix Table 5 (that table groups
"sentence quality" into the surrounding dimensions) -- adjust `DEFAULT_WEIGHTS`
if you need an exact reproduction of a specific ablation.

Usage:
    python pipeline/04_score_text_quality.py
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

from config import TEXT_QUALITY_DIR, TRIMMED_SEGMENTS_DIR, ensure_dirs

DEFAULT_WEIGHTS = {
    "character_quality": 0.20,
    "length_quality": 0.15,
    "sentence_quality": 0.20,
    "repetition_score": 0.15,
    "linguistic_complexity": 0.15,
    "phonetic_coverage": 0.15,
}

# Tier boundaries used for reporting only (Section 3.4.1).
HIGH_QUALITY_THRESHOLD = 0.78
MID_QUALITY_THRESHOLD = 0.62
# Threshold that actually excludes segments from the TTS-ready release (Section 3.6).
TTS_FILTER_THRESHOLD = 0.5


class PersianTextQualityScorer:
    PERSIAN_CHARS = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
    PERSIAN_DIGITS = set("۰۱۲۳۴۵۶۷۸۹")
    ARABIC_CHARS = set("أإئؤةك")  # variant glyphs that should have been normalized to Persian

    COMMON_PERSIAN_WORDS = {
        "که", "در", "از", "به", "با", "این", "آن", "را", "و", "است", "یک", "تا",
        "کرد", "کند", "شد", "شده", "می‌شود", "خواهد", "باید", "دارد", "داشت",
        "بود", "بوده", "هست", "نیست", "برای", "روی", "زیر", "بین", "کنار",
    }

    def normalize_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return ""
        replacements = {"ك": "ک", "ي": "ی", "أ": "ا", "إ": "ا", "ة": "ه", "ؤ": "و", "ئ": "ی"}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return re.sub(r"\s+", " ", text.strip())

    def character_quality(self, text: str) -> float:
        if not text:
            return 0.0
        total = len(text)
        persian_ratio = sum(1 for c in text if c in self.PERSIAN_CHARS) / total
        digit_ratio = sum(1 for c in text if c.isdigit() or c in self.PERSIAN_DIGITS) / total
        punct_ratio = sum(1 for c in text if unicodedata.category(c).startswith("P")) / total
        arabic_penalty = sum(1 for c in text if c in self.ARABIC_CHARS) / total

        score = persian_ratio * 0.6 + min(digit_ratio * 10, 0.2) + min(punct_ratio * 5, 0.15) - arabic_penalty * 0.5
        return min(max(score, 0.0), 1.0)

    def length_quality(self, text: str) -> float:
        word_count = len(text.split())
        char_count = len(text.strip())
        if 5 <= word_count <= 20 and 30 <= char_count <= 150:
            return 1.0
        if 3 <= word_count <= 30 and 20 <= char_count <= 200:
            return 0.8
        if 1 <= word_count <= 50 and 10 <= char_count <= 300:
            return 0.6
        return 0.2

    def sentence_quality(self, text: str) -> float:
        has_end_punct = bool(re.search(r"[.!?؟]$", text.strip()))
        starts_properly = bool(re.match(r"^[؀-ۿ\s]*[؀-ۿ]", text.strip()))
        has_verb = any(word in text for word in ["است", "بود", "شد", "می‌", "خواهد"])
        return (0.3 if has_end_punct else 0.0) + (0.2 if starts_properly else 0.0) + (0.5 if has_verb else 0.0)

    def repetition_score(self, text: str) -> float:
        words = text.split()
        if len(words) < 3:
            return 1.0
        counts = Counter(words)
        repetition_penalty = min(max(counts.values()) / len(words) * 3, 0.5)
        uniqueness = len(counts) / len(words)
        return max(uniqueness - repetition_penalty, 0.0)

    def linguistic_complexity(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        common_ratio = sum(1 for w in words if w in self.COMMON_PERSIAN_WORDS) / len(words)
        avg_word_length = sum(len(w) for w in words) / len(words)
        length_score = 1.0 - min(avg_word_length / 15, 0.5)
        complexity_markers = text.count("،") + text.count("؛")
        complexity_penalty = min(complexity_markers / len(words) * 2, 0.3)
        score = common_ratio * 0.4 + length_score * 0.4 + (1.0 - complexity_penalty) * 0.2
        return min(max(score, 0.0), 1.0)

    def phonetic_coverage(self, text: str) -> float:
        text_lower = text.lower()
        unique_chars = {c for c in text_lower if c in self.PERSIAN_CHARS}
        char_diversity = len(unique_chars) / len(self.PERSIAN_CHARS)
        has_vowels = any(c in text_lower for c in "اآوی")
        has_consonants = any(c in text_lower for c in "بپتدکگ")
        balance_bonus = 0.2 if (has_vowels and has_consonants) else 0.0
        return min(char_diversity + balance_bonus, 1.0)

    def score_text(self, text: str, weights: Dict[str, float] = None) -> Dict[str, float]:
        weights = weights or DEFAULT_WEIGHTS
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return {**{k: 0.0 for k in weights}, "total_score": 0.0}

        normalized = self.normalize_text(text)
        scores = {
            "character_quality": self.character_quality(normalized),
            "length_quality": self.length_quality(normalized),
            "sentence_quality": self.sentence_quality(normalized),
            "repetition_score": self.repetition_score(normalized),
            "linguistic_complexity": self.linguistic_complexity(normalized),
            "phonetic_coverage": self.phonetic_coverage(normalized),
        }
        scores["total_score"] = sum(scores[k] * weights[k] for k in weights)
        return scores


def score_dataframe(df: pd.DataFrame, text_column: str = "final_transcript", path_column: str = "fixed_segment_path") -> pd.DataFrame:
    scorer = PersianTextQualityScorer()
    rows = []
    for _, row in df.iterrows():
        scores = scorer.score_text(row[text_column])
        # "segment_path" here is deliberately the boundary-optimized (Stage 2)
        # path, matching the join key used by stages 4/6/7/8 -- not the
        # pre-trim path from Stage 1.
        rows.append({"segment_path": row[path_column], "text": row[text_column], **scores})

    scores_df = pd.DataFrame(rows)
    scores_df["quality_tier"] = pd.cut(
        scores_df["total_score"],
        bins=[0, MID_QUALITY_THRESHOLD, HIGH_QUALITY_THRESHOLD, 1.0],
        labels=["low", "mid", "high"],
    )
    return scores_df


def load_trimmed_metadata() -> pd.DataFrame:
    """Loads fixed_metadata.csv (Stage 2 output) from every audiobook folder --
    the same universe of segments Stage 3 scores for audio quality."""
    metadata_files = sorted(TRIMMED_SEGMENTS_DIR.glob("*/fixed_metadata.csv"))
    if not metadata_files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in metadata_files], ignore_index=True)


def main():
    ensure_dirs(TEXT_QUALITY_DIR)
    df = load_trimmed_metadata()
    if df.empty:
        print(f"No fixed_metadata.csv files found under {TRIMMED_SEGMENTS_DIR} -- run stage 2 first.")
        return

    print(f"Scoring text quality for {len(df)} transcripts...")
    scores_df = score_dataframe(df, text_column="final_transcript", path_column="fixed_segment_path")

    print(f"Mean score: {scores_df['total_score'].mean():.3f}  Median: {scores_df['total_score'].median():.3f}")
    print(scores_df["quality_tier"].value_counts())

    output_path = TEXT_QUALITY_DIR / "text_quality_scores.csv"
    scores_df.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")

    passing = scores_df[scores_df["total_score"] >= TTS_FILTER_THRESHOLD]
    print(f"Above TTS-release threshold ({TTS_FILTER_THRESHOLD}): {len(passing)}/{len(scores_df)}")


if __name__ == "__main__":
    main()
