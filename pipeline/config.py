"""
Shared configuration for the ParsVoice pipeline.

Every stage reads its input/output locations from environment variables so the
pipeline can run unmodified on a laptop, a server, or inside a notebook. Each
variable has a sensible local default under ./data/, so you can try the
pipeline immediately and override paths later (e.g. via a .env file or
`export PARSVOICE_...=...`) once you move to real data.

The pipeline takes a CSV of audiobooks you already have on disk as input (see
AUDIOBOOKS_CSV below) -- it does not fetch anything itself. Stage outputs
feed the next stage's input by default, mirroring the flow described in
Section 3 of the paper:

    01 (segment & transcribe) -> per-book segments + transcripts
    02 (boundary optimization) -> trimmed segments
    03 (audio quality) -> audio quality scores
    04 (text quality) -> text quality scores
    05 (speaker embeddings) -> per-segment embeddings
    06 (speaker identification) -> local + global speaker IDs
    07 (punctuation restoration) -> release-ready transcripts
"""

import os
from pathlib import Path


def _path(env_var: str, default: str) -> Path:
    return Path(os.environ.get(env_var, default)).expanduser()


# Root directory all default paths live under. Override with PARSVOICE_DATA_ROOT
# to point everything (inputs and outputs alike) somewhere else in one go.
DATA_ROOT = _path("PARSVOICE_DATA_ROOT", "./data")

# --- Pipeline input -----------------------------------------------------------
# A CSV you provide, listing the audiobooks to process. Required columns:
#   title       -- book title (used to derive an output folder name)
#   narrator    -- narrator name, or empty/"unknown" if not known
#   audio_path  -- path to either a single audio file for the whole book, or
#                  a directory containing one audio file per chapter
# See the "Pipeline input format" section of the README for an example.
AUDIOBOOKS_CSV = _path("PARSVOICE_AUDIOBOOKS_CSV", DATA_ROOT / "audiobooks.csv")

# --- Stage 1: Segmentation, ASR transcription, completeness checking --------
SEGMENTS_DIR = _path("PARSVOICE_SEGMENTS_DIR", DATA_ROOT / "01_segments")
# Fine-tuned ParsBERT sentence-completion checkpoint (Appendix C). This is a
# pretrained artifact, not produced by this repository -- point it at your
# local copy or a Hugging Face Hub model id (link omitted for anonymous review).
SENTENCE_COMPLETION_MODEL_PATH = os.environ.get(
    "PARSVOICE_COMPLETION_MODEL_PATH", "<path-to-sentence-completion-model>"
)

# --- Stage 2: Boundary optimization (hybrid binary/linear trim search) ------
TRIMMED_SEGMENTS_DIR = _path("PARSVOICE_TRIMMED_DIR", DATA_ROOT / "02_trimmed_segments")

# --- Stage 3/4: Quality assessment -----------------------------------------
AUDIO_QUALITY_DIR = _path("PARSVOICE_AUDIO_QUALITY_DIR", DATA_ROOT / "03_audio_quality")
TEXT_QUALITY_DIR = _path("PARSVOICE_TEXT_QUALITY_DIR", DATA_ROOT / "04_text_quality")

# --- Stage 5: Speaker embeddings --------------------------------------------
EMBEDDINGS_DIR = _path("PARSVOICE_EMBEDDINGS_DIR", DATA_ROOT / "05_speaker_embeddings")
SPEAKER_EMBEDDING_MODEL = os.environ.get(
    "PARSVOICE_SPEAKER_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
)

# --- Stage 6: Speaker identification (local clustering + global merging) ---
SPEAKER_ID_DIR = _path("PARSVOICE_SPEAKER_ID_DIR", DATA_ROOT / "06_speaker_identification")
GLOBAL_SPEAKER_ID_DIR = _path(
    "PARSVOICE_GLOBAL_SPEAKER_ID_DIR", DATA_ROOT / "06_speaker_identification" / "global"
)

# --- Stage 7: Punctuation restoration + final release -----------------------
RELEASE_DIR = _path("PARSVOICE_RELEASE_DIR", DATA_ROOT / "07_release")
# ParsBERT-based Persian punctuation restoration model (PersianPunc; citation
# omitted for anonymous review). Also a pretrained artifact, not trained by
# this repo.
PUNCTUATION_MODEL_PATH = os.environ.get(
    "PARSVOICE_PUNCTUATION_MODEL_PATH", "<path-to-punctuation-restoration-model>"
)


def ensure_dirs(*paths: Path) -> None:
    """Create directories (including parents) if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
