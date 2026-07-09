#!/usr/bin/env python3
"""
Stage 6 -- Speaker Identification (paper Section 3.5, Appendix D).

Two-stage pipeline that assigns consistent speaker labels across the corpus,
needed because ~40% of IranSeda entries lack narrator metadata and some
books have multiple narrators:

    Stage A (local clustering, per book): the number of speakers k* is
        estimated by consensus among Silhouette, gap-statistic, and DBSCAN
        criteria; clustering itself uses an ensemble of Agglomerative,
        Spectral, and Gaussian Mixture Model methods (best silhouette wins).
        Each segment gets a confidence score combining its silhouette
        contribution and centroid distance; low-confidence segments and
        clusters smaller than 10% of a book's high-confidence pool are
        dropped.

    Stage B (global merging, across books): local cluster centroids are
        compared pairwise by cosine similarity and merged via agglomerative
        clustering at a 0.85 similarity threshold. Cross-book merges further
        require >=100 shared segments between the matched speakers, to avoid
        spurious merges from small samples.

Both stages are checkpointed so a run over thousands of books/speakers can be
resumed.

Usage:
    python pipeline/06_identify_speakers.py local              # Stage A only
    python pipeline/06_identify_speakers.py global             # Stage B only
    python pipeline/06_identify_speakers.py all                # A then B
    python pipeline/06_identify_speakers.py local --retry-failed
"""

import argparse
import hashlib
import json
import os
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from tqdm import tqdm

from config import EMBEDDINGS_DIR, GLOBAL_SPEAKER_ID_DIR, SPEAKER_ID_DIR, ensure_dirs

warnings.filterwarnings("ignore")

# Conservative defaults (Appendix D): higher threshold = fewer, safer merges.
LOCAL_CONFIDENCE_THRESHOLD = 0.4
GLOBAL_SIMILARITY_THRESHOLD = 0.85
GLOBAL_VALIDATION_THRESHOLD = 0.90
GLOBAL_MIN_SAMPLES_FOR_MERGE = 100


# ============================================================================
# Checkpointing (Stage A: local per-book clustering)
# ============================================================================
class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.completed_file = self.checkpoint_dir / "completed_audiobooks.json"
        self.failed_file = self.checkpoint_dir / "failed_audiobooks.json"
        self.completed = self._load(self.completed_file)
        self.failed = self._load(self.failed_file)

    @staticmethod
    def _load(path: Path) -> Dict:
        try:
            return json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            return {}

    @staticmethod
    def _save(data: Dict, path: Path):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(path)

    @staticmethod
    def _hash(audiobook_path, csv_file) -> str:
        try:
            mtime = os.path.getmtime(csv_file)
            return hashlib.md5(f"{audiobook_path}:{csv_file}:{mtime}".encode()).hexdigest()
        except Exception:
            return hashlib.md5(f"{audiobook_path}:{csv_file}".encode()).hexdigest()

    def is_completed(self, audiobook_path, csv_file) -> bool:
        return self._hash(audiobook_path, csv_file) in self.completed

    def is_failed(self, audiobook_path, csv_file) -> bool:
        return self._hash(audiobook_path, csv_file) in self.failed

    def mark_completed(self, audiobook_path, csv_file, result):
        key = self._hash(audiobook_path, csv_file)
        self.completed[key] = {"audiobook_path": str(audiobook_path), "result": result, "completed_at": datetime.now().isoformat()}
        self._save(self.completed, self.completed_file)

    def mark_failed(self, audiobook_path, csv_file, error):
        key = self._hash(audiobook_path, csv_file)
        self.failed[key] = {"audiobook_path": str(audiobook_path), "error": str(error), "failed_at": datetime.now().isoformat()}
        self._save(self.failed, self.failed_file)

    def clear_failed(self, audiobook_path, csv_file):
        key = self._hash(audiobook_path, csv_file)
        if key in self.failed:
            del self.failed[key]
            self._save(self.failed, self.failed_file)


# ============================================================================
# Stage A: local (within-book) speaker clustering
# ============================================================================
class LocalSpeakerClustering:
    """Clusters one audiobook's segment embeddings into local speaker
    identities (Appendix D, "Local clustering")."""

    def __init__(self, embeddings: List[np.ndarray]):
        self.embeddings = np.array(embeddings)
        self.n_samples = len(embeddings)
        self.preprocessed = self._preprocess()

    def _preprocess(self) -> np.ndarray:
        embeddings = self.embeddings.copy()

        # Remove statistical outliers (z-score > 3) before clustering.
        z_scores = np.abs(zscore(embeddings, axis=0))
        outlier_mask = np.any(z_scores > 3, axis=1)
        if np.sum(~outlier_mask) > 10:
            embeddings = embeddings[~outlier_mask]
            self.valid_indices = np.where(~outlier_mask)[0]
        else:
            self.valid_indices = np.arange(len(embeddings))

        embeddings = normalize(embeddings, norm="l2", axis=1)
        if embeddings.shape[1] > 512:
            pca = PCA(n_components=min(512, embeddings.shape[0] - 1), random_state=42)
            embeddings = pca.fit_transform(embeddings)
        return embeddings

    def estimate_num_speakers(self, max_speakers: int = 15) -> Dict[str, int]:
        """Consensus among Silhouette, gap statistic, and DBSCAN estimates."""
        embeddings = self.preprocessed
        results = {}

        silhouette_scores = []
        for k in range(2, min(max_speakers + 1, len(embeddings))):
            labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(embeddings)
            if np.min(np.bincount(labels)) < 2:
                continue
            silhouette_scores.append((k, silhouette_score(embeddings, labels, metric="cosine")))
        if silhouette_scores:
            results["silhouette"] = max(silhouette_scores, key=lambda x: x[1])[0]

        gap_stats = []
        for k in range(1, min(max_speakers + 1, len(embeddings))):
            if k == 1:
                dispersion = np.sum(pdist(embeddings, metric="cosine"))
            else:
                labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(embeddings)
                dispersion = sum(
                    np.sum(pdist(embeddings[labels == label], metric="cosine"))
                    for label in np.unique(labels)
                    if np.sum(labels == label) > 1
                )
            gap_stats.append((k, dispersion))
        if len(gap_stats) > 2:
            second_diffs = np.diff(np.diff([d for _, d in gap_stats]))
            if len(second_diffs) > 0:
                elbow_idx = np.argmax(second_diffs) + 2
                if elbow_idx < len(gap_stats):
                    results["gap_statistic"] = gap_stats[elbow_idx][0]

        dbscan_results = []
        for eps in np.arange(0.1, 1.0, 0.05):
            labels = DBSCAN(eps=eps, metric="cosine", min_samples=3).fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            if n_clusters >= 2 and n_noise < len(embeddings) * 0.5:
                dbscan_results.append((eps, n_clusters, silhouette_score(embeddings, labels, metric="cosine")))
        if dbscan_results:
            results["dbscan"] = max(dbscan_results, key=lambda x: x[2])[1]

        return results

    def _single_clustering(self, embeddings: np.ndarray, n_clusters: int, method: str) -> np.ndarray:
        if method == "agglomerative":
            return AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(embeddings)
        if method == "spectral":
            similarity = 1 - pairwise_distances(embeddings, metric="cosine") + 1e-10 * np.eye(len(embeddings))
            return SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42).fit_predict(similarity)
        if method == "gmm":
            return GaussianMixture(n_components=n_clusters, random_state=42, covariance_type="full").fit_predict(embeddings)
        raise ValueError(method)

    def _co_occurrence_ensemble(self, results: Dict[str, Dict], embeddings: np.ndarray) -> np.ndarray:
        """Combine multiple methods' cluster assignments into a single
        co-occurrence matrix (fraction of methods that agree two samples
        share a cluster), then re-cluster on that agreement matrix."""
        all_labels = np.array([results[method]["labels"] for method in results])
        n_samples = len(embeddings)
        co_occurrence = np.zeros((n_samples, n_samples))
        for labels in all_labels:
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        co_occurrence[i, j] += 1
        co_occurrence /= len(results)
        distance_matrix = 1 - co_occurrence
        return AgglomerativeClustering(
            n_clusters=len(np.unique(all_labels[0])), metric="precomputed", linkage="average"
        ).fit_predict(distance_matrix)

    def cluster(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """Ensemble of Agglomerative / Spectral / GMM: try all three, take
        the best by silhouette, but prefer a co-occurrence combination of
        all methods if it scores even higher (Appendix D)."""
        embeddings = self.preprocessed
        if n_clusters is None:
            estimates = self.estimate_num_speakers()
            n_clusters = int(np.median(list(estimates.values()))) if estimates else 2

        results = {}
        for method in ("agglomerative", "spectral", "gmm"):
            try:
                labels = self._single_clustering(embeddings, n_clusters, method)
                results[method] = {"labels": labels, "silhouette": silhouette_score(embeddings, labels, metric="cosine")}
            except Exception:
                continue

        if not results:
            return self._single_clustering(embeddings, n_clusters, "agglomerative")

        best_method = max(results, key=lambda m: results[m]["silhouette"])
        if len(results) > 1:
            ensemble_labels = self._co_occurrence_ensemble(results, embeddings)
            ensemble_silhouette = silhouette_score(embeddings, ensemble_labels, metric="cosine")
            if ensemble_silhouette > results[best_method]["silhouette"]:
                return ensemble_labels
        return results[best_method]["labels"]

    def confidence_scores(self, labels: np.ndarray) -> np.ndarray:
        """confidence = 0.6 * normalized_silhouette + 0.4 * distance_confidence (Appendix D)."""
        embeddings = self.preprocessed
        sample_silhouette = silhouette_samples(embeddings, labels, metric="cosine")
        confidences = np.zeros(len(embeddings))

        centroids = {label: np.mean(embeddings[labels == label], axis=0) for label in np.unique(labels)}
        for i in range(len(embeddings)):
            label = labels[i]
            cluster_embeddings = embeddings[labels == label]
            if len(cluster_embeddings) > 1:
                dist_to_own = 1 - np.dot(embeddings[i], centroids[label])
                other_dists = [1 - np.dot(embeddings[i], c) for lbl, c in centroids.items() if lbl != label]
                if other_dists:
                    min_dist_to_other = min(other_dists)
                    distance_confidence = (min_dist_to_other - dist_to_own) / min_dist_to_other if min_dist_to_other > dist_to_own else 0.0
                else:
                    distance_confidence = 1.0
            else:
                distance_confidence = 0.5

            normalized_silhouette = (sample_silhouette[i] + 1) / 2
            confidences[i] = max(0, min(1, 0.6 * normalized_silhouette + 0.4 * distance_confidence))
        return confidences


def process_single_audiobook(args: Tuple[str, str, Path, CheckpointManager]) -> Dict:
    audiobook_path, csv_file, output_base_dir, checkpoint = args
    audiobook_name = os.path.basename(audiobook_path)
    start_time = time.time()

    if checkpoint.is_completed(audiobook_path, csv_file):
        cached = checkpoint.completed[checkpoint._hash(audiobook_path, csv_file)]["result"].copy()
        cached["status"] = "cached"
        return cached

    try:
        df = pd.read_csv(csv_file)
        if len(df) < 4:
            result = {"audiobook": audiobook_name, "status": "skipped", "reason": "too_few_files", "files_count": len(df)}
            checkpoint.mark_completed(audiobook_path, csv_file, result)
            return result

        with ThreadPoolExecutor(max_workers=8) as executor:
            embeddings = list(executor.map(lambda p: np.load(p) if os.path.exists(p) else None, df["embedding_path"]))
        valid_idx = [i for i, e in enumerate(embeddings) if e is not None]
        if len(valid_idx) < 4:
            result = {"audiobook": audiobook_name, "status": "skipped", "reason": "too_few_valid_embeddings"}
            checkpoint.mark_completed(audiobook_path, csv_file, result)
            return result

        df_valid = df.iloc[valid_idx].reset_index(drop=True)
        embeddings = [embeddings[i] for i in valid_idx]

        clustering = LocalSpeakerClustering(embeddings)
        labels = clustering.cluster()
        confidences = clustering.confidence_scores(labels)

        # Map clustering results (computed on outlier-filtered embeddings)
        # back onto the full segment set, assigning outliers to their nearest centroid.
        full_labels = np.full(len(df_valid), -1)
        full_confidences = np.full(len(df_valid), 0.0)
        full_labels[clustering.valid_indices] = labels
        full_confidences[clustering.valid_indices] = confidences

        outlier_indices = np.setdiff1d(np.arange(len(df_valid)), clustering.valid_indices)
        if len(outlier_indices) > 0:
            centroids = [np.mean(clustering.preprocessed[labels == c], axis=0) for c in np.unique(labels)]
            for idx in outlier_indices:
                outlier_embedding = normalize(embeddings[idx].reshape(1, -1), norm="l2")[0]
                similarities = [np.dot(outlier_embedding, c) for c in centroids]
                full_labels[idx] = int(np.argmax(similarities))
                full_confidences[idx] = max(similarities) * 0.5  # discounted confidence for outliers

        num_speakers = len(np.unique(full_labels))
        min_speaker_ratio = 1.0 / max(num_speakers, 1)
        high_confidence_mask = full_confidences >= LOCAL_CONFIDENCE_THRESHOLD
        high_conf_count = np.sum(high_confidence_mask)
        min_samples_per_speaker = max(2, int(high_conf_count * min_speaker_ratio))

        speaker_counts = Counter(full_labels[high_confidence_mask])
        valid_speakers = {s for s, c in speaker_counts.items() if c >= min_samples_per_speaker}
        final_mask = high_confidence_mask & np.isin(full_labels, list(valid_speakers))

        df_filtered = df_valid[final_mask].copy()
        if len(df_filtered) > 0:
            df_filtered["speaker_id"] = full_labels[final_mask]
            df_filtered["speaker_confidence"] = full_confidences[final_mask]
            remap = {old: new for new, old in enumerate(sorted(df_filtered["speaker_id"].unique()))}
            df_filtered["speaker_id"] = df_filtered["speaker_id"].map(remap)

        output_dir = output_base_dir / audiobook_name
        output_dir.mkdir(parents=True, exist_ok=True)
        base_filename = Path(csv_file).stem
        if len(df_filtered) > 0:
            df_filtered.to_csv(output_dir / f"{base_filename}_final_speakers.csv", index=False)

        result = {
            "audiobook": audiobook_name,
            "status": "completed",
            "original_files": len(df),
            "valid_embeddings": len(df_valid),
            "final_kept_files": len(df_filtered),
            "retention_rate": len(df_filtered) / len(df_valid) * 100 if len(df_valid) else 0,
            "num_speakers": len(df_filtered["speaker_id"].unique()) if len(df_filtered) else 0,
            "processing_time": time.time() - start_time,
        }
        checkpoint.mark_completed(audiobook_path, csv_file, result)
        print(f"  {audiobook_name}: {len(df_filtered)}/{len(df_valid)} kept, {result['num_speakers']} speakers")
        return result

    except Exception as exc:
        checkpoint.mark_failed(audiobook_path, csv_file, exc)
        return {"audiobook": audiobook_name, "status": "error", "error": str(exc)}


def find_audiobook_embedding_csvs(base_dir: Path) -> List[Tuple[str, str]]:
    pairs = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        csvs = [f for f in entry.glob("*.csv") if "embedding" in f.name.lower() and "backup" not in f.name.lower()]
        if csvs:
            pairs.append((str(entry), str(csvs[0])))
    return pairs


def run_local_clustering(retry_failed: bool = False):
    ensure_dirs(SPEAKER_ID_DIR)
    checkpoint = CheckpointManager(SPEAKER_ID_DIR / ".checkpoints")

    audiobook_pairs = find_audiobook_embedding_csvs(EMBEDDINGS_DIR)
    print(f"Found {len(audiobook_pairs)} audiobooks with embeddings")

    pending = []
    for audiobook_path, csv_file in audiobook_pairs:
        if checkpoint.is_completed(audiobook_path, csv_file):
            continue
        if checkpoint.is_failed(audiobook_path, csv_file):
            if retry_failed:
                checkpoint.clear_failed(audiobook_path, csv_file)
            else:
                continue
        pending.append((audiobook_path, csv_file, SPEAKER_ID_DIR, checkpoint))

    print(f"Processing {len(pending)} audiobooks ({len(audiobook_pairs) - len(pending)} already done)")
    results = []
    with ProcessPoolExecutor(max_workers=min(len(pending), os.cpu_count() or 1) or 1) as executor:
        futures = {executor.submit(process_single_audiobook, p): p[0] for p in pending}
        for future in tqdm(as_completed(futures), total=len(pending), desc="Local clustering"):
            results.append(future.result())

    completed = [r for r in results if r.get("status") == "completed"]
    print(f"Local clustering done: {len(completed)} completed, {len(results) - len(completed)} skipped/failed")
    pd.DataFrame(results).to_csv(SPEAKER_ID_DIR / "local_clustering_summary.csv", index=False)


# ============================================================================
# Stage B: global (cross-book) speaker merging
# ============================================================================
@dataclass
class SpeakerRepresentation:
    book_name: str
    local_speaker_id: int
    centroid_embedding: np.ndarray
    num_samples: int
    avg_confidence: float


@dataclass
class GlobalSpeaker:
    global_id: int
    book_speakers: List[SpeakerRepresentation]
    total_samples: int
    books: List[str]
    confidence_score: float


class CrossBookSpeakerIdentifier:
    """Merges local per-book speaker identities into global speaker IDs
    (Appendix D, "Global merging")."""

    def __init__(self, processed_books_dir: Path, similarity_threshold: float, min_samples_for_merge: int):
        self.processed_books_dir = Path(processed_books_dir)
        self.similarity_threshold = similarity_threshold
        self.min_samples_for_merge = min_samples_for_merge
        self.book_speakers: List[SpeakerRepresentation] = []
        self.global_speakers: List[GlobalSpeaker] = []
        self.similarity_matrix = None

    def load_all_book_speakers(self) -> bool:
        book_dirs = [d for d in self.processed_books_dir.iterdir() if d.is_dir()]
        for book_dir in tqdm(book_dirs, desc="Loading books"):
            final_csv = next(iter(book_dir.glob("*_final_speakers.csv")), None)
            if not final_csv:
                continue
            df = pd.read_csv(final_csv)
            if df.empty:
                continue
            self.book_speakers.extend(self._extract_book_speakers(df, book_dir.name))
        print(f"Loaded {len(self.book_speakers)} local speakers from {len(book_dirs)} books")
        return len(self.book_speakers) > 0

    @staticmethod
    def _extract_book_speakers(df: pd.DataFrame, book_name: str) -> List[SpeakerRepresentation]:
        with ThreadPoolExecutor(max_workers=4) as executor:
            embeddings = list(executor.map(lambda p: np.load(p) if os.path.exists(p) else None, df["embedding_path"]))

        by_speaker = defaultdict(list)
        for i, emb in enumerate(embeddings):
            if emb is not None:
                by_speaker[df.iloc[i]["speaker_id"]].append(
                    {"embedding": normalize(emb.reshape(1, -1), norm="l2")[0], "confidence": df.iloc[i]["speaker_confidence"]}
                )

        speakers = []
        for local_id, samples in by_speaker.items():
            if len(samples) < 2:
                continue
            weights = np.array([s["confidence"] for s in samples])
            weights = weights / np.sum(weights)
            centroid = normalize(
                np.average([s["embedding"] for s in samples], axis=0, weights=weights).reshape(1, -1), norm="l2"
            )[0]
            speakers.append(
                SpeakerRepresentation(
                    book_name=book_name,
                    local_speaker_id=local_id,
                    centroid_embedding=centroid,
                    num_samples=len(samples),
                    avg_confidence=float(np.mean([s["confidence"] for s in samples])),
                )
            )
        return speakers

    def compute_similarities(self) -> bool:
        if len(self.book_speakers) < 2:
            return False
        centroids = np.array([s.centroid_embedding for s in self.book_speakers])
        self.similarity_matrix = 1 - pairwise_distances(centroids, metric="cosine")
        return True

    def _validate_cluster(self, speakers: List[SpeakerRepresentation]) -> float:
        if len(speakers) == 1:
            return 1.0
        similarities = [
            np.dot(speakers[i].centroid_embedding, speakers[j].centroid_embedding)
            for i in range(len(speakers))
            for j in range(i + 1, len(speakers))
        ]
        min_sim = np.min(similarities)
        if min_sim >= GLOBAL_VALIDATION_THRESHOLD:
            return 1.0
        if min_sim >= self.similarity_threshold:
            return 0.5 + 0.5 * (min_sim - self.similarity_threshold) / (GLOBAL_VALIDATION_THRESHOLD - self.similarity_threshold)
        return 0.0

    def identify_global_speakers(self) -> bool:
        if self.similarity_matrix is None:
            return False
        distance_matrix = 1 - self.similarity_matrix
        try:
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=1 - self.similarity_threshold, metric="precomputed", linkage="average"
            ).fit_predict(distance_matrix)
        except Exception:
            labels = np.arange(len(self.book_speakers))

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)

        self.global_speakers = []
        global_id = 0
        for indices in clusters.values():
            cluster_speakers = [self.book_speakers[i] for i in indices]
            total_samples = sum(s.num_samples for s in cluster_speakers)
            books_involved = {s.book_name for s in cluster_speakers}

            if len(books_involved) > 1 and total_samples < self.min_samples_for_merge:
                # Not enough cross-book evidence: keep these as separate single-book speakers.
                for speaker in cluster_speakers:
                    self.global_speakers.append(
                        GlobalSpeaker(global_id, [speaker], speaker.num_samples, [speaker.book_name], 1.0)
                    )
                    global_id += 1
                continue

            confidence = self._validate_cluster(cluster_speakers) if len(cluster_speakers) > 1 else 1.0
            self.global_speakers.append(GlobalSpeaker(global_id, cluster_speakers, total_samples, sorted(books_involved), confidence))
            global_id += 1

        self.global_speakers.sort(key=lambda gs: gs.total_samples, reverse=True)
        for i, gs in enumerate(self.global_speakers):
            gs.global_id = i

        multi_book = sum(1 for gs in self.global_speakers if len(gs.books) > 1)
        print(f"Identified {len(self.global_speakers)} global speakers ({multi_book} span multiple books)")
        return True

    def write_global_ids(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        mapping = {
            (bs.book_name, bs.local_speaker_id): gs.global_id for gs in self.global_speakers for bs in gs.book_speakers
        }

        for book_dir in tqdm([d for d in self.processed_books_dir.iterdir() if d.is_dir()], desc="Writing global IDs"):
            final_csv = next(iter(book_dir.glob("*_final_speakers.csv")), None)
            if not final_csv:
                continue
            df = pd.read_csv(final_csv)
            df["global_speaker_id"] = df["speaker_id"].apply(lambda local_id: mapping.get((book_dir.name, local_id), -1))
            book_output_dir = output_dir / book_dir.name
            book_output_dir.mkdir(exist_ok=True)
            df.to_csv(book_output_dir / final_csv.name, index=False)

        summary = pd.DataFrame(
            [
                {
                    "global_speaker_id": gs.global_id,
                    "total_samples": gs.total_samples,
                    "num_books": len(gs.books),
                    "books": ", ".join(gs.books),
                    "confidence_score": gs.confidence_score,
                }
                for gs in self.global_speakers
            ]
        )
        summary.to_csv(output_dir / "global_speakers_summary.csv", index=False)
        return output_dir


def run_global_merging():
    ensure_dirs(GLOBAL_SPEAKER_ID_DIR)
    identifier = CrossBookSpeakerIdentifier(SPEAKER_ID_DIR, GLOBAL_SIMILARITY_THRESHOLD, GLOBAL_MIN_SAMPLES_FOR_MERGE)

    if not identifier.load_all_book_speakers():
        print("No local speaker data found -- run `local` first.")
        return
    if not identifier.compute_similarities():
        print("Not enough speakers for cross-book merging.")
        return
    identifier.identify_global_speakers()
    output_dir = identifier.write_global_ids(GLOBAL_SPEAKER_ID_DIR)
    print(f"Global speaker identification complete -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 6: speaker identification (local clustering + global merging)")
    parser.add_argument("mode", choices=["local", "global", "all"], help="Which stage to run")
    parser.add_argument("--retry-failed", action="store_true", help="Retry books that previously failed local clustering")
    args = parser.parse_args()

    if args.mode in ("local", "all"):
        run_local_clustering(retry_failed=args.retry_failed)
    if args.mode in ("global", "all"):
        run_global_merging()


if __name__ == "__main__":
    main()
