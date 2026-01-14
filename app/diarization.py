"""
Online speaker diarization using TitaNet + StreamingKMeans.

Key improvement over Mesolitica: Batched GPU inference for speaker embeddings.
Reference: https://github.com/huseinzol05/backup-mesolitica-api/blob/master/app/main.py#L849
"""

import os
import logging
import bisect
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.cuda as cuda

logger = logging.getLogger(__name__)

SPEAKER_EMBEDDING_BATCH_SIZE = int(os.environ.get("SPEAKER_EMBEDDING_BATCH_SIZE", "16"))

# CUDA streams for overlapped host-device communication
h2d_stream = cuda.Stream()
compute_stream = cuda.Stream()

# Minimum audio samples required for TitaNet embedding extraction
# TitaNet uses hop_length=160 at 16kHz, needs multiple frames for mel spectrogram
# 0.5 seconds = 8000 samples at 16kHz is a safe minimum
MIN_CHUNK_SAMPLES = int(os.environ.get("MIN_CHUNK_SAMPLES_FOR_EMBEDDING", "8000"))

_speaker_model = None


def load_speaker_model():
    """
    Load TitaNet Large model for speaker embedding extraction.
    Called once at application startup.

    Uses local implementation with FP16 casting for memory efficiency.
    Reference: backup-mesolitica-api/app/main.py lines 140-144
    """
    global _speaker_model
    if _speaker_model is not None:
        return _speaker_model

    from app.nemo_speaker_vector import nemo_speaker_vector

    logger.info("Loading TitaNet Large speaker embedding model...")
    _speaker_model = nemo_speaker_vector(model="huseinzol05/nemo-titanet_large")
    logger.info("TitaNet Large model loaded successfully")

    return _speaker_model


def get_speaker_model():
    """Get the loaded speaker model instance."""
    global _speaker_model
    if _speaker_model is None:
        raise RuntimeError("Speaker model not loaded. Call load_speaker_model() first.")
    return _speaker_model


def extract_embeddings_batched(
    audio_chunks: List[np.ndarray],
    batch_size: int = SPEAKER_EMBEDDING_BATCH_SIZE,
) -> List[torch.Tensor]:
    """
    Extract speaker embeddings for multiple audio chunks using batched GPU inference.

    Args:
        audio_chunks: List of audio numpy arrays (float32, 16kHz)
        batch_size: Number of chunks to process in one GPU call

    Returns:
        List of embedding vectors (torch.Tensor, one per chunk)
    """
    model = get_speaker_model()
    embeddings = []

    total_chunks = len(audio_chunks)
    logger.debug(
        f"Extracting embeddings for {total_chunks} chunks (batch_size={batch_size})"
    )

    for i in range(0, total_chunks, batch_size):
        batch = audio_chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size

        logger.debug(
            f"Processing embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)"
        )

        # Prepare batch: pad and create pinned tensors
        inputs_pinned, lengths_pinned = model.prep_batch(batch)

        # Transfer to GPU with H2D stream
        with cuda.stream(h2d_stream):
            inputs_gpu = inputs_pinned.to(model.device, non_blocking=True)
            lengths_gpu = lengths_pinned.to(model.device, non_blocking=True)

        # Compute with compute stream, waiting for transfer
        with cuda.stream(compute_stream):
            compute_stream.wait_stream(h2d_stream)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    batch_emb = model.compute_batch(inputs_gpu, lengths_gpu)

        # Synchronize and collect
        compute_stream.synchronize()
        embeddings.extend(list(batch_emb.unbind(dim=0)))

    return embeddings


def assign_skipped_chunks_to_nearest(
    speaker_assignments: Dict[int, int],
    total_chunks: int,
) -> Dict[int, int]:
    """
    Assign skipped chunks (too short for embedding) to nearest valid chunk's speaker.

    Optimized O(n log m) implementation using sorted indices and binary search,
    where n = number of skipped chunks and m = number of valid chunks.

    Args:
        speaker_assignments: Dictionary mapping valid chunk indices to speaker IDs
        total_chunks: Total number of chunks (including skipped ones)

    Returns:
        Updated speaker_assignments dictionary with skipped chunks assigned
    """
    if not speaker_assignments:
        return {idx: 0 for idx in range(total_chunks)}

    all_indices = set(range(total_chunks))
    valid_indices = set(speaker_assignments.keys())
    skipped_indices = all_indices - valid_indices

    if not skipped_indices:
        return speaker_assignments

    sorted_valid_indices = sorted(valid_indices)
    result = speaker_assignments.copy()

    for skipped_idx in skipped_indices:
        pos = bisect.bisect_left(sorted_valid_indices, skipped_idx)

        nearest_idx = None
        min_distance = float("inf")

        if pos > 0:
            left_idx = sorted_valid_indices[pos - 1]
            distance = abs(skipped_idx - left_idx)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = left_idx

        if pos < len(sorted_valid_indices):
            right_idx = sorted_valid_indices[pos]
            distance = abs(skipped_idx - right_idx)
            if distance < min_distance:
                nearest_idx = right_idx

        if nearest_idx is not None:
            result[skipped_idx] = speaker_assignments[nearest_idx]
        else:
            result[skipped_idx] = 0

    return result


def run_online_diarization(
    audio_chunks: List[np.ndarray],
    speaker_similarity: float = 0.3,
    speaker_max_n: int = 10,
) -> Dict[int, int]:
    """
    Run full online diarization pipeline.

    Designed to be called via ProcessPoolExecutor for parallel execution
    alongside transcription. The model is loaded via executor initializer.

    Args:
        audio_chunks: List of audio numpy arrays (float32, 16kHz)
        speaker_similarity: Cosine similarity threshold for same speaker (0.0-1.0)
        speaker_max_n: Maximum number of speakers to detect

    Returns:
        Dictionary mapping chunk_index -> speaker_id (0-indexed)
    """
    from app.clustering_torch import StreamingKMeansMaxClusterTorch
    import torch
    import time

    if not audio_chunks:
        return {}

    t_start = time.time()
    logger.info(f"Starting online diarization for {len(audio_chunks)} chunks...")

    # 1. Filter valid chunks (long enough for embedding)
    valid_chunks = []
    valid_indices = []
    skipped_count = 0

    for idx, chunk in enumerate(audio_chunks):
        if len(chunk) >= MIN_CHUNK_SAMPLES:
            valid_chunks.append(chunk)
            valid_indices.append(idx)
        else:
            skipped_count += 1
            logger.debug(
                f"Chunk {idx} too short for embedding ({len(chunk)} < {MIN_CHUNK_SAMPLES})"
            )

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} chunks too short for embedding")

    if not valid_chunks:
        logger.warning("No chunks long enough for speaker embedding")
        return {i: 0 for i in range(len(audio_chunks))}

    # 2. Extract ALL embeddings at once (batched GPU inference)
    t_embed = time.time()
    embeddings = extract_embeddings_batched(valid_chunks)
    logger.info(f"⏱️ Embedding extraction took: {time.time() - t_embed:.2f}s")

    # 3. Cluster incrementally using StreamingKMeans (PyTorch version)
    t_cluster = time.time()
    cluster = StreamingKMeansMaxClusterTorch(
        threshold=speaker_similarity, max_clusters=speaker_max_n
    )

    # Embeddings are already torch tensors from the updated extract_embeddings_batched

    speaker_assignments = {}
    for i, embedding in enumerate(embeddings):
        speaker_id = cluster.streaming(embedding)
        speaker_assignments[valid_indices[i]] = speaker_id

    logger.info(f"⏱️ Clustering took: {time.time() - t_cluster:.2f}s")

    # 4. Assign skipped chunks to nearest valid chunk's speaker
    result = assign_skipped_chunks_to_nearest(speaker_assignments, len(audio_chunks))

    # Log summary
    speaker_counts = {}
    for speaker_id in result.values():
        speaker_counts[int(speaker_id)] = speaker_counts.get(int(speaker_id), 0) + 1

    logger.info(
        f"⏱️ Online diarization complete in {time.time() - t_start:.2f}s | "
        f"Speakers: {len(speaker_counts)} | Distribution: {speaker_counts}"
    )

    return result


# Keep for backward compatibility, but deprecated
def online_diarize(
    chunks: List[Tuple[np.ndarray, float, float]],
    speaker_similarity: float = 0.3,
    speaker_max_n: int = 10,
) -> Dict[int, int]:
    """
    DEPRECATED: Use run_online_diarization() instead.

    This function exists for backward compatibility.
    """
    audio_chunks = [chunk[0] for chunk in chunks]
    return run_online_diarization(audio_chunks, speaker_similarity, speaker_max_n)


# Keep these for backward compatibility but they're no longer used in main flow
def process_chunk_incremental(
    audio_chunk: np.ndarray,
    diarization_cluster,
) -> int:
    """DEPRECATED: Use run_online_diarization() instead."""
    from app.clustering_torch import StreamingKMeansMaxClusterTorch

    model = get_speaker_model()
    with torch.no_grad():
        embedding = model([audio_chunk])
        if isinstance(embedding, tuple):
            embedding = embedding[1][
                0
            ]  # Get embedding tensor from (logits, embeddings)

    speaker_id = diarization_cluster.streaming(embedding)

    return speaker_id


def process_chunks_batch_incremental(
    audio_chunks: List[np.ndarray],
    diarization_cluster,
    batch_size: int = SPEAKER_EMBEDDING_BATCH_SIZE,
) -> List[int]:
    """DEPRECATED: Use run_online_diarization() instead."""
    from app.clustering_torch import StreamingKMeansMaxClusterTorch

    model = get_speaker_model()
    speaker_ids = []

    for i in range(0, len(audio_chunks), batch_size):
        batch = audio_chunks[i : i + batch_size]
        with torch.no_grad():
            batch_embeddings = model(batch)
            if isinstance(batch_embeddings, tuple):
                batch_embeddings = batch_embeddings[1]  # Get embedding tensors
            if isinstance(batch_embeddings, tuple):
                batch_embeddings = batch_embeddings[1]  # Get embedding tensors

        for embedding in batch_embeddings:
            speaker_id = diarization_cluster.streaming(embedding)
            speaker_ids.append(speaker_id)

    return speaker_ids
