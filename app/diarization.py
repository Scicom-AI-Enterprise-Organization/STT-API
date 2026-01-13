"""
Online speaker diarization using TitaNet + StreamingKMeans.

Key improvement over Mesolitica: Batched GPU inference for speaker embeddings.
Reference: https://github.com/huseinzol05/backup-mesolitica-api/blob/master/app/main.py#L849
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

SPEAKER_EMBEDDING_BATCH_SIZE = int(os.environ.get("SPEAKER_EMBEDDING_BATCH_SIZE", "16"))
SPEAKER_PRECISION_MODE = os.environ.get("SPEAKER_PRECISION_MODE", "FP32")

# Minimum audio samples required for TitaNet embedding extraction
# TitaNet uses hop_length=160 at 16kHz, needs multiple frames for mel spectrogram
# 0.5 seconds = 8000 samples at 16kHz is a safe minimum
MIN_CHUNK_SAMPLES = int(os.environ.get("MIN_CHUNK_SAMPLES_FOR_EMBEDDING", "8000"))

_speaker_model = None

def load_speaker_model():
    """
    Load TitaNet Large model for speaker embedding extraction.
    Called once at application startup.
    
    Reference: backup-mesolitica-api/app/main.py lines 140-144
    
    Precision modes supported by malaya-speech: 'FP16', 'FP32', 'BFLOAT16', 'FP64'
    FP16 provides ~50% memory reduction and faster inference on compatible GPUs.
    """
    global _speaker_model
    if _speaker_model is not None:
        return _speaker_model

    import malaya_speech

    logger.info(f"Loading TitaNet Large speaker embedding model (precision={SPEAKER_PRECISION_MODE})...")
    _speaker_model = malaya_speech.speaker_vector.nemo(
        model='huseinzol05/nemo-titanet_large',
        precision_mode=SPEAKER_PRECISION_MODE
    )
    _ = _speaker_model.eval()
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
) -> List[np.ndarray]:
    """
    Extract speaker embeddings for multiple audio chunks using batched gpu inference.
    
    Meso: speaker_v([sample_wav])[0] - one chunk at a time

    Args:
        audio_chunks: List of audio numpy arrays (float32, 16kHz)
        batch_size: Number of chunks to process in one gpu call

    Returns:
        List of embedding vectors (one per chunk) 
    """
    model = get_speaker_model()
    embeddings = []

    total_chunks = len(audio_chunks)
    logger.debug(f"Extracting embeddings for {total_chunks} chunks (batch_size={batch_size})")

    for i in range(0, total_chunks, batch_size):
        batch = audio_chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size

        logger.debug(f"Processing embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)")

        # TitaNet handles variable-length audio via internal padding
        # This single call processes all chunks in the batch on GPU 

        batch_embeddings = model(batch)
        embeddings.extend(batch_embeddings)

    return embeddings


def online_diarize(
    chunks: List[Tuple[np.ndarray, float, float]],
    speaker_similarity: float = 0.5,
    speaker_max_n: int = 10,
) -> Dict[int, int]:
    """
    Perform online speaker diarization on VAD Chunks.

    Uses StreamkingKMeansMaxCluster for incremental speaker assignment.

    Reference: backup-mesolitica-api/app/main.py lines 849-852, 554-557

    Args:
        chunks: List of (audio_data, start_ts, end_ts) from VAD
        speaker_similarity: Cosine similarity threshold for same speaker (0.0-1.0)
                           Higher = stricter, fewer speakers
                           Lower = looser, more speakers
                           Default: 0.5 (aligned with Mesolitica reference implementation)
        speaker_max_n: Maximum number of speakers to detect
        
    Returns:
        Dictionary mapping chunk_index -> speaker_id (0-indexed)
    """
    if not chunks:
        return {}

    from malaya_speech.model.clustering import StreamingKMeansMaxCluster
    import malaya_speech

    # 1. Filter out chunks that are too short for embedding extraction
    # TitaNet requires minimum audio length for mel spectrogram computation
    valid_indices = []
    valid_audio = []
    skipped_count = 0
    
    for idx, chunk in enumerate(chunks):
        audio = chunk[0]
        if len(audio) >= MIN_CHUNK_SAMPLES:
            valid_indices.append(idx)
            valid_audio.append(audio)
        else:
            skipped_count += 1
            logger.debug(f"Chunk {idx} too short for embedding ({len(audio)} samples < {MIN_CHUNK_SAMPLES})")
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} chunks too short for speaker embedding (min={MIN_CHUNK_SAMPLES} samples)")
    
    if not valid_audio:
        logger.warning("No chunks long enough for speaker embedding extraction")
        # Return all chunks assigned to speaker 0
        return {idx: 0 for idx in range(len(chunks))}

    # 2. batch extract embeddings for valid chunks only
    logger.info(f"Extracting speaker embeddings for {len(valid_audio)} chunks...")
    embeddings = extract_embeddings_batched(valid_audio)

    # 3. streaming clustering
    logger.info(f"Running StreamingKMeans (similarity={speaker_similarity}, max_n={speaker_max_n})...")
    clustering = StreamingKMeansMaxCluster(
        threshold=speaker_similarity,
        max_clusters=speaker_max_n
    )

    # Map valid indices back to original chunk indices
    valid_speaker_assignments = {}
    seen_speakers = set()
    cluster_creation_log = []
    
    for i, embedding in enumerate(embeddings):
        # malaya_speech.diarization.streaming() assigns speaker ID
        # it returns strings like "speaker 0", "speaker 1", etc.
        speaker_label = malaya_speech.diarization.streaming(embedding, clustering)
        # Convert "speaker 0" -> 0, "speaker 1" -> 1, etc.
        try:
            speaker_id = int(speaker_label.replace("speaker ", ""))
        except (ValueError, AttributeError):
            speaker_id = 0
        
        # Track cluster creation vs assignment
        is_new_cluster = speaker_id not in seen_speakers
        if is_new_cluster:
            seen_speakers.add(speaker_id)
            cluster_creation_log.append((i, speaker_id))
            logger.debug(f"Chunk {i}: Created new cluster (speaker {speaker_id})")
        else:
            logger.debug(f"Chunk {i}: Assigned to existing cluster (speaker {speaker_id})")
        
        original_idx = valid_indices[i]
        valid_speaker_assignments[original_idx] = speaker_id
    
    # Log cluster creation summary
    logger.info(f"Clustering complete: {len(seen_speakers)} clusters created from {len(embeddings)} embeddings")
    if cluster_creation_log:
        logger.debug(f"Cluster creation sequence: {cluster_creation_log[:10]}{'...' if len(cluster_creation_log) > 10 else ''}")

    # For skipped chunks, assign based on nearest valid chunk's speaker
    speaker_assignments = {}
    all_indices = list(range(len(chunks)))
    
    for idx in all_indices:
        if idx in valid_speaker_assignments:
            speaker_assignments[idx] = valid_speaker_assignments[idx]
        else:
            # Find nearest valid chunk and use its speaker
            nearest_speaker = 0
            min_distance = float('inf')
            for valid_idx, speaker_id in valid_speaker_assignments.items():
                distance = abs(idx - valid_idx)
                if distance < min_distance:
                    min_distance = distance
                    nearest_speaker = speaker_id
            speaker_assignments[idx] = nearest_speaker

    speaker_counts = {}
    for speaker_id in speaker_assignments.values():
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
    
    # Log final speaker distribution
    logger.info(f"Speaker distribution: {speaker_counts}")
    
    # Validation: Check for diverse speaker assignments
    unique_speakers = len(speaker_counts)
    total_chunks = len(speaker_assignments)
    
    if unique_speakers == 1:
        logger.warning(
            f"All {total_chunks} chunks assigned to single speaker (speaker {list(speaker_counts.keys())[0]}). "
            f"This may indicate: (1) threshold too high (current={speaker_similarity}), "
            f"(2) audio contains only one speaker, or (3) embeddings not discriminative enough. "
            f"Consider lowering speaker_similarity threshold (e.g., 0.3-0.4) or checking audio quality."
        )
    elif unique_speakers == 0:
        logger.error("No speaker assignments made - this should not happen!")
    else:
        # Log distribution statistics
        max_count = max(speaker_counts.values())
        min_count = min(speaker_counts.values())
        max_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
        min_speaker = min(speaker_counts.items(), key=lambda x: x[1])[0]
        
        logger.info(
            f"Speaker diversity: {unique_speakers} speakers detected. "
            f"Largest cluster: speaker {max_speaker} ({max_count} chunks, {max_count/total_chunks*100:.1f}%), "
            f"Smallest cluster: speaker {min_speaker} ({min_count} chunks, {min_count/total_chunks*100:.1f}%)"
        )
        
        # Warn if distribution is very imbalanced (>90% in one cluster)
        if max_count / total_chunks > 0.9:
            logger.warning(
                f"Highly imbalanced speaker distribution: {max_count/total_chunks*100:.1f}% assigned to speaker {max_speaker}. "
                f"Consider adjusting speaker_similarity threshold (current={speaker_similarity})."
            )

    return speaker_assignments