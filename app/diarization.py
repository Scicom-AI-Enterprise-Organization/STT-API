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
    speaker_similarity: float = 0.75,
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
        speaker_max_n: Maximum number of speakers to detect
        
    Returns:
        Dictionary mapping chunk_index -> speaker_id (0-indexed)
    """
    if not chunks:
        return {}

    from malaya_speech.model.clustering import StreamingKMeansMaxCluster
    import malaya_speech

    # 1. extract
    audio_data = [chunk[0] for chunk in chunks]

    # 2. batch
    logger.info(f"Extracting speaker embeddings for {len(audio_data)} chunks...")
    embeddings = extract_embeddings_batched(audio_data)

    # 3. streaming clustering
    logger.info(f"Running StreamingKMeans (similarity={speaker_similarity}, max_n={speaker_max_n})...")
    clustering = StreamingKMeansMaxCluster(
        threshold=speaker_similarity,
        max_clusters=speaker_max_n
    )

    speaker_assignments = {}
    for idx, embedding in enumerate(embeddings):
        # malaya_speech.diarization.streaming() assigns speaker ID
        # it returns strings like "speaker 0", "speaker 1", etc.
        speaker_label = malaya_speech.diarization.streaming(embedding, clustering)
        # Convert "speaker 0" -> 0, "speaker 1" -> 1, etc.
        try:
            speaker_id = int(speaker_label.replace("speaker ", ""))
        except (ValueError, AttributeError):
            speaker_id = 0
        speaker_assignments[idx] = speaker_id

    speaker_counts = {}
    for speaker_id in speaker_assignments.values():
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
    logger.info(f"Speaker distribution: {speaker_counts}")

    return speaker_assignments