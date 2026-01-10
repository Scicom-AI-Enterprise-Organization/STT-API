"""Tests for online diarization module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestOnlineDiarization:
    """Test online diarization functions."""
    
    def test_extract_embeddings_batched_correct_count(self):
        """Verify batched extraction returns correct number of embeddings."""
        from app.diarization import extract_embeddings_batched
        
        # Mock the speaker model
        mock_model = MagicMock()
        mock_model.return_value = [np.random.randn(192) for _ in range(8)]
        
        with patch('app.diarization.get_speaker_model', return_value=mock_model):
            # 20 chunks with batch_size=8 should result in 3 batches
            chunks = [np.random.randn(16000) for _ in range(20)]
            embeddings = extract_embeddings_batched(chunks, batch_size=8)
            
            assert len(embeddings) == 20
            assert mock_model.call_count == 3  # ceil(20/8) = 3 batches
    
    def test_online_diarize_returns_speaker_assignments(self):
        """Verify online_diarize returns valid speaker assignments."""
        from app.diarization import online_diarize
        
        # Create mock chunks (audio, start, end)
        chunks = [
            (np.random.randn(16000).astype(np.float32), 0.0, 1.0),
            (np.random.randn(16000).astype(np.float32), 1.0, 2.0),
            (np.random.randn(16000).astype(np.float32), 2.0, 3.0),
        ]
        
        with patch('app.diarization.extract_embeddings_batched') as mock_extract:
            mock_extract.return_value = [np.random.randn(192) for _ in range(3)]
            
            with patch('malaya_speech.diarization.streaming') as mock_streaming:
                mock_streaming.side_effect = [0, 1, 0]  # Speaker pattern
                
                result = online_diarize(chunks, speaker_similarity=0.75, speaker_max_n=10)
                
                assert result == {0: 0, 1: 1, 2: 0}
    
    def test_online_diarize_empty_chunks(self):
        """Verify online_diarize handles empty input."""
        from app.diarization import online_diarize
        
        result = online_diarize([], speaker_similarity=0.75, speaker_max_n=10)
        assert result == {}