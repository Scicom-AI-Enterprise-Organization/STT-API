"""Tests for online diarization module."""

import pytest
import sys
import numpy as np
from unittest.mock import patch, MagicMock


class TestOnlineDiarization:
    """Test online diarization functions."""
    
    def test_extract_embeddings_batched_correct_count(self):
        """Verify batched extraction returns correct number of embeddings."""
        from app.diarization import extract_embeddings_batched
        
        # Mock the speaker model - return embeddings matching input batch size
        mock_model = MagicMock()
        mock_model.side_effect = lambda batch: [np.random.randn(192) for _ in range(len(batch))]
        
        with patch('app.diarization.get_speaker_model', return_value=mock_model):
            # 20 chunks with batch_size=8 should result in 3 batches (8+8+4)
            chunks = [np.random.randn(16000) for _ in range(20)]
            embeddings = extract_embeddings_batched(chunks, batch_size=8)
            
            assert len(embeddings) == 20
            assert mock_model.call_count == 3  # ceil(20/8) = 3 batches
    
    def test_online_diarize_returns_speaker_assignments(self):
        """Verify online_diarize returns valid speaker assignments."""
        chunks = [
            (np.random.randn(16000).astype(np.float32), 0.0, 1.0),
            (np.random.randn(16000).astype(np.float32), 1.0, 2.0),
            (np.random.randn(16000).astype(np.float32), 2.0, 3.0),
        ]
        
        # Create mock for StreamingKMeansMaxCluster
        mock_clustering_instance = MagicMock()
        mock_clustering_class = MagicMock(return_value=mock_clustering_instance)
        
        # Create mock module hierarchy
        mock_clustering_module = MagicMock()
        mock_clustering_module.StreamingKMeansMaxCluster = mock_clustering_class
        
        mock_model_module = MagicMock()
        mock_model_module.clustering = mock_clustering_module
        
        mock_malaya = MagicMock()
        mock_malaya.model = mock_model_module
        # Return speaker IDs: 0, 1, 0
        mock_malaya.diarization.streaming.side_effect = [0, 1, 0]
        
        mock_modules = {
            'malaya_speech': mock_malaya,
            'malaya_speech.model': mock_model_module,
            'malaya_speech.model.clustering': mock_clustering_module,
        }
        
        # Remove cached module to force re-import with mocks
        if 'app.diarization' in sys.modules:
            del sys.modules['app.diarization']
        
        with patch.dict('sys.modules', mock_modules):
            with patch('app.diarization.extract_embeddings_batched') as mock_extract:
                mock_extract.return_value = [np.random.randn(192) for _ in range(3)]
                
                from app.diarization import online_diarize
                result = online_diarize(chunks, speaker_similarity=0.75, speaker_max_n=10)
                
                # Verify we got assignments for all chunks
                assert len(result) == 3
                assert all(isinstance(v, int) for v in result.values())
    
    def test_online_diarize_empty_chunks(self):
        """Verify online_diarize handles empty input."""
        # Remove cached module that might have malaya_speech dependency
        if 'app.diarization' in sys.modules:
            del sys.modules['app.diarization']
        
        from app.diarization import online_diarize
        result = online_diarize([], speaker_similarity=0.75, speaker_max_n=10)
        assert result == {}
