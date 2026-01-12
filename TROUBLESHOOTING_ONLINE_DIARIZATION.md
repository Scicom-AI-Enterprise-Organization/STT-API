# Online Diarization Troubleshooting

## Issue 1: Diarization Failing with Tensor Error (FIXED)

### Error
```
ERROR:app.main:Diarization failed: normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result in torch.std() returning nan. Make sure your audio length has enough samples for a single feature (ex. at least `hop_length` for Mel Spectrograms).
```

### Root Cause
Some VAD chunks are too short for the TitaNet speaker embedding model. The model requires a minimum audio length to compute Mel spectrograms.

### Solution
Filter out chunks that are too short before extracting embeddings. Added `MIN_CHUNK_SAMPLES = 8000` (0.5 seconds at 16kHz).

### Changes Made
1. `app/diarization.py`: Added `MIN_CHUNK_SAMPLES` constant and filtering in `online_diarize()`
2. Filter chunks shorter than minimum before embedding extraction
3. Map filtered indices back to original indices for speaker assignments
4. Skipped chunks inherit speaker from nearest valid chunk

---

## Issue 2: All Segments Assigned to Same Speaker (INVESTIGATING)

### Symptom
After fixing Issue 1, all 224 chunks are assigned to speaker 0, even though offline diarization correctly identifies 2 speakers (173 vs 163 segments).

### Possible Causes
1. **speaker_similarity threshold too high** (default 0.75) - embeddings may be close enough that all pass the similarity threshold
2. **StreamingKMeansMaxCluster behavior** - may need tuning for this audio type
3. **Embedding quality** - TitaNet embeddings may not be discriminative enough for this audio

### Investigation Steps
1. Try lower speaker_similarity (e.g., 0.5 or 0.6)
2. Add debug logging for embedding distances
3. Compare embedding variance between speakers

### Testing
```bash
# Test with lower similarity threshold
curl -X POST "http://localhost:9091/audio/transcriptions" \
  -F "file=@audios/yousaf-kerolming.mp3" \
  -F "language=ms" \
  -F "response_format=verbose_json" \
  -F "diarization=online" \
  -F "speaker_similarity=0.5"
```

Check that:
1. Response includes `speaker` field in segments
2. Multiple speaker IDs are present
3. Speaker distribution is reasonable

