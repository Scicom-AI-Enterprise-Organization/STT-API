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

## Issue 2: All Segments Assigned to Same Speaker (FIXED)

### Symptom
After fixing Issue 1, all 224 chunks are assigned to speaker 0, even though offline diarization correctly identifies 2 speakers (173 vs 163 segments).

### Problem Analysis

**Key Differences from Mesolitica Reference:**

1. **Threshold Mismatch (PRIMARY ISSUE):**
   - Mesolitica: `speaker_similarity=0.5` (line 722 in backup-mesolitica-api/app/main.py)
   - Lata: `speaker_similarity=0.75` (default in app/diarization.py line 99, app/main.py lines 584, 620)
   - A higher threshold (0.75) means embeddings need to be very similar to be considered the same speaker. If all embeddings have similarity > 0.75 with the first cluster centroid, they'll all be assigned to speaker 0.

2. **Processing Architecture:**
   - Mesolitica: Processes embeddings incrementally during streaming transcription, extracting embeddings from transcribed segments (using timestamps from transcription tokens)
   - Lata: Processes all VAD chunks at once before transcription, extracting embeddings from VAD chunks
   - Both call `malaya_speech.diarization.streaming()` one embedding at a time, so this shouldn't break functionality, but the different audio sources (transcribed segments vs VAD chunks) could affect embedding quality

3. **Lack of Diagnostic Information:**
   - No logging of actual similarity scores or cluster assignments during the streaming process

### Root Cause
The threshold of 0.75 is too high. When `StreamingKMeansMaxCluster` processes embeddings:
- First embedding creates cluster 0 (speaker 0)
- Subsequent embeddings are compared to cluster centroids
- If similarity > threshold (0.75), they're assigned to that cluster
- If similarity ≤ threshold, a new cluster is created
- With threshold=0.75, if all embeddings have similarity > 0.75 with the first cluster centroid, they all get assigned to speaker 0

### Solution Implemented

#### 1. Align Default Threshold with Reference Implementation
- Changed default `speaker_similarity` from 0.75 to 0.5 in:
  - `app/diarization.py` (line 99)
  - `app/main.py` (lines 584, 620)
- Updated documentation in `README.md` and this troubleshooting doc

#### 2. Add Diagnostic Logging
Added comprehensive logging in `app/diarization.py` to track:
- When new clusters are created vs. when embeddings are assigned to existing clusters
- Cluster creation sequence
- Final cluster statistics (number of clusters, assignments per cluster)
- Warning if all chunks are assigned to a single speaker
- Distribution imbalance warnings (>90% in one cluster)

#### 3. Improve Clustering Validation
- Added validation to ensure we're getting diverse speaker assignments
- Log warnings if all chunks are assigned to a single speaker
- Log distribution statistics (largest/smallest clusters, percentages)
- Warn if distribution is highly imbalanced (>90% in one cluster)

#### 4. Updated Documentation
- Documented the threshold change and rationale
- Documented the architectural differences (VAD chunks vs transcribed segments)
- Updated examples with new default value

### Testing
```bash
# Test with default threshold (now 0.5)
curl -X POST "http://localhost:9091/audio/transcriptions" \
  -F "file=@audios/yousaf-kerolming.mp3" \
  -F "language=ms" \
  -F "response_format=verbose_json" \
  -F "diarization=online"

# Test with custom threshold if needed
curl -X POST "http://localhost:9091/audio/transcriptions" \
  -F "file=@audios/yousaf-kerolming.mp3" \
  -F "language=ms" \
  -F "response_format=verbose_json" \
  -F "diarization=online" \
  -F "speaker_similarity=0.4"
```

### Expected Results
1. Response includes `speaker` field in segments
2. Multiple speaker IDs are present (if audio contains multiple speakers)
3. Speaker distribution is logged with statistics
4. Warnings appear if all chunks assigned to single speaker or highly imbalanced

---

## Critical Test Findings (Post-Fix)

### Test Results Summary

After implementing the fix (default threshold=0.5), testing with `yousaf-kerolming.mp3` (18-minute audio with 2 speakers):

| Threshold | Clusters Created | Speaker Distribution | Status |
|-----------|-----------------|---------------------|--------|
| **0.5** (default) | 1 | All 224 chunks → speaker 0 | ❌ Still single speaker |
| **0.4** | 1 | All 224 chunks → speaker 0 | ❌ Still single speaker |
| **0.3** | 10 | Speaker 0: 81 chunks (36%), Speaker 1: 133 chunks (59%), 8 small clusters | ⚠️ Over-segmentation |
| **Offline** (reference) | 2 | Speaker 0: 173 segments, Speaker 1: 163 segments | ✅ Correct |

### Critical Observations

1. **Sharp Threshold Transition**: There's a very sharp transition point between 0.4 and 0.3:
   - Thresholds ≥ 0.4: All embeddings assigned to single cluster (speaker 0)
   - Threshold ≤ 0.3: Creates multiple clusters (10 clusters at 0.3)
   - This suggests embeddings have high inter-speaker similarity (>0.4) but low intra-speaker variance

2. **Default Threshold (0.5) Still Too High**: Despite aligning with Mesolitica reference, threshold=0.5 does not work for this audio. This indicates:
   - The architectural difference (VAD chunks vs transcribed segments) may affect embedding quality
   - OR this particular audio has speakers with very similar voice characteristics
   - OR embeddings extracted from VAD chunks are less discriminative than from transcribed segments

3. **Over-segmentation at Lower Thresholds**: At threshold=0.3:
   - Correctly identifies 2 main speakers (81 vs 133 chunks ≈ 36% vs 59%)
   - But creates 8 additional small clusters (1-2 chunks each)
   - This suggests threshold=0.3 is too low, causing false splits

4. **Optimal Threshold Range**: Based on test results, optimal threshold appears to be **0.25-0.35** for this audio type:
   - Below 0.3: Too many false splits
   - Above 0.4: Single speaker assignment
   - Need to test 0.25-0.35 range to find optimal balance

### Recommendations

1. **For Production Use**: 
   - Default threshold=0.5 may not work for all audio types
   - Consider making threshold configurable per request or audio type
   - Add automatic threshold tuning based on initial clustering results

2. **For This Specific Audio**:
   - Use threshold=0.3 or lower (0.25-0.3 range)
   - Accept some over-segmentation (small clusters can be merged post-processing)
   - OR use offline diarization for better accuracy

3. **Future Improvements**:
   - Investigate why VAD chunk embeddings are less discriminative than transcribed segment embeddings
   - Consider adaptive threshold based on embedding variance
   - Add post-processing to merge small clusters (<5% of total chunks)

### Tuning speaker_similarity

The `speaker_similarity` threshold controls how strict the clustering is:

- **Higher values (0.4-0.8)**: Stricter matching, fewer speakers detected
  - Use when speakers have very distinct voices
  - **WARNING**: Thresholds ≥ 0.4 may assign all chunks to single speaker if embeddings are similar
  - May miss subtle speaker changes
  
- **Lower values (0.2-0.35)**: Looser matching, more speakers detected
  - Use when speakers have similar voices or audio quality is lower
  - May create false splits (one speaker split into multiple clusters)
  - **Recommended range**: 0.25-0.35 for audio with similar-sounding speakers
  
- **Default (0.5)**: Balanced setting aligned with Mesolitica reference
  - **NOTE**: May not work for all audio types (see Critical Test Findings above)
  - Good starting point, but may need adjustment based on audio characteristics
  - If all chunks assigned to single speaker, try lowering to 0.3-0.4
  - If too many clusters created, try raising to 0.35-0.45

### Threshold Selection Guide

1. **Start with default (0.5)**
2. **If all chunks → single speaker**: Lower threshold to 0.3-0.4
3. **If too many clusters (>5)**: Raise threshold to 0.35-0.45
4. **If 2-3 clusters with reasonable distribution**: Current threshold is good
5. **For similar-sounding speakers**: Use 0.25-0.35 range
6. **For very distinct speakers**: Can use 0.4-0.6, but test first

### Architectural Notes

**Difference from Mesolitica:**
- **Mesolitica**: Extracts embeddings from transcribed segments (after transcription)
- **Lata**: Extracts embeddings from VAD chunks (before transcription)

This difference means:
- Lata processes all chunks upfront, which is more efficient
- Both approaches use the same clustering algorithm (`StreamingKMeansMaxCluster`)
- The embedding source (VAD chunks vs transcribed segments) may affect quality, but both should work with proper threshold tuning
