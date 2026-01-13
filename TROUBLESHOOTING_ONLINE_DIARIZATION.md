# Troubleshooting: Online Diarization Issues

## Problem Summary
- **Offline mode**: Works correctly - detects number of speakers and labels correctly
- **Online mode**: Not working properly, even after tuning thresholds (0.3, 0.4, 0.5)
- **Observation**: Sharp change in behavior between threshold 0.3 and 0.4

## Potential Issues Identified

### 1. **Chunk-to-Segment Mapping Logic Flaw** (HIGH PRIORITY)
**Location**: `app/main.py:476-494` in `merge_speakers_with_segments()`

**Problem**: 
- The mapping logic only uses `start_ts` to find which chunk a segment belongs to
- It finds the last chunk where `seg_start >= chunk_start`, but doesn't:
  - Check if the segment actually falls within the chunk's time range
  - Account for chunk end times
  - Handle segments that span multiple chunks
  - Use proper overlap calculation (unlike offline mode which uses `find_speaker_for_timestamp`)

**Code Issue**:
```python
# Current logic (line 488-492):
for i, (chunk_start, idx) in enumerate(chunk_ranges):
    if seg_start >= chunk_start:
        chunk_idx = idx
    else:
        break
```
This will assign a segment to the LAST chunk it's >= to, which might not be the correct chunk if the segment starts after the chunk ends.

**Impact**: Segments may be assigned to wrong speakers because they're mapped to the wrong chunks.

---

### 2. **Missing End Timestamp Information** (HIGH PRIORITY)
**Location**: `app/main.py:476-494`, `app/main.py:718-721`, and `app/main.py:771-806`

**Problem**:
- `chunks_to_transcribe` is `List[Tuple[np.ndarray, float]]` - only contains `(wav_chunk, start_ts)`
- `diarize_chunks` is created with `(wav_chunk, start_ts, end_ts)` at line 718-721, but this is **only used for diarization** and then discarded
- `merge_speakers_with_segments()` receives `chunks_for_diarization` (which is `chunks_to_transcribe`) which doesn't have `end_ts`
- The end timestamps are calculated but then thrown away - they're never passed to the merge function
- Without end timestamps, proper overlap calculation is impossible

**Code Flow**:
```python
# Line 718-721: Create diarize_chunks with end_ts
diarize_chunks = [
    (wav_chunk, start_ts, start_ts + len(wav_chunk) / sample_rate)
    for wav_chunk, start_ts in chunks_to_transcribe
]
# ... diarize_chunks used for diarization, then discarded

# Line 771: Save chunks_to_transcribe (without end_ts!)
chunks_for_diarization = chunks_to_transcribe if diarization == "online" else None

# Line 806: Pass chunks without end_ts to merge function
chunks_to_transcribe=chunks_for_diarization,  # Missing end_ts!
```

**Impact**: Cannot accurately determine which chunk a segment belongs to without knowing chunk boundaries. The end timestamps are calculated but wasted.

---

### 3. **Chunk Filtering Mismatch** (MEDIUM PRIORITY)
**Location**: `app/diarization.py:148-169` and `app/main.py:476-494`

**Problem**:
- `online_diarize()` filters out chunks shorter than `MIN_CHUNK_SAMPLES` (line 156)
- These filtered chunks get assigned speakers based on nearest valid chunk (line 196-212)
- However, `merge_speakers_with_segments()` doesn't know about this filtering
- It tries to map segments to ALL chunks in `chunks_to_transcribe`, including ones that were skipped during embedding extraction

**Impact**: If segments fall within filtered chunks, they might get wrong speaker assignments.

---

### 4. **Threshold Interpretation Issue** (MEDIUM PRIORITY)
**Location**: `app/diarization.py:177-180`

**Problem**:
- `StreamingKMeansMaxCluster` uses `threshold=speaker_similarity` 
- The sharp change between 0.3 and 0.4 suggests the clustering might be:
  - Too sensitive to threshold changes
  - Using distance instead of similarity (or vice versa)
  - Not properly normalized
- Need to verify if threshold is cosine similarity (0-1, higher = more similar) or distance (lower = more similar)

**Impact**: Threshold tuning might not work as expected if the interpretation is wrong.

---

### 5. **Index Mismatch After Filtering** (LOW PRIORITY)
**Location**: `app/diarization.py:150-194`

**Problem**:
- When chunks are filtered, `valid_indices` maps original indices to filtered indices
- The speaker assignments correctly map back to original indices (line 193)
- However, if the order of chunks in `chunks_to_transcribe` doesn't match the order passed to `online_diarize()`, there could be a mismatch

**Impact**: Low, but worth verifying that chunk order is consistent.

---

### 6. **No Overlap Calculation for Online Mode** (HIGH PRIORITY)
**Location**: `app/main.py:456-504`

**Problem**:
- Offline mode uses `find_speaker_for_timestamp()` which calculates overlap between segment and diarization segments
- Online mode uses a simple "find last chunk where seg_start >= chunk_start" without overlap calculation
- This is inconsistent and less accurate

**Impact**: Segments that span chunk boundaries or don't align perfectly with chunk boundaries get incorrect speaker assignments.

---

## Recommended Investigation Steps

1. **Add logging** to see:
   - How many chunks are filtered out in `online_diarize()`
   - What speaker assignments are returned
   - What chunk indices are being used in `merge_speakers_with_segments()`
   - Whether segments are being mapped to correct chunks

2. **Verify threshold interpretation**:
   - Check malaya_speech documentation for `StreamingKMeansMaxCluster`
   - Verify if threshold is similarity (higher = same speaker) or distance (lower = same speaker)
   - Test with very low (0.1) and very high (0.9) thresholds to see behavior

3. **Compare chunk timestamps**:
   - Log the timestamps of chunks passed to `online_diarize()`
   - Log the timestamps of segments from transcription
   - Verify they align correctly

4. **Fix chunk-to-segment mapping**:
   - Add end_ts to `chunks_to_transcribe` or pass `diarize_chunks` to merge function
   - Implement proper overlap calculation similar to offline mode
   - Handle segments that span multiple chunks

---

## Most Likely Root Cause

Based on the analysis, the **most likely issue** is **#1 and #6 combined**: The chunk-to-segment mapping logic is fundamentally flawed because:
1. It doesn't use end timestamps
2. It doesn't calculate overlap
3. It uses a simple "last chunk where seg_start >= chunk_start" which can assign segments to wrong chunks

This would explain why offline works (uses proper overlap calculation) but online doesn't (uses flawed mapping logic).

---

## Fixes Applied

### ✅ Fixed Issues 1, 2, 3, and 6

**Changes made:**

1. **Preserved end timestamps** (`app/main.py:704`):
   - Changed `chunks_to_transcribe` to store `(wav_chunk, start_ts, end_ts)` instead of just `(wav_chunk, start_ts)`
   - Updated all references to handle the new 3-tuple format

2. **Created overlap calculation helper** (`app/main.py:456-483`):
   - Added `find_speaker_for_chunk_timestamp()` function that calculates overlap between segments and chunks
   - Mirrors the logic used in offline mode's `find_speaker_for_timestamp()`

3. **Fixed merge function** (`app/main.py:485-520`):
   - Replaced flawed "last chunk where seg_start >= chunk_start" logic with proper overlap calculation
   - Now uses `find_speaker_for_chunk_timestamp()` to find the chunk with maximum overlap
   - Updated function documentation to reflect new chunk format

4. **Updated diarize_chunks creation** (`app/main.py:718-721`):
   - Now uses preserved `end_ts` from `chunks_to_transcribe` instead of recalculating
   - More accurate and consistent with chunk boundaries

### ⚠️ Issue 4 (Threshold Interpretation) - Needs Verification

The threshold interpretation for `StreamingKMeansMaxCluster` is still unclear. The sharp change between 0.3 and 0.4 suggests:
- Threshold might be interpreted as distance (lower = more similar) instead of similarity (higher = more similar)
- Or the clustering algorithm might have a critical threshold point around 0.35

**Recommendation**: Add logging to track:
- Number of clusters created at different thresholds
- Speaker distribution changes
- Consider testing with very low (0.1) and very high (0.9) thresholds to understand behavior

This can be addressed in a follow-up if threshold tuning is still problematic after the mapping fixes.

