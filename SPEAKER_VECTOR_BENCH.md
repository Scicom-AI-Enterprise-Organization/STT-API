# Speaker vector benchmark

Speed of `POST /audio/speaker_vector` and the TitaNet Large path behind it, and
the two changes that made it **6x faster**.

**Hardware.** One NVIDIA H20-3e, idle. The box has 8 GPUs and 6 were at 100 %
under another tenant at the time; GPU 0 was free and pinned with
`CUDA_VISIBLE_DEVICES=0`. Latency on a contended device is not a property of the
model.

**What is measured.** `extract_embeddings_batched`, the function the endpoint
routes into, so CUDA-stream overlap and host staging are included exactly as
deployed. Model load is excluded (~2 s, once per worker).

**Precision.** Independent runs differ by a few percent; read to two significant
figures. Where a mean and a p50 disagree sharply, the p50 is the real number —
see the decode row below.

## Result

| | before | after | |
|---|---:|---:|---|
| single 3 s clip | 14.3 ms | **4.6 ms** | 3.1x |
| batch of 16 | 57.1 ms | **9.1 ms** | 6.3x |
| sustained throughput | 285 vec/s | **1760 vec/s** | **6.2x** |
| HTTP, 1 file | 21.5 ms | **11.5 ms** | 1.9x |
| HTTP, 16 files | 76.1 ms | **25.3 ms** | 3.0x |
| RTF (30 s clip) | 0.0011 | **0.00011** | 10x |

Both changes are in `stt_api/nemo_speaker_vector.py`, and both were found by
profiling rather than guessing. **The GPU was never the bottleneck** — it was
doing 8 ms of work inside a 57 ms batch.

### 1. Padding went through Python objects

`sequence_1d` called `.tolist()` on every input array, concatenated Python lists
to pad, then re-parsed the result with `np.array`. At batch 16 that is ~768,000
floats converted to Python objects and back — **50 ms per batch against 8 ms of
GPU work**. A vectorised fill into a preallocated array is a memcpy per row.

The original path is kept for non-array input; the fast path triggers only when
every element is a 1-D ndarray, which is what both callers pass. Output is
**bit-identical** (verified, max abs diff 0.0).

### 2. Pinned staging cost 40x the transfer it was optimising

`prep_batch` allocated a page-locked buffer per call and copied into it, so the
host-to-device copy could be non-blocking. Measured on this host:

| | |
|---|---:|
| H2D transfer, 16x48000 batch | 0.07 ms |
| copy *into* the pinned buffer | **10.4 ms** |
| pageable `from_numpy` + blocking H2D | **0.25 ms** |

Page-locked writes run at roughly 280 MB/s here, far below a normal memcpy, so
staging cost 40x the transfer it existed to accelerate. Pageable is **41x faster
end to end**. `non_blocking=True` on pageable memory quietly degrades to a
synchronous copy, which is fine at 0.25 ms — the stream overlap was buying back a
quarter-millisecond at a ten-millisecond price.

Pinning is right on many hosts; it was wrong on this one. `SPEAKER_PIN_MEMORY=1`
restores it — measure before assuming. Output is bit-identical either way
(verified, max abs diff 0.0).

## Batch size

3-second clips, after the optimisation.

| batch | total | per vector | vectors/s | speedup |
|------:|------:|-----------:|----------:|--------:|
| 1  | 4.6 ms | 4.64 ms | 215 | 1.0x |
| 2  | 4.8 ms | 2.41 ms | 415 | 1.9x |
| 4  | 5.3 ms | 1.33 ms | 752 | 3.5x |
| 8  | 5.8 ms | 0.72 ms | 1383 | 6.4x |
| 16 | 9.1 ms | 0.57 ms | 1760 | **8.2x** |
| 32 | 18.1 ms | 0.57 ms | 1763 | 8.2x |
| 64 | 36.3 ms | 0.57 ms | 1763 | 8.2x |

Batching now matters **more**, not less: 8.2x versus 4.1x before, because the
per-item CPU tax that used to dominate is gone. It still saturates at 16 —
`SPEAKER_EMBEDDING_BATCH_SIZE`'s default remains correct — and past that, total
latency grows linearly for no throughput gain.

## Clip duration

Batch 16.

| clip | total | per vector | RTF |
|-----:|------:|-----------:|----:|
| 0.5 s | 6.2 ms | 0.39 ms | 0.00078 |
| 1 s   | 6.2 ms | 0.39 ms | 0.00039 |
| 3 s   | 9.2 ms | 0.57 ms | 0.00019 |
| 10 s  | 19.1 ms | 1.19 ms | 0.00012 |
| 30 s  | 51.3 ms | 3.20 ms | 0.00011 |

Cost tracks audio duration, not file count: 16 half-second clips cost 6 ms, one
30-second clip costs 51 ms. RTF settles near 0.0001 — about **9000x** real time.

## Where the time goes

3-second WAV, steady state.

| stage | p50 |
|---|---:|
| tempfile write + `librosa.load` | 0.4 ms |
| GPU embed (batch 1) | 4.7 ms |
| full HTTP request | 11.5 ms |

Decode was never the bottleneck, which is worth stating because it is the obvious
suspect — the endpoint writes a temp file and decodes with librosa before
touching the GPU.

**A caution about means.** Decode's mean over 30 runs was 8.4 ms with a max of
238 ms, against a p50 of **0.4 ms**. The first call pays librosa/soundfile import
and codec warm-up. On an earlier run that single outlier made decode look like
1.4x the GPU cost, which would have sent this optimisation in entirely the wrong
direction. Read the percentile.

Container format barely matters: WAV 0.4 ms, FLAC 0.8 ms, OGG 1.3 ms at p50.

## End to end

| files | total | per vector |
|------:|------:|-----------:|
| 1  | 11.5 ms | 11.50 ms |
| 4  | 13.5 ms | 3.38 ms |
| 16 | 25.3 ms | 1.58 ms |

~7 ms sits above the raw embed for one file — multipart parsing, decode,
serialisation — and that is now the largest single component of a one-file
request.

**Caveat.** These ran with the executor inlined, so they exclude the
`ProcessPoolExecutor` hop. In production the speaker model lives in a separate
spawn process and each request pays pickling of audio in and vectors out. Treat
this column as a floor, not an SLO.

## A correctness note that predates this work

Batching changes a vector slightly when clips have **ragged lengths**, because
the model sees padding:

| batch | cosine vs computed alone |
|---|---|
| uniform lengths | 1.000 (max abs diff 3e-05, fp16 rounding) |
| ragged 0.6-4.5 s | 0.963-1.000 (max abs diff 1.5e-02) |

The worst case is the shortest clip in a batch padded to the longest. This is not
caused by the optimisation — padding output is bit-identical before and after —
but it matters for an embedding API: **a vector enrolled alone and a vector
computed inside a ragged batch are not quite the same vector.** If you store
enrollment vectors, compute them the same way you will compute the query, or
batch clips of similar length together.

## Reproducing

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_speaker_vector.py          # batch, duration, stage split
CUDA_VISIBLE_DEVICES=0 python benchmark_speaker_vector.py --http   # + the HTTP endpoint
SPEAKER_PIN_MEMORY=1 CUDA_VISIBLE_DEVICES=0 python benchmark_speaker_vector.py   # with pinning
```

Pin to an idle GPU. The noise-cancellation benchmark in
`stt_api/livekit_plugin/noise_cancellation/benchmark/` documents the same trap,
where contention moved a p99 by 8x.
