"""
Speaker vector benchmark: TitaNet Large throughput, latency and stage split.

Drives `extract_embeddings_batched` — the function `POST /audio/speaker_vector`
routes into — rather than a hand-rolled forward pass, so CUDA-stream overlap and
pinned-memory staging are included exactly as deployed. A hand-rolled version
would look faster than production and tell you nothing useful.

Results and interpretation: SPEAKER_VECTOR_BENCH.md

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_speaker_vector.py
    CUDA_VISIBLE_DEVICES=0 python benchmark_speaker_vector.py --http

**Pin it to an idle GPU.** Latency on a contended device is not a property of the
model; the noise-cancellation benchmark in
`stt_api/livekit_plugin/noise_cancellation/benchmark/` documents the same trap,
where contention moved a p99 by 8x.

The package must be importable as `app` (docker-compose mounts ./stt_api as
/app/app). From a checkout:

    PYTHONPATH=. python -c "import stt_api, sys; sys.modules['app'] = stt_api"

or simply run this from a tree where `app/` exists.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time

import numpy as np

SR = 16000


def _ensure_app_package() -> None:
    """Make `app.*` resolve from a `stt_api/` checkout."""
    try:
        import app  # noqa: F401
    except ImportError:
        import stt_api

        sys.modules.setdefault("app", stt_api)


def stat_ms(seconds_list) -> str:
    a = np.asarray(seconds_list) * 1000.0
    return (
        f"mean {a.mean():7.1f}  p50 {np.percentile(a, 50):7.1f}  "
        f"p95 {np.percentile(a, 95):7.1f}  max {a.max():7.1f}"
    )


def noise_chunks(n: int, seconds: float, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [(0.05 * rng.standard_normal(int(seconds * SR))).astype(np.float32) for _ in range(n)]


def wav_bytes(seconds: float, fmt: str = "WAV") -> bytes:
    import soundfile as sf

    t = np.arange(int(seconds * SR)) / SR
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), SR, format=fmt)
    return buf.getvalue()


def timed(fn, repeats: int = 10, warmup: int = 3) -> np.ndarray:
    """Wall-clock per call, after warmup, with the GPU synchronised.

    Warmup is not politeness: the first call pays kernel autotuning and lazy
    imports, and on a 10-run mean that single outlier can dominate. The decode
    stage below shows the same effect at 20.4 ms mean against 0.4 ms p50.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return np.asarray(out)


def bench_batches(extract) -> None:
    print("\n=== latency vs batch size (3s clips) ===")
    print(f"{'batch':>6} {'total ms':>10} {'per-vec ms':>11} {'vec/s':>9} {'speedup':>8}")
    base = None
    for b in (1, 2, 4, 8, 16, 32, 64):
        data = noise_chunks(b, 3.0)
        ts = timed(lambda: extract(data))
        per = ts.mean() * 1000 / b
        base = base or per
        print(f"{b:>6} {ts.mean() * 1000:>10.1f} {per:>11.2f} {1000 / per:>9.0f} {base / per:>7.1f}x")


def bench_durations(extract) -> None:
    print("\n=== clip duration at batch=16 ===")
    print(f"{'seconds':>8} {'total ms':>10} {'per-vec ms':>11} {'RTF':>9}")
    for secs in (0.5, 1.0, 3.0, 10.0, 30.0):
        data = noise_chunks(16, secs)
        ts = timed(lambda: extract(data), repeats=5)
        per = ts.mean() * 1000 / 16
        print(f"{secs:>8} {ts.mean() * 1000:>10.1f} {per:>11.2f} {per / 1000 / secs:>9.5f}")


def bench_stages(extract) -> None:
    import librosa

    print("\n=== stage costs, 3s clip ===")
    blob = wav_bytes(3.0)
    ts_dec = []
    for _ in range(30):
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
            f.write(blob)
            path = f.name
        audio, _ = librosa.load(path, sr=SR, mono=True)
        os.unlink(path)
        ts_dec.append(time.perf_counter() - t0)
    print(f"  decode (tempfile+librosa) : {stat_ms(ts_dec)}")
    print("    ^ read p50, not mean: the first call pays codec warm-up")

    ts_emb = timed(lambda: extract([audio.astype(np.float32)]), repeats=30)
    print(f"  embed (batch=1)           : {stat_ms(ts_emb)}")

    print("\n=== decode cost by container format (3s) ===")
    for fmt in ("WAV", "FLAC", "OGG"):
        try:
            b = wav_bytes(3.0, fmt)
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
                    f.write(b)
                    path = f.name
                librosa.load(path, sr=SR, mono=True)
                os.unlink(path)
                ts.append(time.perf_counter() - t0)
            print(f"  {fmt:5}: {stat_ms(ts)}")
        except Exception as e:  # noqa: BLE001 - a missing codec is not a failure
            print(f"  {fmt:5}: unsupported ({str(e)[:50]})")


def bench_http() -> None:
    """
    Full request through a TestClient against the real model.

    The executor is inlined, so this EXCLUDES the ProcessPoolExecutor hop that
    production pays (pickling audio in, vectors out). A floor, not an SLO.
    """
    import concurrent.futures as cf

    class _Inline(cf.Executor):
        def submit(self, fn, *a, **k):
            fut: cf.Future = cf.Future()
            try:
                fut.set_result(fn(*a, **k))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)
            return fut

    real_pool = cf.ProcessPoolExecutor
    cf.ProcessPoolExecutor = lambda *a, **k: _Inline()
    try:
        import app.main as main
    finally:
        cf.ProcessPoolExecutor = real_pool
    main.get_vad_executor = lambda: _Inline()
    main.get_diarization_executor = lambda: _Inline()

    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    blob = wav_bytes(3.0)
    print("\n=== full HTTP request (executor inlined: a floor, not an SLO) ===")
    for n in (1, 4, 16):
        files = [("file", (f"{i}.wav", blob, "audio/wav")) for i in range(n)]
        for _ in range(3):
            client.post("/audio/speaker_vector", files=files)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = client.post("/audio/speaker_vector", files=files)
            ts.append(time.perf_counter() - t0)
            assert r.status_code == 200, r.text[:200]
        a = np.asarray(ts) * 1000
        print(f"  {n:>2} file(s): total {a.mean():7.1f} ms | per vector {a.mean() / n:6.2f} ms")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--http", action="store_true", help="also time the HTTP endpoint")
    ap.add_argument("--skip-stages", action="store_true", help="skip the decode/embed split")
    args = ap.parse_args()

    os.environ.setdefault("ENABLE_ONLINE_DIARIZATION", "false")
    _ensure_app_package()

    from app.diarization import extract_embeddings_batched, load_speaker_model

    print("loading TitaNet Large ...", flush=True)
    t0 = time.perf_counter()
    load_speaker_model()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s (excluded from every number below)")

    bench_batches(extract_embeddings_batched)
    bench_durations(extract_embeddings_batched)
    if not args.skip_stages:
        bench_stages(extract_embeddings_batched)
    if args.http:
        bench_http()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
