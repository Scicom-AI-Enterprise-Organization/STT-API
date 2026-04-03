"""
Diarization Benchmark Script (Fixed)
Evaluates STT-API diarization (kmeans/birch) on AMI dataset from HuggingFace.
Metrics: DER (Diarization Error Rate) using pyannote.metrics

Usage:
1. Start the STT-API server first:
   STT_API_URL=https://stt-engine-tm-l40.aies.scicom.dev uvicorn stt_api.main:app --host 0.0.0.0 --port 9091

2. Run this benchmark:
   python3.10 benchmark_diarization.py

Requirements:
   pip install pyannote.metrics soundfile "datasets==2.21.0" aiohttp onnxruntime
"""

import os
os.environ["AUDIO_BACKEND"] = "soundfile"

import asyncio
import aiohttp
import time
import json
import tempfile
import numpy as np
import soundfile as sf
from datasets import load_dataset
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_URL = "http://localhost:9091"

METHODS = ["kmeans", "birch"]
SIMILARITIES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

MAX_SAMPLES = 1
CHUNK_DURATION = 15

MAX_RETRIES = 2
TIMEOUT = 120
MAX_CONCURRENT_REQUESTS = 3

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_reference(sample):
    ref = Annotation()
    for s, e, spk in zip(
        sample["timestamps_start"],
        sample["timestamps_end"],
        sample["speakers"]
    ):
        ref[Segment(s, e)] = spk
    return ref


def split_audio(audio, sr):
    chunk_size = sr * CHUNK_DURATION
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunks.append((audio[i:i+chunk_size], i / sr))
    return chunks


def build_hypothesis(segments):
    hyp = Annotation()
    for seg in segments:
        if seg["end"] > seg["start"]:
            hyp[Segment(seg["start"], seg["end"])] = str(seg["speaker"])
    return hyp


# ─────────────────────────────────────────────
# REQUEST (SAFE VERSION)
# ─────────────────────────────────────────────
async def transcribe(session, chunk, sr, method, sim, offset):
    async with semaphore:
        tmp_path = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # save chunk to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, chunk, sr)
                    tmp_path = tmp.name

                # prepare request
                with open(tmp_path, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field("file", f, filename="audio.wav")
                    data.add_field("response_format", "verbose_json")
                    data.add_field("language", "en")
                    data.add_field("diarization", method)
                    data.add_field("speaker_similarity", str(sim))

                    async with session.post(
                        f"{API_URL}/audio/transcriptions",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT)
                    ) as resp:

                        if resp.status == 200:
                            res = await resp.json()
                            segs = res.get("segments", [])
                            
                            if not segs:
                                # empty result is treated as failure
                                print(f"[ERROR] Empty result ({method}, sim={sim}, attempt {attempt})")
                                continue

                            # adjust timestamps
                            for s in segs:
                                s["start"] += offset
                                s["end"] += offset

                            return segs, False  # success

                        else:
                            print(f"[ERROR] HTTP {resp.status} ({method}, sim={sim}, attempt {attempt})")

            except Exception as e:
                print(f"[ERROR] {method} sim={sim} attempt {attempt}: {e}")

            finally:
                # always remove temp file
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            await asyncio.sleep(1)  # small delay before retry

        # after retries, return failure
        return [], True


# ─────────────────────────────────────────────
# BENCHMARK ONE SAMPLE
# ─────────────────────────────────────────────
async def run_sample(session, sample, idx):
    audio = np.array(sample["audio"]["array"], dtype=np.float32)
    sr = sample["audio"]["sampling_rate"]
    ref = build_reference(sample)

    chunks = split_audio(audio, sr)

    results = []

    for method in METHODS:
        for sim in SIMILARITIES:
            print(f"  Sample {idx} | {method} | sim={sim}")

            tasks = [
                transcribe(session, c, sr, method, sim, offset)
                for c, offset in chunks
            ]

            outputs = await asyncio.gather(*tasks)

            segments = []
            fails = 0

            for segs, failed in outputs:
                if failed:
                    fails += 1
                segments.extend(segs)

            fail_ratio = fails / len(chunks)
            print(f"    → failed chunks: {fails}/{len(chunks)} ({fail_ratio:.2f})")

            hyp = build_hypothesis(segments)

            results.append({
                "method": method,
                "similarity": sim,
                "reference": ref,
                "hypothesis": hyp,
                "fail_ratio": fail_ratio
            })

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def main():
    print("Loading dataset...")
    dataset = load_dataset("diarizers-community/ami", "ihm", split="test")
    samples = dataset.select(range(MAX_SAMPLES))

    all_results = []
    start = time.time()

    async with aiohttp.ClientSession() as session:
        for i, s in enumerate(samples):
            print(f"\n[Sample {i+1}/{len(samples)}]")
            r = await run_sample(session, s, i)
            all_results.extend(r)

    # ─────────────────────────
    # DER computation
    # ─────────────────────────
    print("\n" + "="*60)
    print("COMPUTING DER")
    print("="*60)

    summary = []

    for r in all_results:
        metric = DiarizationErrorRate()
        metric(r["reference"], r["hypothesis"])

        summary.append({
            "method": r["method"],
            "similarity": r["similarity"],
            "der": round(abs(metric) * 100, 2),
            "fail_ratio": r["fail_ratio"]
        })

    # sort by DER first, then fail rate
    summary.sort(key=lambda x: (x["der"], x["fail_ratio"]))

    # ─────────────────────────
    # PRINT TABLE
    # ─────────────────────────
    print("\nRESULTS:")
    print(f"{'Method':<10} {'Sim':<6} {'DER%':<8} {'Fail%'}")
    print("-"*40)

    for row in summary:
        print(f"{row['method']:<10} {row['similarity']:<6} {row['der']:<8} {round(row['fail_ratio']*100,2)}")

    # ─────────────────────────
    # WINNER
    # ─────────────────────────
    best = summary[0]

    print("\n" + "="*60)
    print(f"[WINNER] method: {best['method']} | best threshold: {best['similarity']}")
    print(f"         DER: {best['der']}% | fail rate: {round(best['fail_ratio']*100,2)}%")
    print("="*60)

    print(f"\nTime: {(time.time()-start)/60:.2f} min")


if __name__ == "__main__":
    asyncio.run(main())