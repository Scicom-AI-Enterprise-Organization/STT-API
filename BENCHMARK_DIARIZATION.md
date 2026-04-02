cat > BENCHMARK_DIARIZATION.md << 'EOF'
# Benchmark Diarization

Closes #4

## Overview

This report benchmarks the diarization component of STT-API against two standard speaker diarization datasets:

- [diarizers-community/ami](https://huggingface.co/datasets/diarizers-community/ami)
- [diarizers-community/voxconverse](https://huggingface.co/datasets/diarizers-community/voxconverse)

Evaluation was performed using the `Test` class from [huggingface/diarizers](https://github.com/huggingface/diarizers/blob/main/src/diarizers/test.py), which computes standard speaker diarization metrics: DER, false alarm, missed detection, and speaker confusion.

---

## Datasets

| Dataset | Description |
|---|---|
| [AMI](https://huggingface.co/datasets/diarizers-community/ami) | Meeting recordings with multiple speakers, challenging overlapping speech |
| [VoxConverse](https://huggingface.co/datasets/diarizers-community/voxconverse) | Multispeaker audio dataset derived from YouTube videos |

---

## Metrics

| Metric | Description |
|---|---|
| **DER** | Diarization Error Rate — overall error (lower is better) |
| **False Alarm** | Speech detected where there is none |
| **Missed Detection** | Speech not detected where there is |
| **Confusion** | Speech attributed to the wrong speaker |

---

## Results — Baseline Segmentation Model (diarizers Test class)

Evaluated using the `Test` class from diarizers directly against the pyannote segmentation model.

| Dataset | DER | False Alarm | Missed Detection | Confusion |
|---|---|---|---|---|
| AMI | 17.93% | 4.03% | 10.04% | 3.86% |
| VoxConverse | 11.20% | 4.32% | 3.52% | 3.36% |

---

## Results — STT-API Diarization Benchmark (AMI)

Benchmarked the STT-API's online (TitaNet + StreamingKMeans/BIRCH) and offline (pyannote) diarization modes across different `speaker_similarity` thresholds on the AMI dataset.

Total benchmark time: **4.38 hours**

| Algorithm | Similarity | AMI DER (%) |
|---|---|---|
| online | 0.2 | 87.35 |
| online | 0.3 | 75.74 |
| online | 0.4 | 67.02 |
| online | 0.5 | 64.00 |
| **online** | **0.6** | **62.42** ✅ best |
| online | 0.7 | 68.83 |
| online | 0.8 | 70.32 |
| offline | 0.2 | 80.75 |
| offline | 0.3 | 81.26 |
| offline | 0.4 | 81.72 |
| offline | 0.5 | 80.50 |
| offline | 0.6 | 81.01 |
| offline | 0.7 | 81.25 |
| offline | 0.8 | 81.17 |

---

## Key Findings

- **Best configuration: online mode with `speaker_similarity=0.6`**, achieving the lowest DER of **62.42%** on AMI.
- Online mode (TitaNet + StreamingKMeans/BIRCH) consistently outperforms offline mode (pyannote pipeline) on AMI across all similarity thresholds.
- Online DER improves as similarity increases from 0.2 → 0.6, then degrades beyond 0.6, suggesting 0.6 is the optimal threshold.
- Offline mode shows relatively flat DER (~80–82%) regardless of similarity threshold, indicating it is less sensitive to this parameter.
- VoxConverse baseline DER (11.20%) is significantly lower than AMI (17.93%), consistent with VoxConverse being a less overlapping, cleaner dataset.

---

## Recommendation

Use **online diarization mode with `speaker_similarity=0.6`** for best performance. This can be set via the API parameter:
```bash
curl -X POST "http://localhost:9091/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "response_format=verbose_json" \
  -F "diarization=kmeans" \
  -F "speaker_similarity=0.6"
```

---

## Environment

| Component | Details |
|---|---|
| Evaluation framework | [huggingface/diarizers test.py](https://github.com/huggingface/diarizers/blob/main/src/diarizers/test.py) |
| Online diarization | TitaNet Large + StreamingKMeans / BIRCH |
| Offline diarization | pyannote/speaker-diarization-3.1 |
| AMI split | test |
| VoxConverse split | test |
EOF
