# Noise cancellation plugin for LiveKit Agents

Self-hosted noise cancellation. Runs a 48.2 K-parameter ONNX model inside the agent
process — no LiveKit Cloud, no licence key, no external service.

## Why this exists

`livekit-plugins-noise-cancellation` (Krisp NC / BVC / BVCTelephony) authorises
against LiveKit Cloud. On a self-hosted server it loads its native library, fails
the entitlement check, logs `noise cancellation is not authorized (404)` and passes
audio through untouched — so agents look configured but get raw audio. ai-coustics
is the only vendor path that permits self-hosting, and that needs a paid licence key.

The hook itself is not gated. `AudioInputOptions.noise_cancellation` accepts either
Krisp's `rtc.NoiseCancellationOptions` **or** any `rtc.FrameProcessor[rtc.AudioFrame]`
— a plain Python object called once per audio frame, entirely in-process. This plugin
is that second form.

Worth knowing where it pays off: a browser microphone already arrives processed,
because `livekit-client` enables WebRTC's `noiseSuppression`, `echoCancellation` and
`voiceIsolation` by default. **Inbound SIP audio gets none of that** — that is the
gap this closes, and it is also the audio your STT accuracy depends on most.

## Usage

```python
from livekit.agents.voice import room_io
from stt_api.livekit_plugin.noise_cancellation import GTCRN

await session.start(
    agent=MyAgent(),
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            # Match the model's rate: skips resampling, and it is what STT and VAD want.
            sample_rate=16000,
            # A selector, so each participant stream gets its own recurrent caches.
            noise_cancellation=lambda params: GTCRN(),
        ),
    ),
)
```

One instance per audio stream. The three recurrent caches are per-stream state, so
sharing an instance across participants would cross-contaminate them. The ONNX
session holding the weights *is* shared — cached per (model, thread count) — so a
second stream costs no extra memory for the model.

To apply it only to phone callers, where it matters most:

```python
noise_cancellation=lambda params: (
    GTCRN()
    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    else None
),
```

## Measured behaviour

3.3 s of speech, one core of an M-series laptop:

| | |
|---|---|
| CPU | 3 % of one core per stream (RTF 0.031, ~1.6 ms per 50 ms frame) |
| Added delay | **32 ms at 16 kHz**, 56 ms at 24/48 kHz, 109 ms at 8 kHz |
| Denoising | +9 to +10 dB SNR at 3–10 dB input SNR; +5.5 dB at 20 dB |
| Noise floor in pauses | −25 to −39 dB |
| Clean speech | 30 dB fidelity, level unchanged (+0.05 dB) |

Reproduce with `pytest tests/test_noise_cancellation.py -v -s`.

The 32 ms at 16 kHz is two hops: one of output priming, one because the first
analysis window is mostly zeros. Off-rate audio costs two soxr stages on top —
which is why the snippet above pins 16 kHz.

## Why GTCRN and not something else

Measured, not assumed. `benchmark/` runs each candidate frame-by-frame through
this same streaming path — GTCRN literally through `rtc.AudioFrame` and
`_process`, int16 quantisation and internal resamplers included — and scores
quality, realtime cost and downstream WER on the same corpus.

**Quality: the full 824-utterance VoiceBank+DEMAND test set**, the standard
speech-enhancement benchmark. Δ is against unprocessed input.

| model | PESQ | Δ | ESTOI | Δ | SI-SDR | Δ | BAK | OVRL | Δ |
|---|---|---|---|---|---|---|---|---|---|
| unprocessed | 1.971 | — | 0.787 | — | 8.4 | — | 3.13 | 2.70 | — |
| Wiener (classical) | 2.025 | +0.05 | 0.760 | −0.027 | 9.0 | +0.6 | 3.27 | 2.65 | −0.05 |
| RNNoise | 2.124 | +0.15 | 0.783 | −0.004 | 12.3 | +3.8 | 3.84 | 2.89 | +0.20 |
| DTLN | 2.419 | +0.45 | **0.823** | **+0.037** | **17.4** | **+9.0** | 3.81 | 2.90 | +0.20 |
| **GTCRN** | **2.535** | **+0.56** | 0.820 | +0.033 | 14.6 | +6.2 | **3.88** | **2.91** | **+0.22** |
| DeepFilterNet3 † | 2.673 | +0.70 | 0.838 | +0.051 | 15.0 | +6.5 | 4.12 | 3.20 | +0.51 |

† offline, whole-file, unlimited lookahead — a ceiling, not a deployable option.

**Speed: 300 utterances, pinned to one core of an x86-64 server, nothing else
running.** These are *not* the laptop numbers above; different silicon.

| model | RTF | p50 | p95 | **p99** | max | budget | delay |
|---|---|---|---|---|---|---|---|
| Wiener | 0.0031 | 0.14 | 0.20 | 0.25 | 1.97 | 0.5 % | 36 ms |
| DTLN | 0.0392 | 1.84 | 2.25 | **2.54** | 16.79 | 5.1 % | 34 ms |
| **GTCRN** | 0.0535 | 2.52 | 3.34 | **3.68** | 15.16 | 7.4 % | **32 ms** |
| RNNoise | 0.0603 | 2.96 | 3.10 | **3.40** | 17.51 | 6.8 % | 55 ms |

Milliseconds per 50 ms frame; `budget` is p99 against the frame it had to fit
inside. **p99 is the number that matters, not RTF** — `_process` runs inline on
the audio read loop, so a filter that is fine on average and occasionally slow
still stutters the call. At 20 ms frames the ordering is unchanged and per-frame
costs fall to roughly half these values (GTCRN p99 1.88 ms), with RTF flat — the
work scales with audio, not with how it is chopped up.

**The verdict: GTCRN stays.** It wins PESQ (+0.12 over DTLN), background removal
(BAK 3.88) and overall MOS among everything that can actually stream, at the
lowest added delay in the table.

It is not the cheapest, and it is worth being exact about that: GTCRN has the
*highest* p99 of the three neural candidates (3.68 ms vs DTLN's 2.54 ms). At
7.4 % of a 50 ms frame that is comfortable, and it buys the best quality in the
table — but if the CPU budget tightens, this is the row that gives.

Four things worth carrying forward:

- **DTLN is the real alternative, and it is not strictly worse.** It costs ~27 %
  less CPU (RTF 0.039 vs 0.054, p99 2.54 vs 3.68 ms) and clearly wins waveform
  fidelity (SI-SDR 17.4 vs 14.6) and intelligibility by a hair. If CPU per stream
  ever becomes the binding constraint, this is the swap — roughly a quarter off
  the compute for 0.12 PESQ.
- **RNNoise is the worst trade here**: the most expensive model, the highest delay
  (55 ms, half of it resampling a 48 kHz model into a 16 kHz pipeline), and
  +0.15 PESQ with *negative* ESTOI. It posts by far the deepest noise floor
  (−60.7 dB) because it over-suppresses, which is exactly the behaviour that
  wrecks intelligibility.
- **Classical DSP earns nothing.** Wiener buys +0.05 PESQ while *losing* ESTOI and
  overall MOS. So essentially all of GTCRN's gain is attributable to the model,
  not to generic spectral subtraction.
- **~0.14 PESQ and 0.29 OVRL are still on the table**, per the DeepFilterNet3
  ceiling. That is the prize for solving the streaming-export problem described
  below — real, but not large enough to justify the 48 kHz pipeline move on its
  own.

Caveats worth stating: the corpus is 16 kHz, so the 48 kHz models (RNNoise,
DeepFilterNet) cannot use the top octave they were trained on — right for a
16 kHz agent, but a floor rather than their best.

### The ranking is corpus-dependent — check before you trust it

VoiceBank+DEMAND alone would overstate how settled this is. Run across all three
corpora, the winner moves:

| corpus | n | best PESQ | best DNSMOS OVRL |
|---|---|---|---|
| VoiceBank+DEMAND | 824 | **GTCRN** 2.535 (DTLN 2.419) | **GTCRN** 2.91 |
| DNS 2020 synthetic | 150 | **DTLN** 2.360 (GTCRN 2.293) | **RNNoise** 3.16 (GTCRN 3.06) |
| DNS 2020 real recordings | 300 | *no reference* | **RNNoise** 2.77 (GTCRN 2.75, DTLN 2.73) |

So GTCRN sweeps VoiceBank, DTLN takes PESQ on DNS synthetic, and on real
recordings the three are within 0.04 OVRL of each other — a gap far smaller than
the one VoiceBank implies.

The honest reading is that **GTCRN is the most consistent, not the uniformly
best**. It is top-two on every corpus and never the weak one, which is what you
want from a default. But RNNoise, which VoiceBank ranks last among the neural
models, leads DNSMOS on both DNS sets — it over-suppresses, and additively-mixed
benchmarks punish that while a reference-free perceptual metric on real audio
rewards it.

If you change this model, re-run all three. A single-corpus win is not evidence.

## Acoustic vs generative restoration

A second class exists: **generative restorers** that encode speech into a learned
representation, clean the representation, and resynthesise a new waveform through
a vocoder — CallEnhancer (Scicom's own, Sidon-based), resemble-enhance,
voicefixer. Nothing of the original waveform survives, which is how they repair
8 kHz codec'd telephony into something that sounds like studio audio, and also
how they can put words in a caller's mouth that were never said.

Benchmarked on VoiceBank in two conditions — wideband, and `--degrade telephony`
(8 kHz + G.711 µ-law, what inbound SIP actually delivers) — 100 items, scored by
DNSMOS and by Whisper large-v3 WER on the *same* items. large-v3 because that is
what CallEnhancer's published CER table used.

### Telephony (the condition that matters for SIP)

| model | class | OVRL | Δ | WER | Δ |
|---|---|---|---|---|---|
| unprocessed | — | 2.65 | — | 3.57 % | — |
| **CallEnhancer** | generative | 2.84 | +0.19 | **2.68 %** | **−0.89 %** |
| GTCRN | acoustic | 2.92 | +0.27 | 3.82 % | +0.25 % |
| DTLN | acoustic | 2.87 | +0.22 | 4.46 % | +0.89 % |
| voicefixer | generative | 3.13 | +0.48 | 8.28 % | +4.71 % |
| resemble-enhance | generative | **3.19** | **+0.54** | 8.92 % | **+5.35 %** |

### Wideband

| model | OVRL | Δ | WER | Δ |
|---|---|---|---|---|
| unprocessed | 2.68 | — | **1.27 %** | — |
| CallEnhancer | 2.79 | +0.10 | 2.04 % | +0.76 % |
| GTCRN | 2.92 | +0.24 | 2.29 % | +1.02 % |
| DTLN | 2.91 | +0.22 | 3.06 % | +1.78 % |
| resemble-enhance | **3.21** | **+0.52** | 3.57 % | +2.29 % |
| voicefixer | 3.20 | +0.52 | 4.59 % | +3.31 % |

### The finding: perceptual quality and WER are inversely related here

Rank the telephony table by DNSMOS and you get almost exactly the reverse of its
ranking by WER. Spearman **ρ = +0.90 (p = 0.037)**, Pearson **r = +0.98** —
positive meaning *better DNSMOS goes with worse WER*.

The best-sounding model in the table (resemble-enhance, +0.54 OVRL) has the worst
WER (+5.35 %). The worst-sounding generative model (CallEnhancer, +0.19) has the
best WER, and is the only entry that beats doing nothing (−0.89 %). **Choosing
this class on perceptual quality selects almost exactly the wrong model.**

That is not a quirk of DNSMOS, it is what DNSMOS is *for*: it is reference-free,
so it judges whether the output sounds like clean speech, and a vocoder that
resynthesises confident, fluent speech satisfies that whether or not the words
are the ones that were said. Hallucination is this class's characteristic failure
and no reference-free perceptual metric can see it. **Never accept a generative
restorer on MOS alone.**

### What that means

- **CallEnhancer is the only model here that improves WER at all**, and only on
  telephony, the domain it was trained for. Everything else in this benchmark —
  acoustic and generative alike — costs accuracy.
- **CallEnhancer is a restorer, not a suppressor.** It posts the best SIG in the
  table (3.50) and the worst BAK (3.26): it repairs codec and bandwidth damage and
  leaves the noise alone. That makes it *complementary* to GTCRN rather than a
  competitor, and stacking the two is the obvious next experiment.
- **voicefixer and resemble-enhance are disqualified for transcription.** Both add
  4–5 points of word error while sounding the best in the table. They may still be
  right for archival or human listening, where nothing downstream has to be
  correct.

### Caveat: this telephony condition is simulated

`--degrade telephony` is an 8 kHz band limit plus G.711 µ-law. Real call-centre
audio also carries packet loss, AGC, G.729/AMR rather than G.711, room acoustics
and genuine background — and it is much harder than this: CallEnhancer's published
CER on real call-centre audio is 38.91 %, against 71.48 % for the untouched
original, error rates an order of magnitude above anything measured here.

So treat the *absolute* numbers above as specific to this simulation. The robust
finding is the relative ordering of the classes — a telephony-trained restorer
above acoustic suppression above general-purpose restoration, for WER — plus the
DNSMOS/WER inversion, which holds in both conditions. **Before choosing a
restorer for production, re-run this on real call audio.**

### Why none of these replaces GTCRN in the plugin

They need a GPU, they are not causal, and their latency is hundreds of
milliseconds — they cannot be a `FrameProcessor` on the audio read loop at all.
The relevant deployment is an offline pre-STT stage, a separate product decision.
PESQ, STOI and SI-SDR are reported as `n/a` for this class: all three compare
waveforms sample by sample against a signal a vocoder no longer produces, and
would rank them last however good they sound.

```bash
python -m stt_api.livekit_plugin.noise_cancellation.benchmark \
    --models gtcrn,callenhancer,voicefixer,resemble \
    --degrade telephony --asr-whisper --limit 100
```

## Notes

- **Mono only.** Agent input is mono (`AudioInputOptions.num_channels` defaults to
  1). Anything else passes through unfiltered with a single warning, rather than
  being downmixed behind your back.
- **Fails loudly.** A missing model raises at session start instead of degrading to
  silent passthrough — that quiet degradation is the exact failure mode of the Cloud
  plugin this replaces. If `_process` itself ever throws, LiveKit catches it and
  passes the frame through (`rtc/audio_stream.py`), so a bad frame cannot kill a call.
- **`enabled` is live.** Set `nc.enabled = False` to bypass mid-session; the frame is
  then returned untouched.
- `GTCRN_ONNX_PATH` overrides the bundled model.

## Why not DeepFilterNet

Investigated and deliberately not offered. DeepFilterNet3 is the stronger model, but
nothing about it fits a frame-by-frame Python filter on this runtime:

- **Its published ONNX cannot stream.** Both `DeepFilterNet3_onnx.tar.gz` and the
  `_ll` variant export `enc`/`erb_dec`/`df_dec` over a dynamic time axis with **no
  recurrent state inputs or outputs**. `enc` holds a single `GRU` node whose
  `initial_h` is a `ConstantOfShape` of zeros and whose `Y_h` is discarded. Upstream
  streams these with [tract](https://github.com/sonos/tract)'s *pulse* transform,
  which rewrites a sequence model into a streaming one and carries the state
  internally; onnxruntime has no equivalent, so calling it with `S=1` per frame
  resets the GRU every 10 ms. `conv_lookahead=2` over a time-kernel of 3 means the
  convolutions need ring buffers per frame too.
- **`deepfilterlib`** — the Rust DSP, torch-free — exposes only primitives
  (`DF.analysis`/`synthesis`/`erb_widths`/`fft_window`, `erb`, `erb_norm`,
  `unit_norm`), no model runtime, and publishes **no cp312 wheel**. Agents run on
  Python 3.12.
- **`deepfilternet`** — the torch path — pins `numpy>=1.22,<2.0` against a runtime on
  numpy 2.x with onnxruntime 1.28, and torch would add ~2 GB to the agent image.
- Everything else (STFT, 32-band ERB, the `norm_tau` running normalisation, mask
  application, the 5-tap deep filter, ISTFT) lives outside the graphs and would need
  reimplementing in numpy regardless of route.

Two routes exist if it is ever worth revisiting. Patching the graph to expose
`initial_h`/`Y_h` and splitting the encoder so conv buffers live outside gives true
streaming — see [`shimondoodkin/deepfilter-rt`](https://github.com/shimondoodkin/deepfilter-rt),
MIT/Apache like upstream, which does exactly this in Rust. Running the graphs
unmodified over blocks with a warm-up context prefix avoids the surgery but costs
~6× the CPU and 90–110 ms of latency — worse than GTCRN on both axes, better only on
raw denoising quality. Validate either against the official `deep-filter` binary,
published as a prebuilt release asset.

DF3 is also a 48 kHz model (fft 960, hop 480, 32 ERB bands, `nb_df=96`,
`df_order=5`, `df_lookahead=2`), so adopting it would move the agent's whole input
pipeline to 48 kHz.

## Model

[GTCRN](https://github.com/Xiaobin-Rong/gtcrn) by Xiaobin Rong — MIT licensed,
vendored as `resources/gtcrn_simple.onnx` (523 KB) with its licence in
`resources/LICENSE.gtcrn`. 16 kHz, 512-point STFT, 256-sample hop. Cite:

> X. Rong et al., "GTCRN: A Speech Enhancement Model Requiring Ultralow
> Computational Resources", ICASSP 2024.
