# Noise cancellation benchmark

A shootout between candidate denoisers, run under the constraints a LiveKit agent
actually imposes. It exists to answer one question with evidence rather than
vibes: **is GTCRN still the right filter for this plugin, and what would replace
it?**

```bash
uv pip install -e '.[benchmark]'

# which candidates can run on this machine
python -m stt_api.livekit_plugin.noise_cancellation.benchmark --list

# quick look
python -m stt_api.livekit_plugin.noise_cancellation.benchmark --limit 50

# the full standard benchmark, comparable to published numbers
python -m stt_api.livekit_plugin.noise_cancellation.benchmark --models all --limit 0
```

## What makes this different from an offline SE benchmark

Almost every published speech-enhancement number is produced by handing a model a
whole utterance. That measures a model nobody can deploy in a voice agent: it
grants unlimited lookahead, hides buffering latency, and lets a model that cannot
keep up with real time look free.

Here every candidate is driven **frame by frame, in order, at the pipeline's
frame size**, through the same code path LiveKit uses — GTCRN literally goes
through `rtc.AudioFrame` and its own `_process`, int16 quantisation and internal
resamplers included. Three consequences worth knowing:

- **Latency is measured, not assumed.** Each model's algorithmic delay is
  recovered by cross-correlating its output against the clean reference. GTCRN
  measures 32.0 ms, exactly the two hops its own documentation claims — which is
  the cheapest available check that the harness is not lying.
- **Cost is reported at p99, not as a mean.** `_process` runs inline on the audio
  read loop. A filter with a fine RTF that occasionally takes 60 ms on a 20 ms
  frame will stutter, and the mean will never show it.
- **Resampling is charged to whoever needs it.** RNNoise and DeepFilterNet are
  48 kHz models; in a 16 kHz agent that is two resampler stages, and the delay
  those add is theirs.

## The metrics, and which to believe when they disagree

They *will* disagree — that is the main thing this table is for.

| | what it measures | blind spot |
|---|---|---|
| **PESQ** (wideband) | perceptual quality | barely punishes the musical noise neural suppressors invent |
| **STOI / ESTOI** | intelligibility — how much of the message survives | saturates; ESTOI is the one to read |
| **SI-SDR** | raw waveform fidelity, no perceptual model | indifferent to how it sounds |
| **DNSMOS P.835** | SIG / BAK / OVRL, the DNS Challenge ranking metric | it is a model's opinion, not a listener's |
| **WER** | what the STT actually does with it | needs a running `stt-api` |
| **cost** | RTF, p50/p95/p99/max, delay | — |
| **floor** | residual level in the quietest 10% of the reference | — |

`floor` is worth singling out. Aggregate SNR hides the thing users complain
about: a denoiser can post a fine SNR while leaving audible hiss *between* words,
and it is that hiss — not the in-speech noise — that makes a call sound
unprocessed and that a VAD trips on.

**When PESQ and WER disagree, believe WER.** Suppression is tuned to please human
listeners, and what pleases a listener is not what an acoustic model needs.
Fricatives and stop bursts are low-energy and noise-like — the first thing an
aggressive suppressor removes. Gaining a point of PESQ while losing WER is a
normal outcome, not a paradox.

## Corpora

| `--corpus` | what it is | reference? |
|---|---|---|
| `voicebank` | VoiceBank+DEMAND test set, 824 paired utterances | yes |
| `dns` | DNS Challenge 2020 dev test, synthetic no-reverb | yes |
| `dns-real` | DNS Challenge 2020 real recordings | no — DNSMOS only |

`voicebank` is the default because it is *the* standard SE benchmark: GTCRN, DTLN
and DeepFilterNet all publish PESQ/STOI on exactly this set. That is the point —
the absolute numbers can be checked against the papers, so a harness bug shows up
as a model scoring far from its published figure rather than as a plausible
ranking nobody questions.

`dns-real` matters for a different reason: VoiceBank's noise is additively mixed
at known SNRs, which is not what a phone call sounds like.

`--limit N` takes a *deterministic random* subset, not the first N rows —
VoiceBank is ordered by speaker, and the first N rows are two speakers at a
handful of SNRs, a subset that moves the scores by more than the differences
being measured.

Both corpora are 16 kHz, matching the agent pipeline. The 48 kHz models therefore
see upsampled, band-limited input and cannot use the top octave they were trained
on. That is the condition they would face in a real 16 kHz agent, so it is the
right thing to measure — but their scores here are a floor, not their ceiling on
full-band audio.

## The candidates

| `--models` | what | streaming |
|---|---|---|
| `passthrough` | unprocessed control | — |
| `wiener` | decision-directed Wiener + minimum statistics, numpy only | yes |
| `rnnoise` | Xiph RNNoise, 48 kHz, via its real C library | yes |
| `dtln` | 2× LSTM, 512/128 STFT, 16 kHz | yes |
| `gtcrn` | **the incumbent**, 48.2 K params, 16 kHz | yes |
| `dfn3` | DeepFilterNet3 — **offline quality ceiling, not deployable** | no |

Three of these are deliberate rather than obvious:

**`passthrough` is always run**, even if not requested, because every delta is
measured against it. A model that cannot beat doing nothing is worse than no
noise cancellation, and that has to be visible rather than inferred.

**`wiener` exists for attribution.** Without a classical baseline, every neural
gain gets reported against raw noisy input, which overstates what the *learning*
bought — a good part of the improvement on stationary noise is available from
1980s DSP for free. What a neural model wins over this row is the part that
actually needed a neural model.

**`dfn3` is a ceiling, not a candidate.** It is run whole-file, with unlimited
lookahead, on purpose: it answers "how much quality is left on the table?", and
that needs DF3 at its best. It cannot stream here — its published ONNX has no
recurrent state in or out, so per-frame calls reset the GRU every 10 ms; the
plugin README works through this in detail. It runs through the official
`deep-filter` binary rather than the ONNX graph, because that graph is the neural
network *only* — no STFT, no ERB bank, no `norm_tau` normalisation, no 5-tap deep
filter. Reimplementing that stack is how you end up measuring your own DSP bugs
and publishing them as DeepFilterNet's score.

RNNoise is likewise driven through the actual C library, not the ONNX export that
circulates on the Hub — that graph is only the GRU, and expects 42 hand-built
features the caller has to produce.

## The WER axis

```bash
python -m stt_api.livekit_plugin.noise_cancellation.benchmark \
    --limit 100 --asr-url http://127.0.0.1:8000
```

Neither corpus ships transcripts, so the reference is **the STT's own transcript
of the clean signal**. This is a deliberate choice and it changes what the number
means: not "how accurate is the STT" but "how much of the clean-audio result does
this denoiser preserve". That is the right question — a denoiser cannot be blamed
for words the STT would have missed anyway, and pooling both effects into one
number hides which is which. The `passthrough` row gives the degradation the
noise alone causes; anything that does not beat it is losing to doing nothing.

Counts are pooled across utterances rather than averaged as per-utterance rates,
matching `stt_api.evaluation.corpus_metrics`.

## Caching

Models, corpora and the `deep-filter` binary land in
`~/.cache/stt-api/nc-benchmark`, overridable with `NC_BENCHMARK_CACHE`. First run
downloads; after that it is offline. `DEEP_FILTER_BIN` points at your own build if
you are on a platform without a published release.

## Adding a candidate

Subclass `Enhancer`, implement `reset()` and `process(frame, rate)`, add it to
`REGISTRY`. If it has a fixed native rate and a fixed hop, wrap `_HopStreamer` and
it handles resampling, buffering and output priming for you — that class is where
the fiddly "answer every frame with a frame of equal length" logic lives.

Two rules for a contribution to be worth trusting: it must be genuinely causal
(no peeking at future frames), and it must reset all state in `reset()`, or it
will arrive at each utterance already adapted to the noise and post an unearned
score.

## Reading the output

`budget` is p99 as a fraction of the frame duration it had to fit inside. Above
100% the filter cannot keep up with its worst frames. Above roughly 30% there is
no headroom left for the STT, VAD and LLM sharing the core.

`max` is usually the first frame of the stream — model warm-up and resampler
construction — so it reflects session-start cost, not steady state.

### Cost numbers need a quiet machine

The quality columns are deterministic and reproduce exactly regardless of load.
The cost columns do not, and they are far more sensitive than is comfortable:
GTCRN measured p99 1.09 ms / 5% budget on an idle machine and 9.00 ms / 45%
budget on the same audio while another benchmark run was using the other cores —
an eightfold difference that says nothing about GTCRN.

So: run cost comparisons with nothing else happening, and treat any single
absolute figure as suspect. The *relative* ordering between models within one run
survives load, because every model is measured under the same contention; the
absolute budget figure is the one that needs a quiet machine to mean anything.
