# PulseVAD vs silero

Can a **2,118-parameter** VAD replace silero in a LiveKit agent?

**Short answer: it matches silero on accuracy and loses on cost.** On 1,500 real
production turns the two are statistically indistinguishable at detecting speech,
but PulseVAD costs **1.33x more CPU per window** and commits a turn **96 ms
later**. The 2,118-parameter *model* is 4.5x cheaper than silero's (17.3 us vs 78.3 us);
its Python
log-mel front end is what loses the race, and that is fixable — see below.

**Hardware.** One Apple M-series laptop, CPU only, single ORT thread. Cost
numbers were taken with the benchmark as the only significant load; per
`CLAUDE.md` they do not survive contention, and the ordering matters more than
the absolute values.

**Corpus.** 1,500 turns pulled from the internal drift-capture bucket
(`s3://$S3_BUCKET/$S3_PREFIX`, configured in `.env`) — 4.4 hours of real
Malay/English call-centre audio. Each wav has a
sibling `.json` with the Whisper transcript of that turn, which is what makes
this measurable: **1,320 clips with a transcript are speech, 180 with an empty
transcript are not**. The negatives are real production noise, not synthesised.

**What is measured.** Both detectors run through the real
`livekit.agents.vad.VAD` interface, fed identical 20 ms frames, and are scored on
the events an `AgentSession` would receive. Reading probabilities out of an ONNX
session instead would skip resampling, frame chunking, hangover and prefix
padding — which is where a VAD swap actually changes behaviour.

Reproduce:

```bash
pip install ".[benchmark]"
# bucket and prefix come from .env; see .env.example
python -m stt_api.livekit_plugin.pulse_vad.benchmark --limit 1500
```

## Result

Each model at **its own** best threshold — silero 0.65, PulseVAD 0.70. Scoring
both at 0.5 would compare silero's mid-range against PulseVAD's 70th percentile,
because PulseVAD saturates at 0.711 (see below).

| | silero | PulseVAD 2.1k | |
|---|---:|---:|---|
| turns found (1,320 speech clips) | 99.17 % | 99.02 % | *n.s.* (p = 0.84) |
| false fires (180 no-speech clips) | 2.78 % | 4.44 % | *n.s.* (p = 0.57) |
| onset, paired per turn | — | **+96 ms** | later on 90 % of turns |
| model cost per 32 ms window | **0.078 ms** | 0.104 ms | 1.33x |
| weights on disk | 2.22 MB | **76 KB** | 29x |
| frame agreement | — | 97.18 % | κ = 0.933 |

Neither accuracy difference is significant. The false-fire gap is **three
clips** out of 180 (5 vs 8), where one clip is worth 0.56 pp; the detection gap
is **two clips** out of 1,320. The two differences that *are* real and
consistent are the onset delay and the CPU cost.

## Why the 2,118-parameter model is slower than the 1.8 M one

This is the counterintuitive result, and the reason is **numpy dispatch, not
arithmetic**.

| stage | |
|---|---:|
| pre-emphasis | 1.7 us |
| waveform z-norm | 9.7 us |
| reflect pad | 6.7 us |
| frame + Hann | 8.9 us |
| rfft 21x512 | 16.1 us |
| `abs(spec)**2` | 7.2 us |
| mel matmul + log | 5.6 us |
| per-bin z-norm | 11.2 us |
| **front end total** | **67.2 us** |
| ONNX model (2,118 params) | 17.3 us |
| **total** | **84.5 us** |
| silero, everything, one ONNX call | **77.8 us** |

Only the FFT is real compute — 21 double-precision 512-point transforms at
~30 GFLOP/s, near peak. The other 51 us is nine numpy calls on arrays of 1.3-10 k
elements, where per-call cost dominates: `std()` over 1,344 floats measures
**4.94 us** against roughly 0.3 us of actual arithmetic, and `np.pad` of 3,200
floats costs 4.77 us.

Silero takes raw waveform straight into one fused graph and pays that cost
**once**. PulseVAD needs a 64 x 21 log-mel patch built in Python first. The front
end is already vectorised — one `sliding_window_view`, one batched `rfft`,
bit-exact with upstream and 2.1x faster than its per-frame Python loop, which
would have made this 0.181 ms — and it is still 80 % of the bill.

### An optimisation that did not work, and why it is the point

The 200 ms window slides by 32 ms, so **85 % of the STFT is recomputed every
hop**. That looks like an obvious 3x win, and it is available in principle:

* The per-window mean and scale can be applied *after* the STFT, in the power
  domain, via `|X - mW|^2 = |X|^2 - 2m·Re(X·conj W) + m^2·|W|^2` — verified exact
  to 6e-08 relative. So the expensive part can be cached per frame.
* Frames 2..18 read real audio only and sit on a global 160-sample grid, so they
  are cacheable by absolute position; only frames 0, 1, 19, 20 move with the
  window's reflect padding.

A streaming implementation of exactly that reproduces the reference to **1.4e-04
in p(speech)** (60x smaller than the int8 shift already accepted) while doing a
third of the FFT work — and it benchmarks at **90.6 us against the current
78.7 us front end. It is slower.**

Cutting arithmetic cost more than it saved, because it took *more numpy calls* to
do it: a second `np.pad`, a second `rfft` dispatch, the extra `Re(X·conj W)`
term, and the cache shuffling. At these array sizes the call count is the cost
function, not the flop count.

**So the redundancy is not the problem and removing it is not the fix.** The fix
is to make the front end one call instead of ten — fold it into the ONNX graph
(opset 17 has `STFT`, or the usual Conv1d-plus-matmul mel), so ORT runs it in C++
under the single dispatch it already pays for. The prize is measurable: with the
front end free, PulseVAD costs its model alone, **17.3 us against silero's 78.3 us**; add an in-graph
front end back (the FFT is ~8.9 us in float32, plus elementwise in C++) and a
realistic landing zone is **30-35 us, roughly 2.3x cheaper**. Until then the package is slower despite the model
being far faster.

## p(speech) saturates at 0.711, and a higher threshold fails silently

Measured over 12,098 windows of `test_audio/` plus synthetic noise:

| checkpoint | range | p99.9 |
|---|---|---:|
| `2.1k` fp32 | 0.002 – **0.711** | 0.7107 |
| `2.1k` int8 | 0.005 – **0.708** | 0.7078 |
| `81k` fp32 | 0.006 – **0.895** | 0.8949 |

The saturation is flat, not a tail — p99.9 is within 0.0005 of the maximum.

| `activation_threshold` | 2.1k windows firing |
|---:|---:|
| 0.50 | 79.7 % |
| 0.70 | 56.6 % |
| **0.75** | **0.0 %** |
| 0.80 | 0.0 % |

Silero saturates at 1.0, so carrying `activation_threshold=0.8` across is an
ordinary thing to write — and it produces an agent that never hears anyone, with
no exception and no log line. `PulseVAD.load()` raises instead; `model.CEILINGS`
is the source of truth and `tests/test_pulse_vad.py` asserts it.

The plugin defaults to **0.35**, the midpoint of the measured range, not
silero's 0.5.

## int8 is not faster, and the file is bigger

| | size | per window | vs fp32 |
|---|---:|---:|---|
| `2.1k` fp32 | 12.0 KB | **0.018 ms** | — |
| `2.1k` int8 | 26.8 KB | 0.019 ms | mean \|Δp\| 0.008, max 0.052 |
| `81k` fp32 | 325 KB | 0.098 ms | — |

QDQ quantisation pays on the microcontrollers PulseVAD targets. Off one, it adds
Q/DQ nodes to a 2,118-parameter graph for no speedup and more than double the
file size. `precision="fp32"` is therefore the default here, unlike upstream's
`load_pulsevad(quantized=True)`. int8 stays available for parity-testing an edge
deployment; it flips no decisions at 0.5 on this corpus.

## Three things that made the benchmark wrong before they were fixed

Each changed a headline number.

**Both models were "20 % false-positive."** `min_silence_duration` keeps a VAD
speaking for 0.55 s after the last word *by design*, and that hangover drains
into the trailing pad. Counting it rated the setting, not the model. The
false-positive region now excludes a guard band after each speech region.

**Both models had a median onset of exactly 644 ms.** Production turns are cut
by the upstream stack, not trimmed to the first phoneme, so a clip can open with
half a second of line noise that both detectors dutifully wait through. Absolute
onset measured the corpus. Timing is now differenced *within* an item, where the
audio is identical — which is how the real +96 ms gap became visible.

**F1 picked the operating point at random.** It is dominated by the
speech-bearing body, which both detectors get right, so it moved by <0.01 across
the whole threshold range. The operating point is now chosen on
`detect_rate - false_fire_rate`, which is what an agent experiences.

## Validity

The threshold sweep does not re-run inference per threshold — it replays the
streaming state machine offline over recorded probabilities. Every run prints

```
replay check: 1500/1500 items reproduce the live stream's turn count
```

and `tests/test_pulse_vad.py` asserts the same thing. If that ratio is not 1.0,
every swept threshold except the streamed one is fiction.

## Telephony (the SIP condition)

Re-run with `--conditions telephony` — 8 kHz band-limit plus mu-law, what inbound
SIP audio actually is. Same 1,500 clips.

| | silero | PulseVAD |
|---|---:|---:|
| turns found | 99.85 % (99.4-100.0) | 99.09 % (98.4-99.5) |
| false fires | 3.33 % (1.5-7.1), 6/180 | 3.89 % (1.9-7.8), 7/180 |
| onset, paired | — | +96 ms, later on **96 %** of turns |
| frame agreement | — | 96.08 %, κ = 0.908 |

Narrowband costs PulseVAD a little more than silero — detection drops to 99.09 %
where silero holds 99.85 %, and the confidence intervals no longer overlap on
detection. The onset penalty is unchanged but now applies to 96 % of turns rather
than 90 %. Nothing here improves the case for the swap.

## Recommendation

**Keep silero, as the plugin stands.** PulseVAD matches it on accuracy in
wideband and trails slightly on telephony, while costing 1.33x more CPU per
window, committing a turn 96 ms later, and needing a threshold that does not
transfer. There is no deployment argument for the swap today.

**But the cost gap is an implementation artefact, not a property of the model.**
PulseVAD's 2,118-parameter graph runs in 17.3 us against silero's 77.8 us; the
Python log-mel front end adds 67.2 us on top. Fold that front end into the ONNX
graph and PulseVAD lands near **30-35 us, roughly 2.3x cheaper than silero** rather than
1.33x more expensive, at which point the only remaining cost is the 96 ms onset delay --
which the `81k` teacher largely fixes. That is the one piece of work that would
change this recommendation, and it is well-defined: opset 17 `STFT`, or the usual
Conv1d-plus-matmul mel, validated against `frontend.log_mel`.

PulseVAD earns its place where 2.22 MB of weights is the binding constraint — an
embedded device or a microcontroller, which is the problem it was designed for.
Note the honest footprint is **76 KB, not 12 KB**: the graph is 11.7 KB but the
front end is external to it and needs the 64.4 KB mel filterbank at runtime. So
29x smaller, not 150x. Folding the front end into the graph would fix the CPU
problem and make the footprint claim true in the same change.

**And none of this relieves a CPU spike.** Silero as actually run costs 0.42 % of
one core per stream (0.136 ms per 32 ms hop, of which 43 % is the
`run_in_executor` thread hop rather than inference). Deleting the VAD entirely
saves that 0.42 %; switching to PulseVAD *adds* 0.22 %. For comparison GTCRN
noise cancellation is 3 % of a core per stream, seven times the whole VAD. A VAD
swap is not a lever on CPU in either direction.

If the onset delay is the only blocker, the `81k` teacher is the better tradeoff
than tuning the 2.1k: 172 ms median onset against 140 ms, but a far tighter tail
(236 ms worst against 908 ms) and a ceiling of 0.895 that makes thresholds behave
more like silero's. It costs 0.098 ms of model time — still under silero's
0.078 ms only once the front end is out of Python.
