# Can PulseVAD replace silero?

Both detectors run through the real `livekit.agents.vad.VAD` interface, on
identical audio, and are scored on **labelled production turns**.

```bash
pip install ".[benchmark]"

# local files
python -m stt_api.livekit_plugin.pulse_vad.benchmark --audio test_audio/

# labelled production audio from S3 (credentials in .env - see .env.example)
python -m stt_api.livekit_plugin.pulse_vad.benchmark \
    --s3-prefix stt-dev/drift/proxy-.../2026-09-22/ --limit 1500

# one pasted object, or a harder condition
python -m stt_api.livekit_plugin.pulse_vad.benchmark \
    --s3-uri s3://bucket/path/clip.wav --conditions telephony
```

## Where the labels come from

`stt-dev/drift/` holds one wav per user turn with a sibling `.json` carrying the
Whisper transcript of that turn. That gives real ground truth in both
directions:

* **empty transcript → no speech.** ~13 % of the corpus. Any speech reported in
  one of these is a false fire. This is real production noise — line noise,
  breaths, room tone, hold music — not synthesised negatives, and it is the
  failure that matters most: a VAD that fires on silence makes the agent
  interrupt itself.
* **non-empty transcript → speech.** Reporting none is a missed turn, i.e. a
  deaf agent.

The label is evidence, not proof — a large ASR can drop very quiet or very short
speech — but it is the *same* label for both detectors, so the comparison stays
fair even where the label is wrong.

Positive clips are additionally sandwiched between 1.5 s pads of known
non-speech so that timing has a reference instant. Pads default to a 1/f "room"
floor rather than digital silence: a VAD that only has to reject `zeros` is
untested.

## Three methodological points, each of which changed a number

**Each model is scored at its own best threshold.** PulseVAD saturates at 0.711
and silero at ~1.0. Scoring both at 0.5 compares silero's mid-range against
PulseVAD's 70th percentile and reports the difference as accuracy — that is a
measurement of output scale, not of the model.

**The hangover is excluded from the false-positive region.**
`min_silence_duration` keeps a VAD "speaking" for 0.55 s after the last word *by
design*. Counting that as a false positive rates the setting rather than the
model; before this guard existed both detectors scored ~20 % false-positive on
pads, almost all of it hangover draining.

**Timing is paired within an item.** Production turns are cut by the upstream
stack, not trimmed to the first phoneme, so a clip may open with half a second
of line noise that both detectors dutifully wait through. Over 250 real turns
both reported a median onset of *exactly* 644 ms — a fact about the corpus, not
about either model. Differencing within an item cancels the lead-in.

Plain F1 has the same flatness problem and is reported but not optimised: it is
dominated by the speech-bearing body, which both detectors get right, and it
barely moves across thresholds. The operating point is chosen on
`detect_rate - false_fire_rate` instead, which is what an agent experiences.

## Validity check

The threshold sweep does not re-run inference per threshold; it replays the
streaming state machine offline over recorded probabilities
(`metrics.segment`, mirroring `vad.py`). Every run prints

```
replay check: 250/250 items reproduce the live stream's turn count
```

If that ratio is not 1.0, every swept threshold except the streamed one is
fiction and the replay is what to fix. `tests/test_pulse_vad.py` asserts it too.

## Cost

Two columns, deliberately:

* **model only** — the callable each stream invokes, timed in a tight loop.
* **in-stream** — `VADEvent.inference_duration`, which brackets the whole loop
  iteration including frame combining and an `asyncio` thread hop.

At this scale the second is roughly double the first for both detectors, so the
overhead rivals the model and the in-stream figure flatters neither honestly.

Per `CLAUDE.md`: **cost numbers need a quiet machine.** Quality metrics here are
deterministic and reproduce under any load; per-window cost does not. Run cost
comparisons pinned and alone. Relative ordering within one run survives
contention; absolute budget does not.
