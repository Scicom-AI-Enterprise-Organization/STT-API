# Turn detector plugin for LiveKit Agents

End-of-turn detection backed by a vLLM engine instead of LiveKit's bundled
ONNX runner, so the model that decides when a caller has finished speaking can be
one you fine-tuned — here
[`Scicom-intl/Malaysian-Turn-Detector-Qwen3-1.7B`](https://huggingface.co/Scicom-intl/Malaysian-Turn-Detector-Qwen3-1.7B),
a Qwen3-1.7B fine-tune for Malay/English/Chinese/Tamil call-centre speech and the
code-switching between them.

## How it decides

The model never generates text. It is asked for exactly one token and the answer
is the *probability* of that token:

```
prompt  = ChatML render of the conversation so far, with the trailing <|im_end|> stripped
request = max_tokens=1, logprobs=1, allowed_token_ids=[151645]   # <|im_end|>
answer  = exp(choices[0].logprobs.token_logprobs[0])             # P(turn is over)
```

So one forward pass over a short prompt, and the number that comes back is
compared against `unlikely_threshold` (0.5 by default on the remote path). Above
it, the agent takes its turn; below it, it keeps listening.

Two consequences of that design worth holding on to:

- **The whole thing runs on the transcript, not the audio.** Anything the VAD
  admits and the STT transcribes becomes part of this prompt, including a
  background speaker. Turn boundaries can be wrong even when the caller's own
  words are transcribed perfectly, and no amount of acoustic noise cancellation
  fixes it — see [`../noise_cancellation/benchmark/`](../noise_cancellation/benchmark/).
- **`max_tokens=1` with `allowed_token_ids` is why a 1.7B model is affordable
  here.** There is no decode loop; latency is one prefill of at most
  `MAX_HISTORY_TURNS` of context.

## Running the engine

The plugin talks plain OpenAI `/v1/completions`, so any vLLM will do. Three
things are **not** optional:

| requirement | why |
|---|---|
| `--served-model-name livekit/turn-detector` | the plugin hardcodes `model: "livekit/turn-detector"` in the request body (`VLLM_MODEL`). Serve it under its own name and every request 404s. |
| `--max-model-len 2048` (or more) | the prompt is a rendered chat history, not a single turn |
| no `--api-key`, or auth terminated in front | **the plugin sends no `Authorization` header** — see the warning below |

### Docker Compose

[`vllm.yaml`](vllm.yaml) already has this wired up:

```bash
docker network create turn-detector-network   # once
docker compose -f vllm.yaml up -d
curl localhost:9094/v1/models
```

Override the model with `TURN_DETECTOR_MODEL`, and the GPU slice with
`GPU_MEM_UTIL` (default `0.10` — a 1.7B model in bf16 is ~3.5 GB, so it is meant
to share a card).

### Slurm (tm-h20)

A working job lives at
`ucc_slurm-ui-job/jobs/tm-h20/Malaysian-Turn-Detector-Qwen3-1.7B.yaml`. The part
that matters:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME=/mnt/data/huggingface
export VLLM_USE_DEEP_GEMM=0
export OMP_NUM_THREADS=2

CUDA_VISIBLE_DEVICES=6 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 \
  /mnt/data/stt/venv/bin/vllm serve Scicom-intl/Malaysian-Turn-Detector-Qwen3-1.7B \
    --served-model-name livekit/turn-detector \
    --host 0.0.0.0 \
    --port 9094 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 2048 \
  > >(tee "/mnt/data/turn-detector/logs/turn-detector-$SLURM_JOB_ID.log") 2>&1 &
ENGINE_PID=$!
wait $ENGINE_PID
exit 1   # any engine exit is a failure, so self-heal restarts it
```

Notes carried over from that job, each of which was a bug once:

- **Process substitution, not `| tee`.** In a pipeline `$!` is *tee's* pid, so
  `kill`, `wait` and `kill -0` all track the wrong process and crash detection
  silently stops working.
- **`exit 1` even on a clean engine exit.** A serving engine that leaves is never
  a success; returning 0 would let the supervisor mark the attempt finished
  instead of restarting it.
- **Pin the node and bail if you land elsewhere.** Slurm picks the batch node, and
  the job advertises a fixed IP — better to fail than to publish an endpoint
  nothing is listening on.
- **`--time=0`** so a long-lived server is not walltime-killed.
- The `heal` block gives 10 in-place restarts with exponential backoff plus an
  HTTP probe on `/health`, and a 600 s grace period so a cold model load is not
  restart-looped.

The file pins `--account`, `--nodelist` and the advertised IP to one cluster and
one user; change those for your own. It also reuses the `stt-engine` venv
(`vllm 0.17.1`, `torch 2.10+cu128`, matching that node's 570.x driver) rather
than building a second one, since this is a read-only consumer of it.

> **`--api-key` will break inference while looking healthy.**
> The upstream job passes `--api-key "$VLLM_AUTH_KEY"`, but `multilingual.py`
> posts with no `Authorization` header, so every completion returns 401. vLLM
> exempts `/health` from auth, so the self-heal probe keeps passing and the job
> reports healthy. Worse, `_extract_eot_probability` returns **1.0** when it
> cannot parse a response — "the turn is definitely over" — so the symptom is an
> agent that interrupts constantly, with a green job and no errors in the agent
> log. Either drop `--api-key`, terminate auth in a sidecar, or add the header to
> the plugin. Do not leave it half-configured.

## Using it in an agent

```python
import os

# Base URL only — the plugin appends /v1/completions itself.
os.environ["LIVEKIT_REMOTE_EOT_URL"] = "http://10.0.1.166:9094"

from stt_api.livekit_plugin.turn_detector import MultilingualModel

session = AgentSession(
    stt=...,
    llm=...,
    tts=...,
    vad=ctx.proc.userdata["vad"],
    turn_detection=MultilingualModel(),
    preemptive_generation=True,
)
```

`MultilingualModel(unlikely_threshold=0.6)` raises the bar for cutting a turn —
higher means the agent waits longer and interrupts less.

**`LIVEKIT_REMOTE_EOT_URL` must be set before importing the module**, not after.
At import time the module either registers LiveKit's local ONNX inference runner
or does not, based on whether that variable is set; setting it later leaves you
running the bundled model while believing you are on vLLM.

## Operational behaviour

| | |
|---|---|
| Request timeout | 2 s (`REMOTE_INFERENCE_TIMEOUT`) |
| On unparseable response | returns `1.0` — **fails open**, the agent takes the turn |
| On timeout / HTTP error | raises; LiveKit falls back to VAD-only turn taking |
| Prompt history | capped at `MAX_HISTORY_TURNS`, truncated from the left |
| Text normalisation | NFKC, lowercased, punctuation stripped except `'` and `-` |

Failing open is the right default for a voice agent — a detector that fails
closed would leave the agent mute — but it does mean **a broken engine looks like
an over-eager agent, not like an outage**. If an agent starts interrupting, check
this endpoint before touching the VAD.

Consecutive messages from the same role are merged before rendering, so a
partial-transcript stream does not inflate the turn count.

## Model

[`Scicom-intl/Malaysian-Turn-Detector-Qwen3-1.7B`](https://huggingface.co/Scicom-intl/Malaysian-Turn-Detector-Qwen3-1.7B)
— Apache-2.0, fine-tuned from `Qwen/Qwen3-1.7B`. Reported on a 1200-sample test
set (600 positive / 600 negative, 50 conversations per language pair):

| metric | |
|---|---|
| accuracy | 96.67 % |
| precision | 99.82 % |
| recall | 93.50 % |

Precision far above recall is the right shape for this job: the model rarely
declares a turn over when it is not (which would interrupt the caller), and errs
toward waiting. Pair that with the fail-open behaviour above and the two failure
modes are opposite — the *model* errs toward waiting, the *transport* errs toward
interrupting.

The plugin renders prompts with the tokenizer from `livekit/turn-detector`
(revision `v0.4.1-intl`), not from the served checkpoint. That is safe and
verified: both are Qwen-family, `<|im_end|>` is 151645 in both, and their chat
templates render identically. If you swap in a model from another family, check
both before assuming it still holds — a wrong `IM_END_TOKEN_ID` yields a
plausible probability computed from the wrong token.
