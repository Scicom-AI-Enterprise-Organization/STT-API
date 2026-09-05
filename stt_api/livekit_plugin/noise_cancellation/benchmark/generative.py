"""
Generative / semantic restorers: CallEnhancer, Sidon, resemble-enhance, voicefixer.

These are a **different class of system** from everything in `enhancers.py`, and
the difference matters more than the scores do.

An acoustic denoiser estimates a mask and multiplies it onto the noisy spectrum.
Whatever comes out is the input with things removed — the waveform is preserved,
the speaker is preserved, and a sample that was never in the input cannot appear
in the output. A generative restorer instead encodes the audio into a learned
representation, *cleans the representation*, and resynthesises a new waveform
through a vocoder. Nothing of the original waveform survives. That is why they
can repair 8 kHz codec'd telephony into something that sounds like 48 kHz studio
audio, and it is also why they can put words in the speaker's mouth that were
never said.

Three consequences for how they are measured here:

* **PESQ, STOI and SI-SDR are meaningless for them, and worse than meaningless —
  they are actively misleading.** All three compare waveforms sample by sample.
  A vocoder output is a *different waveform* that sounds like the same speech, so
  it scores catastrophically on SI-SDR while sounding far better than anything
  that scores well. The harness reports these columns as `n/a` for this class
  rather than printing a number that would rank them last.
* **DNSMOS and WER are the metrics that transfer.** DNSMOS is reference-free and
  judges the output on its own terms; WER asks the only question that matters
  downstream. A restorer that improves DNSMOS while worsening WER has hallucinated
  fluent speech that is not what the caller said — the characteristic and
  dangerous failure of this class.
* **None of them is a LiveKit inline filter.** They need a GPU, they are not
  causal, and their chunk latency is measured in hundreds of milliseconds. They
  are marked `streaming = False` and belong to a different product decision:
  offline restoration ahead of transcription, not a `FrameProcessor` on the audio
  read loop.

Each model runs in its own venv as a persistent worker subprocess. The venvs are
mandatory — these packages pin mutually incompatible torch, numpy and even Python
versions (resemble-enhance is cp310/cp311 only). The worker is persistent because
the alternative, one subprocess per utterance, would reload a gigabyte of weights
for every three seconds of audio.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from .audio import StreamResampler
from .enhancers import Enhancer, _read_wav, _write_wav

__all__ = ["GENERATIVE", "GenerativeEnhancer"]

_GEN_ROOT = Path(os.environ.get("NC_GEN_ROOT", "/root/nc-bench/gen"))

_READY = "READY"
_OK = "OK"


class GenerativeEnhancer(Enhancer):
    """
    Drives one restorer through a persistent worker in its own interpreter.

    Protocol, deliberately the simplest thing that survives a model crashing
    halfway through a 824-item corpus: the worker prints `READY` once its weights
    are loaded, then for each `<in>\\t<out>` line it prints `OK` or `ERR <msg>`.
    A dead worker surfaces as a failed item rather than a hung benchmark.
    """

    streaming = False
    generative = True
    native_rate = 48000

    #: venv directory name under NC_GEN_ROOT, and the runner script beside it
    venv: str = ""
    runner: str = ""
    #: extra environment for the worker — how the CallEnhancer variants select
    #: their checkpoints without needing three near-identical runner scripts
    extra_env: dict[str, str] = {}
    startup_timeout: float = 900.0
    item_timeout: float = 600.0

    def __init__(self) -> None:
        root = _GEN_ROOT
        py = root / self.venv / "bin" / "python"
        script = root / self.runner
        if not py.exists():
            raise FileNotFoundError(
                f"{self.name}: no interpreter at {py} — run gen/setup.sh first, or "
                f"point NC_GEN_ROOT at the environments"
            )
        if not script.exists():
            raise FileNotFoundError(f"{self.name}: missing runner {script}")
        self._py, self._script = str(py), str(script)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _await(self, proc: subprocess.Popen, max_noise: int = 4000) -> str:
        """
        Read stdout until a protocol line appears, skipping everything else.

        Necessary, not defensive. These libraries write to stdout uninvited and
        there is no way to stop them from the outside: resemble-enhance emits
        DeepSpeed's `[INFO] Setting ds_accelerator to cuda` banner, and various
        loaders print progress. Treating the first line as the handshake made
        every resemble-enhance item fail with the banner text as the "error".
        """
        for _ in range(max_noise):
            line = proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                return ""
            line = line.strip()
            if line == _READY or line == _OK or line.startswith("ERR "):
                return line
        return ""

    def _ensure_worker(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.update(self.extra_env)
        self._proc = subprocess.Popen(
            [self._py, self._script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        if self._await(self._proc) != _READY:
            err = ""
            try:
                self._proc.kill()
                err = (self._proc.stderr.read() or "")[-800:]  # type: ignore[union-attr]
            except Exception:
                pass
            raise RuntimeError(f"{self.name}: worker failed to start. stderr: {err}")
        return self._proc

    def reset(self) -> None:
        # Nothing to reset: these are whole-file, stateless between utterances.
        # The worker is kept alive on purpose — that is the point of it.
        pass

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()
        self._proc = None

    def process_all(self, x: np.ndarray, rate: int) -> np.ndarray:
        with self._lock:
            proc = self._ensure_worker()
            with tempfile.TemporaryDirectory() as td:
                src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
                _write_wav(src, x, rate)
                proc.stdin.write(f"{src}\t{dst}\n")  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
                reply = self._await(proc)
                if reply != _OK:
                    if not reply:
                        # Worker died: surface its stderr, which is where the real
                        # cause is, rather than an empty "worker died".
                        tail = ""
                        try:
                            tail = (proc.stderr.read() or "")[-800:]  # type: ignore[union-attr]
                        except Exception:
                            pass
                        self._proc = None
                        raise RuntimeError(f"{self.name}: worker died. stderr: {tail}")
                    raise RuntimeError(f"{self.name}: {reply}")
                if not dst.exists():
                    raise RuntimeError(f"{self.name}: worker reported OK but wrote nothing")
                y, out_rate = _read_wav(dst)
        # Restorers emit 44.1 or 48 kHz. Bring it back to the pipeline rate so it
        # is scored on the same footing as everything else; the extra bandwidth
        # they synthesise above the corpus Nyquist is not something a 16 kHz agent
        # would ever carry.
        if out_rate != rate:
            y = StreamResampler(out_rate, rate).push(y)
        return y


class CallEnhancer(GenerativeEnhancer):
    """
    Scicom's own telephony restorer: w2v-BERT 2.0 + LoRA feature encoder into a
    DAC decoder, fine-tuned from Sidon for 8 kHz call-centre audio.

    Three variants, all needing a valid `HF_TOKEN` for the private two:

        callenhancer        small, open   — 768M, 48 kHz out
        callenhancer-base   private       — 768M, 48 kHz out, step 896k (v3)
        callenhancer-lite   private       — 89M,  24 kHz out, DistilHuBERT encoder

    Lite is the realtime-oriented one: 8.6x fewer parameters and 5.4x the
    throughput, trading the 580M w2v-BERT encoder for a 37M DistilHuBERT.
    """

    name = "callenhancer"
    note = "Scicom, Sidon-based telephony restoration (small, open)"
    venv = "ce"
    runner = "run_callenhancer.py"


class CallEnhancerBase(CallEnhancer):
    """The larger sibling. `decoder_base_v3` supersedes v2 (896k steps vs 856k)."""

    name = "callenhancer-base"
    note = "Scicom CallEnhancer-Base (private), w2v-BERT + DAC 48 kHz"
    extra_env = {
        "CE_REPO": "Scicom-intl/CallEnhancer-Base",
        "CE_FE": "decoder_base_v3/fe_adapter.pt",
        "CE_DEC": "decoder_base_v3/decoder_only.pt",
        "CE_ARCH": "w2vbert",
        "CE_CHUNK": "0",
    }


class CallEnhancerLite(CallEnhancer):
    """
    The realtime sibling: DistilHuBERT encoder, DAC 1536ch, 24 kHz out.

    `CE_CHUNK=4` is not a tuning choice. DistilHuBERT's feature extractor
    normalises each utterance as a whole, so a single pass over a long file is
    out of distribution and drops speech; the model card is explicit about it.
    """

    name = "callenhancer-lite"
    note = "Scicom CallEnhancer-Lite (private), DistilHuBERT + DAC 24 kHz"
    extra_env = {
        "CE_REPO": "Scicom-intl/CallEnhancer-Lite",
        "CE_FE": "decoder_lite_v2/fe_only.pt",
        "CE_DEC": "decoder_lite_v2/decoder_only.pt",
        "CE_ARCH": "distilhubert",
        "CE_CHUNK": "4",
    }


class Sidon(GenerativeEnhancer):
    """Upstream of CallEnhancer — w2v-BERT feature cleanser plus vocoder."""

    name = "sidon"
    note = "sarulab-speech/sidon-v0.1, general speech restoration"
    venv = "sd"
    runner = "run_sidon.py"


class ResembleEnhance(GenerativeEnhancer):
    """Denoiser + CFM restoration vocoder at 44.1 kHz. cp310/cp311 only."""

    name = "resemble"
    note = "resemble-enhance, 44.1 kHz denoise + restore"
    venv = "re"
    runner = "run_resemble.py"


class VoiceFixer(GenerativeEnhancer):
    """Analysis-synthesis restoration at 44.1 kHz; the oldest of the four."""

    name = "voicefixer"
    note = "voicefixer, 44.1 kHz restoration"
    venv = "vf"
    runner = "run_voicefixer.py"


GENERATIVE: dict[str, type[Enhancer]] = {
    CallEnhancer.name: CallEnhancer,
    CallEnhancerBase.name: CallEnhancerBase,
    CallEnhancerLite.name: CallEnhancerLite,
    Sidon.name: Sidon,
    ResembleEnhance.name: ResembleEnhance,
    VoiceFixer.name: VoiceFixer,
}
