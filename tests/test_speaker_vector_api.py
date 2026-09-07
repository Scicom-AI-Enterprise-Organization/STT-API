"""
Tests for `POST /audio/speaker_vector`.

The GPU embedder is stubbed; the endpoint is not. What these exercise is the
request handling — decoding, the length guard, that several files collapse into
*one* batched call, and that the failure modes return 4xx rather than a vector
that looks plausible.

The batching assertion is the one worth having. Routing through the existing
TitaNet batch path is the entire reason this endpoint accepts repeated `file`
fields, and a refactor that quietly embedded them one at a time would still
return correct-looking vectors — just several times slower.
"""

import contextlib
import io
import os
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="server extra not installed")
pytest.importorskip("torch", reason="server extra not installed")
pytest.importorskip("soundfile", reason="server extra not installed")

RATE = 16000


@pytest.fixture(scope="module")
def client_and_calls():
    """
    Import the app with the GPU pieces neutralised.

    `diarization.py` builds CUDA streams at module import and `main.py` spawns
    process pools at import, neither of which belongs in a unit test. Both are
    pre-existing behaviours; stubbing them is what makes the endpoint testable
    off a GPU box at all.
    """
    os.environ.setdefault("ENABLE_ONLINE_DIARIZATION", "false")

    import torch

    class _NoStream:
        def __init__(self, *a, **k):
            pass

        def synchronize(self):
            pass

        def wait_stream(self, *a):
            pass

    torch.cuda.Stream = _NoStream
    torch.cuda.stream = lambda *a, **k: contextlib.nullcontext()

    import stt_api

    # The container mounts stt_api as `app` (docker-compose: ./stt_api:/app/app),
    # so the absolute imports inside the package resolve under that name.
    sys.modules.setdefault("app", stt_api)

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
        import stt_api.main as main
    finally:
        cf.ProcessPoolExecutor = real_pool

    main.get_vad_executor = lambda: _Inline()
    main.get_diarization_executor = lambda: _Inline()

    calls: list[tuple[int, bool]] = []

    def fake_embed(chunks, normalize=False):
        calls.append((len(chunks), normalize))
        return [[0.1] * 192 for _ in chunks]

    main.embed_chunks_for_api = fake_embed

    from fastapi.testclient import TestClient

    return TestClient(main.app), calls


def wav_bytes(seconds: float, sr: int = RATE) -> bytes:
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_single_file_returns_one_vector(client_and_calls):
    client, _ = client_and_calls
    r = client.post("/audio/speaker_vector", files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["dim"] == 192
    assert len(body["vectors"]) == 1
    assert len(body["vectors"][0]) == 192


def test_several_files_are_embedded_as_one_batch(client_and_calls):
    """
    The point of the endpoint: N files, one GPU batch. Embedding them one at a
    time would return identical output and lose the reason for batching.
    """
    client, calls = client_and_calls
    calls.clear()
    files = [("file", (f"{i}.wav", wav_bytes(1.5), "audio/wav")) for i in range(3)]
    r = client.post("/audio/speaker_vector", files=files)
    assert r.status_code == 200
    assert r.json()["count"] == 3
    assert calls == [(3, False)], f"expected one batched call of 3, got {calls}"


def test_normalize_flag_is_forwarded(client_and_calls):
    client, calls = client_and_calls
    calls.clear()
    r = client.post(
        "/audio/speaker_vector",
        files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")},
        data={"normalize": "true"},
    )
    assert r.status_code == 200
    assert r.json()["normalized"] is True
    assert calls == [(1, True)]


def test_too_short_audio_is_rejected_not_silently_embedded(client_and_calls):
    """
    TitaNet needs enough frames for its mel spectrogram; below that it returns a
    garbage vector rather than raising. A 400 is the only honest answer.
    """
    client, _ = client_and_calls
    r = client.post("/audio/speaker_vector", files={"file": ("s.wav", wav_bytes(0.2), "audio/wav")})
    assert r.status_code == 400
    assert "too short" in r.json()["detail"]


def test_undecodable_audio_is_a_client_error(client_and_calls):
    client, _ = client_and_calls
    r = client.post("/audio/speaker_vector", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert r.status_code == 400
    assert "cannot decode" in r.json()["detail"]
