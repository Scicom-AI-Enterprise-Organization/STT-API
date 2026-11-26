"""
Silero VAD on a GPU server, for LiveKit Agents: a drop-in for `silero.VAD`.

    from stt_api.livekit_plugin.remote_silero import RemoteSileroVAD

    def prewarm(proc):
        proc.userdata["vad"] = RemoteSileroVAD.load("ws://gpu-host:8766")

The server (`python -m stt_api.livekit_plugin.remote_silero.server`) batches every
call's windows into one model run. See README.md for what it does and does not save.
"""

__all__ = ["RemoteSileroVAD", "RemoteSileroVADStream"]


def __getattr__(name: str):
    # Deferred, so that importing the package for its server does not need livekit.
    if name in __all__:
        from . import client

        return getattr(client, name)
    raise AttributeError(name)
