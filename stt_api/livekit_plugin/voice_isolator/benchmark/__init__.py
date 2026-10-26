"""
The voice isolator inside a real LiveKit agent, streamed in real time.

`agent.py` is a LiveKit Agents worker (AgentSession, LiveKit's Silero VAD, the dummy
STT/LLM/TTS) with the filter under test as its audio input's noise cancellation;
`callers.py` opens many rooms at once and publishes recordings into them as
microphones; `report.py` turns the logs into numbers; `run.sh` runs the matrix.
See README.md.
"""
