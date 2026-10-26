#!/bin/bash
# The benchmark matrix on one machine: its own LiveKit server, then per filter a fresh
# agent worker and CALLS callers streaming the recordings in real time.
#
#   LIVEKIT_SERVER=/path/to/livekit-server FILES=calls.txt CALLS=32 \
#   VI_BENCH_MODEL=/models/voice-isolator-cpu VOICE_ISOLATOR_URL=ws://gpu-host:8765 \
#       bash stt_api/livekit_plugin/voice_isolator/benchmark/run.sh none gtcrn local remote
#
# FILES: one directory per line with mix.wav (the call) and caller.wav (the caller
# alone, 16 kHz mono). The reference run publishes caller.wav with no filter; every
# other run publishes mix.wav. Results: $ROOT/report.json and the logs under $ROOT.
# REF=keep reuses a reference run already in $ROOT; RUN_TAG names this matrix's runs
# <filter><RUN_TAG>-<CALLS> (e.g. RUN_TAG=-f20 VI_BENCH_FRAME_MS=20 for 20 ms frames).
#
# The server is always this script's own: its ports (LK_PORT, +1 TCP, +2 UDP) and an API
# key of its own, so that neither its callers nor its worker can ever reach another
# LiveKit server on the machine (whose agents would take the benchmark's rooms, or
# whose rooms the benchmark's worker would take). A port already in use aborts the run.
set -u
PY=${PYTHON:-python}
ROOT=${ROOT:-/tmp/vi_bench}
CALLS=${CALLS:-32}
LK_PORT=${LK_PORT:-17950}
mkdir -p "$ROOT"
for p in $LK_PORT $((LK_PORT + 1)); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
    echo "port $p is in use: refusing to share another LiveKit server" >&2
    exit 1
  fi
done
SECRET=$($PY -c "import secrets; print(secrets.token_hex(24))")
cat > "$ROOT/livekit.yaml" <<EOF
port: $LK_PORT
bind_addresses: ["127.0.0.1"]
rtc:
  tcp_port: $((LK_PORT + 1))
  udp_port: $((LK_PORT + 2))
  use_external_ip: false
  node_ip: 127.0.0.1  # everything runs on this host: offer only the loopback address
keys:
  vibench: $SECRET
logging:
  level: ${LK_LOG_LEVEL:-warn}
EOF
"${LIVEKIT_SERVER:-livekit-server}" --config "$ROOT/livekit.yaml" > "$ROOT/livekit-server.log" 2>&1 &
LK=$!
trap 'kill $LK 2>/dev/null' EXIT
sleep 3
kill -0 $LK || { echo "livekit-server did not start: see $ROOT/livekit-server.log" >&2; exit 1; }
export LIVEKIT_URL=ws://127.0.0.1:$LK_PORT LIVEKIT_API_KEY=vibench LIVEKIT_API_SECRET=$SECRET

one() {  # one() FILTER RUN CHANNEL
  rm -rf "${ROOT:?}/$2"
  VI_BENCH_FILTER=$1 VI_BENCH_OUT="$ROOT/$2" VI_BENCH_IDLE=$((CALLS + 2)) \
    $PY -m stt_api.livekit_plugin.voice_isolator.benchmark.agent start > "$ROOT/agent-$2.log" 2>&1 &
  local W=$!
  # every call's process warm (VAD, model) before the first caller joins
  for _ in $(seq 300); do
    n=$(grep -c '"process initialized"' "$ROOT/agent-$2.log" 2>/dev/null)
    [ "${n:-0}" -ge $((CALLS + 2)) ] && break
    sleep 1
  done
  sleep 2
  $PY -m stt_api.livekit_plugin.voice_isolator.benchmark.callers --files "$FILES" --calls "$CALLS" \
    --run "$2" --channel "$3" --out "$ROOT"
  for _ in $(seq 60); do
    [ "$(ls "$ROOT/$2"/"$2"-*.json 2>/dev/null | wc -l)" -ge "$CALLS" ] && break; sleep 1
  done
  kill -INT $W; wait $W 2>/dev/null
}

if [ "${REF:-}" != keep ] || [ ! -f "$ROOT/ref-$CALLS/callers.json" ]; then
  one none "ref-$CALLS" caller.wav
fi
RUNS=""
for F in "$@"; do
  one "$F" "$F${RUN_TAG:-}-$CALLS" mix.wav
  RUNS="$RUNS $F${RUN_TAG:-}-$CALLS"
done
$PY -m stt_api.livekit_plugin.voice_isolator.benchmark.report --root "$ROOT" --reference "ref-$CALLS" --runs $RUNS
