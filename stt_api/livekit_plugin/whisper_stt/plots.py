"""
plots.py
────────
Regenerates the three figures in README.md from measured data.

    pip install matplotlib        # dev-only; not a dependency of this plugin
    python -m stt_api.livekit_plugin.whisper_stt.plots [--corpus DIR]

`--corpus` points at a cache of production turns (wav + sibling .json carrying a
`transcription` field), as produced by
`stt_api.livekit_plugin.pulse_vad.benchmark`. Without it the level figures fall
back to the recorded summary statistics, so the plots still regenerate on a
machine with no corpus.

To change style: edit the STYLE dict. Palette and conventions follow
`Whisper-Hallucination/bench/plot_benchmark.py` so figures read the same across
both repos. (Note: `ours_color` sits just below the data-viz lightness band at
L=0.369; CVD separation, normal-vision separation and contrast all pass, and
every series is direct-labelled, so the band is the only deviation.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE  ── tweak colors and font sizes here
# ══════════════════════════════════════════════════════════════════════════════
STYLE = dict(
    bg_color="#ffffff",
    grid_color="#e0e0e0",
    ours_color="#203882",  # the measured subject
    bad_color="#922b21",  # the failure mode / cost
    good_color="#1e8449",  # the win
    title_color="#1a1a2e",
    label_color="#333333",
    tick_color="#444444",
    caption_color="#666666",
    anno_bg="#f5f5f5",
    title_fontsize=11,
    axis_fontsize=9.5,
    tick_fontsize=8.5,
    value_fontsize=7.8,
    legend_fontsize=8.5,
    dpi=200,
)

# Measured on livekit-agents 1.3.11 + silero at min_silence_duration=0.55 s:
# VADEvent.speech_duration keeps accumulating through the hangover.
VAD_BURSTS = [(0.15, 0.70), (0.25, 0.70), (0.35, 0.70), (0.49, 1.02), (0.80, 1.38)]
INTERRUPT_THRESHOLD = 0.5
GATE_DBFS = -50.0

FALLBACK = dict(speech_med=-18.4, nonspeech_zero=169, nonspeech_total=180, speech_total=1323)

# Measured end to end on livekit-agents 1.3.11: real in-process AgentSession,
# agent mid-TTS, one 0.35 s VAD false positive. Each cell is (ok?, label).
# Columns are what a caller actually experiences.
OUTCOME_COLUMNS = ["blank bubble\nin the UI", "agent TTS cut\nby silence",
                   "real speech\nstill barges in"]
OUTCOMES = [
    ("baseline\nopenai.STT, min_words=0",
     [(False, "shown  ' '"), (False, "cut"), (True, "yes")]),
    ("+ WhisperSTT\nmin_words=0",
     [(True, "none"), (False, "cut"), (True, "yes")]),
    ("+ min_words=1\nunfiltered STT",
     [(False, "shown  ' '"), (True, "kept"), (True, "yes")]),
    ("both\nWhisperSTT + min_words=1",
     [(True, "none"), (True, "kept"), (True, "yes")]),
]


def _axes(ax, s, axis="both"):
    ax.set_facecolor(s["bg_color"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(s["grid_color"])
    ax.tick_params(colors=s["tick_color"], labelsize=s["tick_fontsize"], length=0)
    ax.grid(True, axis=axis, color=s["grid_color"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def _save(fig, name, s):
    fig.savefig(OUT / name, dpi=s["dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def load_corpus(d: Path):
    """-> (speech dBFS, non-speech dBFS); -240 marks digital zero."""
    import numpy as np
    import soundfile as sf

    sp, ns = [], []
    for j in sorted(d.rglob("*.json")):
        w = j.with_suffix(".wav")
        if not w.exists():
            continue
        try:
            doc = json.loads(j.read_text())
            a, _ = sf.read(w, dtype="float32")
        except Exception:  # noqa: BLE001
            continue
        if a.ndim > 1:
            a = a.mean(axis=1)
        if a.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))
        db = 20.0 * np.log10(max(rms, 1e-12))
        (sp if (doc.get("transcription") or "").strip() else ns).append(db)
    return sp, ns


def fig_levels(sp, ns, s):
    """Why a local gate is free: the two populations barely share an axis."""
    import numpy as np

    floor = -80.0
    fig, ax = plt.subplots(figsize=(8.2, 3.3), facecolor=s["bg_color"])
    _axes(ax, s, axis="y")
    bins = np.arange(floor, 0.1, 2.5)
    ax.hist(np.clip(ns, floor, None), bins=bins, color=s["bad_color"],
            edgecolor=s["bg_color"], linewidth=0.6, label=f"no speech  (n={len(ns)})")
    ax.hist(np.clip(sp, floor, None), bins=bins, color=s["ours_color"],
            edgecolor=s["bg_color"], linewidth=0.6, label=f"speech  (n={len(sp)})")

    ax.axvline(GATE_DBFS, color=s["title_color"], linewidth=1.4, linestyle="--")
    top = ax.get_ylim()[1]
    ax.annotate("gate  −50 dBFS", xy=(GATE_DBFS, top * 0.90), xytext=(6, 0),
                textcoords="offset points", color=s["title_color"],
                fontsize=s["value_fontsize"], fontweight="bold")
    zeros = sum(1 for v in ns if v <= -200)
    ax.annotate(f"{zeros} turns are digitally silent\n(all-zero samples, −∞ dBFS)",
                xy=(floor, top * 0.52), xytext=(9, 0), textcoords="offset points",
                color=s["bad_color"], fontsize=s["value_fontsize"],
                fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=s["anno_bg"],
                          edgecolor="#cccccc", linewidth=0.7))

    ax.set_xlabel("segment level (dBFS)", fontsize=s["axis_fontsize"], color=s["label_color"])
    ax.set_ylabel("turns", fontsize=s["axis_fontsize"], color=s["label_color"])
    ax.set_title("Speech and silence sit ~20 dB apart, so the gate costs nothing",
                 fontsize=s["title_fontsize"], color=s["title_color"],
                 fontweight="bold", loc="left", pad=10)
    ax.set_xlim(floor, 0)
    leg = ax.legend(fontsize=s["legend_fontsize"], facecolor=s["anno_bg"],
                    edgecolor="#cccccc", framealpha=0.95, labelcolor=s["label_color"],
                    loc="upper left")
    leg.get_frame().set_linewidth(0.7)
    _save(fig, "fig-levels.png", s)


def fig_gate(sp, ns, s):
    """The operating window: where the gate skips calls and loses no speech."""
    import numpy as np

    th = np.arange(-70, -9, 1.0)
    skipped = np.array([100 * np.mean([v < t for v in ns]) for t in th])
    lost = np.array([100 * np.mean([v < t for v in sp]) for t in th])

    fig, ax = plt.subplots(figsize=(8.2, 3.3), facecolor=s["bg_color"])
    _axes(ax, s)
    ax.plot(th, skipped, color=s["good_color"], linewidth=2.0,
            label="useless STT calls skipped")
    ax.plot(th, lost, color=s["bad_color"], linewidth=2.0, label="real speech lost")
    ax.axvline(GATE_DBFS, color=s["title_color"], linewidth=1.4, linestyle="--")

    i = int(np.argmin(np.abs(th - GATE_DBFS)))
    for val, col, dy in ((skipped[i], s["good_color"], -3), (lost[i], s["bad_color"], 7)):
        ax.plot([GATE_DBFS], [val], "o", color=col, markersize=7,
                markeredgecolor=s["bg_color"], markeredgewidth=1.6)
        ax.annotate(f"{val:.0f}%", xy=(GATE_DBFS, val), xytext=(8, dy),
                    textcoords="offset points", color=col,
                    fontsize=s["value_fontsize"] + 1, fontweight="bold")
    ax.annotate("default gate", xy=(GATE_DBFS, 62), xytext=(-8, 0),
                textcoords="offset points", ha="right", color=s["title_color"],
                fontsize=s["value_fontsize"], fontweight="bold")

    ax.set_xlabel("silence_floor_dbfs", fontsize=s["axis_fontsize"], color=s["label_color"])
    ax.set_ylabel("% of turns", fontsize=s["axis_fontsize"], color=s["label_color"])
    ax.set_ylim(-3, 103)
    ax.set_title("A ~25 dB window skips nearly every blank call and loses no speech",
                 fontsize=s["title_fontsize"], color=s["title_color"],
                 fontweight="bold", loc="left", pad=10)
    leg = ax.legend(fontsize=s["legend_fontsize"], facecolor=s["anno_bg"],
                    edgecolor="#cccccc", framealpha=0.95, labelcolor=s["label_color"],
                    loc="center left")
    leg.get_frame().set_linewidth(0.7)
    _save(fig, "fig-gate-sweep.png", s)


def fig_hangover(s):
    """Why the blank filter alone cannot stop the interruption."""
    xs = [f"{b:.2f}s" for b, _ in VAD_BURSTS]
    ys = [d for _, d in VAD_BURSTS]

    fig, ax = plt.subplots(figsize=(8.2, 3.3), facecolor=s["bg_color"])
    _axes(ax, s, axis="y")
    bars = ax.bar(xs, ys, color=s["ours_color"], width=0.5,
                  edgecolor=s["bg_color"], linewidth=0.8)
    for b, v in zip(bars, ys):
        ax.annotate(f"{v:.2f}s", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    color=s["ours_color"], fontsize=s["value_fontsize"] + 0.6,
                    fontweight="bold")

    ax.axhline(INTERRUPT_THRESHOLD, color=s["bad_color"], linewidth=1.8, linestyle="--")
    # x in axes fraction, y in data units: a categorical axis clips to
    # (-0.5, n-0.5), so a data-x annotation past the last bar silently vanishes.
    from matplotlib.transforms import blended_transform_factory

    # left side, above the 0.70 s bars: the right side is occupied by the 1.38 s bar
    ax.text(0.015, 0.90, "LiveKit interrupts above 0.5 s  →  every bar qualifies",
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            ha="left", va="center", color=s["bad_color"],
            fontsize=s["value_fontsize"] + 0.6, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=s["anno_bg"],
                      edgecolor="#cccccc", linewidth=0.7))

    ax.set_xlabel("audio actually pushed", fontsize=s["axis_fontsize"], color=s["label_color"])
    ax.set_ylabel("VADEvent.speech_duration peak (s)",
                  fontsize=s["axis_fontsize"], color=s["label_color"])
    ax.set_ylim(0, 1.62)
    ax.set_title("silero's 0.55 s hangover inflates every burst past the interrupt bar",
                 fontsize=s["title_fontsize"], color=s["title_color"],
                 fontweight="bold", loc="left", pad=10)
    _save(fig, "fig-vad-hangover.png", s)


def fig_before_after(s):
    """What a caller experiences, before and after each fix."""
    n_rows, n_cols = len(OUTCOMES), len(OUTCOME_COLUMNS)
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * n_rows + 1.5), facecolor=s["bg_color"])
    ax.set_facecolor(s["bg_color"])

    for r, (_label, cells) in enumerate(OUTCOMES):
        y = n_rows - r - 1
        for c, (ok, text) in enumerate(cells):
            fc = s["good_color"] if ok else s["bad_color"]
            ax.add_patch(plt.Rectangle([c, y], 1, 1, facecolor=fc,
                                       edgecolor=s["bg_color"], linewidth=2.0))
            ax.text(c + 0.5, y + 0.5, text, ha="center", va="center",
                    fontsize=s["value_fontsize"] + 0.8, color="#ffffff",
                    fontweight="bold", fontfamily="monospace")

    for c, name in enumerate(OUTCOME_COLUMNS):
        ax.text(c + 0.5, n_rows + 0.14, name, ha="center", va="bottom",
                fontsize=s["tick_fontsize"], color=s["tick_color"],
                fontfamily="monospace")
    for r, (label, _cells) in enumerate(OUTCOMES):
        head, _, sub = label.partition("\n")
        y = n_rows - r - 1
        is_fix = head.startswith("both")
        ax.text(-0.12, y + 0.62, head, ha="right", va="center",
                fontsize=s["axis_fontsize"], fontweight="bold",
                color=s["ours_color"] if is_fix else s["label_color"])
        ax.text(-0.12, y + 0.28, sub, ha="right", va="center",
                fontsize=s["value_fontsize"], color=s["caption_color"],
                fontfamily="monospace")

    ax.set_xlim(-2.35, n_cols + 0.06)
    ax.set_ylim(0, n_rows + 0.75)
    ax.axis("off")
    ax.set_title("The plugin removes the bubble; min_words removes the interruption",
                 fontsize=s["title_fontsize"], color=s["title_color"],
                 fontweight="bold", loc="left", pad=16, x=-0.28)
    fig.text(0.008, -0.04,
             "livekit-agents 1.3.11 · agent mid-TTS · one 0.35 s VAD false positive · "
             "bottom row is the recommended configuration",
             fontsize=s["value_fontsize"], color=s["caption_color"], style="italic")
    _save(fig, "fig-before-after.png", s)


def main() -> int:
    ap = argparse.ArgumentParser()
    # Any bucket-named subdirectory under the cache root works; the corpus is
    # found by recursive glob so no bucket name is baked in here.
    ap.add_argument("--corpus", type=Path,
                    default=Path.home() / ".cache/stt-api/vad-bench")
    a = ap.parse_args()
    s = STYLE

    if a.corpus.exists():
        sp, ns = load_corpus(a.corpus)
        print(f"corpus: {len(sp)} speech turns, {len(ns)} no-speech turns")
    else:
        import numpy as np

        rng = np.random.default_rng(0)
        sp = list(rng.normal(FALLBACK["speech_med"], 2.0, FALLBACK["speech_total"]))
        ns = [-240.0] * FALLBACK["nonspeech_zero"] + list(
            rng.uniform(-55, -14, FALLBACK["nonspeech_total"] - FALLBACK["nonspeech_zero"]))
        print(f"no corpus at {a.corpus}; using recorded summary statistics")

    fig_before_after(s)
    fig_levels(sp, ns, s)
    fig_gate(sp, ns, s)
    fig_hangover(s)
    print(f"wrote 4 figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
