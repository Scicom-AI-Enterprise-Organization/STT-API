# `stt_api.evaluation` — WER/CER with spelling conventions normalized away

A chunk of measured WER is not recognition error. It is the reference and the hypothesis
disagreeing about how to **write** the same utterance:

| reference | hypothesis | charged | actually |
|---|---|---|---|
| `dua puluh tiga ringgit` | `23 ringgit` | 3 errors | same words |
| `okay lah` | `ok la` | 2 errors | same words |
| `Belum. [laugh] Bodoh.` | `Belum. Bodoh.` | 1 error | nothing was missed |
| `kosong satu dua` | `012` | 3 errors | same digits |

Scoring verbatim is the right default for tracking one model over time, but it charges a
real error for every convention mismatch. This package reports **both** numbers, so you
can see how much of a WER is the model and how much is the transcription style.

Everything here is standard library. The default mode does no network I/O at all.

---

## Install

Already part of `stt-api`, so a plain install is enough:

```bash
pip install -e .                    # core — everything below works
pip install -e '.[evaluation]'      # + huggingface_hub/pyarrow, ONLY for load_canonical
```

---

## The one-call API

```python
from stt_api.evaluation import score

r = score("saya bayar 23 ringgit", "saya bayar dua puluh tiga ringgit")

r.wer                     # [0.5]   as an ordinary scorer charges it
r.normalized_wer          # [0.0]   the whole error was how a number was written
r.cer                     # [0.42]
r.normalized_cer          # [0.0]
r.normalized_hypothesis   # ['saya bayar 23 ringgit']
r.normalized_reference    # ['saya bayar 23 ringgit']
```

### Which argument is which

```python
score(hypothesis, reference)          # ASR output FIRST, ground truth SECOND
score(hypothesis=hyp, reference=ref)  # or by keyword, if you would rather not remember
```

- **`hypothesis`** — what the model produced.
- **`reference`** — what was actually said, as a human wrote it (the dataset's text column).

⚠ WER is not symmetric. Swapping them returns a different, wrong number and raises
nothing: an insertion becomes a deletion, and a short hypothesis against a long reference
scores nothing like the reverse. If in doubt, use keywords.

### Lists in, lists out

A single string pair is just a list of one, so the shape of the result never depends on
the shape of the input:

```python
r = score(["okay lah, saya nak bayar", "nombor 012"],
          ["ok la saya nak bayar",     "nombor kosong satu dua"])

r.wer                     # [0.4, 0.75]      per item
r.normalized_wer          # [0.0, 0.0]
r.corpus_wer              # 0.5556           pooled — NOT the mean of r.wer
r.normalized_corpus_wer   # 0.0
r.recovered_wer           # 0.5556           corpus points that were convention
len(r)                    # 2
```

The two lists must be the same length and in the same order; a mismatch raises
`ValueError`.

**`corpus_wer` is what you publish.** It is total edit distance over total reference
length, so long clips weigh more — which is the standard definition. The mean of
`r.wer` is a different (and worse) statistic: it lets a one-word clip that is wholly
wrong outweigh a thirty-word clip that is nearly right.

### Everything `score()` accepts

```python
r = score(
    hyps, refs,
    mode="deterministic",     # or "llm" / "both" — see below
    drop_fillers=False,       # delete fillers from BOTH sides (changes what is measured)
    categories=[...],         # per-item grouping key, for a breakdown
    ids=[...],                # per-item id, so rows stay identifiable
    variant_maps=[...],       # per-item {variant: canonical} equivalences
    normalizer=shared_norm,   # reuse one cache across several models
    workers=8,                # LLM concurrency
)
```

`r.report` is the full `ScoreReport` — per-row detail, per-category breakdown, rejected
LLM edits.

---

## The full API

`score_pairs()` is the layer underneath, when you have rows rather than two lists:

```python
from stt_api.evaluation import score_pairs, format_report

report = score_pairs([
    {"id": "call-001", "ref": "okay lah", "hyp": "ok la",   "category": "telephony"},
    {"id": "call-002", "ref": "dua puluh tiga", "hyp": "23", "category": "telephony"},
])

report.as_scored.wer          # the headline
report.normalized.wer         # the second reading
report.per_category()         # {"telephony": (Metrics(raw), Metrics(normalized))}
report.as_dict()              # JSON-serialisable, rows included
print(format_report(report))  # the same table the CLI prints
```

It accepts `Pair` objects, `(ref, hyp)` tuples, or dicts. Note that a **dict/tuple row is
`ref` first** — that is the on-disk convention for a results file, and it is why the
ergonomic `score()` wrapper exists with the argument order people actually say out loud.

Individual pieces are importable on their own:

```python
from stt_api.evaluation import (
    deterministic_normalize,  # the rules layer on one string
    normalize,                # tokenizer: NFKC + lowercase + strip punctuation
    levenshtein,              # edit distance over any sequence
    score_one,                # Metrics for one pair
    corpus_metrics, wer_cer,  # pooled totals
    validate,                 # is this edit convention-only?
    load_canonical,           # a dataset's declared variant map
    load_pairs, load_rows,    # read CSV / TSV / JSON / JSONL
)
```

`Metrics` carries counts, not just rates (`word_dist`, `ref_words`, `char_dist`,
`ref_chars`), and adding two `Metrics` gives the correct pooled rate for the union. That
is what lets you slice a corpus without re-scoring it.

---

## Modes

| mode | what it does | cost |
|---|---|---|
| `deterministic` **(default)** | rules only: numbers → digits, filler respells, non-speech tags dropped, NFKC/case/punctuation | free, offline |
| `llm` / `both` | one LLM call per unique text for the residue the rules cannot settle | ~2 calls per clip |
| `pair` | shows the model both sides — **a leakage study, not a number** | ~1 call per clip |

The deterministic layer captures roughly **90% of the recoverable gap** on Malaysian
call-centre audio; the LLM pass added about **0.08 pp** more. That ratio is why
`deterministic` is the default.

---

## Setting up the LLM pass (`.env`)

Only needed for `mode="llm"`, `"both"`, or `"pair"`. Nothing about any endpoint is baked
in — it speaks plain `POST {base_url}/chat/completions`, so vLLM, OpenRouter, the OpenAI
API and any internal proxy all work.

Three variables — copy [`.env.example`](.env.example) and fill it in:

```bash
cp stt_api/evaluation/.env.example stt_api/evaluation/.env
```

```bash
OPENAI_BASE_URL=https://your-endpoint.example.com/v1   # must end in /v1
OPENAI_API_KEY=sk-...                                  # sent as Bearer; non-empty even for a local vLLM
MODEL_NAME=your-model-id                               # or OPENAI_MODEL
```

`.env` is git-ignored (including this one, inside the package); `.env.example` is
tracked.

**From the process environment** (nothing else to do):

```python
from stt_api.evaluation import LLMClient, score

client = LLMClient.from_env()
r = score(hyps, refs, mode="both", client=client)
```

**From a `.env` file** — `load_dotenv` parses it and **never mutates `os.environ`**, so
importing this package can't disturb the host application:

```python
client = LLMClient.from_env(env_file="stt_api/evaluation/.env")
```

**Explicitly**, when the config lives somewhere else entirely:

```python
client = LLMClient(
    base_url="https://your-endpoint.example.com/v1",
    api_key="sk-...",
    model="your-model-id",
    timeout=120, retries=3,
)
assert client.configured        # False when base_url or api_key is missing
```

Precedence is **`.env` file < process environment < explicit argument**. `score()` raises
if an LLM mode is requested without a configured client, rather than silently falling back
to the rules and reporting a number that was never normalized.

`HF_TOKEN` is separate and only matters for `load_canonical` against a gated dataset.

---

## Caching, and why it is not just a speed feature

```python
from stt_api.evaluation import Normalizer, score

norm = Normalizer("both", client=client, cache="llm_norm_cache.json")
model_a = score(hyps_a, refs, normalizer=norm)
model_b = score(hyps_b, refs, normalizer=norm)   # same refs → same normalization
norm.save()
```

The cache is keyed by text hash, so a reference string normalizes **identically across
every model you score against it**. Two arms can then never differ because their shared
reference was normalized two different ways. `cache=` also takes any mutable mapping — a
plain `dict`, or your own Redis-backed shim.

---

## Datasets with a declared variant map

Some benchmark sets ship per-row `variant → canonical` equivalences and their official
scorer folds them into both sides. If yours does, apply it or your baseline is not the
published number:

```python
from stt_api.evaluation import load_canonical, score

canon = load_canonical("Revolab/ASR-Benchmark-Public")       # {id: {variant: canonical}}
r = score(hyps, refs, ids=row_ids, variant_maps=[canon.get(i, {}) for i in row_ids])
```

Needs `pip install 'stt-api[evaluation]'` and, for a gated repo, `HF_TOKEN`.

---

## Command line

```bash
python -m stt_api.evaluation --self-test                      # offline checks, no data needed
python -m stt_api.evaluation --ref "dua puluh tiga ringgit" --hyp "RM23"
python -m stt_api.evaluation --input pairs.csv                # ref,hyp columns
python -m stt_api.evaluation --input rows.jsonl --ref-field reference --hyp-field text
python -m stt_api.evaluation --input pairs.csv --mode both --env-file .env
python -m stt_api.evaluation --input pairs.csv --canonical Revolab/ASR-Benchmark-Public
```

```
samples 820   mode=deterministic
                   WER%     CER%
as scored         12.43     5.25
normalized        11.71     5.05
recovered         +0.73    +0.20   (6% of the WER was convention)

category            as scored  normalized        Δ
read-speech              4.95        4.64    -0.31
short-inputs            18.69       16.19    -2.50
telephony               23.89       17.06    -6.83
```

`--out report.json` writes the whole thing, rows included. `--top N` controls how many
example recoveries are printed.

---

## Reading the numbers honestly

- **Report `as_scored`. Quote `normalized` beside it, never instead of it.** They are two
  different metrics; a normalized WER compared against a figure from an ordinary scorer is
  not a comparison.
- **The delta is an offset, not a re-ranking.** Measured across three models it was
  −1.80 / −1.99 / −2.04 pp and preserved their order exactly. It gives a fairer number for
  the same model — it never picks a better model. **Don't use it to choose a checkpoint.**
- **It is a lower bound on convention error.** Normalizing fixes a *mismatch*; most filler
  error is a *deletion* (the reference writes `aa`, the hypothesis has nothing). No rewrite
  of two strings makes a missing token match — only `drop_fillers` does, and that changes
  what is being measured.
- **Mandarin WER is near-meaningless here.** The word split is whitespace-based, so a
  Chinese utterance is roughly one token. Read CER for `zh`.
- **Rank fixes by error mass (`word_dist`), never by per-item WER.** Per-item WER sorts
  short items to the top — one word, one mismatch, 100% — and will tell you a category is
  mostly artefact when it is not.

## Tests

```bash
pytest tests/test_evaluation.py         # 48 offline checks
python -m stt_api.evaluation --self-test
```

Both run the same cases (`selftest.py`), so they cannot drift apart.
