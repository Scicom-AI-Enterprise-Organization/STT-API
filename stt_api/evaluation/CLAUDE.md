# CLAUDE.md — `stt_api/evaluation/`

Guidance for working on the convention-normalized WER/CER scorer. Read this before
editing anything here. `README.md` is the usage doc; this file is the list of things
that will silently corrupt the measurement if you change them.

## What this is

Score `(hypothesis, reference)` pairs twice: **as an ordinary scorer charges them**, and
again **with spelling conventions folded away** — so a user can see how much of a WER is
the model and how much is the two transcripts disagreeing about how to write the same
words (`23` vs `dua puluh tiga`, `okay` vs `ok`, `[laugh]` vs nothing).

Ported from the STT benchmark harness (`ucc_ai_research/evaluation/stt/llm_normalize_score.py`),
which stays a self-contained single file on purpose. **The two are verified byte-identical
in output** across six arms (three per-sample dumps × `--drop-fillers` on/off). If you
change scoring behaviour here, that parity is gone — say so out loud rather than letting
someone discover it by comparing two numbers that no longer mean the same thing.

## Layout

| file | role |
|---|---|
| `metrics.py` | tokenizer, edit distance, pooled `Metrics`. **No other layer imported.** |
| `deterministic.py` | the free rules: numbers, respells, non-speech tags |
| `validation.py` | the guard — is this edit convention-only? |
| `llm.py` | OpenAI-compatible client + the two prompts + `.env` reading |
| `normalizer.py` | `Normalizer`: mode dispatch, cache, validation |
| `score.py` | `Pair` / `ScoreReport` / `score_pairs()` |
| `simple.py` | `score(hypothesis, reference)` — the ergonomic wrapper |
| `canonical.py` | dataset variant maps (the **only** third-party imports, and lazy) |
| `loaders.py`, `report.py`, `cli.py`, `selftest.py` | file reading, text rendering, CLI, offline checks |

## Invariants

### 1. The LLM sees ONE side per call

Given both texts, a model edits the hypothesis toward the reference. WER collapses, and
nothing in the output looks wrong. Measured on Revolab 820 against a control that pairs
each reference with a *different* clip's hypothesis:

| | WER | vs independent |
|---|---|---|
| independent (each side alone) | 8.61 | — |
| pair (sees its own partner) | 8.12 | −0.49 |
| pair (sees a stranger) | 8.59 | −0.03 |

Convention knowledge is corpus-wide, so the honest part is what survives shuffling:
**+0.03 pp. The other 0.46 pp — 94% — is leakage.** The validator was not asleep; it
rejected *more* in pair mode (229 vs 182). It blocks content edits but cannot stop the
model choosing, among several legal convention-preserving forms, the one matching the
other side. `mode="pair"` exists to study that. **Never quote a number from it.**

### 2. Every LLM edit is validated, and rejection is the normal case

13% (723/5,700) were rejected in a full run and reverted to the original. Allowed:
identity, whitelisted respell, letter-preserving join *or* split, value-preserving number
rewrite. A run that rejects nothing is more suspicious than one that rejects a lot.

### 3. The cache stores RAW model output; validation runs on every read

Cache the post-validation text and a fix to the validator can never be picked up on
re-run — the rejected edit has already been overwritten by its own input.
`test_normalizer_caches_raw_output_not_post_validation_text` pins this.

The cache is also keyed by text hash so a shared reference normalizes identically across
every model compared against it. That is correctness, not speed.

### 4. `metrics.py` is a byte-for-byte copy of the harness scorer

Same tokenizer, same edit distance, same pooling. If you change one, change both or the
two stop being comparable. Corpus WER is **total distance over total reference length**,
never a mean of per-clip rates.

### 5. `load_dotenv` must never mutate `os.environ`

This is a library inside a serving package. A module that edits the process environment
on import breaks the host application. Config precedence is `.env` file < process env <
explicit argument, resolved in `LLMClient.from_env`.

`.env.example` is the tracked template (`OPENAI_BASE_URL` / `OPENAI_API_KEY` /
`MODEL_NAME`, plus `HF_TOKEN` for gated datasets); the real `.env` beside it is
git-ignored by the root `.gitignore`'s bare `.env` pattern, which matches at any depth.
Keep the template in sync with what `LLMClient.from_env` actually reads — a stale
template is how someone ends up debugging an endpoint that was never configured.

### 6. Standard library only, except `canonical.py`

The whole package — including the LLM pass, which speaks `urllib` — must import with no
third-party dependency. `load_canonical` needs `huggingface_hub` + `pyarrow` and imports
them **inside the function**, raising a message naming the extra. Keep it that way: a
scorer that cannot be imported without pyarrow is a scorer nobody embeds.

### 7. Argument order in `score()` is `(hypothesis, reference)`

Chosen because it is the order people say out loud. WER is not symmetric, so a swap
returns a wrong number and raises nothing. **Do not "fix" it to match `score_pairs()`**,
which is `ref`-first because that is the on-disk order of a results row. Both orders are
deliberate; both are documented; changing either silently corrupts existing callers.

## Things that look like bugs and are not

- **`FILLERS` is wrong in both directions and is kept anyway.** It is
  `{herm, ah, la, ya, eh, oh, lo, ma, kan}`: it omits the top hitters (`aa` 52 errors,
  `uh` 42, `um` 19, `haa`, `hmm`, `[event]` tags) and it includes `ya` (*yes*), `kan` (a
  question tag) and `la` — which makes yes/no unscoreable in an IVR transcript. A correct
  interjection set recovers **1.52 pp** where this one recovers **0.35**. It is verbatim
  from the harness so recorded numbers stay reproducible. Callers override it with
  `fillers=`. **Changing the default moves numbers a provider would publish — declare it,
  don't quietly merge it.**
- **`drop_fillers` is off by default.** It changes *what is measured*, not just the
  spelling. It is also the only lever that touches filler **deletions** — 195 of 244
  filler errors are the reference writing `aa` where the hypothesis has nothing, and no
  rewrite of two strings makes a missing token match.
- **A blank hypothesis is scored, not skipped.** `""` costs the full reference as
  deletions. That is a real ASR failure mode (some endpoints return `{"text": ""}` for
  audio they dislike) and dropping those rows flatters the model.
- **Number runs of bare single digits concatenate.** `kosong satu dua` → `012`, not `12`
  and not `0 1 2`: it is an account or phone number. Which convention wins matters far
  less than applying the same one to both sides.
- **`_split_of` re-expands respells.** `baguslah` → `bagus la` reads as letter-losing
  until you notice the model also applied `lah`→`la`. This was 25 of 25 rejections on the
  first real run.

## What this package must not become

- **Not a second scorer.** `metrics.py` is the only place a distance is computed.
  `report.py` formats, `simple.py` reshapes — neither may re-align or re-tokenize.
- **Not a phonetic matcher.** Distance cannot decide proper nouns at any threshold:
  `fallujah`/`faluyah` (same city, misspelt) and `bordentown`/`bordertown` (different
  cities) are indistinguishable to any distance measure — what separates them is whether
  the referent is the same, which is semantics. Entity equivalence must be an explicit
  hand-declared list, which is what `canonical.py` loads.
- **Not a Malay-vs-English error splitter.** A previous attempt classified with
  `^[a-z]+$ and len > 3`, which calls *filem*, *komuniti* and *cerita* English. It is
  retracted. Use `categories=` (per-category breakdown), which needs no language ID.

## Testing

```bash
pytest tests/test_evaluation.py          # 48 offline checks, no network, no key
python -m stt_api.evaluation --self-test # the same deterministic + validator cases
```

Both draw their cases from `selftest.py`, so they cannot drift. Any change to
`deterministic.py` or `validation.py` must keep every case in both directions passing:
the validator has to accept legal convention edits *and* catch inserted, deleted,
substituted and value-changing ones.
