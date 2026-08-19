"""Tests for stt_api.evaluation — the convention-normalized WER/CER scorer.

Offline: no network, no key, no dataset. The deterministic and validator cases
come from `stt_api.evaluation.selftest` so the shipped `--self-test` and this
suite can never disagree.
"""

import json

import pytest

from stt_api.evaluation import (
    LLMClient,
    Metrics,
    Normalizer,
    Pair,
    corpus_metrics,
    deterministic_normalize,
    load_dotenv,
    normalize,
    score,
    score_one,
    score_pairs,
    validate,
    wer_cer,
)
from stt_api.evaluation.selftest import (
    CORRUPTING_EDITS,
    LEGAL_EDITS,
    NORMALIZE_CASES,
)


# ------------------------------------------------------------ deterministic --
@pytest.mark.parametrize("src,want", NORMALIZE_CASES)
def test_deterministic_normalize(src, want):
    assert deterministic_normalize(src) == want


@pytest.mark.parametrize("orig,cand", LEGAL_EDITS)
def test_validator_accepts_convention_edits(orig, cand):
    assert validate(orig, cand) == []


@pytest.mark.parametrize("orig,cand", CORRUPTING_EDITS)
def test_validator_rejects_content_edits(orig, cand):
    """Insert, delete, substitute, or change a value — all must be caught.

    This is the guard that keeps the metric honest; if it ever goes quiet, an LLM
    editing the hypothesis toward the reference stops being visible.
    """
    assert validate(orig, cand)


def test_drop_fillers_removes_from_the_text():
    assert deterministic_normalize("ah okay lah saya tahu") == "ah ok la saya tahu"
    # `ok` survives: it is NOT in the default FILLERS set, which is documented as
    # wrong in both directions and kept only so recorded numbers stay reproducible.
    assert deterministic_normalize("ah okay lah saya tahu", drop_fillers=True) == "ok saya tahu"


def test_custom_filler_set_overrides_the_default():
    """`ya` (yes) is in the default set and should be removable from it."""
    assert deterministic_normalize("ya betul", drop_fillers=True) == "betul"
    assert deterministic_normalize("ya betul", drop_fillers=True, fillers={"herm"}) == "ya betul"


# -------------------------------------------------------------------- metrics --
def test_corpus_wer_is_pooled_not_averaged():
    """A long clip must outweigh a short one — the mean of rates would not."""
    items = [("a b c d e f g h i j", "a b c d e f g h i j"), ("x", "y")]
    m = corpus_metrics(items)
    assert m.word_dist == 1 and m.ref_words == 11
    assert m.wer == pytest.approx(1 / 11)          # not (0.0 + 1.0) / 2


def test_metrics_add_gives_the_union_rate():
    a, b = score_one("a b", "a b"), score_one("c", "d")
    assert (a + b).wer == pytest.approx(1 / 3)


def test_variant_map_folds_both_sides():
    assert score_one("saya nak filem", "saya nak film").wer == pytest.approx(1 / 3)
    folded = score_one("saya nak filem", "saya nak film", {"film": "filem"})
    assert folded.wer == 0.0


def test_empty_reference_does_not_divide_by_zero():
    assert score_one("", "").wer == 0.0
    assert score_one("", "hallucinated text").word_dist == 2


def test_blank_hypothesis_scores_as_all_deletions():
    """The endpoint failure mode: a silent "" must cost the full reference."""
    m = score_one("satu dua tiga", "")
    assert m.word_dist == m.ref_words == 3


def test_wer_cer_tuple_matches_the_harness_signature():
    w, c = wer_cer([("saya nak bayar", "saya nak bayaq")])
    assert w == pytest.approx(1 / 3)
    assert 0 < c < w


def test_tokenizer_keeps_digits_and_drops_punctuation():
    assert normalize("RM23, betul?") == ["rm23", "betul"]


# ---------------------------------------------------------------------- score --
def test_score_single_pair_returns_lists():
    r = score("saya bayar 23 ringgit", "saya bayar dua puluh tiga ringgit")
    assert r.wer == [pytest.approx(0.5)]
    assert r.normalized_wer == [0.0]
    assert r.normalized_hypothesis == ["saya bayar 23 ringgit"]
    assert r.normalized_reference == ["saya bayar 23 ringgit"]
    assert len(r) == 1


def test_score_list_pairs():
    r = score(["okay lah, saya nak bayar", "nombor 012"],
              ["ok la saya nak bayar", "nombor kosong satu dua"])
    assert r.normalized_wer == [0.0, 0.0]
    assert r.corpus_wer > 0                      # raw disagrees, normalized does not
    assert r.normalized_corpus_wer == 0.0
    assert r.recovered_wer == pytest.approx(r.corpus_wer)


def test_score_argument_order_is_hypothesis_first():
    """WER is not symmetric: swapping the arguments changes the number."""
    a = score("satu dua", "satu dua tiga empat")
    b = score("satu dua tiga empat", "satu dua")
    assert a.wer != b.wer


def test_score_length_mismatch_raises():
    with pytest.raises(ValueError):
        score(["a", "b"], ["a"])


def test_score_variant_maps_are_per_item():
    r = score(["saya nak film"], ["saya nak filem"], variant_maps=[{"film": "filem"}])
    assert r.wer == [0.0]


def test_score_empty_input():
    r = score([], [])
    assert r.wer == [] and r.corpus_wer == 0.0


# --------------------------------------------------------------- score_pairs --
def test_score_pairs_accepts_tuples_dicts_and_pairs():
    rows = [("ref one", "ref one"), {"ref": "ref two", "hyp": "ref two"},
            Pair(ref="ref three", hyp="ref three")]
    report = score_pairs(rows)
    assert len(report.rows) == 3 and report.as_scored.wer == 0.0


def test_per_category_breakdown_is_pooled():
    report = score_pairs([
        {"ref": "a b c d", "hyp": "a b c d", "category": "clean"},
        {"ref": "x y", "hyp": "p q", "category": "noisy"},
    ])
    cats = report.per_category()
    assert cats["clean"][0].wer == 0.0
    assert cats["noisy"][0].wer == 1.0


def test_report_as_dict_is_json_serialisable():
    report = score_pairs([("okay lah", "ok la")])
    blob = json.dumps(report.as_dict())
    assert "normalized" in blob


# ---------------------------------------------------------------- normalizer --
def test_normalizer_cache_is_shared_across_arms(tmp_path):
    """A reference shared by two models must normalize identically in both."""
    cache = tmp_path / "cache.json"
    a = Normalizer(cache=cache)
    assert a("dua puluh tiga") == "23"
    a.save()
    b = Normalizer(cache=cache)
    assert b("dua puluh tiga") == "23"
    assert json.loads(cache.read_text()) == {}   # deterministic mode bills no LLM


def test_llm_mode_reads_the_environment_when_no_client_is_passed(monkeypatch):
    """`export OPENAI_*` then `score(..., mode="llm")` must just work.

    This package gets installed with `pip install git+...`, where there is nowhere
    sensible to put a .env — exported variables have to be enough on their own.
    """
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("MODEL_NAME", "my-model")
    norm = Normalizer("llm")                     # no client argument anywhere
    assert norm.client.base_url == "https://example.com/v1"
    assert norm.client.model == "my-model"


def test_llm_mode_raises_when_nothing_is_configured(monkeypatch):
    """It must NOT quietly fall back to the rules and call the result normalized."""
    for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        Normalizer("both")
    with pytest.raises(ValueError):
        Normalizer("both", client=LLMClient())   # explicitly unconfigured
    with pytest.raises(ValueError):
        score("a", "b", mode="llm")


def test_deterministic_mode_never_touches_the_environment(monkeypatch):
    """The default must work with no endpoint configured at all."""
    for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert score("okay lah", "ok la").normalized_wer == [0.0]


def test_normalizer_keeps_the_original_when_validation_fails():
    """The whole point: a content edit from the model must not reach the score."""
    class FakeClient(LLMClient):
        def normalize(self, text):
            return "saya nak bayar bil sekarang juga"   # inserted words

    norm = Normalizer("llm", client=FakeClient(base_url="http://x/v1", api_key="k"),
                      cache={})
    assert norm("saya nak bayar bil") == "saya nak bayar bil"
    assert len(norm.rejected) == 1
    assert "inserted" in norm.rejected[0].violations[0]


def test_normalizer_caches_raw_output_not_post_validation_text():
    """Cache the validated text instead and a validator fix can never be picked up."""
    cache = {}
    class FakeClient(LLMClient):
        def normalize(self, text):
            return text + " extra"

    norm = Normalizer("llm", client=FakeClient(base_url="http://x/v1", api_key="k"),
                      cache=cache)
    norm("saya nak bayar")
    assert list(cache.values()) == ["saya nak bayar extra"]


def test_normalizer_survives_a_dead_endpoint():
    class DeadClient(LLMClient):
        def normalize(self, text):
            raise RuntimeError("connection refused")

    norm = Normalizer("llm", client=DeadClient(base_url="http://x/v1", api_key="k"),
                      cache={})
    assert norm("okay lah") == "ok la"        # falls through to the rules
    assert norm.errors == 1


# ---------------------------------------------------------------------- env --
def test_load_dotenv_does_not_touch_os_environ(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('OPENAI_BASE_URL=https://example.com/v1\nOPENAI_API_KEY="sk-x"\n# comment\n')
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    env = load_dotenv(p)
    assert env["OPENAI_BASE_URL"] == "https://example.com/v1"
    assert env["OPENAI_API_KEY"] == "sk-x"          # quotes stripped
    import os
    assert "OPENAI_BASE_URL" not in os.environ


def test_llm_client_from_env_precedence(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("OPENAI_BASE_URL=https://from-file/v1\nMODEL_NAME=file-model\n")
    monkeypatch.setenv("MODEL_NAME", "env-model")
    c = LLMClient.from_env(env_file=p)
    assert c.base_url == "https://from-file/v1"     # file fills what env lacks
    assert c.model == "env-model"                   # process env wins over the file
    c2 = LLMClient.from_env(env_file=p, model="explicit")
    assert c2.model == "explicit"                   # an argument wins over both
