from __future__ import annotations

import math
import os
import re
import unicodedata
from time import perf_counter

import aiohttp
from transformers import AutoTokenizer

from livekit.agents import LanguageCode, Plugin, llm, utils
from livekit.agents.inference_runner import _InferenceRunner

from livekit.plugins.turn_detector.base import MAX_HISTORY_TURNS, EOUModelBase, EOUPlugin, _EUORunnerBase
from livekit.plugins.turn_detector.log import logger
from livekit.plugins.turn_detector.models import EOUModelType

REMOTE_INFERENCE_TIMEOUT = 2
VLLM_MODEL = "livekit/turn-detector"
MODEL_ID = "livekit/turn-detector"
REVISION = "v0.4.1-intl"
IM_END_TOKEN_ID = 151645

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, truncation_side="left")
    return _tokenizer


class _EUORunnerMultilingual(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_multilingual"

    @classmethod
    def model_type(cls) -> EOUModelType:
        return "multilingual"


class MultilingualModel(EOUModelBase):
    def __init__(self, *, unlikely_threshold: float | None = None):
        super().__init__(
            model_type="multilingual",
            unlikely_threshold=unlikely_threshold,
            load_languages=_remote_inference_url() is None,
        )

    def _inference_method(self) -> str:
        return _EUORunnerMultilingual.INFERENCE_METHOD

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        if not language:
            return None
        if _remote_inference_url():
            return self._unlikely_threshold if self._unlikely_threshold is not None else 0.5
        return await super().unlikely_threshold(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        if _remote_inference_url():
            return True
        return await self.unlikely_threshold(language) is not None

    async def predict_end_of_turn(
        self,
        chat_ctx: llm.ChatContext,
        *,
        timeout: float | None = 3,
    ) -> float:
        url = _remote_inference_url()
        if not url:
            return await super().predict_end_of_turn(chat_ctx, timeout=timeout)

        messages = chat_ctx.copy(
            exclude_function_call=True, exclude_instructions=True, exclude_empty_message=True
        ).truncate(max_items=MAX_HISTORY_TURNS)

        prompt = _build_chatml_prompt(messages)

        request = {
            "model": VLLM_MODEL,
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": 1,
            "allowed_token_ids": [IM_END_TOKEN_ID],
        }

        started_at = perf_counter()
        async with utils.http_context.http_session().post(
            url=url,
            json=request,
            timeout=aiohttp.ClientTimeout(total=REMOTE_INFERENCE_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            probability = _extract_eot_probability(data)
            logger.debug(
                "eou prediction",
                extra={
                    "eou_probability": probability,
                    "duration": perf_counter() - started_at,
                },
            )
            return probability


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text.lower())
    text = "".join(
        ch for ch in text
        if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"])
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_chatml_prompt(chat_ctx: llm.ChatContext) -> str:
    # Normalize and merge consecutive same-role messages
    normalized: list[dict[str, str]] = []
    last: dict[str, str] | None = None
    for msg in chat_ctx.messages():
        if msg.role not in ("user", "assistant"):
            continue
        content = _normalize_text(msg.text_content or "")
        if not content:
            continue
        role = msg.role
        if last and last["role"] == role:
            last["content"] += f" {content}"
        else:
            entry = {"role": role, "content": content}
            normalized.append(entry)
            last = entry

    text = _get_tokenizer().apply_chat_template(
        normalized, add_generation_prompt=False, add_special_tokens=False, tokenize=False
    )
    # Remove the last <|im_end|> so the model predicts whether the turn is over
    ix = text.rfind("<|im_end|>")
    if ix != -1:
        text = text[:ix]
    logger.info("turn detector input prompt", extra={"prompt": text})
    return text


def _extract_eot_probability(data: dict) -> float:
    try:
        token_logprob = data["choices"][0]["logprobs"]["token_logprobs"][0]
        probability = math.exp(token_logprob)
        logger.info(
            "turn detector output",
            extra={"token_logprob": token_logprob, "probability": probability},
        )
        return probability
    except Exception:
        logger.warning("turn detector failed to extract probability", extra={"raw_data": data})
        return 1.0


def _remote_inference_url() -> str | None:
    url_base = os.getenv("LIVEKIT_REMOTE_EOT_URL")
    if not url_base:
        return None
    return f"{url_base}/v1/completions"


if not _remote_inference_url():
    _InferenceRunner.register_runner(_EUORunnerMultilingual)
Plugin.register_plugin(EOUPlugin(_EUORunnerMultilingual))
