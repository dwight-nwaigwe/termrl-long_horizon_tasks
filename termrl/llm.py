"""Thin async client for any OpenAI-compatible chat endpoint (vLLM, OpenAI, OpenRouter, ...)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Reply:
    text: str
    finish_reason: str | None
    # Only filled when the LLM was created with logprobs=True and the server is vLLM:
    token_ids: list[int] | None = None   # exact generated tokens
    logprobs: list[float] | None = None  # log mu(token) under the rollout policy


class ContextTooLong(Exception):
    """The conversation no longer fits the model's context window."""


def parse_token_logprobs(choice: Any) -> tuple[list[int] | None, list[float] | None]:
    """Read vLLM's per-token logprobs (requested with return_tokens_as_token_ids)."""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        return None, None
    ids, lps = [], []
    for item in content:
        tok = item.token if hasattr(item, "token") else item["token"]
        val = item.logprob if hasattr(item, "logprob") else item["logprob"]
        if not isinstance(tok, str) or not tok.startswith("token_id:"):
            return None, None  # server ignored return_tokens_as_token_ids
        ids.append(int(tok.split(":", 1)[1]))
        lps.append(float(val))
    return ids, lps


class LLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        max_tokens_field: str = "max_tokens",
        timeout: float = 600,
        logprobs: bool = False,
    ):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            timeout=timeout,
            max_retries=3,
        )
        self.model = model
        self.extra_body = dict(extra_body or {})
        self.max_tokens_field = max_tokens_field
        self.want_logprobs = logprobs
        if logprobs:
            self.extra_body["return_tokens_as_token_ids"] = True

    async def chat(self, messages: list[dict], temperature: float = 1.0, max_tokens: int = 2048) -> Reply:
        import openai

        kwargs: dict[str, Any] = {self.max_tokens_field: max_tokens}
        if self.want_logprobs:
            kwargs["logprobs"] = True
        try:
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                extra_body=self.extra_body or None,
                **kwargs,
            )
        except openai.BadRequestError as e:
            msg = str(e).lower()
            if "context" in msg or "maximum" in msg or "too long" in msg:
                raise ContextTooLong(str(e)) from e
            raise
        choice = r.choices[0]
        ids, lps = parse_token_logprobs(choice) if self.want_logprobs else (None, None)
        return Reply(text=choice.message.content or "", finish_reason=choice.finish_reason, token_ids=ids, logprobs=lps)


def make_llm(base_url: str, model: str, api_key_env: str | None = None, no_thinking: bool = False, **kw) -> LLM:
    api_key = os.environ.get(api_key_env) if api_key_env else None
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if no_thinking else None
    return LLM(base_url=base_url, model=model, api_key=api_key, extra_body=extra, **kw)
