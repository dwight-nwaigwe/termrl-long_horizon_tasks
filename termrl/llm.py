"""Thin async client for any OpenAI-compatible chat endpoint (vLLM, OpenAI, OpenRouter, ...)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Reply:
    text: str
    finish_reason: str | None


class ContextTooLong(Exception):
    """The conversation no longer fits the model's context window."""


class LLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        max_tokens_field: str = "max_tokens",
        timeout: float = 600,
    ):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            timeout=timeout,
            max_retries=3,
        )
        self.model = model
        self.extra_body = extra_body or None
        self.max_tokens_field = max_tokens_field

    async def chat(self, messages: list[dict], temperature: float = 1.0, max_tokens: int = 2048) -> Reply:
        import openai

        kwargs: dict[str, Any] = {self.max_tokens_field: max_tokens}
        try:
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                extra_body=self.extra_body,
                **kwargs,
            )
        except openai.BadRequestError as e:
            msg = str(e).lower()
            if "context" in msg or "maximum" in msg or "too long" in msg:
                raise ContextTooLong(str(e)) from e
            raise
        choice = r.choices[0]
        return Reply(text=choice.message.content or "", finish_reason=choice.finish_reason)


def make_llm(base_url: str, model: str, api_key_env: str | None = None, no_thinking: bool = False, **kw) -> LLM:
    api_key = os.environ.get(api_key_env) if api_key_env else None
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if no_thinking else None
    return LLM(base_url=base_url, model=model, api_key=api_key, extra_body=extra, **kw)
