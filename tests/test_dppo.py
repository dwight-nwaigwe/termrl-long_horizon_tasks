"""DPPO logic tests that need no GPU or torch (numpy stands in for torch)."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from termrl.dppo import build_samples, dppo_mask_and_ratio, group_advantages
from termrl.llm import Reply, parse_token_logprobs


def mask_np(logp, mu, adv, delta=0.15, div="tv"):
    return dppo_mask_and_ratio(np.array(logp), np.array(mu), adv, delta, div, np.exp, np.log, np.clip)


def test_group_advantages_no_std_and_skip_zero_variance():
    recs = [{"task_id": "a", "reward": r} for r in (1, 0, 0, 0)] + [{"task_id": "b", "reward": 1.0}] * 4 \
        + [{"task_id": "c", "reward": None}]
    out, stats = group_advantages(recs)
    assert [round(a, 3) for _, a in out] == [0.75, -0.25, -0.25, -0.25]  # A = r - mean, no std division
    assert stats == {"groups": 2, "groups_without_signal": 1, "episodes_used": 4}


def test_mask_blocks_only_moves_further_outside_trust_region():
    mu = np.log([0.5, 0.5, 0.5, 0.5])
    # token probs under the current policy: far above, slightly above, slightly below, far below mu
    p = np.log([0.9, 0.55, 0.45, 0.1])
    ratio, mask, d = mask_np(p, mu, adv=+1.0)
    np.testing.assert_allclose(d, [0.4, 0.05, 0.05, 0.4], atol=1e-9)
    # A>0: block only where ratio>1 AND divergence > delta -> first token
    assert mask.tolist() == [0, 1, 1, 1]
    ratio, mask, d = mask_np(p, mu, adv=-1.0)
    # A<0: block only where ratio<1 AND divergence > delta -> last token
    assert mask.tolist() == [1, 1, 1, 0]
    ratio, mask, _ = mask_np(p, mu, adv=0.0)
    assert mask.tolist() == [1, 1, 1, 1]
    np.testing.assert_allclose(ratio, [1.8, 1.1, 0.9, 0.2])


def test_low_probability_tokens_are_not_overconstrained():
    # PPO would clip ratio 3.0; DPPO's TV divergence is tiny (0.001 -> 0.003), so it's allowed.
    _, mask, d = mask_np(np.log([0.003]), np.log([0.001]), adv=+1.0)
    assert mask.tolist() == [1] and d[0] < 0.01


def test_kl_variant():
    mu = np.log([0.5, 0.5])
    p = np.log([0.9, 0.52])
    _, mask, d = mask_np(p, mu, adv=+1.0, delta=0.05, div="kl")
    assert d[0] > 0.05 > d[1] and mask.tolist() == [0, 1]


class FakeTok:
    """Character-level tokenizer with a ChatML-style template."""

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False, **kw):
        s = "".join(f"<{m['role']}>{m['content']}</s>" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_build_samples_uses_rollout_tokens_and_falls_back():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": "ab"}, {"role": "user", "content": "out"},
            {"role": "assistant", "content": "cd"}]
    meta = [{"format_error": False, "token_ids": [97, 98, 9], "logprobs": [-0.1, -0.2, -0.3]}, {"format_error": False}]
    rec = {"task_id": "t", "messages": msgs, "turn_meta": meta}
    samples, stats = build_samples([(rec, 0.5)], FakeTok(), max_length=10_000)
    assert len(samples) == 2 and stats["turns_without_rollout_logprobs"] == 1
    s0, s1 = samples
    assert "".join(map(chr, s0["prompt"])) == "<system>S</s><user>U</s><assistant>"
    assert s0["completion"] == [97, 98, 9] and s0["mu"] == [-0.1, -0.2, -0.3] and s0["adv"] == 0.5
    assert "".join(map(chr, s1["completion"])) == "cd</s>" and s1["mu"] is None
    samples, stats = build_samples([(rec, 0.5)], FakeTok(), max_length=30)
    assert stats["dropped_too_long"] >= 1


def test_parse_vllm_token_logprobs():
    item = lambda t, lp: SimpleNamespace(token=t, logprob=lp)  # noqa: E731
    choice = SimpleNamespace(logprobs=SimpleNamespace(content=[item("token_id:12", -0.5), item("token_id:7", -1.0)]))
    assert parse_token_logprobs(choice) == ([12, 7], [-0.5, -1.0])
    choice = SimpleNamespace(logprobs=SimpleNamespace(content=[item("hello", -0.5)]))
    assert parse_token_logprobs(choice) == (None, None)
    assert parse_token_logprobs(SimpleNamespace(logprobs=None)) == (None, None)
