"""DPPO core: group advantages, the divergence trust-region mask, and turning rollouts into token samples.

Reference: Qi et al., "Rethinking the Trust Region in LLM Reinforcement Learning" (arXiv 2602.04879),
the RL algorithm used by tmax.

For each sampled token a_t with rollout-policy probability mu = mu(a_t|s_t) and current
policy probability p = pi(a_t|s_t), ratio r = p / mu:

  Binary-TV:  D = |mu - p|
  Binary-KL:  D = mu*log(mu/p) + (1-mu)*log((1-mu)/(1-p))

  mask = 0  if (A > 0 and r > 1 and D > delta) or (A < 0 and r < 1 and D > delta), else 1
  loss = - sum_t mask_t * r_t * A

Only updates that move the policy *further* outside the trust region are blocked; updates
back toward the rollout policy are always allowed. Advantages follow GRPO without std
normalization: A = reward - mean(reward of the same task's group).

This module has no torch dependency so the logic can be unit-tested anywhere; the array
functions work on numpy arrays or torch tensors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

EPS = 1e-6


def group_advantages(records: list[dict]) -> tuple[list[tuple[dict, float]], dict]:
    """Return [(record, advantage)] for groups with reward variance, plus stats."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("reward") is not None:
            groups[r["task_id"]].append(r)
    out, n_zero = [], 0
    for recs in groups.values():
        rewards = [r["reward"] for r in recs]
        if len(recs) < 2 or max(rewards) == min(rewards):
            n_zero += 1  # all solved or all failed: no learning signal ("active sampling" filter)
            continue
        mean = sum(rewards) / len(rewards)
        out.extend((r, r["reward"] - mean) for r in recs)
    stats = {"groups": len(groups), "groups_without_signal": n_zero, "episodes_used": len(out)}
    return out, stats


def dppo_mask_and_ratio(
    logp: Any,
    mu_logp: Any,
    adv: float,
    delta: float,
    divergence: str,
    exp: Callable,
    log: Callable,
    clip: Callable,
) -> tuple[Any, Any, Any]:
    """Compute (ratio, mask, divergence) for one sequence.

    logp: log pi of the sampled tokens (may carry gradients); mu_logp: rollout log mu.
    exp/log/clip: array functions (torch.exp, torch.log, torch.clamp or numpy equivalents).
    The mask is computed from detached values by the caller passing detached logp if needed.
    """
    log_ratio = clip(logp - mu_logp, -20.0, 20.0)
    ratio = exp(log_ratio)
    p = exp(logp)
    mu = exp(mu_logp)
    if divergence == "tv":
        d = abs(mu - p)
    elif divergence == "kl":
        p_c, mu_c = clip(p, EPS, 1 - EPS), clip(mu, EPS, 1 - EPS)
        d = mu_c * log(mu_c / p_c) + (1 - mu_c) * log((1 - mu_c) / (1 - p_c))
    else:
        raise ValueError(f"unknown divergence {divergence!r}")
    if adv > 0:
        blocked = (ratio > 1) & (d > delta)
    elif adv < 0:
        blocked = (ratio < 1) & (d > delta)
    else:
        blocked = d < -1  # all False
    mask = 1 - blocked * 1  # works for numpy bool arrays and torch bool tensors
    return ratio, mask, d


def build_samples(
    adv_records: list[tuple[dict, float]],
    tokenizer: Any,
    max_length: int,
    chat_kwargs: dict | None = None,
) -> tuple[list[dict], dict]:
    """One sample per assistant turn: prompt ids (as vLLM saw them), completion ids, rollout logprobs, advantage."""
    chat_kwargs = chat_kwargs or {}
    samples: list[dict] = []
    n_dropped_long = n_no_logprobs = 0
    for rec, adv in adv_records:
        msgs, meta = rec["messages"], rec.get("turn_meta", [])
        a_idx = 0
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            tm = meta[a_idx] if a_idx < len(meta) else {}
            a_idx += 1
            prompt_text = tokenizer.apply_chat_template(msgs[:i], add_generation_prompt=True, tokenize=False, **chat_kwargs)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            comp_ids, mu = tm.get("token_ids"), tm.get("logprobs")
            if not comp_ids or not mu or len(comp_ids) != len(mu):
                # No rollout token data (e.g. not served by vLLM): re-tokenize the reply text.
                full = tokenizer.apply_chat_template(msgs[: i + 1], tokenize=False, **chat_kwargs)
                comp_text = full[len(prompt_text):] if full.startswith(prompt_text) else m["content"]
                comp_ids = tokenizer(comp_text, add_special_tokens=False)["input_ids"]
                mu = None
                n_no_logprobs += 1
            if not comp_ids:
                continue
            if len(prompt_ids) + len(comp_ids) > max_length:
                n_dropped_long += 1
                continue
            samples.append({"prompt": prompt_ids, "completion": list(comp_ids), "mu": mu, "adv": adv, "task_id": rec["task_id"]})
    stats = {"samples": len(samples), "dropped_too_long": n_dropped_long, "turns_without_rollout_logprobs": n_no_logprobs}
    return samples, stats
