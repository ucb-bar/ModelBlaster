"""Shared LLM budget tracker + BudgetExceeded exception.

Used by `pipeline.bedrock_client`, `pipeline.gemini_client`, and
`pipeline.claude_code_client` so the hard-budget-kill semantics are
identical across providers. The arm driver passes `--max-usd N` which
plumbs through as the `MODELBLASTER_MAX_USD` env var; clients
instantiate a `BudgetTracker` at init, ``account()`` after every paid
call, and ``check_before_call()`` before the next one.

Math matches `benchmarks/tools/cost_monitor.price_call` exactly: per
the AWS Bedrock + Anthropic + Gemini prompt-caching docs, the
provider's `inputTokens` field is the non-cached portion only and
cache_read / cache_write are billed separately. So we do NOT subtract
cached counts from input_tokens to derive uncached -- the API already
split them. Symmetric semantics across all three providers.

On trip the tracker prints a single ``MODELBLASTER_BUDGET_EXCEEDED:``
line to stderr (loud + greppable by the arm driver post-hoc) and
flips into "next call raises" mode. The current call's response is
still returned -- we don't void what we already paid for, but no
further calls go out.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional


class BudgetExceeded(RuntimeError):
    """Raised by LLM clients when the cumulative spend has crossed
    the cap set via ``--max-usd`` / ``MODELBLASTER_MAX_USD``. Arm
    drivers catch this and write ``exit_status=budget_exceeded`` into
    ``run.json``. The message includes cumulative + cap so the trace
    in the captured stderr is legible without further context."""


def _default_pricing_path() -> Path:
    return (Path(__file__).resolve().parents[1]
            / "benchmarks" / "config" / "pricing.yaml")


class BudgetTracker:
    """Per-client stateful budget tracker. Construction is cheap;
    pricing.yaml is loaded lazily on the first call that needs rates.

    Parameters
    ----------
    max_usd : Optional[float]
        Cap in dollars. If None, falls back to ``MODELBLASTER_MAX_USD``
        env var. If neither is set, the tracker is a no-op (``account``
        records cumulative for visibility but never trips).
    pricing : Optional[dict]
        Pricing table override (same shape as pricing.yaml). When None,
        loads ``benchmarks/config/pricing.yaml`` lazily on first need.
    label : str
        Free-form provider tag for the stderr trip marker
        ("bedrock" / "gemini" / "claude_code").
    """

    def __init__(self, max_usd: Optional[float] = None,
                 pricing: Optional[dict] = None,
                 label: str = "unknown"):
        env_max = os.environ.get("MODELBLASTER_MAX_USD")
        if max_usd is None and env_max:
            try:
                max_usd = float(env_max)
            except ValueError:
                max_usd = None
        self.max_usd: Optional[float] = max_usd
        self.label = label
        self._pricing_override = pricing
        self._pricing_cache: Optional[dict] = None
        self.cumulative_cost_usd: float = 0.0
        self._tripped: bool = False
        self._trip_announced: bool = False

    # ───────────────────── public API ─────────────────────

    def check_before_call(self) -> None:
        """Raise ``BudgetExceeded`` if a prior call already crossed
        the cap. Cheap to call; designed to live at the top of every
        ``converse()`` so we fail fast before another paid call."""
        if self._tripped:
            raise BudgetExceeded(
                f"[{self.label}] cumulative cost "
                f"${self.cumulative_cost_usd:.4f} already exceeds "
                f"--max-usd ${self.max_usd:.4f}; aborting before next call."
            )

    def account_usage(self, model_id: str,
                      input_tokens: int,
                      output_tokens: int,
                      cache_read_input_tokens: int = 0,
                      cache_write_input_tokens: int = 0) -> Optional[float]:
        """Price a usage record against pricing.yaml and accumulate.
        Returns the per-call cost (or None when the model isn't in
        the pricing table). Marks the tracker tripped if cumulative
        crosses ``max_usd`` -- the NEXT ``check_before_call`` raises."""
        rates = self._rates_for_model(model_id)
        cost = self._price_from_rates(rates, input_tokens, output_tokens,
                                       cache_read_input_tokens,
                                       cache_write_input_tokens)
        return self._account_cost(model_id, cost)

    def account_prepriced(self, model_id: str,
                          cost_usd: float) -> float:
        """For providers (e.g. Claude Code) that return a pre-computed
        cost in their response. Skips the pricing.yaml lookup."""
        return self._account_cost(model_id, float(cost_usd)) or 0.0

    # ───────────────────── internal ─────────────────────

    def _account_cost(self, model_id: str,
                      cost: Optional[float]) -> Optional[float]:
        if cost is None:
            return None
        self.cumulative_cost_usd += cost
        if (self.max_usd is not None
                and self.cumulative_cost_usd >= self.max_usd):
            self._tripped = True
            if not self._trip_announced:
                print(
                    f"MODELBLASTER_BUDGET_EXCEEDED: "
                    f"provider={self.label} "
                    f"cumulative=${self.cumulative_cost_usd:.4f} "
                    f"cap=${self.max_usd:.4f} "
                    f"model={model_id}",
                    file=sys.stderr, flush=True,
                )
                self._trip_announced = True
        return cost

    def _rates_for_model(self, model_id: str) -> Optional[dict]:
        if self._pricing_override is not None:
            entry = (self._pricing_override.get("models") or {}).get(model_id)
            if entry and not entry.get("placeholder"):
                return entry
            return None
        if self._pricing_cache is None:
            try:
                import yaml as _yaml
                p = _default_pricing_path()
                if p.exists():
                    self._pricing_cache = _yaml.safe_load(p.read_text())
                else:
                    self._pricing_cache = {}
            except Exception:
                self._pricing_cache = {}
        entry = (self._pricing_cache.get("models") or {}).get(model_id)
        if entry and not entry.get("placeholder"):
            return entry
        return None

    @staticmethod
    def _price_from_rates(rates: Optional[dict],
                          input_tokens: int, output_tokens: int,
                          cache_read: int, cache_write: int
                          ) -> Optional[float]:
        if rates is None:
            return None
        r_in = rates.get("input_uncached")
        r_cache_read = rates.get("cache_read")
        r_cache_write_5m = rates.get("cache_write_5m")
        r_out = rates.get("output")
        if r_in is None or r_out is None:
            return None
        total = float(input_tokens) * r_in / 1_000_000.0
        if cache_read:
            total += float(cache_read) * (r_cache_read if r_cache_read
                                          is not None else r_in) \
                / 1_000_000.0
        if cache_write:
            total += float(cache_write) * (r_cache_write_5m
                                           if r_cache_write_5m is not None
                                           else r_in) / 1_000_000.0
        total += float(output_tokens) * r_out / 1_000_000.0
        return total
