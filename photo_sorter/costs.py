from __future__ import annotations

from dataclasses import dataclass


# OpenAI model prices verified on 2026-09-02. Values are USD per 1M tokens.
# The exchange rate is deliberately fixed so the desktop app remains offline and the
# displayed amount is reproducible. The UI labels it as an estimate.
JPY_PER_USD = 150.0
PRICING_UPDATED_ON = "2026-09-02"


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float
    cache_write_multiplier: float = 1.25


@dataclass(frozen=True)
class CostEstimate:
    usd: float
    jpy: float
    exchange_rate: float


MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-5.6": ModelPricing(4.00, 0.40, 20.00),
    "gpt-5.6-sol": ModelPricing(4.00, 0.40, 20.00),
    "gpt-5.6-terra": ModelPricing(2.00, 0.20, 12.00),
    "gpt-5.6-luna": ModelPricing(0.20, 0.02, 1.20),
}


def estimate_api_cost(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    cache_write_input_tokens: int,
    output_tokens: int,
    *,
    jpy_per_usd: float = JPY_PER_USD,
) -> CostEstimate | None:
    """Estimate token charges; return None rather than guessing an unknown model price."""
    total_tokens = max(0, input_tokens) + max(0, output_tokens)
    if total_tokens == 0:
        return CostEstimate(usd=0.0, jpy=0.0, exchange_rate=jpy_per_usd)

    pricing = MODEL_PRICING.get(model.strip().lower())
    if pricing is None:
        return None

    billed_input = max(0, input_tokens)
    cached = min(max(0, cached_input_tokens), billed_input)
    cache_write = min(max(0, cache_write_input_tokens), billed_input - cached)
    regular = billed_input - cached - cache_write
    usd = (
        regular * pricing.input_usd_per_million
        + cached * pricing.cached_input_usd_per_million
        + cache_write
        * pricing.input_usd_per_million
        * pricing.cache_write_multiplier
        + max(0, output_tokens) * pricing.output_usd_per_million
    ) / 1_000_000
    return CostEstimate(usd=usd, jpy=usd * jpy_per_usd, exchange_rate=jpy_per_usd)


def format_cost_estimate(estimate: CostEstimate | None, model: str) -> str:
    if estimate is None:
        return f"推定API料金は算出不可（{model} の単価未登録）"
    return (
        f"推定API料金 約{estimate.jpy:.2f}円"
        f"（${estimate.usd:.4f}、1ドル={estimate.exchange_rate:g}円換算）"
    )
