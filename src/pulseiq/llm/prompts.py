"""Prompt templates for strategy recommendations.

Kept separate from the router so prompts can be reviewed and changed without
touching provider or retry logic -- they are the part most likely to be edited,
and the part where a careless change silently degrades output quality.

The system prompt constrains the model to the evidence it is given. This project
measured that no model beats the naive price forecast (median h=1 MAE 0.0000),
so a recommendation confidently predicting price movements would contradict the
project's own findings. The prompt says so explicitly rather than hoping the
model infers restraint.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a pricing analyst. You give short, concrete advice \
grounded strictly in the data provided.

Rules:
- Use only the figures given. Do not invent prices, dates, or competitor names.
- If the data is too thin to support a recommendation, say so plainly.
- Do not predict specific future prices. Analysis of this dataset found that no \
model beats a naive "price stays the same" forecast, so precise predictions \
would be unfounded.
- Be concise: three or four sentences, no preamble, no bullet-point padding."""


def build_recommendation_prompt(
    *,
    product_name: str,
    n_observations: int,
    last_price: float,
    price_change_pct: float | None = None,
    sentiment_summary: str | None = None,
    context: str | None = None,
) -> str:
    """Assemble the user prompt from whatever evidence is available.

    Absent fields are omitted rather than filled with "unknown" or zero -- a
    model handed `sentiment: unknown` tends to reason about the unknown-ness,
    while a model not shown the field simply does not discuss sentiment.
    """
    lines = [
        f"Product: {product_name}",
        f"Price observations on record: {n_observations}",
        f"Most recent price: {last_price:.2f}",
    ]

    if price_change_pct is not None:
        direction = "up" if price_change_pct > 0 else "down"
        lines.append(f"Recent trend: {direction} {abs(price_change_pct):.1f}% over the series")

    if sentiment_summary:
        lines.append(f"Customer sentiment: {sentiment_summary}")

    if context:
        lines.append(f"Additional context: {context}")

    lines.append("")
    lines.append(
        "Give a short pricing recommendation based only on the above. "
        "State plainly if the evidence is insufficient."
    )
    return "\n".join(lines)


def build_sentiment_summary(positive: int, negative: int) -> str:
    """One-line sentiment summary, or an honest note when the sample is too small.

    Below ten reviews the proportion is noise. Reporting "100% positive" from two
    reviews would invite the model to reason from it as if it meant something.
    """
    total = positive + negative
    if total == 0:
        return "no reviews available"
    if total < 10:
        return f"only {total} review(s) available -- too few to summarise reliably"
    share = positive / total * 100
    return f"{share:.0f}% positive across {total} reviews"
