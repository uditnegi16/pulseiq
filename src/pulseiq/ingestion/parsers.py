"""Pure parsing functions for scraped product fields.

Every function here is deterministic, does no I/O, and takes a string in and
returns a number or None. That makes them fully unit-testable without a browser,
a network connection, or a database -- which is why this is the first module
written and the first one covered by CI.

Design rules:
  * Return None for "could not parse", never 0. A price of 0 and a failed parse
    are different facts, and collapsing them corrupts every downstream average.
    (The original scarpe.py returned 0 on failure.)
  * Never raise on bad input. Scrapers see garbage constantly; callers decide
    what to do with None.
  * Assume Western decimal convention (1,234.56). Indian lakh grouping
    (1,29,999) parses correctly since all commas are stripped.
"""

from __future__ import annotations

import re

# Currency symbols / codes stripped before numeric parsing.
_CURRENCY = r"[₹$€£¥]|Rs\.?|INR|USD|EUR|GBP"

# A number with optional thousands separators and optional decimal part.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")

_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_RATING = re.compile(r"(\d+(?:\.\d+)?)\s*out\s+of\s+(\d+(?:\.\d+)?)\s*stars?", re.I)


def _first_number(text: str) -> float | None:
    """Return the first parseable number in `text`, or None."""
    match = _NUMBER.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def extract_price(text: str | None) -> float | None:
    """Parse a price string into a float.

    >>> extract_price("₹24,999.00")
    24999.0
    >>> extract_price("Rs. 1,299.50")
    1299.5
    >>> extract_price("out of stock") is None
    True

    Returns None when no number is present. Negative prices are rejected.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = re.sub(_CURRENCY, " ", text, flags=re.I)
    value = _first_number(cleaned)

    if value is None or value < 0:
        return None
    return value


def extract_discount_pct(text: str | None) -> float | None:
    """Parse a discount percentage from marketing copy.

    Handles the shapes Amazon actually renders: "-15%", "15% off", "(20% off)",
    "Save 30%". The sign is discarded -- a discount is always reported positive.

    Returns None if no percentage is present, or if the value is outside
    0-100 (a "150% off" badge is a parse error, not a bargain).
    """
    if not text or not isinstance(text, str):
        return None

    match = _PERCENT.search(text)
    if not match:
        return None

    try:
        value = float(match.group(1))
    except ValueError:
        return None

    if not 0.0 <= value <= 100.0:
        return None
    return value


def extract_rating(text: str | None) -> float | None:
    """Parse a star rating into a float on a 0-5 scale.

    Accepts the full Amazon string ("4.3 out of 5 stars") and bare numbers
    ("4.3"). Ratings on a non-5 scale are rescaled to 5.

    Rejects malformed values like "4.3.0" -- which appears in the original
    reviews.csv and would otherwise silently become 4.3.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    match = _RATING.search(text)
    if match:
        try:
            value, scale = float(match.group(1)), float(match.group(2))
        except ValueError:
            return None
        if scale <= 0:
            return None
        value = value / scale * 5.0
        return value if 0.0 <= value <= 5.0 else None

    # Bare number: require the WHOLE string to be one clean number, so that
    # corrupt values like "4.3.0" are rejected rather than truncated.
    if not re.fullmatch(r"\d+(?:\.\d+)?", text):
        return None

    value = float(text)
    return value if 0.0 <= value <= 5.0 else None


def compute_discount_pct(selling_price: float | None, original_price: float | None) -> float | None:
    """Derive discount % from the two prices.

    Preferred over trusting the scraped badge, which is inconsistent and
    sometimes absent. Returns None when either price is missing or when
    original_price is 0 (no division by zero, no infinities in the dataset).

    A selling price above the original yields 0.0, not a negative discount.
    """
    if selling_price is None or original_price is None:
        return None
    if original_price <= 0 or selling_price < 0:
        return None

    discount = (1.0 - selling_price / original_price) * 100.0
    return round(max(discount, 0.0), 2)


def clean_review_text(text: str | None) -> str | None:
    """Collapse whitespace and strip. Returns None for empty/whitespace-only."""
    if not text or not isinstance(text, str):
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned or None
