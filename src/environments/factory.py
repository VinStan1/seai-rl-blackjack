"""Construct Blackjack environments from experiment configuration objects."""

from __future__ import annotations

from typing import Any

from src.environments.blackjack import BlackjackEnvironment
from src.environments.finite_blackjack import FiniteBlackjackEnvironment


def make_blackjack_environment(config: dict[str, Any]) -> Any:
    """Create the standard environment or one of the finite-shoe variants."""
    settings = dict(config)
    variant = settings.pop("variant", "standard")
    if variant == "standard":
        return BlackjackEnvironment(**settings)

    observations = {
        "finite_hidden": "hidden",
        "finite_hi_lo": "hi_lo",
        "finite_composition": "composition",
    }
    if variant not in observations:
        raise ValueError(
            f"unknown environment variant {variant!r}; "
            f"expected one of {['standard', *observations]}"
        )
    settings.setdefault("decks", 6)
    settings.setdefault("penetration", 0.75)
    return FiniteBlackjackEnvironment(
        observation=observations[variant], **settings
    )