"""
Common interface every governor (rule-based, logistic, xgboost) implements,
so `evaluate_baselines.py` can call them identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from argus.scenarios.taxonomy import GovernorDecision


class Governor(ABC):
    name: str = "base"

    @abstractmethod
    def decide(self, features: dict[str, int | float]) -> GovernorDecision:
        """Return a decision for one scenario's feature dict."""
        raise NotImplementedError

    def decide_batch(self, feature_dicts: list[dict[str, int | float]]) -> list[GovernorDecision]:
        return [self.decide(f) for f in feature_dicts]
