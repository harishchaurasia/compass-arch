"""Evaluation metrics: ECE, Brier score, selective accuracy, compound failure rate."""


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    return sum((c - o) ** 2 for c, o in zip(confidences, outcomes)) / len(confidences)


def ece(confidences: list[float], outcomes: list[int], n_bins: int = 10) -> float:
    raise NotImplementedError


def selective_success_rate(results: list[dict]) -> tuple[float, float]:
    """Returns (accuracy_on_committed_trials, abstention_rate)."""
    raise NotImplementedError


def compound_failure_rate(results: list[dict]) -> float:
    """Fraction of trials where agent took a destructive action while wrong."""
    raise NotImplementedError
