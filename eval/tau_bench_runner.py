"""τ-bench evaluation runner (40-task subset, retail + airline domains)."""
from pathlib import Path

RESULTS_DB = Path(__file__).parent.parent / "results" / "trials.db"


def run_trial(task: dict, agent, condition: str, model: str) -> dict:
    raise NotImplementedError


def run_tau_bench(tasks: list[dict], agents: dict, db_path: Path = RESULTS_DB) -> None:
    raise NotImplementedError
