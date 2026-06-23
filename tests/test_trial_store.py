"""Tests for SQLite trial persistence."""
import json
from pathlib import Path
import pytest
from eval.trial_store import TrialResult, load_trials, save_trial


@pytest.fixture
def db(tmp_path) -> Path:
    return tmp_path / "trials.db"


def _sample() -> TrialResult:
    return TrialResult(
        task_id="retail_001",
        condition="compass",
        model="claude-sonnet-4-6",
        success=True,
        steps=3,
        abstained=False,
        confidence_scores=[0.9, 0.85, 0.92],
        final_message="Order cancelled successfully.",
    )


def test_save_creates_db_file(db):
    save_trial(_sample(), db)
    assert db.exists()


def test_save_and_load_round_trip(db):
    trial = _sample()
    save_trial(trial, db)
    results = load_trials(db)

    assert len(results) == 1
    r = results[0]
    assert r.task_id == "retail_001"
    assert r.condition == "compass"
    assert r.model == "claude-sonnet-4-6"
    assert r.success is True
    assert r.steps == 3
    assert r.abstained is False
    assert r.confidence_scores == [0.9, 0.85, 0.92]
    assert r.final_message == "Order cancelled successfully."


def test_multiple_trials_stored(db):
    save_trial(_sample(), db)
    save_trial(TrialResult(
        task_id="retail_002",
        condition="vanilla",
        model="claude-sonnet-4-6",
        success=False,
        steps=20,
        abstained=False,
        confidence_scores=[],
        final_message="Max steps reached.",
    ), db)

    results = load_trials(db)
    assert len(results) == 2
    assert {r.task_id for r in results} == {"retail_001", "retail_002"}


def test_abstained_trial_stored_correctly(db):
    trial = TrialResult(
        task_id="retail_003",
        condition="compass",
        model="claude-sonnet-4-6",
        success=False,
        steps=1,
        abstained=True,
        confidence_scores=[0.4],
        final_message="ABSTAINING: low confidence on high-risk action.",
    )
    save_trial(trial, db)
    r = load_trials(db)[0]
    assert r.abstained is True
    assert r.confidence_scores == [0.4]
