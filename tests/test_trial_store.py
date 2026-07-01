"""Tests for SQLite trial persistence."""
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


def test_success_probs_and_mutations_round_trip(db):
    trial = TrialResult(
        task_id="retail_005",
        condition="compass",
        model="claude-sonnet-4-6",
        success=False,
        steps=2,
        abstained=False,
        confidence_scores=[0.9, 0.95],
        final_message="Cancelled the wrong order.",
        success_probs=[0.9, 0.85],
        mutated_order_ids=["#W4444444"],
    )
    save_trial(trial, db)
    r = load_trials(db)[0]
    assert r.success_probs == [0.9, 0.85]
    assert r.mutated_order_ids == ["#W4444444"]


def test_new_fields_default_to_empty_lists(db):
    save_trial(_sample(), db)
    r = load_trials(db)[0]
    assert r.success_probs == []
    assert r.mutated_order_ids == []


def test_migrates_legacy_db_without_new_columns(db):
    """Pilot trials.db predates success_probs / mutated_order_ids. Saving into
    such a DB must add the columns; legacy rows load with empty-list defaults."""
    legacy_create = """
    CREATE TABLE trials (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id           TEXT NOT NULL,
        condition         TEXT NOT NULL,
        model             TEXT NOT NULL,
        success           INTEGER NOT NULL,
        steps             INTEGER NOT NULL,
        abstained         INTEGER NOT NULL,
        confidence_scores TEXT NOT NULL,
        final_message     TEXT NOT NULL,
        created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute(legacy_create)
        conn.execute(
            "INSERT INTO trials (task_id, condition, model, success, steps,"
            " abstained, confidence_scores, final_message)"
            " VALUES ('retail_001', 'vanilla', 'gpt-4o-mini', 1, 3, 0, '[]', 'done')"
        )

    save_trial(_sample(), db)
    results = load_trials(db)
    assert len(results) == 2
    legacy = next(r for r in results if r.condition == "vanilla")
    assert legacy.success_probs == []
    assert legacy.mutated_order_ids == []


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
