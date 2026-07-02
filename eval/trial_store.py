"""SQLite persistence for trial results. One row per trial."""
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrialResult:
    task_id: str
    condition: str          # "vanilla" | "compass"
    model: str
    success: bool
    steps: int
    abstained: bool
    confidence_scores: list[float]   # Compass only; empty list for vanilla
    final_message: str
    success_probs: list[float] = field(default_factory=list)    # calibrated, Compass only
    mutated_order_ids: list[str] = field(default_factory=list)  # orders changed during trial
    risk_levels: list[str] = field(default_factory=list)        # per-step, Compass only


_CREATE = """
CREATE TABLE IF NOT EXISTS trials (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id           TEXT NOT NULL,
    condition         TEXT NOT NULL,
    model             TEXT NOT NULL,
    success           INTEGER NOT NULL,
    steps             INTEGER NOT NULL,
    abstained         INTEGER NOT NULL,
    confidence_scores TEXT NOT NULL,
    final_message     TEXT NOT NULL,
    success_probs     TEXT NOT NULL DEFAULT '[]',
    mutated_order_ids TEXT NOT NULL DEFAULT '[]',
    risk_levels       TEXT NOT NULL DEFAULT '[]',
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

# Columns added after the Phase 2 pilot; ALTER is a no-op error on newer DBs.
_MIGRATIONS = [
    "ALTER TABLE trials ADD COLUMN success_probs TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE trials ADD COLUMN mutated_order_ids TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE trials ADD COLUMN risk_levels TEXT NOT NULL DEFAULT '[]'",
]

_INSERT = """
INSERT INTO trials (task_id, condition, model, success, steps, abstained,
                    confidence_scores, final_message, success_probs, mutated_order_ids,
                    risk_levels)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT = (
    "SELECT task_id, condition, model, success, steps, abstained,"
    " confidence_scores, final_message, success_probs, mutated_order_ids,"
    " risk_levels FROM trials"
)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE)
    for migration in _MIGRATIONS:
        try:
            conn.execute(migration)
        except sqlite3.OperationalError:
            pass  # column already exists


def save_trial(result: TrialResult, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute(_INSERT, (
            result.task_id,
            result.condition,
            result.model,
            int(result.success),
            result.steps,
            int(result.abstained),
            json.dumps(result.confidence_scores),
            result.final_message,
            json.dumps(result.success_probs),
            json.dumps(result.mutated_order_ids),
            json.dumps(result.risk_levels),
        ))


def load_trials(db_path: Path) -> list[TrialResult]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        rows = conn.execute(_SELECT).fetchall()
    return [
        TrialResult(
            task_id=r[0],
            condition=r[1],
            model=r[2],
            success=bool(r[3]),
            steps=r[4],
            abstained=bool(r[5]),
            confidence_scores=json.loads(r[6]),
            final_message=r[7],
            success_probs=json.loads(r[8]),
            mutated_order_ids=json.loads(r[9]),
            risk_levels=json.loads(r[10]),
        )
        for r in rows
    ]
