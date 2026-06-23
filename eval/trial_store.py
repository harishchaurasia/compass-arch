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
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_INSERT = """
INSERT INTO trials (task_id, condition, model, success, steps, abstained, confidence_scores, final_message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT = "SELECT task_id, condition, model, success, steps, abstained, confidence_scores, final_message FROM trials"


def save_trial(result: TrialResult, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE)
        conn.execute(_INSERT, (
            result.task_id,
            result.condition,
            result.model,
            int(result.success),
            result.steps,
            int(result.abstained),
            json.dumps(result.confidence_scores),
            result.final_message,
        ))


def load_trials(db_path: Path) -> list[TrialResult]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
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
        )
        for r in rows
    ]
