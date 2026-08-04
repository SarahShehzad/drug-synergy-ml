"""
SQLite schema + helpers for the drug synergy project.

This is the SQL layer: every screened combination, drug, and cell line
lives here so the rest of the pipeline (features, training, demo app)
queries it instead of re-parsing raw files each time.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "synergy.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS drugs (
    drug_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    smiles      TEXT,               -- structure, pulled from PubChem/DrugBank
    moa         TEXT                -- mechanism of action, e.g. "MEK inhibitor"
);

CREATE TABLE IF NOT EXISTS cell_lines (
    cell_line_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT UNIQUE NOT NULL,   -- e.g. "ipNF05.5mc"
    is_tumor     INTEGER NOT NULL        -- 1 = tumor (NF1-/-), 0 = non-tumor reference
);

CREATE TABLE IF NOT EXISTS screens (
    screen_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id      TEXT NOT NULL,         -- Ferrer et al. BlockID
    drug_a_id     INTEGER NOT NULL REFERENCES drugs(drug_id),
    drug_b_id     INTEGER NOT NULL REFERENCES drugs(drug_id),
    cell_line_id  INTEGER NOT NULL REFERENCES cell_lines(cell_line_id),
    conc_a        REAL NOT NULL,
    conc_b        REAL NOT NULL,
    viability_raw REAL NOT NULL,         -- raw assay readout
    viability_pct REAL,                  -- normalized to solvent-only control (Eq. 1 in paper)
    UNIQUE(block_id, cell_line_id, conc_a, conc_b)
);

CREATE TABLE IF NOT EXISTS synergy_scores (
    score_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id      TEXT NOT NULL,
    drug_a_id     INTEGER NOT NULL REFERENCES drugs(drug_id),
    drug_b_id     INTEGER NOT NULL REFERENCES drugs(drug_id),
    cell_line_id  INTEGER NOT NULL REFERENCES cell_lines(cell_line_id),
    cmrs_score    REAL,     -- reproduced from the paper's method, ground truth #1
    bliss_score   REAL,     -- standard synergy metric, ground truth #2
    UNIQUE(block_id, cell_line_id)
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Initialized database at {DB_PATH}")


def get_or_create_drug(conn: sqlite3.Connection, name: str, smiles: str = None, moa: str = None) -> int:
    cur = conn.execute("SELECT drug_id FROM drugs WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO drugs (name, smiles, moa) VALUES (?, ?, ?)", (name, smiles, moa)
    )
    return cur.lastrowid


def get_or_create_cell_line(conn: sqlite3.Connection, name: str, is_tumor: bool) -> int:
    cur = conn.execute("SELECT cell_line_id FROM cell_lines WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO cell_lines (name, is_tumor) VALUES (?, ?)", (name, int(is_tumor))
    )
    return cur.lastrowid


if __name__ == "__main__":
    init_db()
