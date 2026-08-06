import sqlite3
import pytest

from src.db import SCHEMA, get_or_create_drug, get_or_create_cell_line


@pytest.fixture
def conn():
    """A fresh, throwaway in-memory database for each test -- never
    touches your real data/synergy.db."""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON;")
    c.executescript(SCHEMA)
    yield c
    c.close()

def test_get_or_create_drug_creates_new_row(conn):
    drug_id = get_or_create_drug(conn, "Selumetinib", moa="Mek1/2 inhibitor")
    row = conn.execute(
        "SELECT name, moa FROM drugs WHERE drug_id = ?", (drug_id,)
    ).fetchone()
    assert row == ("Selumetinib", "Mek1/2 inhibitor")


def test_get_or_create_drug_is_idempotent(conn):
    """Calling this twice with the same name must return the same ID, not
    create a duplicate -- the ETL pipeline calls this once per well, and
    the same drug shows up in thousands of wells."""
    id_first = get_or_create_drug(conn, "Cabozantinib")
    id_second = get_or_create_drug(conn, "Cabozantinib")
    assert id_first == id_second

    count = conn.execute(
        "SELECT COUNT(*) FROM drugs WHERE name = 'Cabozantinib'"
    ).fetchone()[0]
    assert count == 1

def test_get_or_create_cell_line_stores_tumor_flag(conn):
    tumor_id = get_or_create_cell_line(conn, "ipNF95.6", is_tumor=True)
    non_tumor_id = get_or_create_cell_line(conn, "ipnNF95.11c", is_tumor=False)

    tumor_flag = conn.execute(
        "SELECT is_tumor FROM cell_lines WHERE cell_line_id = ?", (tumor_id,)
    ).fetchone()[0]
    non_tumor_flag = conn.execute(
        "SELECT is_tumor FROM cell_lines WHERE cell_line_id = ?", (non_tumor_id,)
    ).fetchone()[0]

    assert tumor_flag == 1
    assert non_tumor_flag == 0