"""
Reproduces the paper's Composite Matrix Reduction Score (CMRS) for each
drug pair, using the two tumor cell lines against the non-tumor reference
line (ipnNF95.11c).

Logic (from the paper's Methods section):
  1. Each well is already normalized to its solvent control (viability_pct,
     computed in Phase 2).
  2. For each well, divide the tumor line's viability_pct by the reference
     line's viability_pct at the *same drug pair and same doses*.
  3. Bin that ratio into a score 0-6 (lower ratio = more tumor-selective =
     higher score).
  4. Sum scores across the "combination zone" (both drugs at nonzero dose)
     per drug pair -> that's the CMRS score for that pair, on that tumor
     line.

NOTE: bin thresholds below are reconstructed from the paper's histogram
description (Table 1's exact numbers weren't extractable as text) --
worth a quick visual check against the real Table 1 if you can see it.

Usage:
    python -m src.compute_cmrs
"""

import pandas as pd

from src.db import get_connection


def ratio_to_score(ratio: float) -> int | None:
    if pd.isna(ratio):
        return None
    if ratio <= 0.4:
        return 6
    elif ratio <= 0.5:
        return 5
    elif ratio <= 0.6:
        return 4
    elif ratio <= 0.8:
        return 3
    elif ratio <= 0.9:
        return 2
    elif ratio <= 0.98:
        return 1
    else:
        return 0


def get_cell_line_ids(conn):
    rows = conn.execute("SELECT cell_line_id, name, is_tumor FROM cell_lines").fetchall()
    reference_id = next((cid for cid, name, is_tumor in rows if not is_tumor), None)
    tumor_lines = [(cid, name) for cid, name, is_tumor in rows if is_tumor]
    if reference_id is None:
        raise ValueError("No non-tumor reference cell line found in database.")
    return reference_id, tumor_lines


def compute_for_cell_line(conn, tumor_id: int, tumor_name: str, reference_id: int) -> int:
    """
    Joins tumor and reference wells on drug identity + dose (not on raw
    block_id, since block_id numbering could differ between cell lines --
    joining on drug_a_id/drug_b_id/conc_a/conc_b is the identity that
    actually matters and is robust regardless).
    """
    query = """
    SELECT
        t.block_id AS tumor_block_id,
        t.drug_a_id, t.drug_b_id, t.conc_a, t.conc_b,
        t.viability_pct AS tumor_pct,
        r.viability_pct AS ref_pct
    FROM screens t
    JOIN screens r
        ON t.drug_a_id = r.drug_a_id
       AND t.drug_b_id = r.drug_b_id
       AND t.conc_a = r.conc_a
       AND t.conc_b = r.conc_b
       AND r.cell_line_id = ?
    WHERE t.cell_line_id = ?
      AND t.viability_pct IS NOT NULL
      AND r.viability_pct IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=(reference_id, tumor_id))

    if df.empty:
        print(f"  {tumor_name}: no matching wells found against reference -- check join keys.")
        return 0

    df["ratio"] = df["tumor_pct"] / df["ref_pct"]
    df["score"] = df["ratio"].apply(ratio_to_score)

    # Combination zone: both drugs present at a nonzero dose.
    combo = df[(df["conc_a"] > 0) & (df["conc_b"] > 0)]

    cumulative = (
        combo.groupby(["drug_a_id", "drug_b_id", "tumor_block_id"])["score"]
        .sum()
        .reset_index()
        .rename(columns={"score": "cmrs_score"})
    )

    n_written = 0
    for row in cumulative.itertuples():
        conn.execute(
            """
            INSERT INTO synergy_scores (block_id, drug_a_id, drug_b_id, cell_line_id, cmrs_score)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(block_id, cell_line_id) DO UPDATE SET cmrs_score = excluded.cmrs_score
            """,
            (str(row.tumor_block_id), row.drug_a_id, row.drug_b_id, tumor_id, row.cmrs_score),
        )
        n_written += 1

    conn.commit()
    print(f"  {tumor_name}: computed CMRS for {n_written} drug pairs")
    return n_written


def main():
    conn = get_connection()
    reference_id, tumor_lines = get_cell_line_ids(conn)

    print("Computing CMRS scores...")
    for tumor_id, tumor_name in tumor_lines:
        compute_for_cell_line(conn, tumor_id, tumor_name, reference_id)

    conn.close()


if __name__ == "__main__":
    main()