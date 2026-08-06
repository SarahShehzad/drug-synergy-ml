import pandas as pd
from src.db import get_connection

def compute_bliss_for_cell_line(conn, cell_line_id: int, cell_line_name: str) -> int:
    df = pd.read_sql_query(
        """
        SELECT block_id, drug_a_id, drug_b_id, conc_a, conc_b, viability_pct
        FROM screens
        WHERE cell_line_id = ? AND viability_pct IS NOT NULL
        """,
        conn, params=(cell_line_id,),
    )

    combo = df[(df["conc_a"] > 0) & (df["conc_b"] > 0)].copy()
    single_a = df[(df["conc_a"] > 0) & (df["conc_b"] == 0)][
        ["block_id", "conc_a", "viability_pct"]
    ].rename(columns={"viability_pct": "single_a_pct"})
    single_b = df[(df["conc_a"] == 0) & (df["conc_b"] > 0)][
        ["block_id", "conc_b", "viability_pct"]
    ].rename(columns={"viability_pct": "single_b_pct"})

    merged = combo.merge(single_a, on=["block_id", "conc_a"], how="left")
    merged = merged.merge(single_b, on=["block_id", "conc_b"], how="left")
    merged = merged.dropna(subset=["single_a_pct", "single_b_pct"])

    # independent action prediction, multiply the two survival fractions.
    merged["predicted_pct"] = merged["single_a_pct"] * merged["single_b_pct"] / 100.0
    merged["bliss_excess"] = merged["predicted_pct"] - merged["viability_pct"]

    block_scores = (
        merged.groupby(["block_id", "drug_a_id", "drug_b_id"])["bliss_excess"]
        .mean()
        .reset_index()
        .rename(columns={"bliss_excess": "bliss_score"})
    )

    n = 0
    for row in block_scores.itertuples():
        conn.execute(
            "UPDATE synergy_scores SET bliss_score = ? WHERE block_id = ? AND cell_line_id = ?",
            (row.bliss_score, str(row.block_id), cell_line_id),
        )
        n += 1
    conn.commit()
    print(f"  {cell_line_name}: computed Bliss score for {n} drug pairs")
    return n

def compare_cmrs_and_bliss(conn, cell_line_name: str):
    df = pd.read_sql_query(
        """
        SELECT sc.cmrs_score, sc.bliss_score, da.name AS drug_a, db_.name AS drug_b
        FROM synergy_scores sc
        JOIN cell_lines cl ON sc.cell_line_id = cl.cell_line_id
        JOIN drugs da ON sc.drug_a_id = da.drug_id
        JOIN drugs db_ ON sc.drug_b_id = db_.drug_id
        WHERE cl.name = ? AND sc.cmrs_score IS NOT NULL AND sc.bliss_score IS NOT NULL
        """,
        conn, params=(cell_line_name,),
    )

    pearson = df["cmrs_score"].corr(df["bliss_score"], method="pearson")
    spearman = df["cmrs_score"].corr(df["bliss_score"], method="spearman")

    print(f"\n{cell_line_name}: {len(df)} pairs with both scores")
    print(f"  Pearson correlation:  {pearson:.3f}")
    print(f"  Spearman correlation: {spearman:.3f}")

    top_cmrs = set(df.nlargest(10, "cmrs_score").index)
    top_bliss = set(df.nlargest(10, "bliss_score").index)
    overlap = len(top_cmrs & top_bliss)
    print(f"  Top-10 overlap: {overlap}/10 pairs agree between CMRS and Bliss rankings")

def main():
    conn = get_connection()
    cell_lines = conn.execute(
        "SELECT cell_line_id, name FROM cell_lines WHERE is_tumor = 1"
    ).fetchall()

    print("Computing Bliss scores...")
    for cell_line_id, name in cell_lines:
        compute_bliss_for_cell_line(conn, cell_line_id, name)

    print("\nComparing CMRS vs Bliss...")
    for _, name in cell_lines:
        compare_cmrs_and_bliss(conn, name)

    conn.close()


if __name__ == "__main__":
    main()