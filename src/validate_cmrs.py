import pandas as pd
from src.db import get_connection

def get_ranked_table(conn, cell_line_name: str) -> pd.DataFrame:
    query = """
    SELECT sc.cmrs_score, da.name AS drug_a, db_.name AS drug_b
    FROM synergy_scores sc
    JOIN drugs da ON sc.drug_a_id = da.drug_id
    JOIN drugs db_ ON sc.drug_b_id = db_.drug_id
    JOIN cell_lines cl ON sc.cell_line_id = cl.cell_line_id
    WHERE cl.name = ?
    ORDER BY sc.cmrs_score DESC
    """
    df = pd.read_sql_query(query, conn, params=(cell_line_name,))
    df["rank"] = range(1, len(df) + 1)
    return df

def find_pair_rank(ranked_df: pd.DataFrame, name_a: str, name_b: str):
    match = ranked_df[
        ((ranked_df["drug_a"].str.contains(name_a, case=False, na=False)) &
         (ranked_df["drug_b"].str.contains(name_b, case=False, na=False)))
        |
        ((ranked_df["drug_a"].str.contains(name_b, case=False, na=False)) &
         (ranked_df["drug_b"].str.contains(name_a, case=False, na=False)))
    ]
    if match.empty:
        return None
    return match.iloc[0]

def main():
    conn = get_connection()

    print("=" * 60)
    print("ipNF05.5 (mixed clone)")
    print("=" * 60)
    ranked = get_ranked_table(conn, "ipNF05.5 (mixed clone)")
    print(f"Total ranked pairs: {len(ranked)}\n")

    print("Top 10 by CMRS score:")
    print(ranked.head(10).to_string(index=False))
    print()

    checks = [
        ("Alvespimycin", "Topotecan", "expected #1"),
        ("Panobinostat", "Ganetespib", "expected #2"),
    ]
    for a, b, expectation in checks:
        result = find_pair_rank(ranked, a, b)
        if result is not None:
            print(f"{a} + {b}: rank {int(result['rank'])} of {len(ranked)}, "
                  f"score {result['cmrs_score']:.1f}  ({expectation})")
        else:
            print(f"{a} + {b}: NOT FOUND in this cell line's data  ({expectation})")

    print()
    print("=" * 60)
    print("ipNF95.6")
    print("=" * 60)
    ranked2 = get_ranked_table(conn, "ipNF95.6")
    print(f"Total ranked pairs: {len(ranked2)}\n")

    print("Top 10 by CMRS score:")
    print(ranked2.head(10).to_string(index=False))
    print()

    result = find_pair_rank(ranked2, "Alvespimycin", "Topotecan")
    if result is not None:
        in_top_10 = "YES" if result["rank"] <= 10 else "no"
        print(f"Alvespimycin + Topotecan: rank {int(result['rank'])} of {len(ranked2)} "
              f"(expected top 10 -- in top 10? {in_top_10})")
    else:
        print("Alvespimycin + Topotecan: NOT FOUND in this cell line's data")

    conn.close()

if __name__ == "__main__":
    main()
