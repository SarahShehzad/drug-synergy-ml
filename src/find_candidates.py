import pickle
from pathlib import Path

import pandas as pd

from src.db import get_connection
from src.features import pair_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline_model.pkl"


def find_untested_pairs(conn, cell_line_id: int) -> pd.DataFrame:
    query = """
    SELECT d1.drug_id AS drug_a_id, d1.name AS drug_a_name, d1.smiles AS smiles_a,
           d2.drug_id AS drug_b_id, d2.name AS drug_b_name, d2.smiles AS smiles_b
    FROM drugs d1
    CROSS JOIN drugs d2
    WHERE d1.drug_id < d2.drug_id
      AND NOT EXISTS (
          SELECT 1 FROM screens s
          WHERE s.cell_line_id = ?
            AND ((s.drug_a_id = d1.drug_id AND s.drug_b_id = d2.drug_id)
                 OR (s.drug_a_id = d2.drug_id AND s.drug_b_id = d1.drug_id))
      )
    """
    return pd.read_sql_query(query, conn, params=(cell_line_id,))


def main():
    conn = get_connection()
    tumor_lines = conn.execute(
        "SELECT cell_line_id, name FROM cell_lines WHERE is_tumor = 1"
    ).fetchall()

    for cell_line_id, name in tumor_lines:
        untested = find_untested_pairs(conn, cell_line_id)
        print(f"{name}: {len(untested)} untested pairs found within the panel")

    conn.close()


if __name__ == "__main__":
    main()