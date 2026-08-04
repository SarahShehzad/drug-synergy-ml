"""
Pulls the Ferrer et al. NF1 6x6 drug combination screening data from
Synapse and loads it into the local SQLite database.

Data is open-access on Synapse -- a free account + auth token is enough,
no data use agreement / approval wait needed.

Requires:
  - A free Synapse account
  - Environment variable SYNAPSE_AUTH_TOKEN
    (generate at https://www.synapse.org/#!PersonalAccessTokens:)

Usage:
    python -m src.data_loader
"""

import os
import synapseclient
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from src.db import get_connection, get_or_create_drug, get_or_create_cell_line, init_db

# Each cell line has a "metadata" file (per-block drug names, targets,
# concentration series, IC50s) and a "responses" file (per-well raw
# viability readings). Syn IDs confirmed from the Synapse UI.
CELL_LINE_FILES = {
    "ipNF05.5 (mixed clone)": {
        "metadata": "syn5613591",
        "responses": "syn5613592",
    },
    "ipNF95.6": {
        "metadata": "syn5613594",
        "responses": "syn5613595",
    },
    "ipnNF95.11c": {
        "metadata": "syn5613597",
        "responses": "syn5613598",
    },
}

# All three are NF1-/- tumor-derived Schwann cell lines used in the paper.
# TODO: confirm whether the paper's CMRS calc needs a separate non-tumor
# reference line (e.g. HFF or ipn02.3, both visible as sibling folders in
# the same Synapse project) or derives "non-tumor" comparison some other
# way. For now everything pulled here is marked is_tumor=True.
IS_TUMOR = True


def login() -> synapseclient.Synapse:
    syn = synapseclient.Synapse()
    syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"])
    return syn


def fetch_csv(syn: synapseclient.Synapse, syn_id: str) -> pd.DataFrame:
    entity = syn.get(syn_id)
    return pd.read_csv(entity.path)


def parse_conc_list(conc_str: str) -> list[float]:
    """RowConcs/ColConcs are comma-separated dilution series, e.g. '10.0,2.0,0.4,...'"""
    return [float(x) for x in str(conc_str).split(",") if x.strip() != ""]


def merge_metadata_and_responses(metadata: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the two files on BlockId, then resolves each well's actual
    Row/Col concentration by indexing into the block's RowConcs/ColConcs
    dilution series (Row/Col are 1-indexed).
    """
    merged = responses.merge(metadata, on="BlockId", how="left", suffixes=("", "_meta"))

    def resolve_conc(row, col_name, idx_name):
        concs = parse_conc_list(row[col_name])
        idx = int(row[idx_name]) - 1  # 1-indexed -> 0-indexed
        if 0 <= idx < len(concs):
            return concs[idx]
        return None

    merged["conc_a"] = merged.apply(lambda r: resolve_conc(r, "RowConcs", "Row"), axis=1)
    merged["conc_b"] = merged.apply(lambda r: resolve_conc(r, "ColConcs", "Col"), axis=1)

    return merged


def load_cell_line(conn, syn: synapseclient.Synapse, cell_line_name: str, syn_ids: dict) -> int:
    metadata = fetch_csv(syn, syn_ids["metadata"])
    responses = fetch_csv(syn, syn_ids["responses"])

    merged = merge_metadata_and_responses(metadata, responses)
    cell_line_id = get_or_create_cell_line(conn, cell_line_name, IS_TUMOR)

    n_loaded = 0
    for row in merged.itertuples():
        drug_a_id = get_or_create_drug(conn, row.RowName, moa=getattr(row, "RowTarget", None))
        drug_b_id = get_or_create_drug(conn, row.ColName, moa=getattr(row, "ColTarget", None))

        if row.conc_a is None or row.conc_b is None:
            continue  # couldn't resolve concentration index, skip

        conn.execute(
            """
            INSERT OR IGNORE INTO screens
                (block_id, drug_a_id, drug_b_id, cell_line_id, conc_a, conc_b, viability_raw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(row.BlockId),
                drug_a_id,
                drug_b_id,
                cell_line_id,
                row.conc_a,
                row.conc_b,
                row.Value,
            ),
        )
        n_loaded += 1

    conn.commit()
    print(f"  {cell_line_name}: loaded {n_loaded} wells")
    return n_loaded


def normalize_viability() -> None:
    """
    Normalizes each well to its block's zero-concentration control well
    (conc_a == 0 and conc_b == 0), matching the paper's Equation 1.
    NOTE: verify the dilution series actually includes a true 0 -- some
    6x6 designs use the lowest nonzero dose as the practical reference
    instead. Check after first load.
    """
    conn = get_connection()
    controls = conn.execute(
        """
        SELECT block_id, cell_line_id, viability_raw
        FROM screens
        WHERE conc_a = 0 AND conc_b = 0
        """
    ).fetchall()
    control_lookup = {(b, c): v for b, c, v in controls}

    rows = conn.execute("SELECT screen_id, block_id, cell_line_id, viability_raw FROM screens").fetchall()
    updated = 0
    for screen_id, block_id, cell_line_id, viability_raw in rows:
        control_val = control_lookup.get((block_id, cell_line_id))
        if control_val:
            pct = 100.0 * viability_raw / control_val
            conn.execute("UPDATE screens SET viability_pct = ? WHERE screen_id = ?", (pct, screen_id))
            updated += 1
    conn.commit()
    conn.close()
    print(f"Normalized {updated} wells against solvent-only controls.")
    if updated == 0:
        print("  WARNING: no zero-concentration control wells found -- check dilution series design.")


def main():
    init_db()
    syn = login()
    conn = get_connection()

    print("Loading cell lines...")
    for cell_line_name, syn_ids in CELL_LINE_FILES.items():
        load_cell_line(conn, syn, cell_line_name, syn_ids)

    conn.close()
    normalize_viability()


if __name__ == "__main__":
    main()
