"""
Looks up each drug's SMILES structure from PubChem by name and stores it
in the drugs table. Drugs that only have an NCGC ID (no real name, from
the Phase 2 fallback) will mostly fail this lookup -- that's expected,
they just won't get a fingerprint and the model treats them as unknown.

Usage:
    python -m src.fetch_smiles
"""

import time
import requests

from src.db import get_connection

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/TXT"
)


def lookup_smiles(drug_name: str) -> str | None:
    try:
        resp = requests.get(PUBCHEM_URL.format(name=drug_name), timeout=10)
        if resp.status_code == 200:
            return resp.text.strip()
    except requests.RequestException:
        pass
    return None


def main():
    conn = get_connection()
    drugs = conn.execute(
        "SELECT drug_id, name FROM drugs WHERE smiles IS NULL"
    ).fetchall()

    n_found = 0
    n_missed = 0
    for drug_id, name in drugs:
        smiles = lookup_smiles(name)
        if smiles:
            conn.execute("UPDATE drugs SET smiles = ? WHERE drug_id = ?", (smiles, drug_id))
            n_found += 1
        else:
            n_missed += 1
        time.sleep(0.2)  # be polite to PubChem's free API, avoid rate limiting

    conn.commit()
    conn.close()
    print(f"Found SMILES for {n_found} drugs, missed {n_missed} (likely NCGC-ID-only entries).")


if __name__ == "__main__":
    main()