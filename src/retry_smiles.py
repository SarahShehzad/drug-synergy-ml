import re
import time
import requests

from src.db import get_connection

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/TXT"
)

SALT_SUFFIXES = [
    r"\s*\(hydrochloride\)", r"\s*hydrochloride", r"\s*\(hcl\)", r"\s*hcl",
    r"\s*\(free base\)", r"\s*\(sodium\)", r"\s*sodium salt",
    r"\s*\(.*\)$",  # any trailing parenthetical as a last resort
]


def clean_name(name: str) -> str:
    cleaned = name
    for pattern in SALT_SUFFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def lookup_smiles(name: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            resp = requests.get(PUBCHEM_URL.format(name=name), timeout=10)
            if resp.status_code == 200:
                return resp.text.strip()
            if resp.status_code == 404:
                return None  # genuinely not found, no point retrying
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)  # back off a bit longer each retry
    return None


def main():
    conn = get_connection()
    missing = conn.execute(
        "SELECT drug_id, name FROM drugs WHERE smiles IS NULL"
    ).fetchall()

    print(f"Retrying {len(missing)} drugs missing SMILES...\n")

    n_fixed = 0
    for drug_id, name in missing:
        smiles = lookup_smiles(name)

        if smiles is None:
            cleaned = clean_name(name)
            if cleaned != name:
                print(f"  '{name}' failed, retrying as '{cleaned}'...")
                smiles = lookup_smiles(cleaned)

        if smiles:
            conn.execute("UPDATE drugs SET smiles = ? WHERE drug_id = ?", (smiles, drug_id))
            print(f"  FIXED: {name}")
            n_fixed += 1
        else:
            print(f"  still missing: {name}")

        time.sleep(0.3)

    conn.commit()
    conn.close()
    print(f"\nFixed {n_fixed} of {len(missing)} previously-missing drugs.")


if __name__ == "__main__":
    main()