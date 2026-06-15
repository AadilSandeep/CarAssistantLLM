"""
src/obd_utils.py
================
OBD-II helper utilities: regex parsing, CSV database loading, lookup, and search.

All logic is extracted directly from the notebook (Cells 6 and 10).
Local file path replaces the Google Drive path.
"""

import re
import os
import pandas as pd
from pathlib import Path

from src.config import OBD_CSV_PATH, OBD_HINTS

# ── OBD code regex (from notebook Cell 6) ─────────────────────────────────────
# Matches standard OBD-II DTC format: P0xxx, P1xxx, P2xxx, P3xxx
OBD_RE = re.compile(r"\bP[0-3][0-9A-F]{3}\b", re.IGNORECASE)


def normalize_obd_codes(obd_text: str) -> list[str]:
    """
    Extract and deduplicate OBD-II codes from free text input.
    Returns sorted list of uppercase codes, e.g. ['P0128', 'P0300'].
    """
    return sorted(set(m.group(0).upper() for m in OBD_RE.finditer(obd_text or "")))


def explain_obd(codes: list[str], db: dict | None = None) -> str:
    """
    Return a human-readable explanation of a list of OBD codes.

    To maintain strict parity with the notebook, this function MUST ONLY use
    the inline OBD_HINTS, ignoring the external CSV database.
    """
    if not codes:
        return "None provided."

    lines = []
    for c in codes:
        desc = OBD_HINTS.get(c, "Unknown/General OBD-II code (lookup needed)")
        lines.append(f"{c}: {desc}")
    return "\n".join(lines)


# ── OBD CSV database loader (from notebook Cell 10) ───────────────────────────
def load_obd_db(csv_path: Path = OBD_CSV_PATH) -> dict[str, str]:
    """
    Load the full OBD-II trouble code CSV into a dict {code: description}.

    The CSV has NO header row. Format: "P0100","Mass or Volume Air Flow..."
    Tries candidate column names for robustness; falls back to positional (0, 1).

    Returns a minimal fallback dict if the file does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[obd_utils] WARNING: OBD CSV not found at {path}. Using inline hints only.")
        return dict(OBD_HINTS)

    try:
        # The CSV has no header row — use header=None
        df = pd.read_csv(path, header=None)

        # Assign readable column names based on position
        # Column 0 = code (e.g., P0100), Column 1 = description
        code_col = df.columns[0]
        desc_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        db: dict[str, str] = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip().upper()
            desc = str(row[desc_col]).strip()
            if code and code != "NAN":
                db[code] = desc

        print(f"[obd_utils] Loaded {len(db)} OBD codes from {path.name}")
        return db

    except Exception as e:
        print(f"[obd_utils] ERROR loading OBD CSV: {e}. Using inline hints.")
        return dict(OBD_HINTS)


def obd_list_text(db: dict) -> str:
    """Return all code keys as a sorted newline-separated string."""
    return "\n".join(sorted(db.keys()))


def obd_lookup(code: str, db: dict) -> str:
    """Look up a single OBD code and return its description."""
    c = (code or "").strip().upper()
    if not c:
        return "Enter an OBD code like P0300."
    return f"{c}: {db.get(c, 'Not found in loaded OBD list.')}"


def obd_search(query: str, db: dict) -> str:
    """
    Search the OBD database by code prefix or description keyword.
    Returns all matching code keys separated by newlines.
    """
    q = (query or "").strip().upper()
    if not q:
        return obd_list_text(db)
    matches = [k for k, v in db.items() if q in k or q in str(v).upper()]
    return "\n".join(sorted(matches)) if matches else "No matches found."
