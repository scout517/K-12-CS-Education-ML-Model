#!/usr/bin/env python3
import argparse
import os
import re
import sys
from io import StringIO

import pandas as pd
import numpy as np


def read_schools(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Expect columns like: State, School District Name, School Name, etc.
    required_cols = {"State", "School District Name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Schools file missing required columns: {missing}")
    return df


def _try_read_elsi_csv(path: str) -> pd.DataFrame:
    """
    The ELSI export often wraps a CSV inside a single 'ELSI Export' column.
    This routine reconstructs a proper CSV by detecting the header line.
    """
    try:
        probe = pd.read_csv(path, sep=None, engine="python", nrows=5, low_memory=False)
    except Exception:
        # Fallback to comma separated read
        probe = pd.read_csv(path, engine="python", nrows=5, low_memory=False)

    # If it already parsed to 2+ columns, just read normally
    if probe.shape[1] > 1:
        return pd.read_csv(path, low_memory=False)

    # Otherwise, reconstruct from the single text column
    colname = probe.columns[0]
    whole = pd.read_csv(path, engine="python", low_memory=False)
    lines = whole[colname].astype(str).tolist()

    # Find a CSV header line (look for "Agency Name" with commas)
    start_idx = None
    for i, line in enumerate(lines):
        if "Agency Name" in line and "," in line:
            start_idx = i
            break
    if start_idx is None:
        # Fall back to first line that contains commas
        for i, line in enumerate(lines):
            if "," in line:
                start_idx = i
                break
    if start_idx is None:
        raise ValueError("Could not detect the header line in ELSI export.")

    blob = "\n".join(lines[start_idx:])
    df = pd.read_csv(StringIO(blob), low_memory=False)
    return df


def read_districts(path: str) -> pd.DataFrame:
    """
    Reads the "School District Data.csv" (ELSI export) robustly.
    """
    return _try_read_elsi_csv(path)


def read_any_table(path: str) -> pd.DataFrame:
    """
    Read CSV or Excel with reasonable fallbacks. Prefer CSV for simplicity.
    """
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        # Try multiple engines
        for eng in ("openpyxl", "xlrd", None):
            try:
                if eng is None:
                    return pd.read_excel(path)
                return pd.read_excel(path, engine=eng)
            except Exception:
                continue
        raise
    raise ValueError(f"Unsupported file type: {path}")


NAME_CLEAN_PAT = re.compile(
    r"""\b(
        SCHOOL\s*DIST(RICT)?|PUBLIC\s*SCHOOLS?|SCHOOLS?|UNIFIED|CITY|COUNTY|TOWNSHIP|
        INDEPENDENT|COMMUNITY|EDUCATION|EXCELLENCE|DEPARTMENT|
        SD|USD|ISD|CSD|JSD|ESD|R-?I?|CUSD|CUS|USD|U?S?D|
        #?\d+|NO\.\s*\d+
    )\b|[.,'’"()-]""",
    re.IGNORECASE | re.VERBOSE,
)


def normalize_state_abbr(s) -> str:
    if pd.isna(s):
        return np.nan
    return str(s).strip().upper()


def normalize_district_name(s: str) -> str:
    if pd.isna(s):
        return np.nan
    s0 = str(s)
    s0 = s0.replace("&", " AND ")
    s0 = re.sub(r"\s+", " ", s0).strip()
    s0 = NAME_CLEAN_PAT.sub(" ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0.upper()


def apply_aliases(name: str, state: str) -> str:
    """
    Minimal alias map for stubborn cases. Extend as needed.
    """
    if not isinstance(name, str):
        return name
    # Examples
    if state == "IL" and ("CHICAGO" in name or "299" in name):
        return "CHICAGO"
    if state == "CA" and ("LOS ANGELES" in name or "LA UNIFIED" in name):
        return "LOS ANGELES"
    if state == "NY" and ("NEW YORK CITY" in name or "NYC" in name):
        return "NEW YORK"
    return name


def build_keys_schools(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["state_abbr"] = df["State"].map(normalize_state_abbr)
    df["district_key"] = df["School District Name"].map(normalize_district_name)
    # Apply aliases (after normalization)
    df["district_key"] = [
        apply_aliases(n, st) for n, st in zip(df["district_key"], df["state_abbr"])
    ]
    return df


def find_col(cols, contains: str) -> str:
    for c in cols:
        if contains.lower() in c.lower():
            return c
    raise KeyError(f"Could not find a column containing: {contains}")


def build_keys_districts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Heuristically find the district name and state abbr columns
    name_col = None
    abbr_col = None
    for c in df.columns:
        lc = c.lower()
        if name_col is None and lc.startswith("agency name"):
            name_col = c
        if abbr_col is None and "state abbr" in lc:
            abbr_col = c
    if name_col is None:
        name_col = find_col(df.columns, "Agency Name")
    if abbr_col is None:
        abbr_col = find_col(df.columns, "State Abbr")

    df["state_abbr"] = df[abbr_col].map(normalize_state_abbr)
    df["district_key"] = df[name_col].map(normalize_district_name)
    df["district_key"] = [
        apply_aliases(n, st) for n, st in zip(df["district_key"], df["state_abbr"])
    ]
    return df


def left_join_schools_to_districts(schools: pd.DataFrame, districts: pd.DataFrame) -> pd.DataFrame:
    # Deduplicate districts to avoid many-to-many
    dedup = districts.sort_values(["state_abbr", "district_key"]).drop_duplicates(
        subset=["state_abbr", "district_key"], keep="first"
    )
    merged = schools.merge(
        dedup, on=["state_abbr", "district_key"], how="left", suffixes=("", "_DIST")
    )
    return merged


def maybe_merge_poverty(merged: pd.DataFrame, poverty: pd.DataFrame) -> pd.DataFrame:
    # Try to detect whether poverty is district keyed; if it has state + district name columns, reuse normalization.
    cols = [c.lower() for c in poverty.columns]
    # Common possibilities
    cand_state = None
    for c in poverty.columns:
        if c.lower() in {"state", "state_abbr", "state abbreviation", "state code"} or "abbr" in c.lower():
            cand_state = c
            break
    cand_dist = None
    for c in poverty.columns:
        if "district" in c.lower() and "name" in c.lower():
            cand_dist = c
            break

    # If we find a reasonable pair, join on normalized keys
    if cand_state and cand_dist:
        pov = poverty.copy()
        pov["state_abbr"] = pov[cand_state].map(normalize_state_abbr)
        pov["district_key"] = pov[cand_dist].map(normalize_district_name)
        pov = pov.drop_duplicates(subset=["state_abbr", "district_key"], keep="first")
        merged = merged.merge(pov, on=["state_abbr", "district_key"], how="left", suffixes=("", "_POV"))
        return merged

    # Otherwise, just return unmodified and ask user to provide mapping or convert
    return merged


def main():
    ap = argparse.ArgumentParser(description="Merge school-level file with NCES district data (and optional poverty data).")
    ap.add_argument("--schools", required=True, help="Path to school-level CSV (e.g., Map_Full Data_data.csv)")
    ap.add_argument("--districts", required=True, help="Path to NCES ELSI districts CSV (e.g., School District Data.csv)")
    ap.add_argument("--poverty", default=None, help="Optional poverty file (CSV/XLS/XLSX). Prefer CSV.")
    ap.add_argument("--out", default="merged_school_district_dataset.csv", help="Output merged CSV path")
    ap.add_argument("--unmatched", default="unmatched_sample.csv", help="Output CSV for unmatched sample (first N rows)")
    ap.add_argument("--unmatched_rows", type=int, default=50, help="How many unmatched rows to write to sample CSV")
    args = ap.parse_args()

    # Read inputs
    schools = read_schools(args.schools)
    districts = read_districts(args.districts)

    schools_k = build_keys_schools(schools)
    districts_k = build_keys_districts(districts)

    merged = left_join_schools_to_districts(schools_k, districts_k)

    # Report and sample unmatched
    matched_mask = merged.filter(regex=r"^Agency Name$", axis=1,).shape[1] > 0
    if matched_mask:
        match_rate = merged["Agency Name"].notna().mean()
    else:
        # If "Agency Name" header differs, estimate via any non-null district_key coming from right table
        join_hit = merged["district_key"].notna()
        match_rate = float(join_hit.mean())
    print(f"Join completed. Estimated match rate: {match_rate:.2%}. Rows: {len(merged)}. Columns: {merged.shape[1]}")

    # Optional poverty merge
    if args.poverty:
        try:
            pov = read_any_table(args.poverty)
            merged = maybe_merge_poverty(merged, pov)
            print("Poverty data merge attempted (keyed by normalized state+district name if available).")
        except Exception as e:
            print(f"WARNING: Failed to merge poverty data: {e}", file=sys.stderr)

    # Save merged
    merged.to_csv(args.out, index=False)
    print(f"Wrote merged dataset to: {args.out}")

    # Write unmatched sample
    # Heuristic: rows where we didn't carry over "Agency Name" (district file) are considered unmatched
    if "Agency Name" in merged.columns:
        unmatched = merged[merged["Agency Name"].isna()][["State", "School District Name", "School Name"]].head(args.unmatched_rows)
        unmatched.to_csv(args.unmatched, index=False)
        print(f"Wrote unmatched sample ({len(unmatched)} rows) to: {args.unmatched}")
    else:
        print("Note: 'Agency Name' column not found post-merge; unmatched sample not written.")


if __name__ == "__main__":
    main()
