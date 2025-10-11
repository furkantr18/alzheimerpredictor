"""Normalize raw Alzheimer dataset and write processed CSV.

Usage:
    python data/processed/normalize_raw.py

Behavior:
- Loads the raw CSV located at config.RAW_DATA_PATH
- Drops identifier columns defined in exploratory_plots.IDENTIFIER_COLS if present
- Normalizes numeric columns using min-max scaling to [0,1]
- Converts boolean-like columns (True/False) to 1/0
- Converts explicit binary-like (only 0/1) to integers (0/1)
- Factorizes categorical/object columns to integer codes (keeps a mapping file)
- Writes processed CSV to data/processed/processed_data.csv
- Writes optional mapping files under data/processed/mappings/

This script is intentionally conservative and will not change columns that cannot be
safely transformed. It produces logging to stdout about actions taken.
"""
from pathlib import Path
import pandas as pd
import json
import argparse
import sys

# Project root discovery (same approach as exploratory_plots)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_PATH

RAW_PATH = Path(RAW_DATA_PATH)
OUT_DIR = PROJECT_ROOT / 'data' / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAPPING_DIR = OUT_DIR / 'mappings'
MAPPING_DIR.mkdir(parents=True, exist_ok=True)

# We keep PatientID in the processed output but do not preprocess it.
# Only drop DoctorInCharge (sensitive personnel identifier) if present.
IDENTIFIERS = ('DoctorInCharge',)

# Column to preserve (present in output but excluded from preprocessing)
PRESERVE_COL = 'PatientID'


def load_raw(sample_rows: int | None = None) -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found: {RAW_PATH}")
    if sample_rows is None:
        return pd.read_csv(RAW_PATH)
    return pd.read_csv(RAW_PATH, nrows=sample_rows)


def normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize df in-place and return (df_out, metadata)

    metadata contains mapping info for categorical columns and min/max for numeric columns.
    """
    df = df.copy()
    # drop non-patient identifier columns if present (but preserve PatientID)
    df.drop(columns=[c for c in IDENTIFIERS if c in df.columns], errors='ignore', inplace=True)

    metadata = {
        'numeric': {},
        'categorical_mappings': {},
        'boolean_cols': [],
        'binary_cols': []
    }

    # First pass: detect dtypes and basic transforms
    for col in df.columns:
        # never preprocess the preserved patient identifier
        if col == PRESERVE_COL:
            continue
        ser = df[col]
        # boolean-like -> convert to 0/1
        if ser.dropna().isin([True, False]).all() and ser.dropna().nunique() <= 2:
            df[col] = ser.astype(int)
            metadata['boolean_cols'].append(col)
            continue
        # numeric-like strings -> try coerce
        if ser.dtype == object:
            # heuristics: try to coerce to numeric; if majority converts, use numeric
            coerced = pd.to_numeric(ser, errors='coerce')
            non_null_before = ser.notnull().sum()
            non_null_after = coerced.notnull().sum()
            if non_null_after >= max(1, int(non_null_before * 0.8)):
                df[col] = coerced
        # if numeric, process normalization later

    # Second pass: handle numeric normalization and detect binary numeric columns
    # exclude preserved PatientID from numeric normalization even if numeric-coded
    numeric_cols = [c for c in df.select_dtypes(include=['number']).columns.tolist() if c != PRESERVE_COL]
    for col in numeric_cols:
        ser = df[col]
        if ser.dropna().isin([0, 1]).all() and ser.dropna().nunique() <= 2:
            # binary flag, keep as-is but coerce to int
            df[col] = ser.astype('Int64')
            metadata['binary_cols'].append(col)
            continue
        # compute min/max and scale to [0,1]
        try:
            minv = ser.min()
            maxv = ser.max()
            if pd.isna(minv) or pd.isna(maxv) or minv == maxv:
                # cannot scale constant or missing-only column
                metadata['numeric'][col] = {'min': minv, 'max': maxv, 'scaled': False}
                continue
            df[col] = (ser - minv) / (maxv - minv)
            metadata['numeric'][col] = {'min': float(minv), 'max': float(maxv), 'scaled': True}
        except Exception:
            metadata['numeric'][col] = {'min': None, 'max': None, 'scaled': False}

    # Third pass: factorize remaining object / categorical columns
    # exclude PatientID from factorization
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns.tolist() if c != PRESERVE_COL]
    for col in cat_cols:
        try:
            codes, uniques = pd.factorize(df[col], sort=True)
            df[col] = pd.Series(codes).astype('Int64')
            mapping = {str(code): str(val) for code, val in enumerate(list(uniques))}
            metadata['categorical_mappings'][col] = mapping
            # persist mapping to a JSON file
            with open(MAPPING_DIR / f'{col}_mapping.json', 'w', encoding='utf-8') as fh:
                json.dump(mapping, fh, ensure_ascii=False, indent=2)
        except Exception:
            # skip if factorization fails
            continue

    return df, metadata


def main(sample_rows: int | None = None, out_name: str | None = None) -> Path:
    df = load_raw(sample_rows=sample_rows)
    print(f'Loaded raw data: {len(df)} rows, {len(df.columns)} columns')
    processed_df, meta = normalize_dataframe(df)
    out_name = out_name or 'processed_data.csv'
    out_path = OUT_DIR / out_name
    processed_df.to_csv(out_path, index=False)
    # save metadata for reproducibility
    with open(OUT_DIR / 'processing_metadata.json', 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f'Wrote processed data to: {out_path}')
    print(f'Wrote metadata to: {OUT_DIR / "processing_metadata.json"}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalize raw Alzheimer dataset and write processed CSV')
    parser.add_argument('--sample-rows', type=int, default=None, help='Read only first N rows for quick testing')
    parser.add_argument('--out-name', type=str, default=None, help='Filename for processed CSV (default: processed_data.csv)')
    args = parser.parse_args()
    main(sample_rows=args.sample_rows, out_name=args.out_name)
