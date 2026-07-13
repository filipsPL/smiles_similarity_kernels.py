#!/usr/bin/env python3
"""
Merge pre-computed fingerprint CSVs into combination fingerprints for ML evaluation.

Each dataset directory under 2-string-fp/ contains one CSV per fingerprint type:
  SMILES, name, activity, bit_0, bit_1, ...

This script loads the specified fingerprints, prefixes their feature columns, concatenates
them, and writes a new CSV alongside the originals.

Usage:
    python merge_fingerprints.py [--results-dir DIR] [--combos A+B [A+B ...]] [--dry-run]

Examples:
    # Run default combinations on all dataset directories
    python merge_fingerprints.py

    # Custom results directory
    python merge_fingerprints.py --results-dir path/to/2-string-fp

    # Only specific combinations
    python merge_fingerprints.py --combos smifp38+bpe128_count smifp38+phasmifp12

    # Preview without writing
    python merge_fingerprints.py --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Default combinations to generate. Each tuple is a set of fingerprint names to merge.
# The output filename will be the names joined with "+".
DEFAULT_COMBOS = [
    ("smifp38", "bpe128_count"),
    ("smifp38", "phasmifp12"),
    ("bpe128_count", "phasmifp12"),
    ("smifp38", "bpe128_count", "phasmifp12"),
    ("bpe128_count", "phasmifp12"),
    ("morgan2", "smifp38"),
    ("morgan2", "bpe128_binary"),
    ("morgan2", "phasmifp12_binary"),
    ("morgan2", "smifp38", "bpe128_binary"),
    ("morgan2", "smifp38", "phasmifp12_binary"),
    ("morgan2", "bpe128_binary", "phasmifp12_binary"),

]

META_COLS = ["SMILES", "name", "activity"]


def load_fingerprint(csv_path: Path, fp_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False, dtype={"name": str, "activity": str})
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df.rename(columns={c: f"{fp_name}__{c}" for c in feature_cols})
    return df


def merge_combo(dataset_dir: Path, fp_names: tuple[str, ...]) -> pd.DataFrame | None:
    parts = []
    for fp_name in fp_names:
        csv_path = dataset_dir / f"{fp_name}.csv"
        if not csv_path.exists():
            print(f"  SKIP: {csv_path.name} not found in {dataset_dir.name}", file=sys.stderr)
            return None
        df = load_fingerprint(csv_path, fp_name)
        parts.append(df)

    # Verify all parts have identical meta columns (same molecules, same order).
    # Compare as strings to avoid dtype-mismatch false negatives (e.g. numeric-looking names).
    ref = parts[0][META_COLS].astype(str)
    for i, part in enumerate(parts[1:], 1):
        if not ref.equals(part[META_COLS].astype(str)):
            mismatches = (ref != part[META_COLS].astype(str)).any(axis=1).sum()
            print(
                f"  ERROR: meta columns mismatch between {fp_names[0]} and {fp_names[i]} "
                f"in {dataset_dir.name} ({mismatches} row(s) differ)",
                file=sys.stderr,
            )
            return None

    meta = parts[0][META_COLS]
    feature_blocks = [p.drop(columns=META_COLS) for p in parts]
    merged = pd.concat([meta] + feature_blocks, axis=1)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        default="string_similarity_single_template/2-string-fp",
        help="Root directory containing per-dataset subdirectories (default: %(default)s)",
    )
    parser.add_argument(
        "--combos",
        nargs="+",
        metavar="A+B",
        help="Combinations to generate, e.g. smifp38+bpe128_count (default: built-in list)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing any files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"Results directory not found: {results_dir}")

    if args.combos:
        combos = [tuple(c.split("+")) for c in args.combos]
    else:
        combos = DEFAULT_COMBOS

    dataset_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and any(d.glob("*.csv"))
    )

    if not dataset_dirs:
        sys.exit(f"No dataset directories with CSVs found under {results_dir}")

    print(f"Found {len(dataset_dirs)} dataset(s): {[d.name for d in dataset_dirs]}")
    print(f"Combinations: {['+'.join(c) for c in combos]}\n")

    total_written = 0
    for dataset_dir in dataset_dirs:
        print(f"[{dataset_dir.name}]")
        for fp_names in combos:
            combo_name = "+".join(fp_names)
            out_path = dataset_dir / f"{combo_name}.csv"
            merged = merge_combo(dataset_dir, fp_names)
            if merged is None:
                continue
            n_features = len(merged.columns) - len(META_COLS)
            if args.dry_run:
                print(f"  DRY-RUN: would write {out_path.name} ({n_features} features, {len(merged)} rows)")
            else:
                merged.to_csv(out_path, index=False)
                print(f"  wrote {out_path.name} ({n_features} features, {len(merged)} rows)")
                total_written += 1

    if not args.dry_run:
        print(f"\nDone. {total_written} file(s) written.")


if __name__ == "__main__":
    main()
