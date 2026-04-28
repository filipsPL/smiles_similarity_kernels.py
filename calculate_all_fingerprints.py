#!/usr/bin/env python3
"""
calculate_all_fingerprints.py
Compute all fingerprint types across all supported input representations and
write one CSV per fingerprint per variant.

Pipeline stages applied per variant:
    [convert]   --inchi / --selfies / (none = keep SMILES)
    [normalize] --canonicalize (SMILES only); ELEMENT_REPLACEMENTS auto on/off
    [augment]   --shuffle (random negative control) / --sort (deterministic negative control)

Output file naming:
    {representation}__{modifiers}__{fingerprint}.csv

  representation:  smiles | selfies
  modifiers:       sorted tags joined by '+', or empty string; drawn from:
                     canonical  – --canonicalize applied
                     replaced   – ELEMENT_REPLACEMENTS on (SMILES default)
                     shuffled   – --shuffle applied
                     sorted     – --sort applied

  fingerprint:     smifp34 | smifp34_binary | smifp38 | smifp38_binary |
                   bpe_count | bpe_binary |
                   bpe16_count | bpe16_binary | ... | bpe1024_count | bpe1024_binary

Examples:
    smiles__canonical+replaced__smifp34.csv
    smiles__canonical+replaced__bpe512_count.csv
    smiles__replaced__smifp34_binary.csv
    selfies__selfies_tfidf11.csv  (not applicable here, just for illustration)

Note: InChI is not supported here — fingerprints are SMILES/SELFIES-native.
      For InChI similarity, see calculate_all_similarities.py.

Usage:
    python calculate_all_fingerprints.py \\
        --database  examples/database.smi \\
        --output-dir examples/fingerprints

    # Run only specific variants
    python calculate_all_fingerprints.py ... --variants smiles__replaced smiles__canonical+replaced

    # Run only specific fingerprints
    python calculate_all_fingerprints.py ... --fingerprints smifp34 bpe512_count bpe512_binary

    # Parallel execution (one process per variant×fingerprint combination)
    python calculate_all_fingerprints.py ... --jobs 4

    # Dry run — print commands without executing
    python calculate_all_fingerprints.py ... --dry-run

Parsing the stem back:
    repr_, mods_, fp = stem.split("__")
    modifiers = set(mods_.split("+")) if mods_ else set()
"""

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Variant definitions (input representation + normalization pipeline)
#
# Each entry:
#   "repr"       – representation field in the output filename
#   "modifiers"  – ordered list of modifier tags (combined into filename)
#   "extra_args" – flags forwarded to smiles_similarity_kernels.py
#   "requires"   – Python packages that must be importable for this variant
#   "description"– human-readable label for --list-variants
#   "stage"      – primary pipeline stage for grouping in --list-variants
# ---------------------------------------------------------------------------

VARIANTS = [
    {
        "repr": "smiles",
        "modifiers": ["replaced"],
        "extra_args": [],
        "requires": [],
        "description": "raw SMILES, ELEMENT_REPLACEMENTS on",
        "stage": "normalize",
    },
    {
        "repr": "smiles",
        "modifiers": ["canonical", "replaced"],
        "extra_args": ["--canonicalize"],
        "requires": ["rdkit"],
        "description": "canonicalized SMILES, ELEMENT_REPLACEMENTS on (recommended for ML)",
        "stage": "normalize",
    },
    {
        "repr": "smiles",
        "modifiers": ["replaced", "shuffled"],
        "extra_args": ["--shuffle", "--shuffle-seed", "42"],
        "requires": [],
        "description": "shuffled SMILES, negative control (seed=42)",
        "stage": "augment",
    },
    {
        "repr": "selfies",
        "modifiers": [],
        "extra_args": ["--selfies"],
        "requires": ["selfies"],
        "description": "SELFIES representation",
        "stage": "convert",
    },
]

# Fingerprint types that make sense per representation.
# smifp works on SMILES and SELFIES characters equally (character counting).
# bpe was trained on SMILES, so apply only to SMILES-derived representations.
SMILES_FINGERPRINTS = [
    "smifp34",
    "smifp34_binary",
    "smifp38",
    "smifp38_binary",
    "bpe_count",
    "bpe_binary",
    "bpe16_count",
    "bpe16_binary",
    "bpe32_count",
    "bpe32_binary",
    "bpe64_count",
    "bpe64_binary",
    "bpe128_count",
    "bpe128_binary",
    "bpe256_count",
    "bpe256_binary",
    "bpe512_count",
    "bpe512_binary",
    "bpe1024_count",
    "bpe1024_binary",
]

SELFIES_FINGERPRINTS = [
    "smifp34",
    "smifp34_binary",
    "smifp38",
    "smifp38_binary",
]

ALL_FINGERPRINTS = sorted(set(SMILES_FINGERPRINTS + SELFIES_FINGERPRINTS))


def fingerprints_for_variant(variant: dict) -> list[str]:
    if variant["repr"] == "selfies":
        return SELFIES_FINGERPRINTS
    return SMILES_FINGERPRINTS


def variant_stem(variant: dict) -> str:
    mods = "+".join(sorted(variant["modifiers"])) if variant["modifiers"] else ""
    return f"{variant['repr']}__{mods}"


def check_available(requires: list[str]) -> list[str]:
    missing = []
    for pkg in requires:
        try:
            __import__("rdkit.Chem" if pkg == "rdkit" else pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def run_job(
    database: Path,
    output_dir: Path,
    variant: dict,
    fp_type: str,
    verbose: bool,
    dry_run: bool,
    overwrite: bool,
) -> tuple[bool, str | None, float]:
    """
    Run smiles_similarity_kernels.py --fingerprint for one (variant, fp_type) pair.
    Returns (success, skip_reason, elapsed_seconds).
    skip_reason is None on success/error; elapsed is 0.0 when skipped/dry-run.
    """
    missing = check_available(variant["requires"])
    if missing:
        return False, f"missing: {', '.join(missing)}", 0.0

    stem = variant_stem(variant)
    output_file = output_dir / f"{stem}__{fp_type}.csv"

    if not overwrite and output_file.exists():
        return True, f"exists: {output_file.name}", 0.0

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "smiles_similarity_kernels.py"),
        "--database", str(database),
        "--output", str(output_file),
        "--fingerprint", fp_type,
        *variant["extra_args"],
    ]
    if verbose:
        cmd.append("--verbose")
    if overwrite:
        cmd.append("--overwrite")

    if dry_run:
        return True, None, 0.0

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        return False, None, elapsed

    return True, None, elapsed


def parse_args() -> argparse.Namespace:
    valid_stems = [variant_stem(v) for v in VARIANTS]
    parser = argparse.ArgumentParser(
        description="Compute all fingerprint types across all input representations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--database", "-d", default=None,
        help="Database/library file or directory (.smi, .csv, .tsv)",
    )
    parser.add_argument(
        "--output-dir", "-o", default="fingerprints",
        help="Directory for output CSV files (default: fingerprints/)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=valid_stems,
        default=None,
        metavar="VARIANT",
        help="Run only these representation variants (default: all). Use --list-variants to see choices.",
    )
    parser.add_argument(
        "--fingerprints",
        nargs="+",
        choices=ALL_FINGERPRINTS,
        default=None,
        metavar="FP",
        help="Run only these fingerprint types (default: all applicable per variant). Use --list-fingerprints to see choices.",
    )
    parser.add_argument(
        "--list-variants", action="store_true",
        help="Print all available representation variants and exit",
    )
    parser.add_argument(
        "--list-fingerprints", action="store_true",
        help="Print all available fingerprint types and exit",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, metavar="N",
        help="Run N jobs in parallel (default: 1)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-molecule progress inside each run",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files (default: skip existing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_variants:
        print("Available representation variants:")
        print(f"  {'stem (use with --variants)':45s} {'stage':10s} {'description'}")
        print("  " + "-" * 90)
        for v in VARIANTS:
            missing = check_available(v["requires"])
            needs = f"  [needs: {', '.join(v['requires'])}]" if v["requires"] else ""
            avail = "  ✗ unavailable" if missing else ""
            fps = fingerprints_for_variant(v)
            print(
                f"  {variant_stem(v):45s} {v.get('stage', ''):10s} "
                f"{v['description']}{needs}{avail}  ({len(fps)} fingerprints)"
            )
        return

    if args.list_fingerprints:
        print("Available fingerprint types (by representation):")
        for v in VARIANTS:
            fps = fingerprints_for_variant(v)
            print(f"\n  [{variant_stem(v)}]")
            for fp in fps:
                print(f"    {fp}")
        return

    if not args.database:
        print("Error: --database is required.", file=sys.stderr)
        sys.exit(1)

    database = Path(args.database)
    output_dir = Path(args.output_dir)

    if not database.exists():
        print(f"Error: database path not found: {database}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    selected_variants = set(args.variants) if args.variants else None
    selected_fps = set(args.fingerprints) if args.fingerprints else None

    variants_to_run = [v for v in VARIANTS if selected_variants is None or variant_stem(v) in selected_variants]

    # Build list of (variant, fp_type) jobs
    jobs: list[tuple[dict, str]] = []
    for v in variants_to_run:
        applicable = fingerprints_for_variant(v)
        for fp in applicable:
            if selected_fps is None or fp in selected_fps:
                jobs.append((v, fp))

    print(f"Database   : {database}")
    print(f"Output dir : {output_dir}")
    print(f"Variants   : {len(variants_to_run)}")
    print(f"Jobs total : {len(jobs)}")
    print()

    results: dict[tuple[str, str], tuple[bool, str | None, float]] = {}

    def _run(variant: dict, fp_type: str) -> tuple[str, str, bool, str | None, float]:
        stem = variant_stem(variant)
        label = f"{stem}__{fp_type}"
        ok, reason, elapsed = run_job(database, output_dir, variant, fp_type, args.verbose, args.dry_run, args.overwrite)
        if args.dry_run:
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--database", str(database),
                "--output", str(output_dir / f"{label}.csv"),
                "--fingerprint", fp_type,
                *variant["extra_args"],
            ]
            print(f"  [dry] {' '.join(cmd)}")
        return stem, fp_type, ok, reason, elapsed

    n_jobs = min(args.jobs, len(jobs)) if jobs else 1
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_run, v, fp): (v, fp) for v, fp in jobs}
            for fut in as_completed(futures):
                stem, fp_type, ok, reason, elapsed = fut.result()
                results[(stem, fp_type)] = (ok, reason, elapsed)
    else:
        for variant, fp_type in jobs:
            stem, fp_type, ok, reason, elapsed = _run(variant, fp_type)
            results[(stem, fp_type)] = (ok, reason, elapsed)

    # Summary
    print()
    print("Summary")
    print("-" * 72)
    ok_count = skipped_count = failed_count = 0
    log_lines = ["variant,fingerprint,status,elapsed_s"]

    for v in variants_to_run:
        stem = variant_stem(v)
        applicable = fingerprints_for_variant(v)
        for fp in applicable:
            if selected_fps is not None and fp not in selected_fps:
                continue
            ok, reason, elapsed = results.get((stem, fp), (False, "not run", 0.0))
            label = f"{stem}__{fp}"
            if reason is not None:
                if reason.startswith("exists:"):
                    skipped_count += 1
                    print(f"  SKIP  {label:60s} ({reason})")
                    log_lines.append(f"{stem},{fp},skip,")
                else:
                    skipped_count += 1
                    print(f"  SKIP  {label:60s} ({reason})")
                    log_lines.append(f"{stem},{fp},skip,")
            elif ok:
                ok_count += 1
                label_display = f"  OK    {label:60s} ({elapsed:.1f}s)"
                if args.dry_run:
                    label_display = f"  OK    {label:60s} (dry-run)"
                print(label_display)
                log_lines.append(f"{stem},{fp},ok,{elapsed:.3f}")
            else:
                failed_count += 1
                print(f"  FAIL  {label:60s} ({elapsed:.1f}s)")
                log_lines.append(f"{stem},{fp},fail,{elapsed:.3f}")

    print()
    print(f"Completed: {ok_count}  Skipped: {skipped_count}  Failed: {failed_count}")

    if not args.dry_run:
        timing_path = output_dir / "timing.csv"
        timing_path.write_text("\n".join(log_lines) + "\n")
        print(f"Timing log: {timing_path}")

    if failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
