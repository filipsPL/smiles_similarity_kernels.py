#!/usr/bin/env python3
"""
calculate_all_similarities.py
Run all similarity methods across all supported input representations and
write one CSV per method per variant.

Pipeline stages applied per variant:
    [convert]   --inchi / --selfies / (none = keep SMILES)
    [normalize] --canonicalize (SMILES only); ELEMENT_REPLACEMENTS auto on/off
    [augment]   --shuffle (random negative control) / --sort (deterministic negative control)

Output file naming:
    {representation}__{modifiers}__{method}.csv

  representation:  smiles | inchi_all | inchi_{layer} | selfies
  modifiers:       sorted, joined with +; drawn from:
                     canonical  – --canonicalize applied
                     replaced   – ELEMENT_REPLACEMENTS on (SMILES default)
                     shuffled   – --shuffle applied
                     sorted     – --sort applied
                   empty when none apply (e.g. InChI, SELFIES)

Examples:
    smiles__replaced__lingo.csv
    smiles__canonical+replaced__lingo.csv
    smiles__replaced+shuffled__lingo.csv
    smiles__replaced+sorted__lingo.csv
    smiles__raw__lingo.csv           (--no-preprocess benchmark)
    inchi_all__lingo.csv
    inchi_connections__lingo.csv
    selfies__lingo.csv

Usage:
    python calculate_all_similarities.py \\
        --templates examples/templates.smi \\
        --database  examples/database.smi \\
        --output-dir examples/outputs

    # Run only specific variants
    python calculate_all_similarities.py ... --variants smiles__replaced smiles__raw

    # Parallel execution (one process per variant)
    python calculate_all_similarities.py ... --jobs 4

    # Dry run — print commands without executing
    python calculate_all_similarities.py ... --dry-run

Parsing the stem back:
    repr_, mods_, method = stem.split("__")
    modifiers = set(mods_.split("+")) if mods_ else set()
"""

import argparse
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Variant definitions
#
# Each entry describes one pipeline configuration:
#   "repr"       – representation field in the output filename
#   "modifiers"  – ordered list of modifier tags (combined into filename)
#   "extra_args" – flags forwarded to smiles_similarity_kernels.py
#   "requires"   – Python packages that must be importable for this variant
#   "description"– human-readable label for --list-variants
#   "stage"      – primary pipeline stage for grouping in --list-variants
# ---------------------------------------------------------------------------

INCHI_LAYERS = [
    "connections",
    "formula",
    "hydrogens",
    "stereo_tet",
]


def variant_stem(variant: dict) -> str:
    """
    Build the filename stem for a variant:
        {repr}__{modifiers}
    where modifiers is sorted tags joined by '+', or empty string.
    The method name is prepended by smiles_similarity_kernels.py at runtime.
    """
    mods = "+".join(sorted(variant["modifiers"])) if variant["modifiers"] else ""
    return f"{variant['repr']}__{mods}"


VARIANTS = [
    # ── SMILES variants ────────────────────────────────────────────────────────
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
        "modifiers": ["raw"],
        "extra_args": ["--no-preprocess"],
        "requires": [],
        "description": "raw SMILES, no ELEMENT_REPLACEMENTS (benchmark)",
        "stage": "normalize",
    },
    {
        "repr": "smiles",
        "modifiers": ["canonical", "replaced"],
        "extra_args": ["--canonicalize"],
        "requires": ["rdkit"],
        "description": "canonicalized SMILES, ELEMENT_REPLACEMENTS on",
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
        "repr": "smiles",
        "modifiers": ["replaced", "sorted"],
        "extra_args": ["--sort"],
        "requires": [],
        "description": "sorted SMILES, deterministic negative control",
        "stage": "augment",
    },
    # ── InChI variants ─────────────────────────────────────────────────────────
    # ELEMENT_REPLACEMENTS is auto-disabled by the pipeline for InChI
    {
        "repr": "inchi_all",
        "modifiers": [],
        "extra_args": ["--inchi"],
        "requires": ["rdkit"],
        "description": "full InChI (all layers)",
        "stage": "convert",
    },
    *[
        {
            "repr": f"inchi_{layer}",
            "modifiers": [],
            "extra_args": ["--inchi", "--inchi-layer", layer],
            "requires": ["rdkit"],
            "description": f"InChI {layer} layer",
            "stage": "convert",
        }
        for layer in INCHI_LAYERS
    ],
    # ── SELFIES variant ────────────────────────────────────────────────────────
    # ELEMENT_REPLACEMENTS is auto-disabled by the pipeline for SELFIES
    {
        "repr": "selfies",
        "modifiers": [],
        "extra_args": ["--selfies"],
        "requires": ["selfies"],
        "description": "SELFIES",
        "stage": "convert",
    },
]


def check_available(requires: list[str]) -> list[str]:
    """Return list of missing packages from a requires list."""
    missing = []
    for pkg in requires:
        try:
            __import__("rdkit.Chem" if pkg == "rdkit" else pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def run_variant(
    templates: Path,
    database: Path,
    output_dir: Path,
    variant: dict,
    verbose: bool,
    dry_run: bool,
) -> tuple[bool, str | None, float, list[tuple[str, str, str]]]:
    """
    Run smiles_similarity_kernels.py --all-methods for one variant.
    Returns (success, skip_reason, elapsed_seconds, method_rows).
    method_rows is a list of (method, status, elapsed_s) strings from --timing-log.
    skip_reason is None on success or error; elapsed/method_rows are empty when skipped/dry-run.
    """
    missing = check_available(variant["requires"])
    if missing:
        return False, f"missing: {', '.join(missing)}", 0.0, []

    stem = variant_stem(variant)
    output_file = output_dir / f"{stem}.csv"

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        timing_log = Path(tf.name)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "smiles_similarity_kernels.py"),
        "--templates",
        str(templates),
        "--database",
        str(database),
        "--output",
        str(output_file),
        "--all-methods",
        "--timing-log",
        str(timing_log),
        *variant["extra_args"],
    ]
    if verbose:
        cmd.append("--verbose")

    if dry_run:
        timing_log.unlink(missing_ok=True)
        return True, None, 0.0, []

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    elapsed = time.perf_counter() - t0

    method_rows: list[tuple[str, str, str]] = []
    if timing_log.exists():
        for line in timing_log.read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                method_rows.append(tuple(parts[:3] + [""] * (3 - len(parts[:3]))))
        timing_log.unlink(missing_ok=True)

    if result.returncode != 0:
        return False, None, elapsed, method_rows

    return True, None, elapsed, method_rows


def count_outputs(output_dir: Path, stem: str) -> int:
    """Count files written for a variant (pattern: {stem}__{method}.csv)."""
    clean = stem.rstrip("_")
    return len(list(output_dir.glob(f"{clean}__*.csv")))


def parse_args() -> argparse.Namespace:
    valid_stems = [variant_stem(v) for v in VARIANTS]
    parser = argparse.ArgumentParser(
        description="Calculate all similarity methods across all input representations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--templates", "-t", default=None, help="Templates file or directory (.smi, .csv, .tsv)")
    parser.add_argument("--database", "-d", default=None, help="Database/library file or directory (.smi, .csv, .tsv)")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Directory for output CSV files (default: outputs/)")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=valid_stems,
        default=None,
        metavar="VARIANT",
        help=f"Run only these variants (default: all). Use --list-variants to see choices.",
    )
    parser.add_argument("--list-variants", action="store_true", help="Print all available variants and exit")
    parser.add_argument("--jobs", "-j", type=int, default=1, metavar="N", help="Run N variants in parallel (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-method progress inside each variant run")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_variants:
        print("Available variants:")
        print(f"  {'stem (use with --variants)':40s} {'stage':10s} {'description'}")
        print("  " + "-" * 80)
        for v in VARIANTS:
            missing = check_available(v["requires"])
            needs = f"  [needs: {', '.join(v['requires'])}]" if v["requires"] else ""
            avail = "  ✗ unavailable" if missing else ""
            print(f"  {variant_stem(v):40s} {v.get('stage', ''):10s} {v['description']}{needs}{avail}")
        return

    if not args.templates or not args.database:
        print("Error: --templates and --database are required.", file=sys.stderr)
        sys.exit(1)

    templates = Path(args.templates)
    database = Path(args.database)
    output_dir = Path(args.output_dir)

    if not templates.exists():
        print(f"Error: templates path not found: {templates}", file=sys.stderr)
        sys.exit(1)
    if not database.exists():
        print(f"Error: database path not found: {database}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.variants) if args.variants else None
    variants_to_run = [v for v in VARIANTS if selected is None or variant_stem(v) in selected]

    print(f"Templates : {templates}")
    print(f"Database  : {database}")
    print(f"Output dir: {output_dir}")
    print(f"Variants  : {len(variants_to_run)}")
    print()

    results: dict[str, tuple[bool, str | None, float, list]] = {}

    def _run(variant: dict) -> tuple[str, bool, str | None, float, list]:
        stem = variant_stem(variant)
        print(f"  → [{stem}] {variant['description']}")
        ok, reason, elapsed, method_rows = run_variant(templates, database, output_dir, variant, args.verbose, args.dry_run)
        if args.dry_run:
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--templates",
                str(templates),
                "--database",
                str(database),
                "--output",
                str(output_dir / f"{stem}.csv"),
                "--all-methods",
                *variant["extra_args"],
            ]
            print(f"     {' '.join(cmd)}")
        return stem, ok, reason, elapsed, method_rows

    jobs = min(args.jobs, len(variants_to_run))
    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_run, v): v for v in variants_to_run}
            for fut in as_completed(futures):
                suffix, ok, reason, elapsed, method_rows = fut.result()
                results[suffix] = (ok, reason, elapsed, method_rows)
    else:
        for variant in variants_to_run:
            suffix, ok, reason, elapsed, method_rows = _run(variant)
            results[suffix] = (ok, reason, elapsed, method_rows)

    # Summary
    print()
    print("Summary")
    print("-" * 60)
    ok_count = skipped_count = failed_count = total_files = 0
    # variant-level log (one row per variant)
    variant_log_lines = ["variant,status,elapsed_s,n_files"]
    # method-level log (one row per method per variant)
    method_log_lines = ["variant,method,status,elapsed_s"]
    for v in variants_to_run:
        stem = variant_stem(v)
        ok, reason, elapsed, method_rows = results[stem]
        if reason is not None:
            skipped_count += 1
            print(f"  SKIP  {stem:45s} ({reason})")
            variant_log_lines.append(f"{stem},skip,,")
        elif ok:
            ok_count += 1
            n = count_outputs(output_dir, stem) if not args.dry_run else 0
            total_files += n
            label = f"{n} files" if not args.dry_run else "dry-run"
            print(f"  OK    {stem:45s} ({elapsed:.1f}s, {label})")
            variant_log_lines.append(f"{stem},ok,{elapsed:.3f},{n}")
        else:
            failed_count += 1
            print(f"  FAIL  {stem:45s} ({elapsed:.1f}s)")
            variant_log_lines.append(f"{stem},fail,{elapsed:.3f},")
        for row in method_rows:
            method, status, method_elapsed = row[0], row[1], row[2] if len(row) > 2 else ""
            method_log_lines.append(f"{stem},{method},{status},{method_elapsed}")

    print()
    print(f"Completed: {ok_count}  Skipped: {skipped_count}  Failed: {failed_count}")
    if not args.dry_run:
        print(f"Total output files: {total_files}")
        (output_dir / "timing_variants.csv").write_text("\n".join(variant_log_lines) + "\n")
        (output_dir / "timing_methods.csv").write_text("\n".join(method_log_lines) + "\n")
        print(f"Timing logs       : {output_dir}/timing_variants.csv, timing_methods.csv")

    if failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
