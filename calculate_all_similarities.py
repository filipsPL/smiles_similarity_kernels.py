#!/usr/bin/env python3
"""
calculate_all_similarities.py
Run all similarity methods across all supported input representations and
write one CSV per method per variant.

Pipeline stages applied per variant:
    [convert]   --inchi / --selfies / (none = keep SMILES)
    [normalize] --canonicalize (SMILES only); ELEMENT_REPLACEMENTS auto on/off
    [augment]   --shuffle (random ordering ablation) / --sort (deterministic ordering ablation)
                (NOT a random/null baseline -- both preserve character composition;
                see smiles_similarity_kernels.py's shuffle_smiles/sort_string docstrings)

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
# Method exclusion rules
#
# Each prefix listed here will exclude any method whose name starts with it.
# Add new prefixes here when adding representations or method families.
#
# Rationale:
#   tok-smiles / tok-schwaller / tok-bpe  — tokenize on SMILES-specific tokens
#       (Cl, Br, @@, ring digits…); meaningless on InChI or SELFIES strings
#   tok-selfies   — splits on [...] SELFIES brackets; meaningless on SMILES/InChI
#   smifp         — counts SMILES-alphabet characters after ELEMENT_REPLACEMENTS;
#       SELFIES uses a different bracket alphabet, so counts are not comparable
#   lingo*        — applies ring-number normalization (SMILES-specific step);
#       semantically wrong on SELFIES
# ---------------------------------------------------------------------------

# methods invalid for InChI representations
_INCHI_EXCLUDED = frozenset(["tok-smiles_tfidf", "tok-schwaller_tfidf", "tok-bpe", "tok-selfies_tfidf"])

# methods invalid for SELFIES representation
# "lingo" catches lingo/lingo3/lingo_tversky/lingo_tfidf/etc. by prefix, but
# "spaced_lingo" doesn't start with "lingo" despite being the same LINGO
# q-gram family (get_lingos on raw characters) -- needs its own entry or it
# silently slips through onto SELFIES's bracket-token strings.
_SELFIES_EXCLUDED = frozenset(["tok-smiles_tfidf", "tok-schwaller_tfidf", "tok-bpe", "smifp", "lingo", "spaced_lingo"])

# ---------------------------------------------------------------------------
# TF-IDF pruning
#
# Running the full ngram grid (1,1)–(6,6) = 21 combos per tokenizer family is
# a hyperparameter sweep; _TFIDF_KEEP_NGRAMS/_BPE_KEEP_SIZES below can narrow
# it to a representative subset (empirically (4,4) performs best) to cut
# compute time. This mattered a great deal when TF-IDF/BPE was the pipeline's
# dominant cost by orders of magnitude (see git history / root CLAUDE.md) —
# it no longer is, now that both the retokenization redundancy and
# SMILESTokenizerBPE's own algorithm are fixed (µs/pair, not ms–s/pair), so
# _PRUNE_TFIDF_GRID defaults to False (full grid) as of 2026-07-15. Flip it
# back to True for a quick/cheap exploratory run; the keep-sets are left
# in place either way.
# ---------------------------------------------------------------------------

_PRUNE_TFIDF_GRID = False

# ngram suffixes to keep when _PRUNE_TFIDF_GRID is True, e.g. "11", "12", "44"
_TFIDF_KEEP_NGRAMS = frozenset(["11", "12", "44"])

# BPE merge counts to keep when _PRUNE_TFIDF_GRID is True (plus the unsuffixed
# full-vocabulary variant)
_BPE_KEEP_SIZES = frozenset(["64", "256", "1024"])

# ---------------------------------------------------------------------------
# Redundant methods (2026-09-26 benchmark review)
#
# Each of these is an exact monotonic transform of, or an unconditional
# delegation to, another registered method (documented in
# smiles_similarity_kernels.py's own docstrings/AVAILABLE_METHODS
# descriptions) -- it produces identical rankings and therefore identical
# AUROC/BEDROC/EF to the method it transforms. Computing and storing it adds
# wall-clock and disk with zero new information for the benchmark. Excluded
# from --all-methods by default; pass --keep-redundant-methods to include
# them anyway (e.g. to re-verify the equivalence empirically).
#   - lingo_dice_binary    = 2*J/(1+J) of lingo_jaccard_binary
#   - lingo_ruzicka        = lingo_tversky_similarity(alpha=1, beta=1) (delegates)
#   - lingo_tversky_sym    = lingo_tversky_similarity(alpha=0.5, beta=0.5) = lingo_dice
#   - spectrum_tversky_sym = spectrum_kernel_similarity(coefficient="tversky", alpha=beta=0.5)
#                            = multiset Dice, same arithmetic as lingo_dice's family
# ---------------------------------------------------------------------------
REDUNDANT_METHODS = frozenset(
    [
        "lingo_dice_binary",
        "lingo_ruzicka",
        "lingo_tversky_sym",
        "spectrum_tversky_sym",
    ]
)

# ---------------------------------------------------------------------------
# Expensive methods (2026-09-29 benchmark review)
#
# Real, measured cost -- not a heuristic guess -- from running the full
# 14-dataset lit-pcba pass plus the (removed) MoleculeNet-HIV run:
#   - monge_elkan_sym : 35,212s (9.8h) for ONE InChI variant on adrb2's 5.3M
#     pairs, even with --sim-jobs 4. O(tokens1 x tokens2) per pair, run
#     bidirectionally -- the single worst per-pair cost measured anywhere in
#     this module.
#   - monge_elkan     : same mechanism, single-direction (~half the cost).
#   - mismatch5       : 18.0 CPU-days on HIV's 56.6M pairs (single-threaded,
#     pre --sim-jobs) -- the worst HIV number by a wide margin. Also the
#     method implicated in a ~900GB RSS blowup when combined with SELFIES
#     representation on adrb2 (root cause not yet isolated -- see CLAUDE.md).
#   - mismatch / mismatch3 : same combinatorial neighbourhood-expansion
#     kernel, smaller k. HIV: 9.2 / 5.0 CPU-days respectively.
# None of these five are in BATCH_FEATURIZERS (no vectorized/featurize-once
# path exists for them), so they are also exactly the methods --sim-jobs
# helps -- and even parallelized, the numbers above hold. Excluded from
# --all-methods by default, like REDUNDANT_METHODS; pass
# --keep-expensive-methods to include them anyway. Deliberately NOT cutting
# edit/nlcs/clcs/token_edit/lcs_substring here -- expensive too (HIV:
# 12.6h-1.4 CPU-days) but nowhere near this tier, and they're more
# standard/citable methods for a benchmarking paper.
# ---------------------------------------------------------------------------
EXPENSIVE_METHODS = frozenset(
    [
        "monge_elkan",
        "monge_elkan_sym",
        "mismatch",
        "mismatch3",
        "mismatch5",
    ]
)


def slow_methods() -> frozenset[str]:
    """
    Return the set of registered methods with no batch/vectorized fast path.

    Derived directly from smiles_similarity_kernels.py's own BATCH_FEATURIZERS
    registry and its TF-IDF family detector (_tfidf_family), rather than a
    hand-maintained duplicate list, so it can't silently go stale as methods
    are added or moved between the two paths.

    These are exactly the methods smiles_similarity_kernels.py's own --jobs/
    n_jobs multiprocessing path exists for ("only speeds up methods that fall
    through to the general per-pair path"), and empirically the ones whose
    per-pair cost is dominated by an O(len^2) DP (edit/nlcs/clcs/token_edit/
    lcs_substring) or a combinatorial neighbourhood expansion (mismatch*) or
    an O(tokens^2) double loop (monge_elkan*) -- on MoleculeNet-HIV's
    inchi_all variant (1409 x 40166 = 56.6M pairs) these 14 methods alone
    took a combined ~46+ CPU-days on a single core, while every batched
    method (lingo*/spectrum*/smifp*/tfidf*/ncd/substring/subsequence*)
    finished the same variant in under a day combined. See the redesign
    notes for the 2026-09-26 benchmark review.
    """
    import smiles_similarity_kernels as _ssk

    return frozenset(name for name in _ssk.AVAILABLE_METHODS if name not in _ssk.BATCH_FEATURIZERS and _ssk._tfidf_family(name) is None)


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
    # "stereo_tet",
]

# Multi-layer InChI combinations -- extract_inchi_layers()/smiles_to_inchi_layers()
# already accept a list, but each combo needs its own VARIANTS entry to actually
# be run. "connections+hydrogens" is all-structure/no-stereo: heavy-atom
# connectivity plus H placement, without the digit-and-symbol noise the full
# InChI's stereo layers add for molecules that don't have stereocentres.
INCHI_LAYER_COMBOS = [
    ["connections", "hydrogens"],
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
        "description": "shuffled SMILES, ordering ablation (seed=42)",
        "stage": "augment",
    },
    {
        "repr": "smiles",
        "modifiers": ["replaced", "sorted"],
        "extra_args": ["--sort"],
        "requires": [],
        "description": "sorted SMILES, deterministic ordering ablation",
        "stage": "augment",
    },
    # ── Murcko scaffold variant ────────────────────────────────────────────────
    # Ring systems + linkers only, substituents stripped -- turns the whole
    # method battery into scaffold similarity. Canonicalized so scaffold SMILES
    # ordering isn't an artifact of the parent molecule's original atom order.
    {
        "repr": "murcko_scaffold",
        "modifiers": ["canonical", "replaced"],
        "extra_args": ["--murcko-scaffold", "--canonicalize"],
        "requires": ["rdkit"],
        "description": "Bemis-Murcko scaffold SMILES, canonicalized",
        "stage": "convert",
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
        "excluded_method_prefixes": _INCHI_EXCLUDED,
    },
    *[
        {
            "repr": f"inchi_{layer}",
            "modifiers": [],
            "extra_args": ["--inchi", "--inchi-layer", layer],
            "requires": ["rdkit"],
            "description": f"InChI {layer} layer",
            "stage": "convert",
            "excluded_method_prefixes": _INCHI_EXCLUDED,
        }
        for layer in INCHI_LAYERS
    ],
    *[
        {
            "repr": f"inchi_{'_'.join(layers)}",
            "modifiers": [],
            # smiles_to_inchi_layers/extract_inchi_layers already accept a list
            # of layers; --inchi-layer just needs the comma-separated form.
            "extra_args": ["--inchi", "--inchi-layer", ",".join(layers)],
            "requires": ["rdkit"],
            "description": f"InChI {'+'.join(layers)} layers combined",
            "stage": "convert",
            "excluded_method_prefixes": _INCHI_EXCLUDED,
        }
        for layers in INCHI_LAYER_COMBOS
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
        "excluded_method_prefixes": _SELFIES_EXCLUDED,
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


def _is_tfidf_method_kept(name: str) -> bool:
    """
    Return False for TF-IDF methods that fall outside the pruned ngram/BPE grid.

    A no-op (always returns True) when _PRUNE_TFIDF_GRID is False — the
    current default, full-grid coverage. When True:

    Kept:
      - Non-TF-IDF methods (always kept).
      - tok-{family}_tfidf{mn} where mn ∈ _TFIDF_KEEP_NGRAMS.
      - tok-bpe{k}_tfidf{mn} where k ∈ _BPE_KEEP_SIZES and mn ∈ _TFIDF_KEEP_NGRAMS.
      - The unsuffixed aliases (tok-smiles_tfidf, tok-bpe_tfidf, …) which map to (1,2).
      - tok-bpe{k}_tfidf (unsuffixed alias) where k ∈ _BPE_KEEP_SIZES.

    Excluded:
      - tok-{family}_tfidf{mn} where mn ∉ _TFIDF_KEEP_NGRAMS.
      - tok-bpe{k}_tfidf{mn/alias} where k ∉ _BPE_KEEP_SIZES.
    """
    if not _PRUNE_TFIDF_GRID:
        return True
    if not name.startswith("tok-"):
        return True
    # Identify tfidf methods by the "_tfidf" marker
    tfidf_idx = name.find("_tfidf")
    if tfidf_idx == -1:
        return True  # not a tfidf method

    ngram_suffix = name[tfidf_idx + len("_tfidf") :]  # e.g. "44", "12", "" (alias)
    prefix = name[:tfidf_idx]  # e.g. "tok-smiles", "tok-bpe64", "tok-bpe"

    # Check ngram suffix: "" (alias → maps to "12") is always kept
    if ngram_suffix and ngram_suffix not in _TFIDF_KEEP_NGRAMS:
        return False

    # For BPE methods, also check the merge-count size
    # prefix is "tok-bpe{k}" for fixed-size or "tok-bpe" for full-vocab
    if prefix.startswith("tok-bpe"):
        size_str = prefix[len("tok-bpe") :]  # e.g. "64", "256", "" (full-vocab)
        if size_str and size_str not in _BPE_KEEP_SIZES:
            return False

    return True


def get_valid_methods(
    variant: dict,
    exclude_methods: frozenset[str] = frozenset(),
    only_methods: list[str] | None = None,
) -> list[str] | None:
    """
    Return the list of method names valid for this variant.

    Applies TF-IDF/BPE ngram-grid pruning (_TFIDF_KEEP_NGRAMS, _BPE_KEEP_SIZES)
    when _PRUNE_TFIDF_GRID is True; a no-op otherwise (current default: full
    grid). Additionally filters by variant's 'excluded_method_prefixes' when
    present — this is a *representation-validity* filter (e.g. tok-smiles_tfidf*
    is meaningless on InChI), independent of _PRUNE_TFIDF_GRID, and always applies.

    exclude_methods : method names to drop regardless of the above (e.g.
        REDUNDANT_METHODS, or slow_methods() when splitting a run in two —
        see run_variant).
    only_methods : if given, restricts the candidate set to exactly these
        names before the other filters apply (e.g. slow_methods(), for a
        template-subsampled "slow methods only" sub-run). None means "every
        registered method is a candidate" (the normal case).

    Returns None only when no filtering at all is needed (no exclusions, no
    only_methods restriction, and TF-IDF pruning is off), so the caller can
    use --all-methods instead.
    """
    excluded_prefixes = variant.get("excluded_method_prefixes", frozenset())

    import smiles_similarity_kernels as _ssk  # lazy import

    candidates = _ssk.AVAILABLE_METHODS if only_methods is None else [m for m in only_methods if m in _ssk.AVAILABLE_METHODS]

    valid = [
        name
        for name in candidates
        if not any(name.startswith(pfx) for pfx in excluded_prefixes) and _is_tfidf_method_kept(name) and name not in exclude_methods
    ]

    # If nothing was filtered out, signal the caller to use --all-methods
    if only_methods is None and not exclude_methods and len(valid) == len(_ssk.AVAILABLE_METHODS):
        return None

    return valid


def run_variant(
    templates: Path,
    database: Path,
    output_dir: Path,
    variant: dict,
    verbose: bool,
    dry_run: bool,
    overwrite: bool = False,
    exclude_methods: frozenset[str] = frozenset(),
    only_methods: list[str] | None = None,
    sim_jobs: int = 1,
) -> tuple[bool, str | None, float, list[tuple[str, str, str]]]:
    """
    Run smiles_similarity_kernels.py --all-methods for one variant.
    Returns (success, skip_reason, elapsed_seconds, method_rows).
    method_rows is a list of (method, status, elapsed_s) strings from --timing-log.
    skip_reason is None on success or error; elapsed/method_rows are empty when skipped/dry-run.

    exclude_methods / only_methods : see get_valid_methods.
    sim_jobs : forwarded as smiles_similarity_kernels.py's own --jobs, i.e. the
        number of worker *processes* used for the general per-pair path inside
        ONE method's matrix computation (edit/nlcs/clcs/mismatch*/monge_elkan*/
        jellyfish-based methods/...). This is independent of, and multiplies
        with, this script's own --jobs (which parallelizes across *variants*,
        each already its own OS process) -- size sim_jobs so
        (variant --jobs) x sim_jobs x (dataset-level parallelism, if any)
        does not oversubscribe the machine. Default 1 preserves the previous
        behaviour (sequential per-pair loop) exactly.
    """
    missing = check_available(variant["requires"])
    if missing:
        return False, f"missing: {', '.join(missing)}", 0.0, []

    stem = variant_stem(variant)
    output_file = output_dir / f"{stem}.csv.gz"

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        timing_log = Path(tf.name)

    valid_methods = get_valid_methods(variant, exclude_methods=exclude_methods, only_methods=only_methods)
    if valid_methods is None:
        method_args = ["--all-methods"]
    elif not valid_methods:
        # Every candidate method was filtered out for this variant (e.g. an
        # only_methods=slow_methods() sub-run against a variant whose
        # excluded_method_prefixes happens to remove all of them) -- nothing
        # to do, not an error.
        timing_log.unlink(missing_ok=True)
        return True, "no applicable methods for this variant", 0.0, []
    else:
        method_args = ["--methods", *valid_methods]

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "smiles_similarity_kernels.py"),
        "--templates",
        str(templates),
        "--database",
        str(database),
        "--output",
        str(output_file),  # .csv.gz — smiles_similarity_kernels.py detects extension
        *method_args,
        "--timing-log",
        str(timing_log),
        *variant["extra_args"],
    ]
    if sim_jobs != 1:
        cmd += ["--jobs", str(sim_jobs)]
    if verbose:
        cmd.append("--verbose")
    if overwrite:
        cmd.append("--overwrite")

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
    """Count files written for a variant (pattern: {stem}__{method}.csv.gz)."""
    clean = stem.rstrip("_")
    return len(list(output_dir.glob(f"{clean}__*.csv.gz")))


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
    parser.add_argument(
        "--sim-jobs",
        type=int,
        default=1,
        metavar="N",
        help="Worker PROCESSES for the per-pair similarity computation INSIDE each variant "
        "(forwarded as smiles_similarity_kernels.py's own --jobs; default 1 = sequential, -1 = all "
        "CPUs). Independent of --jobs above, which parallelizes across variants -- the two multiply, "
        "so size them so the product does not oversubscribe the machine. Only affects methods with no "
        "batch/vectorized path (edit/nlcs/clcs/mismatch*/monge_elkan*/jellyfish-based methods/...); "
        "a no-op for lingo*/spectrum*/smifp*/tfidf* (already vectorized).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-method progress inside each variant run")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files. Without this flag, existing files are skipped with a warning."
    )
    parser.add_argument(
        "--exclude-methods",
        nargs="+",
        default=[],
        metavar="METHOD",
        help="Additional method names to exclude on top of REDUNDANT_METHODS/EXPENSIVE_METHODS (see "
        "--keep-redundant-methods/--keep-expensive-methods to disable those default exclusions instead "
        "of adding to them).",
    )
    parser.add_argument(
        "--keep-redundant-methods",
        action="store_true",
        help="Do not exclude REDUNDANT_METHODS (lingo_dice_binary, lingo_ruzicka, lingo_tversky_sym, "
        "spectrum_tversky_sym) by default -- they are exact monotonic transforms/delegations of other "
        "registered methods and add no new ranking information, so they are skipped unless this is passed.",
    )
    parser.add_argument(
        "--keep-expensive-methods",
        action="store_true",
        help="Do not exclude EXPENSIVE_METHODS (monge_elkan, monge_elkan_sym, mismatch, mismatch3, "
        "mismatch5) by default -- measured as the worst per-pair cost in the module (monge_elkan_sym: "
        "9.8h for one InChI variant on 5.3M pairs even parallelized; mismatch5: 18 CPU-days on 56.6M "
        "pairs), so they are skipped unless this is passed.",
    )
    parser.add_argument(
        "--only-methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Restrict the run to exactly these method names (e.g. the categorically slow, non-batch "
        "methods, for a template-subsampled sub-run). Combines with --exclude-methods/"
        "--keep-redundant-methods/--keep-expensive-methods and each variant's own representation-validity "
        "exclusions.",
    )
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

    exclude_methods = (
        frozenset(args.exclude_methods)
        | (frozenset() if args.keep_redundant_methods else REDUNDANT_METHODS)
        | (frozenset() if args.keep_expensive_methods else EXPENSIVE_METHODS)
    )

    print(f"Templates : {templates}")
    print(f"Database  : {database}")
    print(f"Output dir: {output_dir}")
    print(f"Variants  : {len(variants_to_run)}")
    if exclude_methods:
        print(f"Excluding : {', '.join(sorted(exclude_methods))}")
    if args.only_methods:
        print(f"Only      : {', '.join(args.only_methods)}")
    if args.sim_jobs != 1:
        print(f"Sim-jobs  : {args.sim_jobs} (per-variant per-pair worker processes)")
    print()

    results: dict[str, tuple[bool, str | None, float, list]] = {}

    def _run(variant: dict) -> tuple[str, bool, str | None, float, list]:
        stem = variant_stem(variant)
        print(f"  → [{stem}] {variant['description']}")
        ok, reason, elapsed, method_rows = run_variant(
            templates,
            database,
            output_dir,
            variant,
            args.verbose,
            args.dry_run,
            args.overwrite,
            exclude_methods=exclude_methods,
            only_methods=args.only_methods,
            sim_jobs=args.sim_jobs,
        )
        if args.dry_run:
            valid_methods = get_valid_methods(variant, exclude_methods=exclude_methods, only_methods=args.only_methods)
            method_args = ["--all-methods"] if valid_methods is None else ["--methods", *valid_methods]
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--templates",
                str(templates),
                "--database",
                str(database),
                "--output",
                str(output_dir / f"{stem}.csv.gz"),
                *method_args,
                *(["--jobs", str(args.sim_jobs)] if args.sim_jobs != 1 else []),
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
