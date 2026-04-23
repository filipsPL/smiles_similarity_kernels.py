# SMILES-based Similarity Kernels

## **tl;dr**

calculate molecular similarity based on SMILES strings only, using multiple similarity measures (eg for VS):

```bash
# Calculate similarities between templates and library molecules
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method lingo

# $ cat examples/output.csv 
# Name,Similarity_0054-0090,Similarity_0133-0086
# 0054-0090,1.00000,0.39080
# 0061-0013,0.06061,0.12500
# 0062-0039,0.00000,0.00000
# 0082-0017,0.08333,0.18431
# 0083-0114,0.14815,0.18939
# ...
```

## About

Python implementation of SMILES-based compound similarity functions for ligand-based virtual screening. Partially inspired by the methods described in Öztürk et al. (2016) which were originally implemented in Java. This library re-implements, corrects, and substantially extends that work with additional algorithms, chemically-aware preprocessing, SMILES canonicalization, InChI support, and new string-similarity methods not present in the original. And more, as work is in progress.

> [!CAUTION]
> The original Java implementation contains inconsistencies with the manuscript. This implementation corrects those issues (see [Differences from Java Implementation](#differences-from-java-implementation)).

[![Python manual install](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/python-install.yml/badge.svg)](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/python-install.yml) [![CodeQL Advanced](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/codeql.yml/badge.svg)](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/codeql.yml) [![osv scanner](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/osv-scanner.yml/badge.svg)](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/osv-scanner.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18457244.svg)](https://doi.org/10.5281/zenodo.18457244)


## Overview

This module provides **71 similarity methods** for comparing chemical compounds represented as SMILES strings (or InChI/SELFIES strings). It can be used as a Python library or run directly from the command line.

**Key extensions beyond the original Java implementation:**
- Corrected formulas for NLCS, Edit, LINGO edge cases, and SMIfp
- Expanded multi-character element encoding covering stereochemistry (`@@`, `@TH1`…), rare metals, and lanthanides
- Regex-based preprocessing (safe longest-match, no sequential-replace corruption)
- SMILES canonicalization via RDKit (`--canonicalize`)
- InChI conversion (`--inchi`) for representation-independent comparison, with optional per-layer selection (`--inchi-layer formula,connections,...`)
- Layer-respecting InChI preprocessing (`preprocess_inchi`, `extract_inchi_layers`, `smiles_to_inchi_layers`) that does **not** mangle the formula layer
- SELFIES conversion (`--selfies`) — 100% robust molecular string representation; all similarity methods apply directly to SELFIES tokens (`SELFIESTokenizer`)
- TF-IDF cosine similarity with chemically-aware tokenization (`SMILESTokenizer`) — full n-gram grid `smiles_tfidf{m}{n}` / `selfies_tfidf{m}{n}` for m∈{1..6}, n∈{m..6} (best average performance at `(4,4)`)
- Five additional string metrics: Damerau-Levenshtein, Jaro, Jaro-Winkler, Hamming, and Normalized Compression Distance (NCD)
- Classical spectrum kernel, mismatch `(k, m)` kernel, query-weighted asymmetric Tversky on LINGOs, Sørensen-Dice on LINGOs, and stand-alone longest-common-substring similarity
- Character shuffle (`--shuffle`) and alphabetical sort (`--sort`) for negative-control experiments — both destroy chemistry while preserving string length and character composition; shuffle is random (optional seed), sort is deterministic


### Processing pipeline

```
[READ]    Input always as SMILES (.smi / .csv / .tsv)
    ↓
[CONVERT] Select string representation (default: keep SMILES)
          --inchi [--inchi-layer LAYER]   → InChI string
          --selfies                       → SELFIES string
    ↓
[NORMALIZE] Applied after conversion
          --canonicalize                  → canonical SMILES (SMILES only, requires rdkit)
          ELEMENT_REPLACEMENTS            → multi-char atom substitution
                                            auto ON for SMILES, auto OFF for InChI/SELFIES
                                            override with --no-preprocess
    ↓
[AUGMENT] Applied to the final string, type-agnostic
          --shuffle [--shuffle-seed SEED] → random character shuffle (negative control)
          --sort                          → alphabetical character sort (deterministic negative control)
    ↓
[SIMILARITY] All 71 methods available for all string types
```


## Installation

```bash
# Core (required for most methods)
pip install -r requirements.txt
```

## Quick Start

### As a Python Module

```python
from smiles_similarity_kernels import (
    lingo_similarity,
    edit_similarity,
    nlcs_similarity,
    # new in this release:
    lingo_tversky_similarity,
    spectrum_kernel_similarity,
    mismatch_kernel_similarity,
)

smiles1 = "CCO"     # ethanol
smiles2 = "CCCO"    # propanol

print(f"LINGO  (q=4):             {lingo_similarity(smiles1, smiles2, q=4):.3f}")
print(f"Edit:                     {edit_similarity(smiles1, smiles2):.3f}")

# Query-weighted asymmetric Tversky on LINGO q-grams (α=0.9, β=0.1).
# The first argument is treated as the query (template), the second as
# the database candidate — swapping them will in general give different
# values.  Motivated by Bajusz et al. (2025) for nucleic-acid screening.
print(f"Tversky (query=s1):       {lingo_tversky_similarity(smiles1, smiles2):.3f}")
print(f"Tversky (query=s2):       {lingo_tversky_similarity(smiles2, smiles1):.3f}")

# Classical spectrum kernel (Leslie et al. 2002) with k=4, Tanimoto
print(f"Spectrum (k=4):           {spectrum_kernel_similarity(smiles1, smiles2, k=4):.3f}")

# Mismatch kernel — tolerates up to m atom substitutions per k-mer
print(f"Mismatch (k=4, m=1):      {mismatch_kernel_similarity(smiles1, smiles2, k=4, m=1):.3f}")
```

### Command Line

```bash
# Calculate similarities between templates and library molecules
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method lingo

# $ cat examples/output.csv 
# Name,Similarity_0054-0090,Similarity_0133-0086
# 0054-0090,1.00000,0.39080
# 0061-0013,0.06061,0.12500
# 0062-0039,0.00000,0.00000
# 0082-0017,0.08333,0.18431
# 0083-0114,0.14815,0.18939
# ...


# Use all available methods (creates one output file per method) (see example folder for outputs)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/outputs/output.csv --all-methods

# Query-weighted Tversky on LINGOs (recommended for screening)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method lingo_tversky

# Classical spectrum kernel (k=4) and mismatch kernel (k=4, m=1)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method spectrum
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method mismatch

# Canonicalize SMILES before comparison (requires rdkit)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method lingo --canonicalize

# Use InChI representation instead of SMILES (requires rdkit)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method edit --inchi

# Compare using only the connection table of the InChI (topology only)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method lingo --inchi --inchi-layer connections

# Convert to SELFIES before comparison (requires selfies; ELEMENT_REPLACEMENTS auto-disabled)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method edit --selfies

# SELFIES-aware TF-IDF (best-performing n-gram range)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method selfies_tfidf44 --selfies

# Shuffle SMILES characters — random negative control
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method lingo --shuffle --shuffle-seed 42

# Sort SMILES characters alphabetically — deterministic negative control
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output.csv --method lingo --sort

# List available methods
python smiles_similarity_kernels.py --list-methods

# Run built-in demo
python smiles_similarity_kernels.py --demo
```

Expected output format:
```
Name,Similarity_0054-0090,Similarity_0133-0086
0054-0090,1.00000,0.39080
0061-0013,0.06061,0.12500
0062-0039,0.00000,0.00000
...
```

## Available Methods

### String-based (original Öztürk et al. methods)

| CLI name           | Function                      | Description                                                                   | Requires |
| ------------------ | ----------------------------- | ----------------------------------------------------------------------------- | -------- |
| `edit`             | `edit_similarity`             | Levenshtein edit distance, normalized by `max(len1, len2)`                    | —        |
| `nlcs`             | `nlcs_similarity`             | Normalized Longest Common Subsequence: LCS²/(len1×len2)                       | —        |
| `clcs`             | `clcs_similarity`             | Combined LCS: weighted sum of NLCS + NMCLCS1 + NMCLCSn                        | —        |
| `substring`        | `substring_kernel_similarity` | Substring kernel: normalized inner product of all-substring frequency vectors | —        |
| `smifp_cbd`        | `smifp_similarity_cityblock`  | SMILES fingerprint 34D, City Block Distance: 1/(1+CBD)                        | scipy    |
| `smifp_tanimoto`   | `smifp_similarity_tanimoto`   | SMILES fingerprint 34D, Tanimoto coefficient                                  | —        |
| `smifp38_cbd`      | `smifp_similarity_cityblock`  | SMILES fingerprint 38D (+ `/`, `\`, `@@`), City Block Distance                | scipy    |
| `smifp38_tanimoto` | `smifp_similarity_tanimoto`   | SMILES fingerprint 38D, Tanimoto coefficient                                  | —        |
| `lingo`            | `lingo_similarity`            | LINGO q-gram Tanimoto, q=4 (default)                                          | —        |
| `lingo3`           | `lingo_similarity`            | LINGO q-gram Tanimoto, q=3                                                    | —        |
| `lingo5`           | `lingo_similarity`            | LINGO q-gram Tanimoto, q=5                                                    | —        |

### LINGO variants with alternative coefficients (extensions)

Motivated by [our paper](https://doi.org/10.1093/bib/bbaf620) where query-weighted Tversky consistently outperformed Tanimoto on nucleic-acid targets.

| CLI name            | Function                   | Description                                                                  | Requires |
| ------------------- | -------------------------- | ---------------------------------------------------------------------------- | -------- |
| `lingo_tversky`     | `lingo_tversky_similarity` | **Asymmetric Tversky on LINGO q-grams** (q=4, α=0.9, β=0.1) — query-weighted | —        |
| `lingo_tversky_sym` | `lingo_tversky_similarity` | Symmetric Tversky (α=β=0.5, equivalent to Dice) on LINGO q-grams             | —        |
| `lingo_dice`        | `lingo_dice_similarity`    | Sørensen–Dice coefficient on LINGO q-gram counts (q=4)                       | —        |

> **Asymmetry note:** `lingo_tversky` treats the *first* argument as the query (template) and the *second* as the database candidate. Swapping arguments will in general yield different values. This mirrors the "query-weighted Tversky" convention used in our paper [Bajusz et al. (2025)](https://doi.org/10.1093/bib/bbaf620).

### Spectrum and mismatch kernels (extensions)

Classical string-kernel methods from the biological-sequence literature, ported to SMILES. Unlike `lingo`/`substring`, these return a single inner-product-based coefficient (Tanimoto, Dice, or cosine) over the full k-mer count vector.

| CLI name          | Function                              | Description                                                                   | Requires |
| ----------------- | ------------------------------------- | ----------------------------------------------------------------------------- | -------- |
| `spectrum`        | `spectrum_kernel_similarity`          | **Spectrum kernel** (Leslie et al. 2002), k=4, Tanimoto                       | —        |
| `spectrum3`       | `spectrum_kernel_similarity`          | Spectrum kernel, k=3, Tanimoto                                                | —        |
| `spectrum5`       | `spectrum_kernel_similarity`          | Spectrum kernel, k=5, Tanimoto                                                | —        |
| `spectrum_cosine` | `spectrum_kernel_similarity`          | Spectrum kernel, k=4, cosine normalisation                                    | —        |
| `mismatch`        | `mismatch_kernel_similarity`          | **Mismatch kernel** (Leslie et al. 2004), k=4, m=1 — tolerates 1-atom swaps   | —        |
| `mismatch3`       | `mismatch_kernel_similarity`          | Mismatch kernel, k=3, m=1                                                     | —        |
| `mismatch5`       | `mismatch_kernel_similarity`          | Mismatch kernel, k=5, m=1                                                     | —        |
| `lcs_substring`   | `longest_common_substring_similarity` | Normalised Longest Common **Substring** (contiguous): LCSubstr² / (len1·len2) | —        |

> **Mismatch cost note:** the neighbourhood size grows roughly as `C(k, m) * (|alphabet|-1)^m`. For SMILES alphabets of ~30–50 symbols, m=1 with k ≤ 5 is practical; m=2 is expensive and rarely useful.

### TF-IDF (extensions)

Two tokenizer-backed TF-IDF families, each covering the full n-gram grid m∈{1..6}, n∈{m..6} (21 combinations per family):

| CLI name pattern      | Function                   | Tokenizer          | Description                                                | Requires     |
| --------------------- | -------------------------- | ------------------ | ---------------------------------------------------------- | ------------ |
| `smiles_tfidf`        | `smiles_tfidf_similarity`  | `SMILESTokenizer`  | Alias for `smiles_tfidf12` (backward-compatible default)   | scikit-learn |
| `smiles_tfidf{m}{n}`  | `smiles_tfidf_similarity`  | `SMILESTokenizer`  | Chemical-token TF-IDF, ngram (m, n); e.g. `smiles_tfidf44` | scikit-learn |
| `selfies_tfidf`       | `selfies_tfidf_similarity` | `SELFIESTokenizer` | Alias for `selfies_tfidf12` (backward-compatible default)  | scikit-learn |
| `selfies_tfidf{m}{n}` | `selfies_tfidf_similarity` | `SELFIESTokenizer` | SELFIES-token TF-IDF, ngram (m, n); e.g. `selfies_tfidf44` | scikit-learn |

**Tokenizers:**
- `SMILESTokenizer` — treats multi-character atoms (`Cl`, `Br`, `Si`, …) and `@@` as indivisible tokens; operates on chemical units rather than raw characters.
- `SELFIESTokenizer` — splits on `[token]` bracket groups; each SELFIES token is one semantically atomic unit.

**N-gram range selection:** empirical experiments show best average performance around n-gram ranges (3,3)–(5,5), with **(4,4) performing best on average**. Ranges with m=n (single n-gram size) tend to outperform mixed ranges at the same scale.

> **TF-IDF on InChI:** no dedicated InChI tokenizer is provided. When `--inchi` is active, `smiles_tfidf{m}{n}` runs on the InChI string with `preprocess=False` (SMILES substitution is auto-disabled for non-SMILES types), treating InChI characters as raw tokens. This is functional but not semantically optimized for InChI structure.

### SELFIES (extensions)

SELFIES (Self-Referencing Embedded Strings) are a 100% robust molecular representation — every string decodes to a valid molecule. All existing similarity methods apply directly to SELFIES strings; use `--selfies` to convert inputs automatically.

```python
from smiles_similarity_kernels import smiles_to_selfies, SELFIESTokenizer, selfies_tfidf_similarity

selfies = smiles_to_selfies("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
# → '[C][C][=Branch1][C][=O][O][C][=C][C][=C][C][=C][Ring1][=A][C][=Branch1][C][=O][O]'

tok = SELFIESTokenizer()
tok.tokenize(selfies)
# → ['[C]', '[C]', '[=Branch1]', '[C]', '[=O]', '[O]', ...]
```

CLI: pass `--selfies` alongside any `--method`. `ELEMENT_REPLACEMENTS` substitution is automatically disabled for SELFIES (and InChI) — it only runs for SMILES. Works with all 71 methods.

### Negative controls: character shuffle and sort

Two type-agnostic augmentations are available for negative-control experiments. Both destroy chemical meaning while preserving string length and character composition.

**`--shuffle`** randomly permutes characters (with optional `--shuffle-seed` for reproducibility). Scores should approach random-pair baseline; a method scoring well above baseline likely has length bias.

**`--sort`** sorts characters alphabetically — deterministic, no seed needed. Provides a fixed lower-bound baseline that is identical across runs, useful for comparing runs or methods without variance from randomness.

```bash
# Random negative control (reproducible with seed)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output_shuffled.csv --method lingo --shuffle --shuffle-seed 42

# Deterministic negative control
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output examples/output_sorted.csv --method lingo --sort
```

### Additional string metrics (extensions)

| CLI name              | Function                         | Description                                                         | Requires  |
| --------------------- | -------------------------------- | ------------------------------------------------------------------- | --------- |
| `damerau_levenshtein` | `damerau_levenshtein_similarity` | Like edit distance but transpositions cost 1 (not 2)                | jellyfish |
| `jaro`                | `jaro_similarity`                | Jaro similarity                                                     | jellyfish |
| `jaro_winkler`        | `jaro_winkler_similarity`        | Jaro-Winkler (prefix-weighted)                                      | jellyfish |
| `hamming`             | `hamming_similarity`             | Hamming distance, shorter string padded                             | jellyfish |
| `ncd`                 | `ncd_similarity`                 | Normalized Compression Distance via gzip; universal, parameter-free | —         |

> **NCD note:** compression-based similarity is semantically unaware of chemistry — it detects string-level patterns, not structural features. Best used with `--canonicalize` and for near-duplicate detection or benchmarking. See source docstring for a full assessment.


## SMILES Preprocessing

All string methods apply `preprocess_smiles()` before comparison. This replaces multi-character atoms and stereochemistry tokens with single Unicode characters so every atom counts as exactly one character:

```python
from smiles_similarity_kernels import preprocess_smiles

preprocess_smiles("CCCl")          # → 'CCCL'
preprocess_smiles("c1ccc(Br)cc1")  # → 'c1ccc(R)cc1'
preprocess_smiles("C[C@@H](Cl)Br") # → 'C[C¡H](L)R'
```

The full replacement table covers: halogens (`Cl`, `Br`), metalloids (`Si`, `Se`, `As`, `Te`, …), common metals (`Na`, `Ca`, `Mg`, `Fe`, `Zn`, `Cu`, …), stereochemistry tokens (`@@`, `@TH1`–`@SP3`, `@TB`, `@OH`), rare metals (`Ru`, `Rh`, `Ir`, `Mo`, …), single-character element symbols used as atoms (`W`, `V`, `U`), and lanthanides.

### Canonicalization and InChI

When RDKit is available, you can normalize input representations before comparison:

```python
from smiles_similarity_kernels import canonicalize_smiles, smiles_to_inchi

canonicalize_smiles("OCC")   # → 'CCO'  (same as canonicalize_smiles("CCO"))
smiles_to_inchi("CCO")       # → '1S/C2H6O/c1-2-3/h3H,2H2,1H3'  (no 'InChI=' prefix)
```

CLI flags: `--canonicalize` and `--inchi`.

### InChI layer extraction

InChI strings are **layered**: `<version>/<formula>/c<connections>/h<H>/q<charge>/...`. The SMILES-oriented `preprocess_smiles` must **not** be applied to InChI because it would corrupt the formula layer (e.g. `C6H5Cl` → `C6H5L`, which destroys the element-count encoding). The library therefore provides a dedicated set of InChI helpers:

```python
from smiles_similarity_kernels import (
    preprocess_inchi,
    extract_inchi_layers,
    smiles_to_inchi_layers,
    INCHI_LAYERS,
)

# Strip 'InChI=' and '1S/' version tag; keep layer separators
preprocess_inchi("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
# → 'C2H6O/c1-2-3/h3H,2H2,1H3'

# Select a single layer
extract_inchi_layers("InChI=1S/C9H8O4/c1-6(10)13-...", "connections")
# → 'c1-6(10)13-...'

# Select multiple layers (order is preserved)
extract_inchi_layers(inchi, ["formula", "connections"])
# → 'C9H8O4/c1-6(10)13-...'

# One-shot: SMILES → InChI-layer subset
smiles_to_inchi_layers("CC(=O)Oc1ccccc1C(=O)O", ["formula", "connections"])
# → 'C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12'
```

Supported layer names (keys of `INCHI_LAYERS`):

| Name            | Prefix | Content                           |
| --------------- | ------ | --------------------------------- |
| `formula`       | —      | Molecular formula (e.g. `C9H8O4`) |
| `connections`   | `c`    | Atom-connection table (topology)  |
| `hydrogens`     | `h`    | Hydrogen layer                    |
| `charge`        | `q`    | Net charge                        |
| `protons`       | `p`    | Mobile-proton layer               |
| `stereo_db`     | `b`    | Double-bond stereochemistry       |
| `stereo_tet`    | `t`    | Tetrahedral stereochemistry       |
| `stereo_parity` | `m`    | Parity layer                      |
| `stereo_type`   | `s`    | Stereo type (abs/rel/rac)         |
| `isotope`       | `i`    | Isotope layer                     |
| `fixedH`        | `f`    | Fixed-H (non-standard InChI)      |
| `reconnected`   | `r`    | Reconnected-metals layer          |

The CLI mirrors this with `--inchi-layer`:

```bash
# Full InChI (default)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output out.csv --method lingo --inchi

# Compare using only the connection table (topology, no elements/stereochemistry)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output out.csv --method lingo --inchi --inchi-layer connections

# Formula + connections (most discriminating combination without stereochemistry)
python smiles_similarity_kernels.py \
    --templates examples/templates.smi --database examples/database.smi \
    --output out.csv --method lingo --inchi --inchi-layer formula,connections
```

> **Design note:** `ELEMENT_REPLACEMENTS` substitution (`preprocess`) is automatically **on** when string type is SMILES, and **off** for InChI and SELFIES — the pipeline tracks the current string type and sets `preprocess` accordingly. Use `--no-preprocess` to disable it for SMILES (e.g. for benchmarking raw strings). If calling similarity functions directly from Python with InChI or SELFIES, pass `preprocess=False` explicitly.

## Batch Processing

```python
from smiles_similarity_kernels import compute_similarity_matrix, compute_cross_similarity_matrix

# Pairwise n×n matrix
smiles_list = ["CCO", "CCC", "CCCC", "CCOC"]
sim_matrix = compute_similarity_matrix(smiles_list, method='lingo')

# Cross-similarity: library × templates
templates = ["CCO", "CCC"]
library   = ["CCCC", "CCOC", "CCCCl", "CCOCC"]
sim_matrix = compute_cross_similarity_matrix(templates, library, method='lingo')
# shape: (4, 2)
```

## Input / Output Formats

**SMILES files (`.smi`)** — space/tab-separated, no header:
```
CCO ethanol
CCC propane
```

**CSV files** — with header, configurable column names:
```csv
Name,SMILES
ethanol,CCO
propane,CCC
```

**Output CSV:**
```csv
Name,Similarity_template1,Similarity_template2
mol1,0.85000,0.62300
mol2,0.23400,0.89100
```

## Command Line Reference

```bash
python smiles_similarity_kernels.py --templates TEMPLATES --database DATABASE --output OUTPUT [OPTIONS]
```

| Option                        | Short | Description                                                                                                                                         |
| ----------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--templates TEMPLATES`       | `-t`  | Templates file or directory (.smi, .csv, .tsv)                                                                                                      |
| `--database DATABASE`         | `-d`  | Database/library file or directory (.smi, .csv, .tsv)                                                                                               |
| `--output OUTPUT`             | `-o`  | Output CSV file path                                                                                                                                |
| `--method METHOD`             | `-m`  | Similarity method (default: `lingo`)                                                                                                                |
| `--all-methods`               |       | Run all methods; output named `METHOD_output.csv`                                                                                                   |
| `--list-methods`              |       | Print all available methods and exit                                                                                                                |
| `--demo`                      |       | Run a demonstration with example molecules and exit                                                                                                 |
| `--canonicalize`              |       | **[normalize]** Canonicalize SMILES with RDKit (SMILES only, requires rdkit)                                                                        |
| `--inchi`                     |       | **[convert]** Convert SMILES → InChI (strips `InChI=` prefix, requires rdkit)                                                                       |
| `--inchi-layer LAYER[,...]`   |       | **[convert]** With `--inchi`, restrict to selected layer(s). Comma-separated. Default: `all`. See [InChI layer extraction](#inchi-layer-extraction)  |
| `--selfies`                   |       | **[convert]** Convert SMILES → SELFIES (requires `selfies`)                                                                                         |
| `--no-preprocess`             |       | **[normalize]** Disable `ELEMENT_REPLACEMENTS` substitution for SMILES (auto-disabled for InChI/SELFIES). Useful for benchmarking raw strings.       |
| `--shuffle`                   |       | **[augment]** Randomly shuffle characters — **negative control**, type-agnostic, applied after all conversions                                      |
| `--shuffle-seed SEED`         |       | **[augment]** Random seed for `--shuffle` (default: non-reproducible).                                                                              |
| `--sort`                      |       | **[augment]** Sort characters alphabetically — **deterministic negative control**, type-agnostic, applied after all conversions                     |
| `--verbose`, `-v`             |       | Print progress                                                                                                                                      |
| `--templates-smiles-col COL`  |       | SMILES column name/index in templates file                                                                                                          |
| `--templates-name-col COL`    |       | Name column in templates file                                                                                                                       |
| `--templates-delimiter DELIM` |       | Delimiter for templates file                                                                                                                        |
| `--templates-no-header`       |       | Templates file has no header                                                                                                                        |
| `--database-smiles-col COL`   |       | SMILES column in database file                                                                                                                      |
| `--database-name-col COL`     |       | Name column in database file                                                                                                                        |
| `--database-delimiter DELIM`  |       | Delimiter for database file                                                                                                                         |
| `--database-no-header`        |       | Database file has no header                                                                                                                         |

## Differences from Java Implementation

### Corrected formulas

| Method                   | Python (this)                              | Java (original)         | Error        |
| ------------------------ | ------------------------------------------ | ----------------------- | ------------ |
| NLCS                     | `LCS² / (len1×len2)`                       | incorrect variant       | 24–29%       |
| Edit                     | normalized by `max(len1, len2)`            | different normalization | 5–13%        |
| LINGO (both=0 LINGOs)    | returns `1.0` (equally empty)              | returns `0.0`           | wrong        |
| LINGO (no common LINGOs) | returns `0.0`                              | may return non-zero     | wrong        |
| SMIfp                    | preprocesses `Cl`→`L` etc. before counting | counts raw characters   | inconsistent |

### Design differences

- **Preprocessing:** multi-character atoms (`Cl`→`L`, `Br`→`R`, etc.) are always substituted before string operations, giving consistent character-level representations.
- **Substring kernel:** normalized to [0,1] by default (`normalized=True`); Java returns raw kernel values.
- **Ring normalization:** LINGO normalizes all ring digits to `0` before q-gram extraction; Java does not.

## Performance

| Method                                            | Complexity     | Notes                                                       |
| ------------------------------------------------- | -------------- | ----------------------------------------------------------- |
| `lingo`, `lingo_tversky`, `lingo_dice`, `smifp_*` | O(n)           | Fastest — recommended for large-scale screening             |
| `spectrum`                                        | O(n)           | Very fast, equivalent cost to LINGO                         |
| `mismatch` (k=4, m=1)                             | O(n·k·\|Σ\|)   | ~20–50× slower than `spectrum` for typical SMILES alphabets |
| `mismatch` (m≥2)                                  | O(n·k²·\|Σ\|²) | Expensive — use only for short SMILES or small alphabets    |
| `lcs_substring`                                   | O(m×n)         | DP — same cost as `nlcs`                                    |
| `edit`, `nlcs`, `clcs`                            | O(m×n)         | DP — slow for long SMILES                                   |
| `substring`                                       | O(m²+n²)       | Can be slow for long SMILES                                 |
| `smiles_tfidf{m}{n}`, `selfies_tfidf{m}{n}`       | O(corpus)      | Fit once on full corpus for batch use; cost grows with n    |
| `ncd`                                             | O(n log n)     | Compression overhead; fine for millions                     |
| jellyfish methods                                 | O(n)           | Very fast via C extension                                   |


## Citation

Based on methods described in:

> Öztürk, H., Ozkirimli, E., & Özgür, A. (2016). A comparative study of SMILES-based compound similarity functions for drug-target interaction prediction. *BMC Bioinformatics*, 17, 128. [DOI: 10.1186/s12859-016-0977-x](https://doi.org/10.1186/s12859-016-0977-x)

Original Java implementation: https://github.com/hkmztrk/SMILESbasedSimilarityKernels

Cite **THIS** implementation using DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18457244.svg)](https://doi.org/10.5281/zenodo.18457244)
