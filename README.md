# SMILES-based Similarity Kernels

## **tl;dr**

❶ calculate molecular similarity based on SMILES strings only, using multiple similarity measures (eg for VS):

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

❷ calculate SMILES fingerprints (for ML or VS):

```bash
# BPE 512-bit binary fingerprint
python smiles_similarity_kernels.py \
    --fingerprint bpe512_binary \
    --database examples/database.smi \
    --output examples/fingerprints_bpe.csv

# $ cat examples/fingerprints_bpe.csv
# Name,bit_0,bit_1,bit_2,bit_3,bit_4,bit_5,bit_6,bit_7 [...]
# 0054-0090,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
# 0061-0013,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
# 0062-0039,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
# 0082-0017,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
# 0083-0114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
# 0086-0080,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
# [...]

```

## About

Python implementation of SMILES-based compound similarity functions for ligand-based virtual screening. Partially inspired by the methods described in Öztürk et al. (2016) which were originally implemented in Java. This library re-implements, corrects, and substantially extends that work with additional algorithms, chemically-aware preprocessing, SMILES canonicalization, InChI support, and new string-similarity methods not present in the original. And more, as work is in progress.

> [!CAUTION]
> The original Java implementation contains inconsistencies with the manuscript. This implementation corrects those issues (see [Differences from Java Implementation](#differences-from-java-implementation)).

[![Python manual install](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/python-install.yml/badge.svg)](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/python-install.yml) [![CodeQL Advanced](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/codeql.yml/badge.svg)](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/codeql.yml) [![osv scanner](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/osv-scanner.yml/badge.svg)](https://github.com/filipsPL/smiles_similarity_kernels.py/actions/workflows/osv-scanner.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18457244.svg)](https://doi.org/10.5281/zenodo.18457244)

## Overview

This module provides **247 similarity methods** for comparing chemical compounds represented as SMILES strings (or InChI/SELFIES strings). It can be used as a Python library or run directly from the command line.

**Key extensions beyond the original Java implementation:**

- Corrected formulas for NLCS, Edit, LINGO edge cases, and SMIfp
- Expanded multi-character element encoding covering stereochemistry (`@@`, `@TH1`…), rare metals, and lanthanides
- Regex-based preprocessing (safe longest-match, no sequential-replace corruption)
- SMILES canonicalization via RDKit (`--canonicalize`)
- InChI conversion (`--inchi`) for representation-independent comparison, with optional per-layer selection (`--inchi-layer formula,connections,...`)
- Layer-respecting InChI preprocessing (`preprocess_inchi`, `extract_inchi_layers`, `smiles_to_inchi_layers`) that does **not** mangle the formula layer
- SELFIES conversion (`--selfies`) — 100% robust molecular string representation; all similarity methods apply directly to SELFIES tokens (`SELFIESTokenizer`)
- TF-IDF cosine similarity with four tokenizer families: `tok-smiles_tfidf{m}{n}` (`SMILESTokenizer`), `tok-schwaller_tfidf{m}{n}` (`SMILESTokenizerSchwaller`, Schwaller et al. atom-level), `tok-bpe_tfidf{m}{n}` (`SMILESTokenizerBPE`, data-driven BPE trained on ChEMBL), and `tok-selfies_tfidf{m}{n}` — full n-gram grid for m∈{1..6}, n∈{m..6} (best average performance at `(4,4)`)
- Five additional string metrics: Damerau-Levenshtein, Jaro, Jaro-Winkler, Hamming, and Normalized Compression Distance (NCD)
- Classical spectrum kernel, mismatch `(k, m)` kernel, query-weighted asymmetric Tversky on LINGOs, Sørensen-Dice on LINGOs, and stand-alone longest-common-substring similarity
- Character shuffle (`--shuffle`) and alphabetical sort (`--sort`) for negative-control experiments — both destroy chemistry while preserving string length and character composition; shuffle is random (optional seed), sort is deterministic

**SMILES fingerprints**:

- brand new, fixed-length fingerprint vectors for each molecule — suitable as ML feature matrices, for clustering, or for direct comparison with other fingerprint-based tools

### Processing pipeline

```text
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
[SIMILARITY] All 247 methods available for all string types
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
    --output examples/output.csv --method tok-selfies_tfidf44 --selfies

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

Four tokenizer-backed TF-IDF families, each covering the full n-gram grid m∈{1..6}, n∈{m..6} (21 combinations per family):

| CLI name pattern            | Function                     | Tokenizer                  | Description                                                             | Requires     |
| --------------------------- | ---------------------------- | -------------------------- | ----------------------------------------------------------------------- | ------------ |
| `tok-smiles_tfidf`          | `smiles_tfidf_similarity`    | `SMILESTokenizer`          | Alias for `tok-smiles_tfidf12` (default n-gram range)                   | scikit-learn |
| `tok-smiles_tfidf{m}{n}`    | `smiles_tfidf_similarity`    | `SMILESTokenizer`          | Chemical-token TF-IDF, ngram (m, n); e.g. `tok-smiles_tfidf44`         | scikit-learn |
| `tok-schwaller_tfidf`       | `schwaller_tfidf_similarity` | `SMILESTokenizerSchwaller` | Alias for `tok-schwaller_tfidf12` (default n-gram range)                | scikit-learn |
| `tok-schwaller_tfidf{m}{n}` | `schwaller_tfidf_similarity` | `SMILESTokenizerSchwaller` | Schwaller atom-level TF-IDF, ngram (m, n); e.g. `tok-schwaller_tfidf44` | scikit-learn |
| `tok-bpe_tfidf`             | `bpe_tfidf_similarity`       | `SMILESTokenizerBPE`       | Alias for `tok-bpe_tfidf12` (uses all merges from vocab file)           | scikit-learn |
| `tok-bpe_tfidf{m}{n}`       | `bpe_tfidf_similarity`       | `SMILESTokenizerBPE`       | BPE TF-IDF (all merges), ngram (m, n); e.g. `tok-bpe_tfidf44`          | scikit-learn |
| `tok-bpe{k}_tfidf`          | `bpe_tfidf_similarity`       | `SMILESTokenizerBPE`       | Alias for `tok-bpe{k}_tfidf12`; k ∈ {16, 32, 64, 256, 512, 1024}       | scikit-learn |
| `tok-bpe{k}_tfidf{m}{n}`    | `bpe_tfidf_similarity`       | `SMILESTokenizerBPE`       | BPE TF-IDF using first k merges, ngram (m, n); e.g. `tok-bpe64_tfidf44` | scikit-learn |
| `tok-selfies_tfidf`         | `selfies_tfidf_similarity`   | `SELFIESTokenizer`         | Alias for `tok-selfies_tfidf12` (default n-gram range)                  | scikit-learn |
| `tok-selfies_tfidf{m}{n}`   | `selfies_tfidf_similarity`   | `SELFIESTokenizer`         | SELFIES-token TF-IDF, ngram (m, n); e.g. `tok-selfies_tfidf44`         | scikit-learn |

**Tokenizers:**

- `SMILESTokenizer` — treats multi-character bare elements (`Cl`, `Br`, `Si`, …) and `@@` as indivisible tokens; everything else is a single character.
- `SMILESTokenizerSchwaller` — Schwaller et al. (*ACS Central Science* 2019) atom-level tokenization: bracket atoms (`[nH+]`, `[13C@@H]`, …) are single tokens, every bond/branch/stereo symbol is its own token, two-digit ring closures (`%10`) are single tokens. De-facto standard for sequence-to-sequence chemical models.
- `SMILESTokenizerBPE` — data-driven BPE tokenizer trained on ~1M ChEMBL drug-like molecules. Starts from Schwaller atom-level tokens and iteratively merges the most frequent adjacent pair, producing variable-granularity tokens where common fragments (`C(=O)N`, `c1ccccc1`, …) become single units. Vocabulary JSON produced by `train_bpe_tokenizer.py`. The `num_merges` parameter controls how many merges to apply (default: all), allowing different granularities from a single large vocab file. CLI exposes fixed counts: 16, 32, 64, 256, 512, 1024 (e.g. `tok-bpe64_tfidf44`). **A note**: There is a very similar approach by @XinhaoLi74 [described here](https://github.com/XinhaoLi74/SmilesPE) I was not aware of. See below for details.
- `SELFIESTokenizer` — splits on `[token]` bracket groups; each SELFIES token is one semantically atomic unit.

##### Notes on SMILESTokenizerBPE tokenizer

It was trained on a set of set of small molecules fetched from ChemBL database on 2026/04/23, preprocessed in KNIME (strip salts, keep organic molecules). A random subsample of 200k molecules was selected, next for each SMILES additional 5 equivalent SMILES were generated using [SMILES-enumeration](https://github.com/EBjerrum/SMILES-enumeration) by @EBjerrum. Resulting training set consisted of 1,199,970 SMILES strings.

The algorithm:

```
Given: an ordered list of 512 merge pairs, e.g.:

[('c','c'), ('C','C'), ('O',')'), ('c','1'), ('=','O)'), ...]
For each input SMILES string:

0. Base tokenization — split with the Schwaller regex into atom-level tokens:
"CC(=O)N" → ['C', 'C', '(', '=', 'O', ')', 'N']

1. Apply merge 1 (c+c → cc): scan left-to-right, replace every adjacent ('c','c') pair with 'cc'. No match here, list unchanged.

2. Apply merge 2 (C+C → CC): find C followed by C at position 0→1, replace:
['CC', '(', '=', 'O', ')', 'N']

3. Apply merge 3 (O+) → O)): find at positions 3→4:
['CC', '(', '=', 'O)', 'N']

4. Apply merge 5 (=+O) → =O)): find at positions 2→3:
['CC', '(', '=O)', 'N']

5. Apply merge 11 (C(+=O) → C(=O)): find at positions 1→2... wait, CC ≠ C(, no match.

6. ... continue through all 512 merges ...

Final result: ['CC(=O)N'] — the whole amide becomes one token after enough merges chain together.

The key property: each merge pass is a single left-to-right scan — O(len(tokens)) per merge, so tokenizing one molecule costs O(512 × len). For a 50-token molecule that's ~25k operations, which is fast. The merge order is critical — earlier (more frequent) merges produce the tokens that later merges can combine further.
```

**Using BPE in Python** (pass `vocab_path` through `vectorizer` for batch use):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from smiles_similarity_kernels import SMILESTokenizerBPE, bpe_tfidf_similarity

# Use all merges (default) — CLI: tok-bpe_tfidf{m}{n}
tok = SMILESTokenizerBPE(vocab_path="smiles_bpe_vocab.json")

# Use only first 64 merges (finer tokenization) — CLI: tok-bpe64_tfidf{m}{n}
tok = SMILESTokenizerBPE(vocab_path="smiles_bpe_vocab.json", num_merges=64)

vec = TfidfVectorizer(tokenizer=tok, analyzer="word", lowercase=False,
                      token_pattern=None, ngram_range=(1,1), min_df=1, sublinear_tf=True)
vec.fit(corpus)
s = bpe_tfidf_similarity(smi1, smi2, vectorizer=vec)

# Or pass num_merges directly (vectorizer built internally)
s = bpe_tfidf_similarity(smi1, smi2, num_merges=64)
```

**N-gram range selection:** empirical experiments show best average performance around n-gram ranges (3,3)–(5,5), with **(4,4) performing best on average**. Ranges with m=n (single n-gram size) tend to outperform mixed ranges at the same scale.

> **TF-IDF on InChI:** no dedicated InChI tokenizer is provided. When `--inchi` is active, `tok-smiles_tfidf{m}{n}` runs on the InChI string with `preprocess=False` (SMILES substitution is auto-disabled for non-SMILES types), treating InChI characters as raw tokens. This is functional but not semantically optimized for InChI structure.

### SELFIES (extensions)

SELFIES (Self-Referencing Embedded Strings) are a 100% robust molecular representation — every string decodes to a valid molecule. All existing similarity methods apply directly to SELFIES strings; use `--selfies` to convert inputs automatically.

```python
from smiles_similarity_kernels import smiles_to_selfies, SELFIESTokenizer, selfies_tfidf_similarity  # CLI: tok-selfies_tfidf{m}{n}

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

## Fingerprints

In addition to pairwise similarity, the library can produce **fixed-length fingerprint vectors** for each molecule — suitable as ML feature matrices, for clustering, or for direct comparison with other fingerprint-based tools.

All fingerprints are:

- **deterministic** — same SMILES string always gives the same vector
- **corpus-free** — computed from a single molecule, no dataset fitting required
- **float64 arrays** — count or binary (0/1) values

> [!IMPORTANT]
> **Canonicalize before fingerprinting for ML use.** The BPE-pattern fingerprint scans tokens left-to-right, so two SMILES strings representing the *same* molecule but written differently (e.g. `CC(=O)Nc1ccccc1` and `c1ccc(NC(C)=O)cc1`) will produce **different vectors**. SMIfp is order-independent (character counts) but can still differ across SMILES variants for molecules with multi-character elements encoded positionally. For any ML application — training, prediction, or embedding — canonicalize first to ensure one molecule → one fingerprint:
>
> ```bash
> # CLI: add --canonicalize (requires rdkit)
> python smiles_similarity_kernels.py \
>     --fingerprint bpe512_count --canonicalize \
>     --database database.smi --output fingerprints.csv
> ```
>
> ```python
> # Python API: canonicalize explicitly before calling the fingerprint function
> from smiles_similarity_kernels import canonicalize_smiles, bpe_pattern_fingerprint
>
> fp = bpe_pattern_fingerprint(canonicalize_smiles(smiles), num_merges=512)
> ```
>
> Without canonicalization the fingerprint is still deterministic given a fixed input string, which is fine for benchmarking raw SMILES databases or when the input source already guarantees canonical form.

### SMIfp fingerprint

The SMIfp fingerprint counts occurrences of a fixed character set in the preprocessed SMILES string.

| Type | CLI name | Dimensions | Values |
|---|---|---|---|
| Count (default) | `smifp34` | 34 | character counts |
| Binary | `smifp34_binary` | 34 | 0/1 presence |
| Extended count | `smifp38` | 36* | character counts |
| Extended binary | `smifp38_binary` | 36* | 0/1 presence |

*The "38D" variant removes `%` from the 34D set and adds `/`, `\`, `@@`; actual size is 36.

```python
from smiles_similarity_kernels import smifp_fingerprint, SMIFP_CHARS_34, SMIFP_CHARS_38

fp = smifp_fingerprint("CC(=O)Nc1ccccc1")          # 34D count vector
fp = smifp_fingerprint("CC(=O)Nc1ccccc1", binary=True)  # 34D binary
fp = smifp_fingerprint("CC(=O)Nc1ccccc1", chars=SMIFP_CHARS_38)  # extended
```

### BPE-pattern fingerprint

Uses the BPE merge table (trained on ChEMBL) as a fixed pattern dictionary.  Each dimension corresponds to one *merged* token; its value is how many times that token appears in the BPE-tokenized SMILES.

**Key properties:**

- Patterns are learned from ChEMBL but applied to any SMILES without refitting
- Fixed length = `num_merges` (e.g. 512), set at training time
- Complementary to SMIfp: focuses on multi-atom fragments (`C(=O)N`, `c1ccccc1`, …) rather than raw characters
- Base single-character tokens are excluded — those are already captured by SMIfp

| Type | CLI name | Dimensions | Values |
|---|---|---|---|
| Count (all merges) | `bpe_count` | all merges in vocab | token counts |
| Binary (all merges) | `bpe_binary` | all merges in vocab | 0/1 presence |
| Count (k merges) | `bpe{k}_count` | k | token counts |
| Binary (k merges) | `bpe{k}_binary` | k | 0/1 presence |

Available fixed-k values: 16, 32, 64, 128, 256, 512, 1024.

```python
from smiles_similarity_kernels import bpe_pattern_fingerprint

fp = bpe_pattern_fingerprint("CC(=O)Nc1ccccc1")             # all merges, count
fp = bpe_pattern_fingerprint("CC(=O)Nc1ccccc1", num_merges=512)  # fixed 512-bit
fp = bpe_pattern_fingerprint("CC(=O)Nc1ccccc1", num_merges=512, binary=True)
```

### Batch fingerprints

```python
from smiles_similarity_kernels import compute_fingerprint_matrix

smiles = ["CC(=O)Nc1ccccc1", "c1ccccc1", "CCO"]

# SMIfp 34D
matrix, feature_names = compute_fingerprint_matrix(smiles, fp_type="smifp34")
# matrix.shape == (3, 34)

# BPE 512-bit count
matrix, feature_names = compute_fingerprint_matrix(smiles, fp_type="bpe512_count")
# matrix.shape == (3, 512)
```

### Fingerprint CLI

```bash
# Compute SMIfp 34D for all molecules in a database file
python smiles_similarity_kernels.py \
    --fingerprint smifp34 \
    --database examples/database.smi \
    --output fingerprints.csv

# BPE 512-bit binary fingerprint
python smiles_similarity_kernels.py \
    --fingerprint bpe512_binary \
    --database examples/database.smi \
    --output fingerprints_bpe.csv

# List all available fingerprint types
python smiles_similarity_kernels.py --list-fingerprints
```

Output format — one row per molecule, columns `Name, bit_0, bit_1, …`:

```csv
Name,bit_0,bit_1,bit_2,...
mol1,3,0,1,...
mol2,5,2,0,...
```

The convert/normalize/augment pipeline flags (`--canonicalize`, `--inchi`, `--selfies`, `--shuffle`, `--sort`) work with `--fingerprint` in the same way as with `--method`.

### Available fingerprint types

```bash
python smiles_similarity_kernels.py --list-fingerprints
```

| CLI name | Length | Description |
|---|---|---|
| `smifp34` | 34 | SMIfp character-frequency count |
| `smifp34_binary` | 34 | SMIfp binary (presence/absence) |
| `smifp38` | 36 | SMIfp extended count (adds `/`, `\`, `@@`; removes `%`) |
| `smifp38_binary` | 36 | SMIfp extended binary |
| `bpe_count` | all merges | BPE-pattern count (all merges in vocab) |
| `bpe_binary` | all merges | BPE-pattern binary (all merges in vocab) |
| `bpe{k}_count` | k | BPE-pattern count, k ∈ {16,32,64,128,256,512,1024} |
| `bpe{k}_binary` | k | BPE-pattern binary, k ∈ {16,32,64,128,256,512,1024} |

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
| `--inchi-layer LAYER[,...]`   |       | **[convert]** With `--inchi`, restrict to selected layer(s). Comma-separated. Default: `all`. See [InChI layer extraction](#inchi-layer-extraction) |
| `--selfies`                   |       | **[convert]** Convert SMILES → SELFIES (requires `selfies`)                                                                                         |
| `--no-preprocess`             |       | **[normalize]** Disable `ELEMENT_REPLACEMENTS` substitution for SMILES (auto-disabled for InChI/SELFIES). Useful for benchmarking raw strings.      |
| `--shuffle`                   |       | **[augment]** Randomly shuffle characters — **negative control**, type-agnostic, applied after all conversions                                      |
| `--shuffle-seed SEED`         |       | **[augment]** Random seed for `--shuffle` (default: non-reproducible).                                                                              |
| `--sort`                      |       | **[augment]** Sort characters alphabetically — **deterministic negative control**, type-agnostic, applied after all conversions                     |
| `--overwrite`                 |       | Overwrite existing output files. Without this flag, existing files are **skipped with a warning** printed to stderr.                                |
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

| Method                                              | Complexity     | Notes                                                       |
| --------------------------------------------------- | -------------- | ----------------------------------------------------------- |
| `lingo`, `lingo_tversky`, `lingo_dice`, `smifp_*`   | O(n)           | Fastest — recommended for large-scale screening             |
| `spectrum`                                          | O(n)           | Very fast, equivalent cost to LINGO                         |
| `mismatch` (k=4, m=1)                               | O(n·k·\|Σ\|)   | ~20–50× slower than `spectrum` for typical SMILES alphabets |
| `mismatch` (m≥2)                                    | O(n·k²·\|Σ\|²) | Expensive — use only for short SMILES or small alphabets    |
| `lcs_substring`                                     | O(m×n)         | DP — same cost as `nlcs`                                    |
| `edit`, `nlcs`, `clcs`                              | O(m×n)         | DP — slow for long SMILES                                   |
| `substring`                                         | O(m²+n²)       | Can be slow for long SMILES                                 |
| `tok-smiles_tfidf{m}{n}`, `tok-selfies_tfidf{m}{n}` | O(corpus)      | Fit once on full corpus for batch use; cost grows with n    |
| `ncd`                                               | O(n log n)     | Compression overhead; fine for millions                     |
| jellyfish methods                                   | O(n)           | Very fast via C extension                                   |

## Citation

Based on methods described in:

> Öztürk, H., Ozkirimli, E., & Özgür, A. (2016). A comparative study of SMILES-based compound similarity functions for drug-target interaction prediction. *BMC Bioinformatics*, 17, 128. [DOI: 10.1186/s12859-016-0977-x](https://doi.org/10.1186/s12859-016-0977-x)

Original Java implementation: <https://github.com/hkmztrk/SMILESbasedSimilarityKernels>

Cite **THIS** implementation using DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18457244.svg)](https://doi.org/10.5281/zenodo.18457244)
