# SMILES-based Similarity Kernels

Python implementation of SMILES-based compound similarity functions for ligand-based virtual screening. The original methods are described in Öztürk et al. (2016) and were originally implemented in Java — this library re-implements, corrects, and substantially extends that work with additional algorithms, chemically-aware preprocessing, SMILES canonicalization, InChI support, and new string-similarity methods not present in the original.

> [!CAUTION]
> The original Java implementation contains inconsistencies with the manuscript. This implementation corrects those issues (see [Differences from Java Implementation](#differences-from-java-implementation)).

## Overview

This module provides **21 similarity methods** for comparing chemical compounds represented as SMILES strings (or InChI strings). It can be used as a Python library or run directly from the command line.

**Key extensions beyond the original Java implementation:**
- Corrected formulas for NLCS, Edit, LINGO edge cases, and SMIfp
- Expanded multi-character element encoding covering stereochemistry (`@@`, `@TH1`…), rare metals, and lanthanides
- Regex-based preprocessing (safe longest-match, no sequential-replace corruption)
- SMILES canonicalization via RDKit (`--canonicalize`)
- InChI conversion (`--inchi`) for representation-independent comparison
- TF-IDF cosine similarity with chemically-aware tokenization (`SMILESTokenizer`)
- Five additional string metrics: Damerau-Levenshtein, Jaro, Jaro-Winkler, Hamming, and Normalized Compression Distance (NCD)

## Citation

Based on methods described in:

> Öztürk, H., Ozkirimli, E., & Özgür, A. (2016). A comparative study of SMILES-based compound similarity functions for drug-target interaction prediction. *BMC Bioinformatics*, 17, 128. [DOI: 10.1186/s12859-016-0977-x](https://doi.org/10.1186/s12859-016-0977-x)

Original Java implementation: https://github.com/hkmztrk/SMILESbasedSimilarityKernels

Cite THIS implementation using DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18457244.svg)](https://doi.org/10.5281/zenodo.18457244)

## Installation

```bash
# Core (required for most methods)
pip install numpy pandas

# Full installation (all methods)
pip install numpy pandas scipy scikit-learn rdkit jellyfish
```

See `requirements.txt` for version constraints.

## Quick Start

### As a Python Module

```python
from smiles_similarity_kernels import lingo_similarity, edit_similarity, nlcs_similarity

smiles1 = "CCO"     # ethanol
smiles2 = "CCCO"    # propanol

sim = lingo_similarity(smiles1, smiles2, q=4)
print(f"LINGO similarity: {sim:.3f}")

sim = edit_similarity(smiles1, smiles2)
print(f"Edit similarity: {sim:.3f}")
```

### Command Line

```bash
# Calculate similarities between templates and library molecules
python smiles_similarity_kernels.py \
    templates.smi library.smi output.csv --method lingo

# Canonicalize SMILES before comparison (requires rdkit)
python smiles_similarity_kernels.py \
    templates.smi library.smi output.csv --method lingo --canonicalize

# Use InChI representation instead of SMILES (requires rdkit)
python smiles_similarity_kernels.py \
    templates.smi library.smi output.csv --method edit --inchi

# Use all available methods (creates one output file per method)
python smiles_similarity_kernels.py \
    templates.smi library.smi output.csv --all-methods

# List available methods
python smiles_similarity_kernels.py --list-methods
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

### TF-IDF (extensions)

| CLI name         | Function                  | Description                                                      | Requires     |
| ---------------- | ------------------------- | ---------------------------------------------------------------- | ------------ |
| `smiles_tfidf`   | `smiles_tfidf_similarity` | TF-IDF cosine similarity with chemical tokenization, ngram (1,2) | scikit-learn |
| `smiles_tfidf13` | `smiles_tfidf_similarity` | TF-IDF cosine similarity with chemical tokenization, ngram (1,3) | scikit-learn |
| `smiles_tfidf23` | `smiles_tfidf_similarity` | TF-IDF cosine similarity with chemical tokenization, ngram (2,3) | scikit-learn |
| `smiles_tfidf14` | `smiles_tfidf_similarity` | TF-IDF cosine similarity with chemical tokenization, ngram (1,4) | scikit-learn |

The `SMILESTokenizer` treats multi-character atoms (`Cl`, `Br`, `Si`, …) and `@@` as indivisible tokens, so TF-IDF operates on chemical units rather than raw characters.

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
python smiles_similarity_kernels.py TEMPLATES LIBRARY OUTPUT [OPTIONS]
```

| Option                        | Description                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| `--method METHOD`             | Similarity method (default: `lingo`)                               |
| `--all-methods`               | Run all methods; output named `METHOD_output.csv`                  |
| `--list-methods`              | Print all available methods and exit                               |
| `--canonicalize`              | Canonicalize SMILES with RDKit before comparison                   |
| `--inchi`                     | Convert SMILES to InChI (strips `InChI=` prefix) before comparison |
| `--verbose`, `-v`             | Print progress                                                     |
| `--templates-smiles-col COL`  | SMILES column name/index in templates file                         |
| `--templates-name-col COL`    | Name column in templates file                                      |
| `--templates-delimiter DELIM` | Delimiter for templates file                                       |
| `--templates-no-header`       | Templates file has no header                                       |
| `--database-smiles-col COL`   | SMILES column in database file                                     |
| `--database-name-col COL`     | Name column in database file                                       |
| `--database-delimiter DELIM`  | Delimiter for database file                                        |
| `--database-no-header`        | Database file has no header                                        |

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

| Method                 | Complexity | Notes                                           |
| ---------------------- | ---------- | ----------------------------------------------- |
| `lingo`, `smifp_*`     | O(n)       | Fastest — recommended for large-scale screening |
| `edit`, `nlcs`, `clcs` | O(m×n)     | DP — slow for long SMILES                       |
| `substring`            | O(m²+n²)   | Can be slow for long SMILES                     |
| `smiles_tfidf`         | O(corpus)  | Fit once on full corpus for batch use           |
| `ncd`                  | O(n log n) | Compression overhead; fine for millions         |
| jellyfish methods      | O(n)       | Very fast via C extension                       |
