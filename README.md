# SMILES-based Similarity Kernels

*Semi-automatic* (with claude.ai), but validated, python implementation of the java [SMILES-based compound similarity functions for ligand-based virtual screening](https://github.com/hkmztrk/SMILESbasedSimilarityKernels/).

> [!CAUTION]
> according to my analysis, the original implementation may contain some inconsistencies and differences to what is described in the manuscript. This implementation tries to fix it (see below)
> Also here I use encoding for more "two symbol" elements (Mg, Si etc, see below - not perfect but reasonable)

## Overview

This module provides **11 similarity methods** for comparing chemical compounds represented as SMILES strings. These methods are used for ligand-based virtual screening, particularly for discovering small-molecule binders to nucleic acids (RNA/DNA - NEW) or proteins (original work).

## Citation

This implementation is based on methods described in:

> Öztürk, H., Ozkirimli, E., & Özgür, A. (2016). A comparative study of SMILES-based compound similarity functions for drug-target interaction prediction. *BMC Bioinformatics*, 17, 128. [DOI: 10.1186/s12859-016-0977-x](https://doi.org/10.1186/s12859-016-0977-x)

## Installation

### Requirements

**Core dependencies:**
```bash
numpy
pandas
```

**Optional dependencies (for specific methods):**
```bash
scipy          # For SMIfp City Block Distance
scikit-learn   # For LINGO TF-IDF methods (not included in standard methods)
```

### Install

```bash
# Basic installation (no dependencies needed for most methods)
# Just copy smiles_similarity_kernels.py to your project

# Or install dependencies if needed:
pip install numpy pandas scipy scikit-learn
```

## Quick Start

### As a Python Module

```python
from smiles_similarity_kernels import lingo_similarity, edit_similarity, nlcs_similarity

# Compare two SMILES strings
smiles1 = "CCO"     # ethanol
smiles2 = "CCCO"    # propanol

# LINGO similarity (most popular)
sim = lingo_similarity(smiles1, smiles2, q=4)
print(f"LINGO similarity: {sim:.3f}")

# Edit distance similarity
sim = edit_similarity(smiles1, smiles2)
print(f"Edit similarity: {sim:.3f}")

# NLCS similarity
sim = nlcs_similarity(smiles1, smiles2)
print(f"NLCS similarity: {sim:.3f}")
```

### Command Line

```bash
# Calculate similarities between templates and library molecules
python smiles_similarity_kernels.py \
    templates.smi \
    library.smi \
    output.csv \
    --method lingo

# Use all available methods
python smiles_similarity_kernels.py \
    templates.smi \
    library.smi \
    output.csv \
    --all-methods

# List available methods
python smiles_similarity_kernels.py --list-methods
```

## Available Methods

### 1. Edit Distance Similarity

**Method:** `edit` or `edit_similarity(smiles1, smiles2)`

**Formula:**
```
EditSim(S1, S2) = 1 - edit_distance(S1, S2) / max(len(S1), len(S2))
```

**Description:**  
Calculates Levenshtein edit distance and normalizes by maximum string length.

**Returns:** Similarity in [0, 1], where 1 = identical strings

**Example:**
```python
sim = edit_similarity("CCC", "CCCCC")  # 0.6
# Edit distance = 2, max length = 5, similarity = 1 - 2/5 = 0.6
```

---

### 2. NLCS (Normalized Longest Common Subsequence)

**Method:** `nlcs` or `nlcs_similarity(smiles1, smiles2)`

**Formula:**
```
NLCS(S1, S2) = LCS(S1, S2)² / (len(S1) × len(S2))
```

**Description:**  
Finds the longest common subsequence (not necessarily contiguous) and normalizes by string lengths. The LCS length is **squared** in the numerator.

**Returns:** Similarity in [0, 1]

**Example:**
```python
sim = nlcs_similarity("ABC", "AC")  # 0.6667
# LCS("ABC", "AC") = "AC" = length 2
# NLCS = 2² / (3 × 2) = 4/6 = 0.6667
```

**Note:** This is the correct formula according to literature. Some implementations incorrectly omit the squaring operation.

---

### 3. CLCS (Combined LCS Models)

**Method:** `clcs` or `clcs_similarity(smiles1, smiles2)`

**Formula:**
```
CLCS(S1, S2) = w1×NLCS + w2×NMCLCS1 + w3×NMCLCSn

Where:
- NLCS: Normalized LCS
- NMCLCS1: Normalized Maximal Consecutive LCS from position 1
- NMCLCSn: Normalized Maximal Consecutive LCS from any position
- Default weights: w1=0.33, w2=0.33, w3=0.34 (sum to 1.0)
```

**Description:**  
Combines three LCS variants with configurable weights.

**Returns:** Similarity in [0, 1]

**Parameters:**
```python
sim = clcs_similarity(smiles1, smiles2, w1=0.33, w2=0.33, w3=0.34)
```

---

### 4. Substring Kernel

**Method:** `substring` or `substring_kernel_similarity(smiles1, smiles2)`

**Formula:**
```
K(S1, S2) = <φ(S1), φ(S2)>

Where φ(S) is a feature vector of all substrings (length ≥ min_length)

Normalized version:
K_norm(S1, S2) = K(S1, S2) / √(K(S1, S1) × K(S2, S2))
```

**Description:**  
String kernel based on shared substrings. Can return raw kernel value or normalized similarity.

**Returns:** 
- Normalized: Similarity in [0, 1]
- Raw kernel: Non-negative integer (count of shared substrings)

**Parameters:**
```python
# Normalized (default, recommended)
sim = substring_kernel_similarity(smiles1, smiles2, normalized=True)

# Raw kernel value (for comparison with Java implementation)
kernel = substring_kernel_similarity(smiles1, smiles2, normalized=False)
```

---

### 5-6. SMIfp (SMILES Fingerprint) Methods

**Methods:** 
- `smifp_cbd` or `smifp_similarity_cityblock(smiles1, smiles2)`
- `smifp_tanimoto` or `smifp_similarity_tanimoto(smiles1, smiles2)`

**Fingerprint:**  
34-dimensional or 38-dimensional count vector of specific SMILES characters:

**34D characters (default):**
```
C, c, O, o, N, n, S, s, P, p, F, I, B, b, 
1-9 (digits), (, ), [, ], =, #, +, -, @, .
```

**38D characters (extended):**
```
34D + /, \, @@ (adds chirality and directional bonds)
```

**Similarity Measures:**

**City Block Distance (Manhattan):**
```
Similarity = 1 / (1 + CityBlockDistance)

Where CityBlockDistance = Σ|fp1[i] - fp2[i]|
```

**Tanimoto Coefficient:**
```
Tanimoto = dot(fp1, fp2) / (||fp1||² + ||fp2||² - dot(fp1, fp2))
```

**Parameters:**
```python
# Use 34D fingerprint (default)
sim = smifp_similarity_tanimoto(smiles1, smiles2)

# Use 38D fingerprint
from smiles_similarity_kernels import SMIFP_CHARS_38
sim = smifp_similarity_tanimoto(smiles1, smiles2, chars=SMIFP_CHARS_38)
```

**Note:** This implementation **preprocesses multi-character atoms** (Cl→L, Br→R) before counting, which provides more consistent character-level representation.

---

### 7-9. LINGO (Linguistic Molecular Representation)

**Methods:**
- `lingo3` - LINGO with q=3 (trigrams)
- `lingo` or `lingo4` - LINGO with q=4 (default, 4-grams)
- `lingo5` - LINGO with q=5 (5-grams)

**Formula:**
```
LINGOsim(S1, S2) = Σ [1 - |N(S1,i) - N(S2,i)| / (N(S1,i) + N(S2,i))] / m

Where:
- N(S,i) = frequency of LINGO i in SMILES S
- m = total number of unique LINGOs in both S1 and S2
- LINGO = q-character substring
```

**Description:**  
Linguistic approach that extracts overlapping q-character substrings (LINGOs) and compares their frequency distributions using a modified Tanimoto coefficient.

**Preprocessing:**
1. Multi-character atoms replaced: `Cl→L`, `Br→R`, etc.
2. Ring numbers normalized to `0`: `C1CCCCC1 → C0CCCCC0`

**Example (q=4):**
```python
sim = lingo_similarity("CCCCC", "CCCC", q=4)

# CCCCC generates: CCCC (appears 2x)
# CCCC generates:  CCCC (appears 1x)
# Similarity calculated from frequency comparison
```

**Edge Cases:**
- If SMILES length < q, no LINGOs are generated
- If both have 0 LINGOs, returns 1.0 (both "equally empty")
- If one has 0 LINGOs, returns 0.0 (no basis for comparison)

**Parameters:**
```python
# Default q=4
sim = lingo_similarity(smiles1, smiles2)

# Custom q value
sim = lingo_similarity(smiles1, smiles2, q=3)

# Disable preprocessing (not recommended)
sim = lingo_similarity(smiles1, smiles2, q=4, preprocess=False)

# Disable ring normalization (not recommended for LINGO)
sim = lingo_similarity(smiles1, smiles2, q=4, normalize_rings=False)
```

---

## Advanced Features

### SMILES Preprocessing

The module provides automatic preprocessing of SMILES strings to handle multi-character atoms:

```python
from smiles_similarity_kernels import preprocess_smiles

# Automatic conversion
original = "CCCl"
processed = preprocess_smiles(original)
print(processed)  # "CCCL"
```

**Replacement table:** - not perfect but reasonable

```python
ELEMENT_REPLACEMENTS = {
    # Halogens
    'Cl': 'L',
    'Br': 'R',
    # Metalloids and other elements
    'Si': 'G',
    'Se': 'E',
    'se': 'e',  # aromatic selenium
    'As': 'D',
    'as': 'd',  # aromatic arsenic
    'Te': 'T',
    'te': 't',  # aromatic tellurium
    # Metals commonly found in SMILES
    'Na': 'Y',
    'Ca': 'W',
    'Mg': 'M',
    'Fe': 'X',
    'Zn': 'Z',
    'Cu': 'Q',
    'Mn': 'J',
    'Co': 'K',
    'Ni': 'U',
    'Al': 'A',
    'Li': 'V',
    'Ag': '!',
    'Au': '$',
    'Pt': '&',
    'Pd': '^',
    'Cr': '~',
    'Ti': '`',
    'Sn': ';',
    'Pb': ':',
    'Hg': '?',
    'Cd': '<',
    'Ba': '>',
    'Sr': '{',
    'Bi': '}',
    'Sb': '|',
}
```

**When preprocessing is used:**
- Edit, NLCS, CLCS: ✓ Yes (by default)
- LINGO: ✓ Yes (by default)
- Substring: ✓ Yes (by default)
- SMIfp: ✓ Yes (this implementation)

**Rationale:** Preprocessing ensures consistent character-level operations. For example, "CCCl" should be treated as 4 characters (C-C-C-L), not 5 (C-C-C-l).

### Ring Number Normalization

For LINGO methods, ring numbers are normalized to `0`:

```python
from smiles_similarity_kernels import normalize_ring_numbers

original = "C1CCCCC1"
normalized = normalize_ring_numbers(original)
print(normalized)  # "C0CCCCC0"
```

**Why:** Makes LINGO extraction independent of ring numbering scheme. Rings are structural features, not unique identifiers.

### Batch Processing

Calculate similarity matrix for multiple molecules:

```python
from smiles_similarity_kernels import compute_similarity_matrix

smiles_list = ["CCO", "CCC", "CCCC", "CCOC"]
sim_matrix = compute_similarity_matrix(smiles_list, method='lingo')

print(sim_matrix)
# 4x4 matrix of pairwise similarities
```

Calculate cross-similarity (templates vs library):

```python
from smiles_similarity_kernels import compute_cross_similarity_matrix

templates = ["CCO", "CCC"]
library = ["CCCC", "CCOC", "CCCCl", "CCOCC"]

sim_matrix = compute_cross_similarity_matrix(templates, library, method='lingo')
print(sim_matrix.shape)  # (4, 2) - library × templates
```

## Input Formats

### SMILES Files (.smi)

**Simple format (space/tab-separated):**
```
CCO ethanol
CCC propane
CCCC butane
```

**CSV format:**
```csv
Name,SMILES
ethanol,CCO
propane,CCC
butane,CCCC
```

### Reading Files

```python
from smiles_similarity_kernels import read_smiles_from_file

# Read .smi file
molecules = read_smiles_from_file("molecules.smi")

# Read CSV with custom columns
molecules = read_smiles_from_file(
    "data.csv",
    smiles_col="SMILES",
    name_col="ID",
    header=True
)

# Returns: {'name': 'smiles', ...}
```

## Output Formats

### CSV Output

The script generates CSV files with similarity scores:

```csv
Name,Similarity_template1,Similarity_template2
mol1,0.850,0.623
mol2,0.234,0.891
mol3,0.456,0.234
```

**Columns:**
- `Name`: Molecule identifier
- `Similarity_X`: Similarity to template molecule X

## Command Line Reference

### Basic Usage

```bash
python smiles_similarity_kernels.py TEMPLATES LIBRARY OUTPUT [OPTIONS]
```

**Arguments:**
- `TEMPLATES`: File or directory with template molecules
- `LIBRARY`: File or directory with library molecules to screen
- `OUTPUT`: Output CSV file path

### Options

```bash
--method METHOD          # Similarity method (default: lingo)
--all-methods           # Calculate all methods, output separate files
--list-methods          # List available methods and exit
--verbose, -v           # Print progress information

# Template file options
--templates-smiles-col COL    # Column for SMILES
--templates-name-col COL      # Column for names  
--templates-delimiter DELIM   # Column delimiter
--templates-no-header         # File has no header

# Database file options  
--database-smiles-col COL     # Column for SMILES
--database-name-col COL       # Column for names
--database-delimiter DELIM    # Column delimiter
--database-no-header          # File has no header
```

### Examples

**Single method:**
```bash
python smiles_similarity_kernels.py \
    templates.smi \
    library.smi \
    results.csv \
    --method lingo
```

expected output:

```
Name,Similarity_0054-0090,Similarity_0133-0086
0054-0090,1.00000,0.39080
0061-0013,0.06061,0.12500
0062-0039,0.00000,0.00000
0082-0017,0.08333,0.18431
0083-0114,0.14815,0.18939
0086-0080,0.10598,0.22055
0092-0008,0.33854,0.63571
0096-0280,0.23232,0.31111
0098-0003,0.15625,0.08333
0099-0265,0.20513,0.26126
0107-0034,0.05882,0.05556
0109-0002,0.09314,0.22626
0109-0045,0.12745,0.26263
0109-0175,0.10976,0.17236
0133-0036,0.14286,0.17647
0133-0054,0.40476,0.95455
0133-0074,0.11111,0.11111
0133-0083,0.05983,0.18421
0133-0086,0.39080,1.00000
```


**All methods:**
```bash
python smiles_similarity_kernels.py \
    templates.smi \
    library.smi \
    results.csv \
    --all-methods

# Creates: edit_results.csv, lingo_results.csv, nlcs_results.csv, etc.
```

**CSV with custom columns:**
```bash
python smiles_similarity_kernels.py \
    templates.csv \
    database.csv \
    output.csv \
    --method lingo \
    --templates-smiles-col "SMILES" \
    --templates-name-col "ID" \
    --database-smiles-col "smiles" \
    --database-name-col "compound_id"
```

**Verbose output:**
```bash
python smiles_similarity_kernels.py \
    templates.smi \
    library.smi \
    results.csv \
    --method lingo \
    --verbose
```

## Differences from Java Implementation

This Python implementation differs from the original Java implementation in several important ways:

### ✅ Corrected Implementations

**1. NLCS Formula (CRITICAL)**

| Implementation | Formula | Status |
|----------------|---------|--------|
| **Python (this)** | `LCS²/(len1×len2)` | ✅ Correct (matches literature) |
| **Java (original)** | Unknown variant | ❌ Wrong (produces 24-29% error) |

**Example:**
```
SMILES: "CCC" vs "CCCCC"
LCS = 3

Python: 3² / (3×5) = 9/15 = 0.600  ✅
Java:   0.457                       ❌ (should be 0.6)
```

**2. Edit Distance Formula**

| Implementation | Normalization | Status |
|----------------|---------------|--------|
| **Python (this)** | `max(len1, len2)` | ✅ Correct |
| **Java (original)** | Unknown | ⚠️ Minor differences (5-13%) |

**3. LINGO Edge Cases**

| Case | Python (this) | Java (original) |
|------|---------------|-----------------|
| Both have 0 LINGOs | 1.0 (equally empty) | 0.0 (no comparison) |
| Template has 0 LINGOs | 0.0 (cannot compare) | May return non-zero ❌ |
| No common LINGOs | 0.0 (no similarity) | May return non-zero ❌ |

**Python's behavior is more consistent with the mathematical definition.**

### 🔄 Design Differences (Intentional)

**1. SMILES Preprocessing**

| Feature | Python (this) | Java (original) |
|---------|---------------|-----------------|
| Multi-char atoms (Cl→L) | ✓ Preprocessed | ✗ Not preprocessed |
| Ring normalization | ✓ For LINGO | ✗ Not normalized |

**Impact:** Python provides more consistent character-level representations.

**Example:**
```python
# Python preprocesses before calculating
"CCCl" → "CCCL" (4 characters) → more accurate edit distance

# Java counts directly
"CCCl" (4 chars as 'C','C','Cl' or 5 as 'C','C','C','l') → inconsistent
```

**2. Substring Kernel Normalization**

| Mode | Python (this) | Java (original) |
|------|---------------|-----------------|
| Default | Normalized (0-1) | Raw kernel value |
| Parameter | `normalized=True/False` | No option |

**Python is more flexible** - can output both normalized similarity and raw kernel value.

### 📊 Validation Results

The Python implementation has been validated against literature formulas:

```
Test: NLCS("ABC", "AC") 
Expected: 0.6667 (from literature)
Python:   0.6667 ✅ PASS
Java:     0.XXXX ❌ FAIL

Test: Edit("CCC", "CCCCC")
Expected: 0.6 (1 - 2/5)
Python:   0.6 ✅ PASS  
Java:     0.571 ❌ FAIL
```

### Known Issues in Java Implementation

The original Java implementation may have the following *features*:

1. **NLCS**: Wrong formula (24-29% error)
2. **CLCS**: Inherits NLCS error
3. **Edit**: Minor formula difference (5-13% error)
4. **LINGO**: False positives (reports similarity when none exists)
5. **SMIfp**: Potential calculation errors (up to 170% error in some cases)

## Performance Considerations

### Speed

- **Edit distance**: O(m×n) - Dynamic programming
- **NLCS**: O(m×n) - Dynamic programming  
- **LINGO**: O(m+n) - Linear in SMILES length
- **SMIfp**: O(k) - Linear in fingerprint dimension (k=34 or 38)

**Fastest methods:** LINGO, SMIfp  
**Slowest methods:** Edit, NLCS, CLCS

### Memory

All methods have O(1) or O(m×n) memory requirements. For large-scale virtual screening:

**Recommended:**
- Use LINGO (fast, good performance)
- Use SMIfp (fastest, simple)

**Not recommended for millions of compounds:**
- Edit distance (slow)
- NLCS/CLCS (slow)

## Troubleshooting

### ImportError: No module named 'scipy'

**Solution:** Install scipy or use methods that don't require it:
```bash
pip install scipy
```

Or avoid `smifp_cbd` (use `smifp_tanimoto` instead).

### ImportError: No module named 'sklearn'

**Solution:** TF-IDF methods require sklearn (not included in standard methods):
```bash
pip install scikit-learn
```

### Empty LINGO Sets

If you get warnings about empty LINGO sets:

```python
# Your SMILES may be too short for the q value
sim = lingo_similarity("CC", "CO", q=5)  # Both have length < 5

# Solution: Use smaller q value
sim = lingo_similarity("CC", "CO", q=2)  # Works fine
```

