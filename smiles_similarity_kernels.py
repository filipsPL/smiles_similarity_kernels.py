#!/usr/bin/env python3
"""
SMILES-based Similarity Kernels

Python implementation of SMILES-based compound similarity functions
as described in:

Öztürk, H., Ozkirimli, E., & Özgür, A. (2016). A comparative study of
SMILES-based compound similarity functions for drug-target interaction
prediction. BMC Bioinformatics, 17, 128.

Original Java implementation: https://github.com/hkmztrk/SMILESbasedSimilarityKernels

WARNING: The original implementation may contain inconsistencies with the manuscript.
This implementation aims to correct those issues. See README.md for details.

This module can be:
1. Imported and used in Python programs
2. Run from command line to calculate similarity matrices

Usage as module:
    from smiles_similarity_kernels import lingo_similarity, edit_similarity
    sim = lingo_similarity(smiles1, smiles2, q=4)

Usage from command line:
    python smiles_similarity_kernels.py templates_dir library_dir output.csv --method lingo

Author: fstefaniak@iimcb.gov.pl, based on the original work by Öztürk, H., Ozkirimli, E., & Özgür, A
"""

import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union, Callable
from pathlib import Path

# Optional imports for TF-IDF (sklearn)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional import for City Block Distance
try:
    from scipy.spatial.distance import cityblock

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Optional import for RDKit (SMILES canonicalization and InChI conversion)
try:
    from rdkit import Chem

    try:
        from rdkit.Chem.inchi import MolToInchi  # RDKit >= 2020
    except ImportError:
        from rdkit.Chem.rdinchi import MolToInchi  # older RDKit
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Optional import for jellyfish (Damerau-Levenshtein, Jaro, Jaro-Winkler, Hamming)
try:
    import jellyfish

    JELLYFISH_AVAILABLE = True
except ImportError:
    JELLYFISH_AVAILABLE = False


# ============================================================================
# SMILES Preprocessing
# ============================================================================

# Complete mapping of multi-character elements to single characters.
# Unicode characters are used for metals/rare elements to guarantee no
# collision with any standard SMILES character or with each other.
# Longer patterns (e.g. '@@', '@TH1') MUST be matched before shorter
# prefixes — the regex-based preprocess_smiles handles this automatically.
ELEMENT_REPLACEMENTS = {
    # --- Stereochemistry (must precede bare '@') ---
    "@@": "¡",  # counterclockwise chirality
    "@TH1": "¢",
    "@TH2": "£",
    "@AL1": "¤",
    "@AL2": "¥",
    "@SP1": "¦",
    "@SP2": "§",
    "@SP3": "¨",
    "@TB": "©",  # trigonal bipyramidal (followed by digits)
    "@OH": "ª",  # octahedral (followed by digits)
    # --- Halogens ---
    "Cl": "L",
    "Br": "R",
    # --- Metalloids and chalcogens ---
    "Si": "G",
    "Se": "E",
    "se": "e",  # aromatic selenium
    "As": "D",
    "as": "d",  # aromatic arsenic
    "Te": "T",
    "te": "t",  # aromatic tellurium
    "Ge": "«",
    "Ga": "¬",
    # --- Common metals ---
    "Na": "Y",
    "Ca": "Ω",
    "Mg": "M",
    "Fe": "X",
    "Zn": "Z",
    "Cu": "Q",
    "Mn": "J",
    "Co": "K",
    "Ni": "Θ",
    "Al": "A",
    "Li": "Λ",
    "Ag": "!",
    "Au": "$",
    "Pt": "&",
    "Pd": "^",
    "Cr": "~",
    "Ti": "`",
    "Sn": ";",
    "Pb": ":",
    "Hg": "?",
    "Cd": "<",
    "Ba": ">",
    "Sr": "{",
    "Bi": "}",
    "Sb": "|",
    # --- Extended / rare metals ---
    "In": "®",
    "Tl": "¯",
    "Be": "°",
    "Ra": "±",
    "Ru": "²",
    "Rh": "³",
    "Os": "´",
    "Ir": "µ",
    "Mo": "¶",
    "Nb": "¹",
    "Ta": "º",
    "Re": "»",
    "Tc": "¼",
    # Single-character element symbols that would otherwise be confused
    # with SMILES atom tokens if left unencoded when they appear inside
    # bracket atoms (e.g. [W], [V], [U]).  We encode them here so that
    # downstream string-similarity methods treat them as atomic units.
    "W": "·",  # Tungsten
    "V": "¸",  # Vanadium
    "U": "Ë",  # Uranium
    # --- Lanthanides / actinides ---
    "La": "½",
    "Ce": "¾",
    "Pr": "¿",
    "Nd": "À",
    "Sm": "Á",
    "Eu": "Â",
    "Gd": "Ã",
    "Tb": "Ä",
    "Dy": "Å",
    "Ho": "Æ",
    "Er": "Ç",
    "Tm": "È",
    "Yb": "É",
    "Lu": "Ê",
}

# Reverse mapping for decoding (if needed) - not needed, indeed, but may be useful.
ELEMENT_REVERSE = {v: k for k, v in ELEMENT_REPLACEMENTS.items()}

# Pre-compiled regex for fast, correct multi-character element replacement.
# Keys are sorted longest-first so that longer patterns (e.g. '@@', '@TH1')
# are always matched before shorter prefixes (e.g. '@'), avoiding partial
# replacements that sequential str.replace() calls would produce.
_PREPROCESS_PATTERN = re.compile("|".join(re.escape(k) for k in sorted(ELEMENT_REPLACEMENTS.keys(), key=len, reverse=True)))


def preprocess_smiles(smiles: str) -> str:
    """
    Preprocess SMILES string by replacing multi-character atoms with single characters.
    This is required for accurate string-based similarity calculations.

    As specified in Öztürk et al. (2016):
    "All SMILES strings are modified such that atoms represented with
    two characters such as 'Cl' and 'Br' are replaced with single characters."

    Parameters
    ----------
    smiles : str
        Input SMILES string

    Returns
    -------
    str
        Preprocessed SMILES string with all multi-character elements
        replaced by single characters

    Examples
    --------
    >>> preprocess_smiles("CCCCl")
    'CCCL'
    >>> preprocess_smiles("c1ccc(Br)cc1")
    'c1ccc(R)cc1'
    >>> preprocess_smiles("C[C@@H](Cl)Br")
    'C[C¡H](L)R'
    """
    return _PREPROCESS_PATTERN.sub(lambda m: ELEMENT_REPLACEMENTS[m.group(0)], smiles)


def normalize_ring_numbers(smiles: str) -> str:
    """
    Normalize ring numbers in SMILES string by replacing all digits with '0'.

    As specified in Ã–ztÃ¼rk et al. (2016) for LINGO method:
    "Before the LINGO creation process, all ring numbers in the SMILES
    string are set to '0'."

    Parameters
    ----------
    smiles : str
        Input SMILES string

    Returns
    -------
    str
        SMILES with all ring numbers normalized to '0'

    Examples
    --------
    >>> normalize_ring_numbers("c1ccccc1")
    'c0ccccc0'
    >>> normalize_ring_numbers("C1CC2CCCCC2C1")
    'C0CC0CCCCC0C0'
    """
    return re.sub(r"[0-9]", "0", smiles)


def canonicalize_smiles(smiles: str) -> str:
    """
    Return the canonical SMILES for a molecule using RDKit.

    Ensures that two different SMILES strings representing the same molecule
    (e.g. "CCO" and "OCC") produce identical strings before any string-based
    comparison.  Falls back to the original string when RDKit is unavailable
    or the SMILES cannot be parsed.

    Parameters
    ----------
    smiles : str
        Input SMILES string

    Returns
    -------
    str
        Canonical SMILES, or the original string if canonicalization fails
    """
    if not smiles or not RDKIT_AVAILABLE:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


def smiles_to_inchi(smiles: str) -> str:
    """
    Convert a SMILES string to an InChI string, stripping the leading
    'InChI=' prefix so downstream string-similarity methods operate on
    the information-bearing part only.

    Requires RDKit.  Returns an empty string when conversion fails or
    RDKit is unavailable.

    Parameters
    ----------
    smiles : str
        Input SMILES string

    Returns
    -------
    str
        InChI string with 'InChI=' prefix removed, or '' on failure

    Examples
    --------
    >>> smiles_to_inchi("CCO")
    '1S/C2H6O/c1-2-3/h3H,2H2,1H3'
    """
    if not smiles or not RDKIT_AVAILABLE:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        inchi = MolToInchi(mol)
        if inchi is None:
            return ""
        if inchi.startswith("InChI="):
            inchi = inchi[6:]
        return inchi
    except Exception:
        return ""


# ============================================================================
# 1. Edit Distance Similarity
# ============================================================================


def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns
    -------
    int
        Number of edit operations (insert, delete, substitute)
    """
    m, n = len(s1), len(s2)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])  # deletion  # insertion  # substitution

    return dp[m][n]


def edit_similarity(smiles1: str, smiles2: str, preprocess: bool = True) -> float:
    """
    Calculate edit distance-based similarity between two SMILES strings.

    EditSim(S1, S2) = 1 - edit(S1, S2) / max(len(S1), len(S2))

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    preprocess : bool
        Whether to preprocess SMILES (replace multi-char atoms)

    Returns
    -------
    float
        Similarity score in [0, 1]
    """
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)

    if len(smiles1) == 0 and len(smiles2) == 0:
        return 1.0

    ed = edit_distance(smiles1, smiles2)
    max_len = max(len(smiles1), len(smiles2))

    return 1.0 - (ed / max_len)


# ============================================================================
# 2. Normalized Longest Common Subsequence (NLCS)
# ============================================================================


def lcs_length(s1: str, s2: str) -> int:
    """
    Calculate length of longest common subsequence.

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns
    -------
    int
        Length of LCS
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def nlcs_similarity(smiles1: str, smiles2: str, preprocess: bool = True) -> float:
    """
    Calculate Normalized Longest Common Subsequence similarity.

    NLCS(S1, S2) = LCS(S1, S2)^2 / (len(S1) * len(S2))

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    preprocess : bool
        Whether to preprocess SMILES

    Returns
    -------
    float
        Similarity score in [0, 1]
    """
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)

    if len(smiles1) == 0 or len(smiles2) == 0:
        return 0.0 if len(smiles1) != len(smiles2) else 1.0

    lcs_len = lcs_length(smiles1, smiles2)

    return (lcs_len**2) / (len(smiles1) * len(smiles2))


# ============================================================================
# 3. Combined LCS Models (CLCS)
# ============================================================================


def mclcs1_length(s1: str, s2: str) -> int:
    """
    Maximal Consecutive LCS starting from character 1.
    Common subsequence must be consecutive and start from index 0.
    """
    min_len = min(len(s1), len(s2))
    length = 0

    for i in range(min_len):
        if s1[i] == s2[i]:
            length += 1
        else:
            break

    return length


def mclcsn_length(s1: str, s2: str) -> int:
    """
    Maximal Consecutive LCS starting from any position.
    Finds the longest contiguous common substring.
    """
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0

    # Dynamic programming for longest common substring
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])

    return max_length


def clcs_similarity(smiles1: str, smiles2: str, w1: float = 0.33, w2: float = 0.33, w3: float = 0.34, preprocess: bool = True) -> float:
    """
    Combined LCS similarity - weighted combination of NLCS, NMCLCS1, NMCLCSn.

    K(S1, S2) = w1*NLCS + w2*NMCLCS1 + w3*NMCLCSn

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    w1, w2, w3 : float
        Weights for each component (should sum to 1)
    preprocess : bool
        Whether to preprocess SMILES

    Returns
    -------
    float
        Similarity score
    """
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)

    if len(smiles1) == 0 or len(smiles2) == 0:
        return 0.0 if len(smiles1) != len(smiles2) else 1.0

    denominator = len(smiles1) * len(smiles2)

    # NLCS
    lcs_len = lcs_length(smiles1, smiles2)
    v1 = (lcs_len**2) / denominator

    # NMCLCS1
    mclcs1_len = mclcs1_length(smiles1, smiles2)
    v2 = (mclcs1_len**2) / denominator

    # NMCLCSn
    mclcsn_len = mclcsn_length(smiles1, smiles2)
    v3 = (mclcsn_len**2) / denominator

    return w1 * v1 + w2 * v2 + w3 * v3


# ============================================================================
# 4. SMILES-based Substring Kernel
# ============================================================================


def get_all_substrings(s: str, min_length: int = 2) -> Counter:
    """
    Get frequency counts of all substrings with length >= min_length.

    Parameters
    ----------
    s : str
        Input string
    min_length : int
        Minimum substring length

    Returns
    -------
    Counter
        Dictionary of substring frequencies
    """
    substrings = Counter()
    n = len(s)

    for i in range(n):
        for j in range(i + min_length, n + 1):
            substrings[s[i:j]] += 1

    return substrings


def substring_kernel_similarity(smiles1: str, smiles2: str, min_length: int = 2, normalized: bool = True, preprocess: bool = True) -> float:
    """
    SMILES representation-based string kernel.

    Calculates inner product of substring frequency vectors.
    K(S1, S2) = <Î¸(S1), Î¸(S2)>

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    min_length : int
        Minimum substring length to consider
    normalized : bool
        If True, normalize by self-similarities
    preprocess : bool
        Whether to preprocess SMILES

    Returns
    -------
    float
        Kernel value (normalized similarity if normalized=True)
    """
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)

    freq1 = get_all_substrings(smiles1, min_length)
    freq2 = get_all_substrings(smiles2, min_length)

    # Inner product
    common_substrings = set(freq1.keys()) & set(freq2.keys())
    k12 = sum(freq1[s] * freq2[s] for s in common_substrings)

    if not normalized:
        return float(k12)

    # Normalized version
    k11 = sum(v * v for v in freq1.values())
    k22 = sum(v * v for v in freq2.values())

    if k11 == 0 or k22 == 0:
        return 0.0

    return k12 / np.sqrt(k11 * k22)


# ============================================================================
# 5. SMILES Fingerprint (SMIfp)
# ============================================================================

# Original 34 characters from SMIfp paper
SMIFP_CHARS_34 = [
    "C",
    "c",
    "O",
    "o",
    "N",
    "n",
    "S",
    "s",
    "P",
    "p",
    "F",
    "I",
    "B",
    "b",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "(",
    ")",
    "[",
    "]",
    "=",
    "#",
    "+",
    "-",
    "@",
    "%",
    ".",
]

# Extended 38 characters (adding chirality and directional bonds)
SMIFP_CHARS_38 = SMIFP_CHARS_34.copy()
SMIFP_CHARS_38.remove("%")  # Remove '%' as it's rare
SMIFP_CHARS_38.extend(["/", "\\", "@@"])


def smiles_to_fingerprint(smiles: str, chars: List[str] = None) -> np.ndarray:
    """
    Convert SMILES to fingerprint vector based on character frequencies.

    Parameters
    ----------
    smiles : str
        SMILES string
    chars : List[str]
        List of characters to count (default: 34D SMIfp)

    Returns
    -------
    np.ndarray
        Fingerprint vector
    """
    if chars is None:
        chars = SMIFP_CHARS_34

    fp = np.zeros(len(chars), dtype=float)

    for i, char in enumerate(chars):
        fp[i] = smiles.count(char)

    return fp


def smifp_similarity_cityblock(smiles1: str, smiles2: str, chars: List[str] = None, preprocess: bool = True) -> float:
    """
    SMIfp similarity using City Block Distance.

    Similarity = 1 / (1 + CBD)

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    chars : List[str]
        Character set for fingerprint
    preprocess : bool
        Whether to preprocess SMILES

    Returns
    -------
    float
        Similarity score
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for City Block Distance")

    # Note: SMIfp typically doesn't preprocess multi-char elements
    # as it counts raw SMILES characters
    fp1 = smiles_to_fingerprint(smiles1, chars)
    fp2 = smiles_to_fingerprint(smiles2, chars)

    cbd = cityblock(fp1, fp2)

    return 1.0 / (1.0 + cbd)


def smifp_similarity_tanimoto(smiles1: str, smiles2: str, chars: List[str] = None, preprocess: bool = True) -> float:
    """
    SMIfp similarity using Tanimoto coefficient.

    Tanimoto = dot(fp1, fp2) / (|fp1|^2 + |fp2|^2 - dot(fp1, fp2))

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    chars : List[str]
        Character set for fingerprint
    preprocess : bool
        Whether to preprocess SMILES

    Returns
    -------
    float
        Similarity score
    """
    fp1 = smiles_to_fingerprint(smiles1, chars)
    fp2 = smiles_to_fingerprint(smiles2, chars)

    dot_product = np.dot(fp1, fp2)
    norm1_sq = np.dot(fp1, fp1)
    norm2_sq = np.dot(fp2, fp2)

    denominator = norm1_sq + norm2_sq - dot_product

    if denominator == 0:
        return 1.0 if norm1_sq == 0 and norm2_sq == 0 else 0.0

    return dot_product / denominator


# ============================================================================
# 6. LINGO Similarity
# ============================================================================


def get_lingos(smiles: str, q: int = 4, normalize_rings: bool = True, preprocess: bool = True) -> Counter:
    """
    Extract LINGOs (q-character substrings) from SMILES.

    Parameters
    ----------
    smiles : str
        SMILES string
    q : int
        LINGO length (default 4)
    normalize_rings : bool
        Whether to normalize ring numbers to '0'
    preprocess : bool
        Whether to preprocess multi-char elements

    Returns
    -------
    Counter
        LINGO frequency counts
    """
    if preprocess:
        smiles = preprocess_smiles(smiles)

    if normalize_rings:
        smiles = normalize_ring_numbers(smiles)

    lingos = Counter()
    n = len(smiles)

    for i in range(n - q + 1):
        lingo = smiles[i : i + q]
        lingos[lingo] += 1

    return lingos


def lingo_similarity(smiles1: str, smiles2: str, q: int = 4, preprocess: bool = True) -> float:
    """
    LINGOsim - LINGO-based Tanimoto similarity.

    LINGOsim = Î£(1 - |N(S1,i) - N(S2,i)| / (N(S1,i) + N(S2,i))) / m

    where m is total number of unique LINGOs, N(S,i) is frequency of LINGO i in S.

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    q : int
        LINGO length
    preprocess : bool
        Whether to preprocess SMILES

    Returns
    -------
    float
        Similarity score in [0, 1]
    """
    lingos1 = get_lingos(smiles1, q, normalize_rings=True, preprocess=preprocess)
    lingos2 = get_lingos(smiles2, q, normalize_rings=True, preprocess=preprocess)

    # Get all unique LINGOs
    all_lingos = set(lingos1.keys()) | set(lingos2.keys())

    if len(all_lingos) == 0:
        return 1.0

    similarity_sum = 0.0

    for lingo in all_lingos:
        n1 = lingos1.get(lingo, 0)
        n2 = lingos2.get(lingo, 0)

        if n1 + n2 > 0:
            similarity_sum += 1.0 - abs(n1 - n2) / (n1 + n2)

    return similarity_sum / len(all_lingos)


# ============================================================================
# 7 & 8. LINGO-based TF and TF-IDF Cosine Similarity
# ============================================================================


class LingoVectorizer:
    """
    Vectorizer for LINGO-based TF and TF-IDF representations.

    This class creates LINGO tokens from SMILES and builds
    TF or TF-IDF weighted vectors for similarity calculation.
    """

    def __init__(self, q: int = 4, use_idf: bool = True, preprocess: bool = True, normalize_rings: bool = True):
        """
        Initialize the vectorizer.

        Parameters
        ----------
        q : int
            LINGO length
        use_idf : bool
            If True, use TF-IDF; if False, use TF only
        preprocess : bool
            Whether to preprocess SMILES
        normalize_rings : bool
            Whether to normalize ring numbers
        """
        self.q = q
        self.use_idf = use_idf
        self.preprocess = preprocess
        self.normalize_rings = normalize_rings
        self.vectorizer = None
        self.is_fitted = False

    def _smiles_to_lingo_string(self, smiles: str) -> str:
        """Convert SMILES to space-separated LINGO string for sklearn."""
        if self.preprocess:
            smiles = preprocess_smiles(smiles)

        if self.normalize_rings:
            smiles = normalize_ring_numbers(smiles)

        lingos = []
        n = len(smiles)
        for i in range(n - self.q + 1):
            lingos.append(smiles[i : i + self.q])

        return " ".join(lingos)

    def fit(self, smiles_list: List[str]):
        """
        Fit the vectorizer on a corpus of SMILES strings.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for TF-IDF vectorization")

        # Convert SMILES to LINGO strings
        lingo_strings = [self._smiles_to_lingo_string(s) for s in smiles_list]

        if self.use_idf:
            self.vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"[^\s]+",  # Any non-whitespace sequence
                sublinear_tf=True,  # Use 1 + log(tf) as in the paper
            )
        else:
            self.vectorizer = CountVectorizer(analyzer="word", token_pattern=r"[^\s]+")

        self.vectorizer.fit(lingo_strings)
        self.is_fitted = True

    def transform(self, smiles_list: List[str]):
        """
        Transform SMILES strings to TF or TF-IDF vectors.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings

        Returns
        -------
        sparse matrix
            Matrix of vectors
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")

        lingo_strings = [self._smiles_to_lingo_string(s) for s in smiles_list]
        return self.vectorizer.transform(lingo_strings)

    def fit_transform(self, smiles_list: List[str]):
        """Fit and transform in one step."""
        self.fit(smiles_list)
        return self.transform(smiles_list)


def lingo_tfidf_similarity(smiles1: str, smiles2: str, q: int = 4, corpus: List[str] = None, vectorizer: LingoVectorizer = None) -> float:
    """
    LINGO-based TF-IDF cosine similarity.

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    q : int
        LINGO length
    corpus : List[str]
        Corpus for IDF calculation (required for meaningful IDF)
    vectorizer : LingoVectorizer
        Pre-fitted vectorizer (optional, for efficiency)

    Returns
    -------
    float
        Cosine similarity
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for TF-IDF similarity")

    if vectorizer is not None and vectorizer.is_fitted:
        vec1 = vectorizer.transform([smiles1])
        vec2 = vectorizer.transform([smiles2])
    else:
        if corpus is None:
            corpus = [smiles1, smiles2]

        vectorizer = LingoVectorizer(q=q, use_idf=True)
        vectorizer.fit(corpus)

        vec1 = vectorizer.transform([smiles1])
        vec2 = vectorizer.transform([smiles2])

    sim = sklearn_cosine_similarity(vec1, vec2)[0, 0]
    return sim


# ============================================================================
# 9. SMILES TF-IDF Cosine Similarity (chemical tokenization)
# ============================================================================


class SMILESTokenizer:
    """
    Chemically-aware SMILES tokenizer for use with sklearn TF-IDF.

    Recognises multi-character elements (Cl, Br, Si, …) and stereochemistry
    markers (@@, @) as single tokens so that TF-IDF operates on chemical
    units rather than raw characters.
    """

    # Ordered longest-first so that '@@' beats '@', '@TH1' beats '@', etc.
    _PATTERNS = sorted(
        [
            "@@",
            "Br",
            "Cl",
            "Si",
            "Se",
            "se",
            "As",
            "as",
            "Te",
            "te",
            "Na",
            "Ca",
            "Mg",
            "Fe",
            "Zn",
            "Cu",
            "Mn",
            "Co",
            "Ni",
            "Al",
            "Li",
            "Ag",
            "Au",
            "Pt",
            "Pd",
            "Cr",
            "Ti",
            "Sn",
            "Pb",
            "Hg",
            "Cd",
            "Ba",
            "Sr",
            "Bi",
            "Sb",
            "Ge",
            "Ga",
            "In",
            "Tl",
            "Be",
            "Ra",
            "Ru",
            "Rh",
            "Os",
            "Ir",
            "Mo",
            "Nb",
            "Ta",
            "Re",
            "Tc",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        ],
        key=len,
        reverse=True,
    )

    _TOKEN_RE = re.compile("|".join(re.escape(p) for p in _PATTERNS) + "|.")

    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string into chemical tokens."""
        return self._TOKEN_RE.findall(smiles)

    def __call__(self, smiles: str) -> List[str]:
        return self.tokenize(smiles)


def smiles_tfidf_similarity(
    smiles1: str, smiles2: str, corpus: List[str] = None, ngram_range: Tuple[int, int] = (1, 2), vectorizer=None
) -> float:
    """
    TF-IDF cosine similarity with chemically-aware tokenization.

    Uses SMILESTokenizer so that multi-character atoms (Cl, Br, …) and
    stereochemistry markers (@@) are treated as indivisible tokens.
    When a pre-fitted vectorizer is supplied it is reused directly,
    which is strongly recommended for batch/matrix calculations.

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    corpus : List[str]
        Corpus used to fit the IDF weights.  Defaults to [smiles1, smiles2].
    ngram_range : Tuple[int, int]
        N-gram range passed to TfidfVectorizer (default (1, 2)).
    vectorizer : fitted TfidfVectorizer or None
        Pre-fitted vectorizer for efficiency in batch use.

    Returns
    -------
    float
        Cosine similarity in [0, 1]
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for smiles_tfidf_similarity")

    if vectorizer is None:
        if corpus is None:
            corpus = [smiles1, smiles2]
        tokenizer = SMILESTokenizer()
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            analyzer="word",
            lowercase=False,
            token_pattern=None,
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True,
        )
        vectorizer.fit(corpus)

    vec1 = vectorizer.transform([smiles1])
    vec2 = vectorizer.transform([smiles2])
    return float(sklearn_cosine_similarity(vec1, vec2)[0, 0])


# ============================================================================
# 10. Jellyfish-based string similarity metrics
# ============================================================================


def damerau_levenshtein_similarity(smiles1: str, smiles2: str, preprocess: bool = True) -> float:
    """
    Damerau-Levenshtein similarity (transpositions count as one edit).

    Like edit_similarity but also treats adjacent-character transpositions
    as a single operation, which can better capture SMILES typos/variants.
    Requires the ``jellyfish`` package.

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings to compare
    preprocess : bool
        Replace multi-character atoms before comparison

    Returns
    -------
    float
        Similarity in [0, 1]
    """
    if not JELLYFISH_AVAILABLE:
        raise ImportError("jellyfish is required for damerau_levenshtein_similarity")
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
    max_len = max(len(smiles1), len(smiles2))
    if max_len == 0:
        return 1.0
    return 1.0 - jellyfish.damerau_levenshtein_distance(smiles1, smiles2) / max_len


def jaro_similarity(smiles1: str, smiles2: str, preprocess: bool = True) -> float:
    """
    Jaro similarity between two SMILES strings.

    Particularly sensitive to common characters and transpositions;
    less meaningful for long strings.  Requires ``jellyfish``.

    Parameters
    ----------
    smiles1, smiles2 : str
    preprocess : bool

    Returns
    -------
    float
        Similarity in [0, 1]
    """
    if not JELLYFISH_AVAILABLE:
        raise ImportError("jellyfish is required for jaro_similarity")
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
    return jellyfish.jaro_similarity(smiles1, smiles2)


def jaro_winkler_similarity(smiles1: str, smiles2: str, preprocess: bool = True) -> float:
    """
    Jaro-Winkler similarity — Jaro with extra weight for common prefixes.

    Can capture cases where two SMILES share a common scaffold prefix.
    Requires ``jellyfish``.

    Parameters
    ----------
    smiles1, smiles2 : str
    preprocess : bool

    Returns
    -------
    float
        Similarity in [0, 1]
    """
    if not JELLYFISH_AVAILABLE:
        raise ImportError("jellyfish is required for jaro_winkler_similarity")
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
    return jellyfish.jaro_winkler_similarity(smiles1, smiles2)


def hamming_similarity(smiles1: str, smiles2: str, preprocess: bool = True) -> float:
    """
    Hamming similarity between two SMILES strings.

    Strings are right-padded with spaces to equal length before comparison.
    Requires ``jellyfish``.

    Parameters
    ----------
    smiles1, smiles2 : str
    preprocess : bool

    Returns
    -------
    float
        Similarity in [0, 1]
    """
    if not JELLYFISH_AVAILABLE:
        raise ImportError("jellyfish is required for hamming_similarity")
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
    max_len = max(len(smiles1), len(smiles2))
    if max_len == 0:
        return 1.0
    s1 = smiles1.ljust(max_len)
    s2 = smiles2.ljust(max_len)
    try:
        return 1.0 - jellyfish.hamming_distance(s1, s2) / max_len
    except Exception:
        return 0.0


# ============================================================================
# 11. Normalized Compression Distance (NCD) similarity
# ============================================================================


def _compress_bytes(data: bytes, compresslevel: int = 9) -> int:
    """Return compressed size of *data* using gzip with mtime=0 (deterministic)."""
    import gzip as _gzip
    import io as _io

    buf = _io.BytesIO()
    with _gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=compresslevel, mtime=0) as f:
        f.write(data)
    return len(buf.getvalue())


def ncd_similarity(smiles1: str, smiles2: str, preprocess: bool = True) -> float:
    """
    Normalized Compression Distance (NCD) similarity using gzip.

    NCD(x,y) = (C(x|y) - min(C(x), C(y))) / max(C(x), C(y))
    similarity = 1 - NCD, clamped to [0, 1].

    Both concatenation orders (x|y and y|x) are tried; the minimum
    compressed size is used for robustness.  A '|' separator (not valid
    in SMILES) is inserted between the two strings.

    This is a universal, parameter-free metric — but it is semantically
    unaware of chemistry.  It works best for detecting near-duplicate
    SMILES and for comparison benchmarks.

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings (or InChI strings) to compare
    preprocess : bool
        Replace multi-character atoms before comparison (recommended for SMILES)

    Returns
    -------
    float
        Similarity in [0, 1]
    """
    if not smiles1 or not smiles2:
        return 0.0
    if smiles1 == smiles2:
        return 1.0
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
    a = smiles1.encode("utf-8")
    b = smiles2.encode("utf-8")
    sep = b"|"
    c_a = _compress_bytes(a)
    c_b = _compress_bytes(b)
    c_ab = _compress_bytes(a + sep + b)
    c_ba = _compress_bytes(b + sep + a)
    c_xy = min(c_ab, c_ba)
    denominator = max(c_a, c_b)
    if denominator == 0:
        return 1.0
    ncd = (c_xy - min(c_a, c_b)) / denominator
    return max(0.0, min(1.0, 1.0 - ncd))


# ============================================================================
# Available Methods Registry
# ============================================================================

AVAILABLE_METHODS = {
    "edit": {"function": edit_similarity, "description": "Edit distance similarity", "params": {}},
    "nlcs": {"function": nlcs_similarity, "description": "Normalized Longest Common Subsequence", "params": {}},
    "clcs": {"function": clcs_similarity, "description": "Combined LCS models", "params": {}},
    "substring": {
        "function": lambda s1, s2: substring_kernel_similarity(s1, s2, normalized=True),
        "description": "Substring kernel (normalized)",
        "params": {},
    },
    "smifp_cbd": {
        "function": smifp_similarity_cityblock,
        "description": "SMILES fingerprint 34D with City Block Distance (Manhattan)",
        "params": {},
        "requires": "scipy",
    },
    "smifp_tanimoto": {"function": smifp_similarity_tanimoto, "description": "SMILES fingerprint 34D with Tanimoto", "params": {}},
    "smifp38_cbd": {
        "function": lambda s1, s2: smifp_similarity_cityblock(s1, s2, chars=SMIFP_CHARS_38),
        "description": "SMILES fingerprint 38D with City Block Distance (Manhattan)",
        "params": {},
        "requires": "scipy",
    },
    "smifp38_tanimoto": {
        "function": lambda s1, s2: smifp_similarity_tanimoto(s1, s2, chars=SMIFP_CHARS_38),
        "description": "SMILES fingerprint 38D with Tanimoto",
        "params": {},
    },
    "lingo": {"function": lingo_similarity, "description": "LINGO similarity (q=4)", "params": {"q": 4}},
    "lingo3": {"function": lambda s1, s2: lingo_similarity(s1, s2, q=3), "description": "LINGO similarity (q=3)", "params": {"q": 3}},
    "lingo5": {"function": lambda s1, s2: lingo_similarity(s1, s2, q=5), "description": "LINGO similarity (q=5)", "params": {"q": 5}},
    "smiles_tfidf": {
        "function": smiles_tfidf_similarity,
        "description": "TF-IDF cosine similarity with chemical tokenization (ngram (1,2))",
        "params": {"ngram_range": (1, 2)},
        "requires": "sklearn",
    },
    "smiles_tfidf13": {
        "function": lambda s1, s2: smiles_tfidf_similarity(s1, s2, ngram_range=(1, 3)),
        "description": "TF-IDF cosine similarity with chemical tokenization (ngram (1,3))",
        "params": {"ngram_range": (1, 3)},
        "requires": "sklearn",
    },
    "damerau_levenshtein": {
        "function": damerau_levenshtein_similarity,
        "description": "Damerau-Levenshtein similarity (transpositions as 1 edit)",
        "params": {},
        "requires": "jellyfish",
    },
    "jaro": {
        "function": jaro_similarity,
        "description": "Jaro similarity",
        "params": {},
        "requires": "jellyfish",
    },
    "jaro_winkler": {
        "function": jaro_winkler_similarity,
        "description": "Jaro-Winkler similarity (prefix-weighted Jaro)",
        "params": {},
        "requires": "jellyfish",
    },
    "hamming": {
        "function": hamming_similarity,
        "description": "Hamming similarity (shorter string padded with spaces)",
        "params": {},
        "requires": "jellyfish",
    },
    "ncd": {
        "function": ncd_similarity,
        "description": "Normalized Compression Distance similarity (gzip, universal/parameter-free)",
        "params": {},
    },
}


def get_similarity_function(method: str) -> Callable:
    """
    Get similarity function by method name.

    Parameters
    ----------
    method : str
        Method name (e.g., 'lingo', 'edit', 'nlcs')

    Returns
    -------
    Callable
        Similarity function
    """
    if method not in AVAILABLE_METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(AVAILABLE_METHODS.keys())}")

    method_info = AVAILABLE_METHODS[method]

    if "requires" in method_info:
        req = method_info["requires"]
        if req == "scipy" and not SCIPY_AVAILABLE:
            raise ImportError(f"Method '{method}' requires scipy")
        if req == "sklearn" and not SKLEARN_AVAILABLE:
            raise ImportError(f"Method '{method}' requires scikit-learn")
        if req == "jellyfish" and not JELLYFISH_AVAILABLE:
            raise ImportError(f"Method '{method}' requires jellyfish")

    return method_info["function"]


# ============================================================================
# Batch Processing & Similarity Matrix Generation
# ============================================================================


def compute_similarity_matrix(smiles_list: List[str], method: str = "lingo", **kwargs) -> np.ndarray:
    """
    Compute pairwise similarity matrix for a list of SMILES.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings
    method : str
        Similarity method name
    **kwargs : dict
        Additional arguments for the similarity function

    Returns
    -------
    np.ndarray
        n x n similarity matrix
    """
    n = len(smiles_list)
    sim_matrix = np.zeros((n, n))

    sim_func = get_similarity_function(method)

    for i in range(n):
        sim_matrix[i, i] = 1.0  # Self-similarity
        for j in range(i + 1, n):
            sim = sim_func(smiles_list[i], smiles_list[j], **kwargs)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


def compute_cross_similarity_matrix(templates: List[str], library: List[str], method: str = "lingo", **kwargs) -> np.ndarray:
    """
    Compute similarity matrix between templates and library molecules.

    Parameters
    ----------
    templates : List[str]
        List of template SMILES strings
    library : List[str]
        List of library SMILES strings
    method : str
        Similarity method name
    **kwargs : dict
        Additional arguments for the similarity function

    Returns
    -------
    np.ndarray
        len(library) x len(templates) similarity matrix
    """
    n_lib = len(library)
    n_templates = len(templates)
    sim_matrix = np.zeros((n_lib, n_templates))

    sim_func = get_similarity_function(method)

    for i, lib_smiles in enumerate(library):
        for j, template_smiles in enumerate(templates):
            sim = sim_func(lib_smiles, template_smiles, **kwargs)
            sim_matrix[i, j] = sim

    return sim_matrix


# ============================================================================
# File I/O Functions
# ============================================================================


def read_smiles_file(filepath: str) -> Tuple[str, str]:
    """
    Read SMILES from a .smi file.

    Expected format: SMILES string (optionally followed by name/id)

    Parameters
    ----------
    filepath : str
        Path to .smi file

    Returns
    -------
    Tuple[str, str]
        (SMILES string, molecule name)
    """
    with open(filepath, "r") as f:
        content = f.read().strip()

    parts = content.split()
    smiles = parts[0] if parts else ""

    # Try to get name from file or content
    if len(parts) > 1:
        name = parts[1]
    else:
        name = Path(filepath).stem

    return smiles, name


def read_smiles_from_file(
    filepath: str,
    smiles_col: Optional[Union[int, str]] = None,
    name_col: Optional[Union[int, str]] = None,
    delimiter: Optional[str] = None,
    header: bool = True,
    skip_errors: bool = True,
) -> Dict[str, str]:
    """
    Read multiple SMILES from a single file.

    Supports various formats:
    - .smi/.smiles: Space/tab-separated SMILES and optional name
    - .csv: Comma-separated with header
    - .tsv: Tab-separated with header
    - Generic text files with configurable delimiter

    Parameters
    ----------
    filepath : str
        Path to file containing SMILES
    smiles_col : int or str, optional
        Column index (0-based) or name for SMILES.
        Default: 0 for .smi, auto-detect for .csv/.tsv
    name_col : int or str, optional
        Column index (0-based) or name for molecule names.
        Default: 1 for .smi, auto-detect for .csv/.tsv
    delimiter : str, optional
        Column delimiter. Auto-detected from file extension if not specified.
    header : bool
        Whether file has a header row (default: True for .csv/.tsv, False for .smi)
    skip_errors : bool
        If True, skip lines that can't be parsed; if False, raise exception

    Returns
    -------
    Dict[str, str]
        Dictionary mapping molecule names to SMILES strings

    Examples
    --------
    >>> molecules = read_smiles_from_file("library.smi")
    >>> molecules = read_smiles_from_file("data.csv", smiles_col="SMILES", name_col="ID")
    >>> molecules = read_smiles_from_file("data.tsv", smiles_col=0, name_col=1)
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    # Auto-detect format based on extension
    if delimiter is None:
        if ext in [".csv"]:
            delimiter = ","
        elif ext in [".tsv"]:
            delimiter = "\t"
        else:
            # For .smi, .smiles, .txt - use whitespace
            delimiter = None  # Will use split() for any whitespace

    # Default header behavior based on extension
    if ext in [".smi", ".smiles"]:
        header = False
        if smiles_col is None:
            smiles_col = 0
        if name_col is None:
            name_col = 1
    else:
        if smiles_col is None:
            smiles_col = 0
        if name_col is None:
            name_col = 1

    molecules = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines:
        return molecules

    # Process header if present
    start_idx = 0
    col_names = None

    if header and lines:
        header_line = lines[0].strip()
        if delimiter:
            col_names = header_line.split(delimiter)
        else:
            col_names = header_line.split()
        start_idx = 1

        # Convert column names to indices if strings were provided
        if isinstance(smiles_col, str):
            try:
                smiles_col = col_names.index(smiles_col)
            except ValueError:
                raise ValueError(f"SMILES column '{smiles_col}' not found in header: {col_names}")
        if isinstance(name_col, str):
            try:
                name_col = col_names.index(name_col)
            except ValueError:
                raise ValueError(f"Name column '{name_col}' not found in header: {col_names}")

    # Ensure indices are integers
    smiles_col = int(smiles_col) if smiles_col is not None else 0
    name_col = int(name_col) if name_col is not None else 1

    # Process data lines
    unnamed_counter = 0
    for line_num, line in enumerate(lines[start_idx:], start=start_idx + 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            if delimiter:
                parts = line.split(delimiter)
            else:
                parts = line.split()

            if len(parts) == 0:
                continue

            # Get SMILES
            if smiles_col >= len(parts):
                if skip_errors:
                    continue
                raise ValueError(f"Line {line_num}: SMILES column {smiles_col} out of range")
            smiles = parts[smiles_col].strip()

            if not smiles:
                continue

            # Get name
            if name_col is not None and name_col < len(parts):
                name = parts[name_col].strip()
            else:
                # Generate name if not available
                unnamed_counter += 1
                name = f"mol_{unnamed_counter}"

            # Handle duplicate names
            original_name = name
            counter = 1
            while name in molecules:
                name = f"{original_name}_{counter}"
                counter += 1

            molecules[name] = smiles

        except Exception as e:
            if skip_errors:
                continue
            raise ValueError(f"Error parsing line {line_num}: {e}")

    return molecules


def read_molecules_from_source(
    source: str,
    smiles_col: Optional[Union[int, str]] = None,
    name_col: Optional[Union[int, str]] = None,
    delimiter: Optional[str] = None,
    header: Optional[bool] = None,
) -> Dict[str, str]:
    """
    Read molecules from either a directory or a file.

    Automatically detects whether source is a directory (reads .smi files)
    or a file (reads multi-molecule format).

    Parameters
    ----------
    source : str
        Path to directory containing .smi files OR path to a single file
        containing multiple SMILES
    smiles_col : int or str, optional
        Column for SMILES (for file input)
    name_col : int or str, optional
        Column for names (for file input)
    delimiter : str, optional
        Column delimiter (for file input)
    header : bool, optional
        Whether file has header (for file input)

    Returns
    -------
    Dict[str, str]
        Dictionary mapping molecule names to SMILES strings
    """
    source_path = Path(source)

    if source_path.is_dir():
        return read_smiles_directory(str(source_path))
    elif source_path.is_file():
        kwargs = {}
        if smiles_col is not None:
            kwargs["smiles_col"] = smiles_col
        if name_col is not None:
            kwargs["name_col"] = name_col
        if delimiter is not None:
            kwargs["delimiter"] = delimiter
        if header is not None:
            kwargs["header"] = header
        return read_smiles_from_file(str(source_path), **kwargs)
    else:
        raise FileNotFoundError(f"Source not found: {source}")


def read_smiles_directory(dirpath: str) -> Dict[str, str]:
    """
    Read all SMILES files from a directory.

    Parameters
    ----------
    dirpath : str
        Path to directory containing .smi files

    Returns
    -------
    Dict[str, str]
        Dictionary mapping molecule names to SMILES strings
    """
    molecules = {}
    dirpath = Path(dirpath)

    for filepath in sorted(dirpath.glob("*.smi")):
        smiles, name = read_smiles_file(str(filepath))
        if smiles:
            molecules[name] = smiles

    return molecules


def write_similarity_csv(output_path: str, library_names: List[str], template_names: List[str], sim_matrix: np.ndarray):
    """
    Write similarity matrix to CSV file.

    Output format:
    Name,Similarity_{template1},Similarity_{template2},...

    Parameters
    ----------
    output_path : str
        Output CSV file path
    library_names : List[str]
        Names of library molecules (rows)
    template_names : List[str]
        Names of template molecules (columns)
    sim_matrix : np.ndarray
        Similarity matrix (library x templates)
    """
    # Create column names
    columns = ["Name"] + [f"Similarity_{name}" for name in template_names]

    # Create DataFrame
    data = {"Name": library_names}
    for j, template_name in enumerate(template_names):
        data[f"Similarity_{template_name}"] = sim_matrix[:, j]

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format="%.5f")


# ============================================================================
# Command Line Interface
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate SMILES-based similarity between molecules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate LINGO similarity between templates and library (directories)
  python smiles_similarity_kernels.py templates/ library/ output.csv --method lingo
  
  # Calculate all methods (creates multiple output files)
  python smiles_similarity_kernels.py templates/ library/ output.csv --all-methods
  
  # Use files instead of directories
  python smiles_similarity_kernels.py templates.smi library.smi output.csv --method lingo
  
  # Use CSV files with column specification
  python smiles_similarity_kernels.py templates.csv database.csv output.csv \\
      --method lingo --templates-smiles-col SMILES --templates-name-col ID \\
      --database-smiles-col SMILES --database-name-col MolID
  
  # Mix directory and file inputs
  python smiles_similarity_kernels.py templates/ library.smi output.csv --method edit
  
  # Use edit distance similarity
  python smiles_similarity_kernels.py templates/ library/ output.csv --method edit
  
  # List available methods
  python smiles_similarity_kernels.py --list-methods

Input formats:
  - Directory: Reads all .smi files from the directory
  - .smi/.smiles file: Space/tab-separated, no header (SMILES Name)
  - .csv file: Comma-separated with header
  - .tsv file: Tab-separated with header

Available methods: edit, nlcs, clcs, substring, smifp_cbd, smifp_tanimoto, 
                   lingo, lingo3, lingo5
        """,
    )

    parser.add_argument("templates", nargs="?", type=str, help="Directory or file containing template molecules (.smi, .csv, .tsv)")
    parser.add_argument("database", nargs="?", type=str, help="Directory or file containing database/library molecules (.smi, .csv, .tsv)")
    parser.add_argument("output", nargs="?", type=str, help="Output CSV file path")

    parser.add_argument(
        "--method", "-m", type=str, default="lingo", choices=list(AVAILABLE_METHODS.keys()), help="Similarity method to use (default: lingo)"
    )

    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Calculate similarities using all available methods. Output files will be named METHOD_output.csv",
    )

    # Template file options
    templates_group = parser.add_argument_group(
        "Template file options", "Options for reading templates from a file (ignored for directory input)"
    )
    templates_group.add_argument(
        "--templates-smiles-col", type=str, default=None, metavar="COL", help="Column name or index (0-based) for SMILES in templates file"
    )
    templates_group.add_argument(
        "--templates-name-col", type=str, default=None, metavar="COL", help="Column name or index (0-based) for names in templates file"
    )
    templates_group.add_argument(
        "--templates-delimiter",
        type=str,
        default=None,
        metavar="DELIM",
        help="Column delimiter for templates file (auto-detected if not specified)",
    )
    templates_group.add_argument("--templates-no-header", action="store_true", help="Templates file has no header row")

    # Database file options
    database_group = parser.add_argument_group("Database file options", "Options for reading database from a file (ignored for directory input)")
    database_group.add_argument(
        "--database-smiles-col", type=str, default=None, metavar="COL", help="Column name or index (0-based) for SMILES in database file"
    )
    database_group.add_argument(
        "--database-name-col", type=str, default=None, metavar="COL", help="Column name or index (0-based) for names in database file"
    )
    database_group.add_argument(
        "--database-delimiter",
        type=str,
        default=None,
        metavar="DELIM",
        help="Column delimiter for database file (auto-detected if not specified)",
    )
    database_group.add_argument("--database-no-header", action="store_true", help="Database file has no header row")

    parser.add_argument("--list-methods", action="store_true", help="List available similarity methods and exit")

    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Canonicalize SMILES with RDKit before comparison (requires rdkit). " 'Ensures "CCO" and "OCC" are treated as the same molecule.',
    )

    parser.add_argument(
        "--inchi",
        action="store_true",
        help='Convert SMILES to InChI (stripping the "InChI=" prefix) before '
        "comparison (requires rdkit). Useful for representation-independent "
        "similarity; implies --canonicalize.",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress information")

    return parser.parse_args()


def _parse_col_arg(col_arg: Optional[str]) -> Optional[Union[int, str]]:
    """
    Parse column argument - convert to int if numeric, otherwise keep as string.

    Parameters
    ----------
    col_arg : str or None
        Column argument from command line

    Returns
    -------
    int, str, or None
        Parsed column specification
    """
    if col_arg is None:
        return None
    try:
        return int(col_arg)
    except ValueError:
        return col_arg


def main():
    """Main function for command line execution."""
    args = parse_args()

    # List methods if requested
    if args.list_methods:
        print("\nAvailable similarity methods:")
        print("-" * 60)
        for name, info in AVAILABLE_METHODS.items():
            req = f" (requires {info.get('requires', 'nothing')})" if "requires" in info else ""
            print(f"  {name:20s} - {info['description']}{req}")
        print()
        return

    # Check required arguments
    if not args.templates or not args.database or not args.output:
        print("Error: templates, database, and output are required.")
        print("Use --help for usage information.")
        sys.exit(1)

    # Parse column arguments
    templates_smiles_col = _parse_col_arg(args.templates_smiles_col)
    templates_name_col = _parse_col_arg(args.templates_name_col)
    database_smiles_col = _parse_col_arg(args.database_smiles_col)
    database_name_col = _parse_col_arg(args.database_name_col)

    # Read templates
    if args.verbose:
        source_type = "directory" if Path(args.templates).is_dir() else "file"
        print(f"Reading templates from {source_type}: {args.templates}")

    templates = read_molecules_from_source(
        args.templates,
        smiles_col=templates_smiles_col,
        name_col=templates_name_col,
        delimiter=args.templates_delimiter,
        header=None if not args.templates_no_header else False,
    )

    # Read database/library
    if args.verbose:
        source_type = "directory" if Path(args.database).is_dir() else "file"
        print(f"Reading database from {source_type}: {args.database}")

    library = read_molecules_from_source(
        args.database,
        smiles_col=database_smiles_col,
        name_col=database_name_col,
        delimiter=args.database_delimiter,
        header=None if not args.database_no_header else False,
    )

    if not templates:
        print(f"Error: No molecules found in templates source: {args.templates}")
        sys.exit(1)

    if not library:
        print(f"Error: No molecules found in database source: {args.database}")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(templates)} templates and {len(library)} database molecules")
        if args.all_methods:
            print(f"Using all methods: {', '.join(AVAILABLE_METHODS.keys())}")
        else:
            print(f"Using method: {args.method}")

    # Get ordered lists
    template_names = list(templates.keys())
    template_smiles = [templates[n] for n in template_names]
    library_names = list(library.keys())
    library_smiles = [library[n] for n in library_names]

    # Optional SMILES canonicalization / InChI conversion
    if args.inchi or args.canonicalize:
        if not RDKIT_AVAILABLE:
            print("Warning: --canonicalize/--inchi requested but RDKit is not installed. " "Install with: pip install rdkit", file=sys.stderr)
        else:
            if args.inchi:
                if args.verbose:
                    print("Converting SMILES to InChI (stripping 'InChI=' prefix)...")

                def _transform(s):
                    return smiles_to_inchi(s) or s

            else:
                if args.verbose:
                    print("Canonicalizing SMILES with RDKit...")

                def _transform(s):
                    return canonicalize_smiles(s)

            template_smiles = [_transform(s) for s in template_smiles]
            library_smiles = [_transform(s) for s in library_smiles]

    # Determine which methods to use
    if args.all_methods:
        methods_to_run = list(AVAILABLE_METHODS.keys())
    else:
        methods_to_run = [args.method]

    # Calculate similarities for each method
    for method in methods_to_run:
        if args.all_methods:
            # Generate output filename for this method
            output_path = Path(args.output)
            method_output = output_path.parent / f"{method}_{output_path.name}"
        else:
            method_output = args.output

        if args.verbose:
            if args.all_methods:
                print(f"\nProcessing method: {method}")
            print("Calculating similarities...")
            total_comparisons = len(library) * len(templates)
            print(f"  Total comparisons: {total_comparisons:,}")

        sim_matrix = compute_cross_similarity_matrix(template_smiles, library_smiles, method=method)

        # Write output
        if args.verbose:
            print(f"Writing output to: {method_output}")

        write_similarity_csv(method_output, library_names, template_names, sim_matrix)

    if args.verbose:
        if args.all_methods:
            print(f"\nCompleted! Generated {len(methods_to_run)} output files.")
        print("Done!")


# ============================================================================
# Demo / Test
# ============================================================================


def demo():
    """Run a demonstration of the similarity functions."""
    print("=" * 60)
    print("SMILES-based Similarity Kernels - Demo")
    print("=" * 60)

    # Example SMILES strings
    smiles1 = "OC(O)=O"  # Carbonic acid
    smiles2 = "CCCCC(O)=C"  # Example
    smiles3 = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    smiles4 = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    smiles5 = "c1ccc(Cl)cc1"  # Chlorobenzene
    smiles6 = "c1ccc(Br)cc1"  # Bromobenzene

    print(f"\nTest SMILES strings:")
    print(f"  S1: {smiles1} (Carbonic acid)")
    print(f"  S2: {smiles2}")
    print(f"  S3: {smiles3} (Aspirin)")
    print(f"  S4: {smiles4} (Caffeine)")
    print(f"  S5: {smiles5} (Chlorobenzene)")
    print(f"  S6: {smiles6} (Bromobenzene)")

    print("\n--- Preprocessing Demo ---")
    print(f"  Chlorobenzene original: {smiles5}")
    print(f"  Chlorobenzene processed: {preprocess_smiles(smiles5)}")
    print(f"  Bromobenzene original: {smiles6}")
    print(f"  Bromobenzene processed: {preprocess_smiles(smiles6)}")

    # Test with more complex SMILES
    complex_smiles = "[Si](C)(C)O[Si](C)(C)C"  # Siloxane
    print(f"  Siloxane original: {complex_smiles}")
    print(f"  Siloxane processed: {preprocess_smiles(complex_smiles)}")

    print("\n--- Pairwise Similarities (Chlorobenzene vs Bromobenzene) ---")
    print(f"  Edit Distance:     {edit_similarity(smiles5, smiles6):.4f}")
    print(f"  NLCS:              {nlcs_similarity(smiles5, smiles6):.4f}")
    print(f"  CLCS:              {clcs_similarity(smiles5, smiles6):.4f}")
    print(f"  Substring Kernel:  {substring_kernel_similarity(smiles5, smiles6, normalized=True):.4f}")
    print(f"  SMIfp (Tanimoto):  {smifp_similarity_tanimoto(smiles5, smiles6):.4f}")
    print(f"  LINGO (q=4):       {lingo_similarity(smiles5, smiles6, q=4):.4f}")

    print("\n--- Similarity Matrix (4 molecules, LINGO method) ---")
    test_smiles = [smiles1, smiles2, smiles3, smiles4]
    test_names = ["Carbonic", "S2", "Aspirin", "Caffeine"]
    sim_matrix = compute_similarity_matrix(test_smiles, method="lingo")

    # Print header
    print("\n" + " " * 12, end="")
    for name in test_names:
        print(f"{name:>10s}", end="")
    print()

    # Print matrix
    for i, (name, row) in enumerate(zip(test_names, sim_matrix)):
        print(f"{name:>12s}", end="")
        for val in row:
            print(f"{val:>10.3f}", end="")
        print()

    print("\n--- Multi-character element list ---")
    print(f"  Elements handled: {', '.join(sorted(ELEMENT_REPLACEMENTS.keys()))}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    # If no arguments provided, run demo
    if len(sys.argv) == 1:
        demo()
    else:
        main()
