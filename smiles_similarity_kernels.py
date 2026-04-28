#!/usr/bin/env python3
"""
SMILES-based Similarity Kernels

Python implementation of SMILES-based compound similarity functions. Partially inspired by
and extended:

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

Author: fstefaniak@iimcb.gov.pl, https://github.com/filipsPL/smiles_similarity_kernels.py

Cite all versions by using the DOI 10.5281/zenodo.18457244
"""

import re
import os
import sys
import json
import time
import random
import warnings
import argparse
import functools
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

# Optional import for selfies (SELFIES molecular string representation)
try:
    import selfies as sf

    SELFIES_AVAILABLE = True
except ImportError:
    SELFIES_AVAILABLE = False


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


# ----------------------------------------------------------------------------
# InChI preprocessing and layer extraction
# ----------------------------------------------------------------------------

# Standard InChI layer prefixes.  Each appears after a '/' separator in the
# string (except the formula, which is the first field after the version).
#
# Standard InChI format:
#   InChI=<version>/<formula>/c<conn>/h<H>/q<charge>/p<protons>
#                  /b<dbStereo>/t<tetraStereo>/m<parity>/s<stereoType>
#                  /i<isotope>/h<mobileH>/f<fixedH>/r<reconnectedMetals>
#
# The version is always the first segment (e.g. '1S' for standard InChI).
# The formula layer has NO leading letter; every other layer starts with a
# single-letter prefix indicating the layer type.
INCHI_LAYERS = {
    "formula": None,  # special: first segment after version, no prefix letter
    "connections": "c",
    "hydrogens": "h",
    "charge": "q",
    "protons": "p",
    "stereo_db": "b",
    "stereo_tet": "t",
    "stereo_parity": "m",
    "stereo_type": "s",
    "isotope": "i",
    "fixedH": "f",
    "reconnected": "r",
}

# Set of all single-letter layer prefixes for quick lookup
_INCHI_LAYER_PREFIXES = {v for v in INCHI_LAYERS.values() if v is not None}


def preprocess_inchi(inchi: str, strip_version: bool = True) -> str:
    """
    Minimal, layer-respecting preprocessing for InChI strings.

    Unlike SMILES, InChI strings are *layered* (segments separated by '/')
    and multi-character element symbols appear with count suffixes in the
    formula layer (e.g. 'C6H5Cl').  Character-level substitution — as used
    for SMILES — would therefore break the parseability of the layers and
    create meaningless q-grams.

    This function performs only minimal cleanup:
      - strips the leading 'InChI=' prefix if present (idempotent with
        :func:`smiles_to_inchi`, which already removes it)
      - optionally strips the version tag ('1S/' or '1/') so that string
        similarity is not artificially inflated by a shared constant prefix

    Layer separators '/' are deliberately **not** modified — they serve as
    natural boundaries that prevent q-grams from straddling unrelated
    layers.

    Parameters
    ----------
    inchi : str
        Input InChI string (with or without 'InChI=' prefix)
    strip_version : bool
        If True (default), strip the '1S/' or '1/' version tag.  Every
        standard InChI shares this prefix, so keeping it inflates the
        pairwise similarity of short molecules.

    Returns
    -------
    str
        Cleaned InChI string

    Examples
    --------
    >>> preprocess_inchi("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
    'C2H6O/c1-2-3/h3H,2H2,1H3'
    >>> preprocess_inchi("1S/CH4/h1H4", strip_version=False)
    '1S/CH4/h1H4'
    """
    if not inchi:
        return inchi
    if inchi.startswith("InChI="):
        inchi = inchi[6:]
    if strip_version:
        for prefix in ("1S/", "1/"):
            if inchi.startswith(prefix):
                inchi = inchi[len(prefix) :]
                break
    return inchi


def extract_inchi_layers(inchi: str, layers: Union[str, List[str]]) -> str:
    """
    Extract one or more layers from an InChI string.

    Allows comparison of molecules based on a selected subset of structural
    information — for example, formula-only (very coarse), connections-only
    (topology without hydrogens or stereochemistry), or connections+hydrogens
    (most structural information, stereochemistry excluded).

    The leading 'InChI=' prefix and '1S/'/'1/' version tag are stripped
    before extraction.  Layers are returned concatenated with '/' separators
    in the order given, each still carrying its single-letter prefix
    (except the formula layer, which has no prefix).  If a requested layer
    is absent from the input, it is silently omitted.

    Parameters
    ----------
    inchi : str
        Input InChI string (with or without 'InChI=' prefix)
    layers : str or List[str]
        Layer name(s) to extract.  Supported names are the keys of
        :data:`INCHI_LAYERS`:

        - 'formula'      — molecular formula (e.g. 'C9H8O4')
        - 'connections'  — atom-connection layer ('c...')
        - 'hydrogens'    — hydrogen layer ('h...')
        - 'charge'       — charge layer ('q...')
        - 'protons'      — proton layer ('p...')
        - 'stereo_db'    — double-bond stereo ('b...')
        - 'stereo_tet'   — tetrahedral stereo ('t...')
        - 'stereo_parity'— parity ('m...')
        - 'stereo_type'  — stereo type ('s...')
        - 'isotope'      — isotope ('i...')
        - 'fixedH'       — fixed-H ('f...')
        - 'reconnected'  — reconnected-metals ('r...')

        A single string is treated as a one-element list.  Use 'all' to
        return the full preprocessed InChI (equivalent to
        ``preprocess_inchi``).

    Returns
    -------
    str
        The extracted layer(s) concatenated with '/' separators.  Empty
        string if the input is empty or no requested layers are present.

    Examples
    --------
    >>> inchi = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
    >>> extract_inchi_layers(inchi, "formula")
    'C9H8O4'
    >>> extract_inchi_layers(inchi, "connections")
    'c1-6(10)13-8-5-3-2-4-7(8)9(11)12'
    >>> extract_inchi_layers(inchi, ["formula", "connections"])
    'C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12'
    """
    if not inchi:
        return ""

    if isinstance(layers, str):
        if layers == "all":
            return preprocess_inchi(inchi, strip_version=True)
        layers = [layers]

    # Validate layer names
    for layer in layers:
        if layer not in INCHI_LAYERS:
            raise ValueError(f"Unknown InChI layer: '{layer}'. " f"Available: {list(INCHI_LAYERS.keys())}")

    cleaned = preprocess_inchi(inchi, strip_version=True)
    if not cleaned:
        return ""

    # Split on '/' and classify each segment by its prefix letter.
    # The first segment is always the formula (no prefix letter).
    segments = cleaned.split("/")
    if not segments:
        return ""

    layer_contents: Dict[str, str] = {}
    layer_contents["formula"] = segments[0]

    for seg in segments[1:]:
        if not seg:
            continue
        prefix = seg[0]
        # Find which layer name corresponds to this prefix
        for name, pfx in INCHI_LAYERS.items():
            if pfx == prefix:
                layer_contents[name] = seg
                break

    # Assemble requested layers in the order the user asked for them.
    parts = [layer_contents[name] for name in layers if name in layer_contents]
    return "/".join(parts)


def smiles_to_inchi_layers(smiles: str, layers: Union[str, List[str]] = "all") -> str:
    """
    Convert a SMILES string to selected InChI layer(s) in one step.

    Convenience wrapper around :func:`smiles_to_inchi` and
    :func:`extract_inchi_layers`.  Useful for batch pipelines where
    every molecule is to be represented by the same subset of layers.

    Parameters
    ----------
    smiles : str
        Input SMILES string
    layers : str or List[str]
        Layer name(s) to retain; see :func:`extract_inchi_layers`.
        Use 'all' (default) to retain the full InChI (minus 'InChI='
        prefix and version tag).

    Returns
    -------
    str
        Selected InChI layers, or empty string on failure.

    Examples
    --------
    >>> smiles_to_inchi_layers("CCO", "connections")
    'c1-2-3'
    >>> smiles_to_inchi_layers("CCO", ["formula", "connections"])
    'C2H6O/c1-2-3'
    """
    inchi = smiles_to_inchi(smiles)
    if not inchi:
        return ""
    return extract_inchi_layers(inchi, layers)


def shuffle_smiles(smiles: str, seed: Optional[int] = None) -> str:
    """
    Randomly shuffle the characters of a SMILES string.

    This is a **negative control** transformation: the result is a chemically
    meaningless string of the same length and character composition as the
    input.  Similarity scores computed against shuffled strings should be
    close to the baseline expected for random string pairs.

    Parameters
    ----------
    smiles : str
        Input SMILES string
    seed : int or None
        Optional random seed for reproducibility

    Returns
    -------
    str
        Character-shuffled version of the input string

    Examples
    --------
    >>> sorted(shuffle_smiles("CCO")) == sorted("CCO")
    True
    """
    chars = list(smiles)
    rng = random.Random(seed)
    rng.shuffle(chars)
    return "".join(chars)


def sort_string(s: str) -> str:
    """
    Sort the characters of a string alphabetically.

    Like :func:`shuffle_smiles`, this is a **negative control** transformation:
    the result is chemically meaningless but preserves the length and character
    composition of the input.  Sorting is deterministic (no seed needed), which
    makes it a reproducible fixed-order baseline complementary to the random
    shuffle.

    Parameters
    ----------
    s : str
        Input string (SMILES, InChI, SELFIES, or any string representation)

    Returns
    -------
    str
        Character-sorted version of the input string

    Examples
    --------
    >>> sort_string("CCO")
    'CCO'
    >>> sort_string("c1ccccc1")
    '1cccccc'
    """
    return "".join(sorted(s))


def smiles_to_selfies(smiles: str) -> str:
    """
    Convert a SMILES string to a SELFIES string.

    SELFIES (Self-Referencing Embedded Strings) are a 100% robust molecular
    string representation — every string decodes to a valid molecule.  Unlike
    SMILES, string-similarity methods on SELFIES cannot produce invalid
    intermediates, making them useful for generative and similarity tasks.

    Requires the ``selfies`` package (``pip install selfies``).  Returns an
    empty string when conversion fails or the package is unavailable.

    Parameters
    ----------
    smiles : str
        Input SMILES string

    Returns
    -------
    str
        SELFIES string, or '' on failure

    Examples
    --------
    >>> smiles_to_selfies("CCO")
    '[C][C][O]'
    >>> smiles_to_selfies("c1ccccc1")
    '[C][=C][C][=C][C][=C][Ring1][=A]'
    """
    if not smiles or not SELFIES_AVAILABLE:
        return ""
    try:
        return sf.encoder(smiles)
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

    # Two-row rolling DP — only the previous row is ever needed.
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])  # deletion  # insertion  # substitution
        prev = curr

    return prev[n]


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

    # Two-row rolling DP — only the previous row is ever needed.
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    return prev[n]


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

    # Two-row rolling DP — only the previous row is ever needed.
    prev = [0] * (n + 1)
    max_length = 0

    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > max_length:
                    max_length = curr[j]
        prev = curr

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
    if abs(w1 + w2 + w3 - 1.0) > 1e-9:
        warnings.warn(f"clcs_similarity: weights w1={w1}, w2={w2}, w3={w3} sum to {w1+w2+w3:.6g}, not 1. Scores will be off-scale.", stacklevel=2)

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

    counts = Counter(smiles)
    return np.array([counts.get(char, 0) for char in chars], dtype=float)


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

    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
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
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
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


def lingo_tversky_similarity(
    smiles1: str,
    smiles2: str,
    q: int = 4,
    alpha: float = 0.9,
    beta: float = 0.1,
    preprocess: bool = True,
) -> float:
    """
    Asymmetric Tversky similarity on LINGO (q-gram) count vectors.

    The Tversky index generalises Tanimoto/Dice with two weights that
    control how strongly missing features on each side are penalised:

        S(A, B) = |A ∩ B| / (|A ∩ B| + alpha * |A \\ B| + beta * |B \\ A|)

    where A, B are the multisets of q-grams of smiles1 and smiles2.
    Intersection/difference are computed on multiset counts so that
    repeated q-grams contribute appropriately.

    Setting alpha = beta = 1 recovers the Tanimoto-style coefficient;
    alpha = beta = 0.5 recovers Dice.  The default (alpha=0.9, beta=0.1)
    is the "query-weighted" asymmetric Tversky used by Bajusz et al.
    (2025) for nucleic-acid ligand screening, where smiles1 is treated
    as the *query* (reference template) and smiles2 as the *database*
    candidate: q-grams present in the query but missing in the database
    are penalised more than q-grams present only in the database.

    Parameters
    ----------
    smiles1 : str
        Query (template/reference) SMILES string
    smiles2 : str
        Database/candidate SMILES string
    q : int
        LINGO length (default 4)
    alpha : float
        Weight applied to q-grams unique to the query (smiles1).
        Default 0.9 (query-weighted).
    beta : float
        Weight applied to q-grams unique to the database (smiles2).
        Default 0.1 (query-weighted).
    preprocess : bool
        Whether to preprocess SMILES before q-gram extraction.

    Returns
    -------
    float
        Tversky similarity in [0, 1].  Asymmetric when alpha != beta.

    References
    ----------
    Tversky A. "Features of similarity." Psychological Review 84, 327–352 (1977).

    Bajusz D., Rácz A., Stefaniak F. "Evaluation of single-template
    ligand-based methods for the discovery of small-molecule nucleic
    acid binders." Briefings in Bioinformatics, 2025.
    """
    lingos1 = get_lingos(smiles1, q, normalize_rings=True, preprocess=preprocess)
    lingos2 = get_lingos(smiles2, q, normalize_rings=True, preprocess=preprocess)

    if not lingos1 and not lingos2:
        return 1.0
    if not lingos1 or not lingos2:
        return 0.0

    # Multiset intersection and differences
    intersection = 0
    only1 = 0
    only2 = 0

    all_keys = set(lingos1) | set(lingos2)
    for k in all_keys:
        c1 = lingos1.get(k, 0)
        c2 = lingos2.get(k, 0)
        intersection += min(c1, c2)
        only1 += max(c1 - c2, 0)
        only2 += max(c2 - c1, 0)

    denominator = intersection + alpha * only1 + beta * only2
    if denominator == 0:
        return 0.0
    return intersection / denominator


def lingo_dice_similarity(smiles1: str, smiles2: str, q: int = 4, preprocess: bool = True) -> float:
    """
    Sørensen–Dice coefficient on LINGO (q-gram) count vectors.

    Equivalent to Tversky with alpha = beta = 0.5.  Compared to Tanimoto,
    Dice weights shared q-grams more heavily and typically yields higher
    values for moderately similar molecule pairs, which can improve
    early-enrichment performance on some targets.

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings to compare.
    q : int
        LINGO length (default 4).
    preprocess : bool
        Whether to preprocess SMILES.

    Returns
    -------
    float
        Dice similarity in [0, 1].
    """
    return lingo_tversky_similarity(smiles1, smiles2, q=q, alpha=0.5, beta=0.5, preprocess=preprocess)


# ============================================================================
# 7. Spectrum Kernel (fixed k, no ring normalization)
# ============================================================================


def spectrum_kernel_similarity(
    smiles1: str,
    smiles2: str,
    k: int = 4,
    coefficient: str = "tanimoto",
    normalize_rings: bool = False,
    preprocess: bool = True,
) -> float:
    """
    Spectrum kernel similarity between two SMILES strings.

    The spectrum kernel (Leslie, Eskin & Noble, 2002) represents each
    sequence as a vector of counts of every k-mer, then compares two
    sequences through an inner product (or a normalised coefficient).
    It is the canonical fixed-k string kernel and the most widely
    benchmarked alignment-free method in biological sequence work.

    Differs from :func:`lingo_similarity` in two ways:

    1. The similarity coefficient is a single inner-product-based
       measure (Tanimoto / Dice / cosine) on the full count vector,
       rather than an averaged per-q-gram agreement.
    2. Ring digits are **not** normalised to '0' by default, preserving
       ring-closure identity.  Set ``normalize_rings=True`` to match
       the LINGO convention.

    Parameters
    ----------
    smiles1 : str
        First SMILES string
    smiles2 : str
        Second SMILES string
    k : int
        k-mer length (default 4).
    coefficient : {'tanimoto', 'dice', 'cosine'}
        Normalisation of the kernel inner product.
    normalize_rings : bool
        Whether to replace all ring-closure digits with '0'.
    preprocess : bool
        Whether to apply SMILES multi-character preprocessing.

    Returns
    -------
    float
        Normalised similarity in [0, 1].

    References
    ----------
    Leslie C., Eskin E., Noble W. "The spectrum kernel: a string kernel
    for SVM protein classification." PSB 2002, 564–575.
    """
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
    if normalize_rings:
        smiles1 = normalize_ring_numbers(smiles1)
        smiles2 = normalize_ring_numbers(smiles2)

    if len(smiles1) < k and len(smiles2) < k:
        return 1.0 if smiles1 == smiles2 else 0.0
    if len(smiles1) < k or len(smiles2) < k:
        return 0.0

    # Build k-mer count vectors
    counts1: Counter = Counter(smiles1[i : i + k] for i in range(len(smiles1) - k + 1))
    counts2: Counter = Counter(smiles2[i : i + k] for i in range(len(smiles2) - k + 1))

    # Inner product, self-inner-products
    dot = 0.0
    for kmer, c in counts1.items():
        if kmer in counts2:
            dot += c * counts2[kmer]
    norm1 = sum(c * c for c in counts1.values())
    norm2 = sum(c * c for c in counts2.values())
    if norm1 == 0 or norm2 == 0:
        return 0.0

    coef = coefficient.lower()
    if coef == "cosine":
        return dot / (np.sqrt(norm1) * np.sqrt(norm2))
    if coef == "tanimoto":
        denominator = norm1 + norm2 - dot
        if denominator <= 0:
            return 0.0
        return dot / denominator
    if coef == "dice":
        denominator = norm1 + norm2
        if denominator <= 0:
            return 0.0
        return 2.0 * dot / denominator
    raise ValueError(f"Unknown coefficient: '{coefficient}'. " "Supported: 'tanimoto', 'dice', 'cosine'.")


# ============================================================================
# 8. Mismatch Kernel (spectrum-(k, m) kernel)
# ============================================================================


@functools.lru_cache(maxsize=4096)
def _generate_mismatches(kmer: str, m: int, alphabet: str) -> List[str]:
    """
    Generate all strings at Hamming distance <= m from ``kmer``.

    For small m (typically 1 or 2) and moderate k this is tractable;
    the total number of mismatched strings is
    sum(C(k, i) * (|alphabet| - 1)**i for i in 0..m).

    Used internally by :func:`mismatch_kernel_similarity`.
    """
    if m < 0:
        return []
    results = {kmer}
    current = {kmer}
    for _ in range(m):
        nxt = set()
        for s in current:
            for i in range(len(s)):
                for ch in alphabet:
                    if ch != s[i]:
                        candidate = s[:i] + ch + s[i + 1 :]
                        if candidate not in results:
                            nxt.add(candidate)
        results |= nxt
        current = nxt
        if not current:
            break
    return list(results)


def mismatch_kernel_similarity(
    smiles1: str,
    smiles2: str,
    k: int = 4,
    m: int = 1,
    coefficient: str = "tanimoto",
    normalize_rings: bool = False,
    preprocess: bool = True,
    alphabet: Optional[str] = None,
) -> float:
    """
    Mismatch (spectrum-(k, m)) kernel similarity between two SMILES.

    The mismatch kernel (Leslie, Eskin, Weston & Noble, 2004) extends the
    spectrum kernel so that a pair of k-mers is considered a match when
    they differ in at most *m* positions (Hamming distance <= m).  For
    SMILES this captures the intuition that "CCCCN" and "CCCCO" encode
    nearly the same molecule (one-atom swap), which pure q-gram methods
    score very low.

    Implementation: each k-mer in *smiles1* contributes to the inner
    product against every k-mer in *smiles2* that lies within the
    m-mismatch neighbourhood.  For m = 0 this reduces to the exact
    spectrum kernel.

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings to compare.
    k : int
        k-mer length (default 4).
    m : int
        Maximum number of allowed mismatches per k-mer (default 1).
    coefficient : {'tanimoto', 'dice', 'cosine'}
        Normalisation of the kernel inner product.
    normalize_rings : bool
        Whether to replace all ring-closure digits with '0'.
    preprocess : bool
        Whether to apply SMILES multi-character preprocessing.
    alphabet : str, optional
        Alphabet used to enumerate mismatches.  If not given, the union
        of characters that actually appear in the two preprocessed
        SMILES strings is used — this keeps the neighbourhood small
        while still capturing every biologically meaningful substitution.

    Returns
    -------
    float
        Normalised similarity in [0, 1].

    Notes
    -----
    Computational cost grows with the neighbourhood size, roughly
    ``O(|S| * C(k, m) * (|alphabet|-1)**m)``.  For SMILES with
    alphabets of ~30–50 symbols, m = 1 (and k <= 5) is practical;
    m = 2 is expensive and rarely needed.

    References
    ----------
    Leslie C., Eskin E., Weston J., Noble W. "Mismatch string kernels
    for discriminative protein classification." Bioinformatics 20,
    467–476 (2004).
    """
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
    if normalize_rings:
        smiles1 = normalize_ring_numbers(smiles1)
        smiles2 = normalize_ring_numbers(smiles2)

    if m < 0:
        raise ValueError("m must be >= 0")
    if m == 0:
        # Fall back to the plain spectrum kernel — same semantics, faster path.
        return spectrum_kernel_similarity(smiles1, smiles2, k=k, coefficient=coefficient, normalize_rings=False, preprocess=False)

    if len(smiles1) < k and len(smiles2) < k:
        return 1.0 if smiles1 == smiles2 else 0.0
    if len(smiles1) < k or len(smiles2) < k:
        return 0.0

    if alphabet is None:
        alphabet = "".join(sorted(set(smiles1) | set(smiles2)))
    if len(alphabet) < 2:
        # Degenerate: no mismatches possible, reduces to spectrum kernel
        return spectrum_kernel_similarity(smiles1, smiles2, k=k, coefficient=coefficient, normalize_rings=False, preprocess=False)

    # Build exact k-mer counts
    counts1 = Counter(smiles1[i : i + k] for i in range(len(smiles1) - k + 1))
    counts2 = Counter(smiles2[i : i + k] for i in range(len(smiles2) - k + 1))

    # Expand each k-mer to its m-mismatch neighbourhood.  Each (neighbour,
    # source_count) contributes to the feature vector indexed by ``neighbour``.
    def _expanded(counts: Counter) -> Counter:
        exp: Counter = Counter()
        for kmer, c in counts.items():
            for nb in _generate_mismatches(kmer, m, alphabet):
                exp[nb] += c
        return exp

    exp1 = _expanded(counts1)
    exp2 = _expanded(counts2)

    # Inner products in the expanded feature space
    dot = 0.0
    for kmer, c in exp1.items():
        if kmer in exp2:
            dot += c * exp2[kmer]
    norm1 = sum(c * c for c in exp1.values())
    norm2 = sum(c * c for c in exp2.values())
    if norm1 == 0 or norm2 == 0:
        return 0.0

    coef = coefficient.lower()
    if coef == "cosine":
        return dot / (np.sqrt(norm1) * np.sqrt(norm2))
    if coef == "tanimoto":
        denominator = norm1 + norm2 - dot
        if denominator <= 0:
            return 0.0
        return dot / denominator
    if coef == "dice":
        denominator = norm1 + norm2
        if denominator <= 0:
            return 0.0
        return 2.0 * dot / denominator
    raise ValueError(f"Unknown coefficient: '{coefficient}'. " "Supported: 'tanimoto', 'dice', 'cosine'.")


# ============================================================================
# 9. Longest Common Substring similarity (normalised, interpretable)
# ============================================================================


def longest_common_substring_similarity(smiles1: str, smiles2: str, preprocess: bool = True) -> float:
    """
    Normalised Longest Common *Substring* (contiguous) similarity.

    Returns ``len(LCSubstr)^2 / (len(s1) * len(s2))``, analogous to NLCS
    but requiring the common part to be *contiguous* (a substring, not a
    subsequence).  This exposes the longest-common-substring logic from
    :func:`mclcsn_length` — already used internally by :func:`clcs_similarity` —
    as a stand-alone method, because the contiguous shared stretch is
    often directly interpretable as a shared scaffold-ish fragment.

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings to compare.
    preprocess : bool
        Whether to apply SMILES multi-character preprocessing.

    Returns
    -------
    float
        Similarity in [0, 1].
    """
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)
    if not smiles1 or not smiles2:
        return 0.0 if (smiles1 or smiles2) else 1.0
    lcs = mclcsn_length(smiles1, smiles2)
    return (lcs * lcs) / (len(smiles1) * len(smiles2))


# ============================================================================
# 10. LINGO-based TF and TF-IDF Cosine Similarity
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
        try:
            vectorizer.fit(corpus)
        except ValueError:
            return 0.0

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
    smiles1: str,
    smiles2: str,
    corpus: List[str] = None,
    ngram_range: Tuple[int, int] = (1, 2),
    vectorizer=None,
    preprocess: bool = False,
) -> float:
    """
    TF-IDF cosine similarity with chemically-aware tokenization.

    Uses SMILESTokenizer so that multi-character atoms (Cl, Br, …) and
    stereochemistry markers (@@) are treated as indivisible tokens.
    When a pre-fitted vectorizer is supplied it is reused directly,
    which is strongly recommended for batch/matrix calculations.

    Note: the ``preprocess`` argument is accepted for API compatibility
    with other similarity functions but is ignored — this method relies
    on the chemical tokenizer rather than on character substitution, so
    it works correctly on both SMILES and InChI inputs without any
    additional preprocessing step.

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
        try:
            vectorizer.fit(corpus)
        except ValueError:
            return 0.0

    vec1 = vectorizer.transform([smiles1])
    vec2 = vectorizer.transform([smiles2])
    return float(sklearn_cosine_similarity(vec1, vec2)[0, 0])


# ============================================================================
# 10. Schwaller TF-IDF Cosine Similarity
# ============================================================================


class SMILESTokenizerSchwaller:
    """
    Schwaller-style SMILES tokenizer for use with sklearn TF-IDF.

    Implements the regex-based atom-level tokenization from Schwaller et al.
    (Molecular Transformer, 2019).  Key differences from :class:`SMILESTokenizer`:

    - Bracket atoms ``[nH+]``, ``[NH3+]``, ``[13C]`` etc. are captured as a
      single indivisible token (the whole ``[...]`` group).
    - Bare two-character elements (``Br``, ``Cl``) are single tokens.
    - Every bond symbol (``=``, ``#``), branch delimiter (``(``, ``)``),
      ring-closure digit, and stereochemistry marker (``@``, ``/``, ``\\``)
      is its own token.
    - ``%dd`` two-digit ring closures are a single token.

    This gives a chemically complete atom-level tokenization that is the
    de-facto standard in sequence-to-sequence chemical models.

    Reference: Schwaller et al. *ACS Central Science* 2019, 5, 1572–1583.
    """

    # Schwaller et al. tokenization regex (longest match wins via ordering).
    # Bracket atoms first, then two-char elements, then single chars/symbols.
    _TOKEN_RE = re.compile(
        r"\[[^\]]+\]"  # bracket atom: [nH+], [NH3+], [13C@H], ...
        r"|Br?"  # Br or bare B
        r"|Cl?"  # Cl or bare C
        r"|N|O|S|P|F|I"  # other common single-char bare atoms
        r"|b|c|n|o|s|p"  # aromatic bare atoms
        r"|\(|\)"  # branch open/close
        r"|\.|\=|\#"  # disconnect, double bond, triple bond
        r"|-|\+"  # charge signs used as bond or in SMARTS
        r"|\\|/"  # stereo bond directions
        r"|:|~|@"  # aromatic bond, unspecified, chirality
        r"|\?|>|\*|\$"  # query atoms / reaction arrow / wildcard
        r"|\%[0-9]{2}"  # two-digit ring closure %10, %99, ...
        r"|[0-9]"  # single-digit ring closure
        r"|."  # catch-all for anything else
    )

    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string into Schwaller atom-level tokens."""
        return self._TOKEN_RE.findall(smiles)

    def __call__(self, smiles: str) -> List[str]:
        return self.tokenize(smiles)


def schwaller_tfidf_similarity(
    smiles1: str,
    smiles2: str,
    corpus: List[str] = None,
    ngram_range: Tuple[int, int] = (1, 2),
    vectorizer=None,
    preprocess: bool = False,
) -> float:
    """
    TF-IDF cosine similarity using Schwaller atom-level tokenization.

    Uses :class:`SMILESTokenizerSchwaller` so that bracket atoms (``[nH+]``,
    ``[13C]``, …) are indivisible tokens and every bond/branch/stereo symbol
    is its own token.  This gives a chemically complete atom-level vocabulary
    consistent with the Molecular Transformer standard.

    The ``preprocess`` argument is accepted for API compatibility but ignored.

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings to compare.
    corpus : List[str]
        Corpus used to fit IDF weights.  Defaults to [smiles1, smiles2].
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
        raise ImportError("sklearn is required for schwaller_tfidf_similarity")

    if vectorizer is None:
        if corpus is None:
            corpus = [smiles1, smiles2]
        tokenizer = SMILESTokenizerSchwaller()
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            analyzer="word",
            lowercase=False,
            token_pattern=None,
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True,
        )
        try:
            vectorizer.fit(corpus)
        except ValueError:
            return 0.0

    vec1 = vectorizer.transform([smiles1])
    vec2 = vectorizer.transform([smiles2])
    return float(sklearn_cosine_similarity(vec1, vec2)[0, 0])


# ============================================================================
# 10b. BPE TF-IDF Cosine Similarity
# ============================================================================


# Default BPE vocabulary: smiles_bpe_vocab.json next to this module file.
_DEFAULT_BPE_VOCAB = Path(__file__).parent / "smiles_bpe_vocab.json"


class SMILESTokenizerBPE:
    """
    Data-driven BPE tokenizer for use with sklearn TF-IDF.

    Applies the merge table produced by ``train_bpe_tokenizer.py``: starts
    from the Schwaller atom-level tokens, then greedily applies learned BPE
    merges in order.  The result is a variable-granularity vocabulary where
    common fragments (``C(=O)N``, ``c1ccccc1``, …) are single tokens.

    Parameters
    ----------
    vocab_path : str or Path or None
        Path to the JSON vocabulary file written by ``train_bpe_tokenizer.py``.
        Must contain a ``"merges"`` list of ``[a, b]`` pairs.
        Defaults to ``smiles_bpe_vocab.json`` in the same directory as this
        module.  Raises ``FileNotFoundError`` if neither the default file nor
        an explicit path can be found.
    """

    # Schwaller base regex (same as SMILESTokenizerSchwaller)
    _BASE_RE = re.compile(
        r"\[[^\]]+\]"
        r"|Br?"
        r"|Cl?"
        r"|N|O|S|P|F|I"
        r"|b|c|n|o|s|p"
        r"|\(|\)"
        r"|\.|\=|\#"
        r"|-|\+"
        r"|\\|/"
        r"|:|~|@"
        r"|\?|>|\*|\$"
        r"|\%[0-9]{2}"
        r"|[0-9]"
        r"|."
    )

    def __init__(self, vocab_path=None, num_merges=None):
        path = Path(vocab_path) if vocab_path is not None else _DEFAULT_BPE_VOCAB
        if not path.exists():
            raise FileNotFoundError(
                f"BPE vocabulary file not found: {path}\n"
                "Train one with train_bpe_tokenizer.py and place it next to "
                "smiles_similarity_kernels.py, or pass vocab_path= explicitly."
            )
        data = json.loads(path.read_text())
        all_merges: list[tuple[str, str]] = [tuple(pair) for pair in data.get("merges", [])]
        self._merges = all_merges[:num_merges] if num_merges is not None else all_merges

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string using BPE merges."""
        tokens = self._BASE_RE.findall(smiles)
        for a, b in self._merges:
            merged = a + b
            out = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    def __call__(self, smiles: str) -> List[str]:
        return self.tokenize(smiles)


def bpe_tfidf_similarity(
    smiles1: str,
    smiles2: str,
    corpus: List[str] = None,
    ngram_range: Tuple[int, int] = (1, 2),
    vectorizer=None,
    vocab_path=None,
    num_merges=None,
    preprocess: bool = False,
) -> float:
    """
    TF-IDF cosine similarity using BPE tokenization trained on ChEMBL.

    Uses :class:`SMILESTokenizerBPE` which applies the learned BPE merge
    table so that frequent fragments (``C(=O)N``, ``c1ccccc1``, …) become
    single tokens.  The vocabulary JSON is produced by ``train_bpe_tokenizer.py``.

    The ``preprocess`` argument is accepted for API compatibility but ignored.

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings to compare.
    corpus : List[str]
        Corpus used to fit IDF weights.  Defaults to [smiles1, smiles2].
    ngram_range : Tuple[int, int]
        N-gram range passed to TfidfVectorizer (default (1, 2)).
    vectorizer : fitted TfidfVectorizer or None
        Pre-fitted vectorizer for efficiency in batch use.
    vocab_path : str or Path or None
        Path to BPE vocabulary JSON.
    num_merges : int or None
        Use only the first ``num_merges`` merges from the vocabulary file.
        ``None`` (default) uses all merges.  Allows exploring different
        vocabulary granularities from a single large JSON file.

    Returns
    -------
    float
        Cosine similarity in [0, 1]
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for bpe_tfidf_similarity")

    if vectorizer is None:
        if corpus is None:
            corpus = [smiles1, smiles2]
        tokenizer = SMILESTokenizerBPE(vocab_path=vocab_path, num_merges=num_merges)
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            analyzer="word",
            lowercase=False,
            token_pattern=None,
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True,
        )
        try:
            vectorizer.fit(corpus)
        except ValueError:
            return 0.0

    vec1 = vectorizer.transform([smiles1])
    vec2 = vectorizer.transform([smiles2])
    return float(sklearn_cosine_similarity(vec1, vec2)[0, 0])


# ============================================================================
# 10c. SELFIES TF-IDF Cosine Similarity
# ============================================================================


class SELFIESTokenizer:
    """
    SELFIES-aware tokenizer for use with sklearn TF-IDF.

    Splits a SELFIES string on its natural token boundaries: each ``[...]``
    bracket group is one indivisible token.  Characters outside brackets
    (which should not appear in valid SELFIES) are returned as individual
    single-character tokens so that malformed input does not silently lose
    information.
    """

    _TOKEN_RE = re.compile(r"\[[^\[\]]*\]|.")

    def tokenize(self, selfies: str) -> List[str]:
        """Split a SELFIES string into its constituent tokens."""
        return self._TOKEN_RE.findall(selfies)

    def __call__(self, selfies: str) -> List[str]:
        return self.tokenize(selfies)


def selfies_tfidf_similarity(
    selfies1: str,
    selfies2: str,
    corpus: List[str] = None,
    ngram_range: Tuple[int, int] = (1, 2),
    vectorizer=None,
    preprocess: bool = False,
) -> float:
    """
    TF-IDF cosine similarity with SELFIES-aware tokenization.

    Uses :class:`SELFIESTokenizer` so that each ``[token]`` in the SELFIES
    string is treated as an indivisible unit.  Intended to be called with
    pre-converted SELFIES strings (use :func:`smiles_to_selfies` first, or
    pass ``--selfies`` on the CLI).

    The ``preprocess`` argument is accepted for API compatibility but ignored:
    SELFIES tokens are already semantically atomic and do not benefit from
    SMILES-style character substitution.

    Requires ``scikit-learn`` and ``selfies``.

    Parameters
    ----------
    selfies1, selfies2 : str
        SELFIES strings to compare
    corpus : List[str]
        Corpus used to fit IDF weights.  Defaults to [selfies1, selfies2].
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
        raise ImportError("sklearn is required for selfies_tfidf_similarity")

    if vectorizer is None:
        if corpus is None:
            corpus = [selfies1, selfies2]
        tokenizer = SELFIESTokenizer()
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            analyzer="word",
            lowercase=False,
            token_pattern=None,
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True,
        )
        try:
            vectorizer.fit(corpus)
        except ValueError:
            return 0.0

    vec1 = vectorizer.transform([selfies1])
    vec2 = vectorizer.transform([selfies2])
    return float(sklearn_cosine_similarity(vec1, vec2)[0, 0])


# ============================================================================
# 11. Jellyfish-based string similarity metrics
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
# Fingerprint Functions
# ============================================================================
#
# Each fingerprint function has the signature:
#   fp_func(smiles: str, **kwargs) -> np.ndarray
#
# The returned array is always 1-D and float64.  Binary fingerprints use
# 0.0 / 1.0 values; count fingerprints use non-negative integer counts
# stored as float64 for uniform downstream handling.
#
# All fingerprints are deterministic and corpus-free: they can be computed
# for a single molecule without fitting on a dataset.


def smifp_fingerprint(
    smiles: str,
    chars: List[str] = None,
    binary: bool = False,
    preprocess: bool = True,
) -> np.ndarray:
    """
    SMIfp character-frequency fingerprint (34D or 38D).

    Wraps :func:`smiles_to_fingerprint` as a standalone fingerprint function
    compatible with :data:`AVAILABLE_FINGERPRINTS`.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    chars : List[str] or None
        Character set to count.  Defaults to :data:`SMIFP_CHARS_34` (34D).
        Pass :data:`SMIFP_CHARS_38` for the extended 38D variant.
    binary : bool
        If True, binarise the count vector (count > 0 → 1).
    preprocess : bool
        Replace multi-character atoms before counting.

    Returns
    -------
    np.ndarray
        1-D float64 array of length ``len(chars)``.
    """
    if chars is None:
        chars = SMIFP_CHARS_34
    if preprocess:
        smiles = preprocess_smiles(smiles)
    fp = smiles_to_fingerprint(smiles, chars)
    if binary:
        fp = (fp > 0).astype(float)
    return fp


def bpe_pattern_fingerprint(
    smiles: str,
    vocab_path=None,
    num_merges: Optional[int] = None,
    binary: bool = False,
) -> np.ndarray:
    """
    BPE-pattern count fingerprint.

    Uses the merge table from a BPE vocabulary JSON (produced by
    ``train_bpe_tokenizer.py``) as a fixed pattern dictionary.  Each
    dimension corresponds to one *merged* token (base single-character
    tokens are excluded — they are nearly always present and are already
    captured by SMIfp).  The value is how many times that merged token
    appears in the Schwaller-tokenized SMILES after all BPE merges up to
    ``num_merges`` have been applied.

    The fingerprint is:

    - **deterministic** — no corpus required at inference time
    - **fixed-length** — always ``num_merges`` (or total merges if None) dimensions
    - **corpus-free** — patterns were learned from ChEMBL but applied to any SMILES
    - **complementary to SMIfp** — focuses on multi-atom fragments, not raw characters

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    vocab_path : str or Path or None
        Path to the BPE vocabulary JSON file.  Defaults to
        ``smiles_bpe_vocab.json`` in the same directory as this module.
    num_merges : int or None
        Use only the first ``num_merges`` merges (and thus only that many
        dimensions).  ``None`` uses all merges in the file.
    binary : bool
        If True, binarise the count vector (count > 0 → 1).

    Returns
    -------
    np.ndarray
        1-D float64 array of length ``num_merges`` (or total merges).

    Notes
    -----
    The BPE tokenizer applies merges greedily in order.  A merged token
    ``"C(=O)N"`` only appears if the full fragment is present contiguously
    in the token stream after all prior merges have been applied.  This
    means rare merged tokens at the end of the merge list are only set for
    molecules that contain the exact corresponding substructure.
    """
    path = Path(vocab_path) if vocab_path is not None else _DEFAULT_BPE_VOCAB
    if not path.exists():
        raise FileNotFoundError(
            f"BPE vocabulary file not found: {path}\n"
            "Train one with train_bpe_tokenizer.py or pass vocab_path= explicitly."
        )
    data = json.loads(path.read_text())
    all_merges: list = [tuple(pair) for pair in data.get("merges", [])]
    merges = all_merges[:num_merges] if num_merges is not None else all_merges

    # Build the merged token vocabulary (one dimension per merge).
    merged_tokens = [a + b for a, b in merges]

    # Tokenize using BPE (same logic as SMILESTokenizerBPE.tokenize).
    _base_re = SMILESTokenizerBPE._BASE_RE
    tokens = _base_re.findall(smiles)
    for a, b in merges:
        ab = a + b
        out = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                out.append(ab)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out

    # Count occurrences of each merged token.
    token_counts = Counter(tokens)
    fp = np.array([float(token_counts.get(tok, 0)) for tok in merged_tokens], dtype=float)
    if binary:
        fp = (fp > 0).astype(float)
    return fp


# ---------------------------------------------------------------------------
# Fingerprint registry
# ---------------------------------------------------------------------------
#
# Each entry:
#   "function"    : (smiles, **kwargs) -> np.ndarray
#   "description" : str
#   "length"      : int or None (None = depends on vocab / num_merges)
#   "params"      : dict of fixed kwargs forwarded to the function
#   "requires"    : optional str, dependency flag name (same as AVAILABLE_METHODS)

AVAILABLE_FINGERPRINTS: Dict[str, dict] = {
    # ── SMIfp ────────────────────────────────────────────────────────────────
    "smifp34": {
        "function": smifp_fingerprint,
        "description": "SMIfp 34D character-frequency fingerprint (count)",
        "length": 34,
        "params": {"chars": SMIFP_CHARS_34, "binary": False},
    },
    "smifp34_binary": {
        "function": lambda smi, **kw: smifp_fingerprint(smi, chars=SMIFP_CHARS_34, binary=True, **kw),
        "description": "SMIfp 34D binary fingerprint (presence/absence)",
        "length": 34,
        "params": {},
    },
    "smifp38": {
        "function": lambda smi, **kw: smifp_fingerprint(smi, chars=SMIFP_CHARS_38, binary=False, **kw),
        "description": "SMIfp extended character-frequency fingerprint (count); 34D - '%' + '/', '\\\\', '@@'",
        "length": len(SMIFP_CHARS_38),
        "params": {},
    },
    "smifp38_binary": {
        "function": lambda smi, **kw: smifp_fingerprint(smi, chars=SMIFP_CHARS_38, binary=True, **kw),
        "description": "SMIfp extended binary fingerprint (presence/absence); 34D - '%' + '/', '\\\\', '@@'",
        "length": len(SMIFP_CHARS_38),
        "params": {},
    },
    # ── BPE pattern fingerprints ─────────────────────────────────────────────
    "bpe_count": {
        "function": bpe_pattern_fingerprint,
        "description": "BPE-pattern count fingerprint (all merges, count)",
        "length": None,
        "params": {"binary": False},
    },
    "bpe_binary": {
        "function": lambda smi, **kw: bpe_pattern_fingerprint(smi, binary=True, **kw),
        "description": "BPE-pattern binary fingerprint (all merges, presence/absence)",
        "length": None,
        "params": {},
    },
    # Fixed-merge-count BPE variants
    **{
        f"bpe{k}_count": {
            "function": (lambda _k: lambda smi, **kw: bpe_pattern_fingerprint(smi, num_merges=_k, binary=False, **kw))(k),
            "description": f"BPE-pattern count fingerprint ({k} merges)",
            "length": k,
            "params": {"num_merges": k},
        }
        for k in (16, 32, 64, 128, 256, 512, 1024)
    },
    **{
        f"bpe{k}_binary": {
            "function": (lambda _k: lambda smi, **kw: bpe_pattern_fingerprint(smi, num_merges=_k, binary=True, **kw))(k),
            "description": f"BPE-pattern binary fingerprint ({k} merges)",
            "length": k,
            "params": {"num_merges": k},
        }
        for k in (16, 32, 64, 128, 256, 512, 1024)
    },
}


def get_fingerprint_function(fp_type: str):
    """Return the fingerprint function for *fp_type*, checking availability."""
    if fp_type not in AVAILABLE_FINGERPRINTS:
        raise ValueError(
            f"Unknown fingerprint type: '{fp_type}'. "
            f"Available: {list(AVAILABLE_FINGERPRINTS.keys())}"
        )
    entry = AVAILABLE_FINGERPRINTS[fp_type]
    req = entry.get("requires")
    if req == "sklearn" and not SKLEARN_AVAILABLE:
        raise ImportError(f"Fingerprint '{fp_type}' requires scikit-learn")
    return entry["function"]


def compute_fingerprint_matrix(
    smiles_list: List[str],
    fp_type: str = "bpe_count",
    names: List[str] = None,
    **kwargs,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute fingerprints for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : List[str]
        Input SMILES strings.
    fp_type : str
        Fingerprint type key from :data:`AVAILABLE_FINGERPRINTS`.
    names : List[str] or None
        Molecule names (used only for the returned list; not required).
    **kwargs
        Extra kwargs forwarded to the fingerprint function (e.g.
        ``vocab_path``, ``num_merges``).

    Returns
    -------
    matrix : np.ndarray
        Shape ``(n_molecules, n_bits)``.
    feature_names : List[str]
        Feature labels (``"bit_0"``, ``"bit_1"``, …) or BPE token strings
        when applicable.
    """
    fp_func = get_fingerprint_function(fp_type)
    fps = [fp_func(smi, **kwargs) for smi in smiles_list]
    matrix = np.vstack(fps)
    feature_names = [f"bit_{i}" for i in range(matrix.shape[1])]
    return matrix, feature_names


def write_fingerprint_csv(
    output_path: str,
    molecule_names: List[str],
    matrix: np.ndarray,
    feature_names: List[str],
    fp_type: str,
) -> None:
    """Write fingerprint matrix to CSV (rows = molecules, columns = bits)."""
    cols = {"Name": molecule_names}
    for i, fname in enumerate(feature_names):
        cols[fname] = matrix[:, i]
    df = pd.DataFrame(cols)
    df.to_csv(output_path, index=False, float_format="%.0f")


# ============================================================================
# Available Methods Registry
# ============================================================================

AVAILABLE_METHODS = {
    "edit": {"function": edit_similarity, "description": "Edit distance similarity", "params": {}},
    "nlcs": {"function": nlcs_similarity, "description": "Normalized Longest Common Subsequence", "params": {}},
    "clcs": {"function": clcs_similarity, "description": "Combined LCS models", "params": {}},
    "substring": {
        "function": lambda s1, s2, **kw: substring_kernel_similarity(s1, s2, normalized=True, **kw),
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
        "function": lambda s1, s2, **kw: smifp_similarity_cityblock(s1, s2, chars=SMIFP_CHARS_38, **kw),
        "description": "SMILES fingerprint 38D with City Block Distance (Manhattan)",
        "params": {},
        "requires": "scipy",
    },
    "smifp38_tanimoto": {
        "function": lambda s1, s2, **kw: smifp_similarity_tanimoto(s1, s2, chars=SMIFP_CHARS_38, **kw),
        "description": "SMILES fingerprint 38D with Tanimoto",
        "params": {},
    },
    "lingo": {"function": lingo_similarity, "description": "LINGO similarity (q=4)", "params": {"q": 4}},
    "lingo3": {
        "function": lambda s1, s2, **kw: lingo_similarity(s1, s2, q=3, **kw),
        "description": "LINGO similarity (q=3)",
        "params": {"q": 3},
    },
    "lingo5": {
        "function": lambda s1, s2, **kw: lingo_similarity(s1, s2, q=5, **kw),
        "description": "LINGO similarity (q=5)",
        "params": {"q": 5},
    },
    "lingo_tversky": {
        "function": lingo_tversky_similarity,
        "description": "Asymmetric Tversky on LINGO q-grams (q=4, alpha=0.9, beta=0.1) — query-weighted",
        "params": {"q": 4, "alpha": 0.9, "beta": 0.1},
    },
    "lingo_tversky_sym": {
        "function": lambda s1, s2, **kw: lingo_tversky_similarity(s1, s2, q=4, alpha=0.5, beta=0.5, **kw),
        "description": "Symmetric Tversky (alpha=beta=0.5, equivalent to Dice) on LINGO q-grams",
        "params": {"q": 4, "alpha": 0.5, "beta": 0.5},
    },
    "lingo_dice": {
        "function": lingo_dice_similarity,
        "description": "Sørensen-Dice coefficient on LINGO q-gram counts (q=4)",
        "params": {"q": 4},
    },
    "spectrum": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(s1, s2, k=4, coefficient="tanimoto", **kw),
        "description": "Spectrum kernel (k=4, Tanimoto) — classical fixed-k string kernel",
        "params": {"k": 4, "coefficient": "tanimoto"},
    },
    "spectrum3": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(s1, s2, k=3, coefficient="tanimoto", **kw),
        "description": "Spectrum kernel (k=3, Tanimoto)",
        "params": {"k": 3, "coefficient": "tanimoto"},
    },
    "spectrum5": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(s1, s2, k=5, coefficient="tanimoto", **kw),
        "description": "Spectrum kernel (k=5, Tanimoto)",
        "params": {"k": 5, "coefficient": "tanimoto"},
    },
    "spectrum_cosine": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(s1, s2, k=4, coefficient="cosine", **kw),
        "description": "Spectrum kernel (k=4, cosine normalisation)",
        "params": {"k": 4, "coefficient": "cosine"},
    },
    "mismatch": {
        "function": lambda s1, s2, **kw: mismatch_kernel_similarity(s1, s2, k=4, m=1, coefficient="tanimoto", **kw),
        "description": "Mismatch (spectrum-(k,m)) kernel (k=4, m=1, Tanimoto) — tolerates 1 atom swap",
        "params": {"k": 4, "m": 1, "coefficient": "tanimoto"},
    },
    "mismatch3": {
        "function": lambda s1, s2, **kw: mismatch_kernel_similarity(s1, s2, k=3, m=1, coefficient="tanimoto", **kw),
        "description": "Mismatch kernel (k=3, m=1, Tanimoto)",
        "params": {"k": 3, "m": 1, "coefficient": "tanimoto"},
    },
    "mismatch5": {
        "function": lambda s1, s2, **kw: mismatch_kernel_similarity(s1, s2, k=5, m=1, coefficient="tanimoto", **kw),
        "description": "Mismatch kernel (k=5, m=1, Tanimoto)",
        "params": {"k": 5, "m": 1, "coefficient": "tanimoto"},
    },
    "lcs_substring": {
        "function": longest_common_substring_similarity,
        "description": "Normalised Longest Common Substring (contiguous) — LCSubstr²/(len1×len2)",
        "params": {},
    },
    **{
        f"tok-smiles_tfidf{m}{n}": {
            "function": (lambda _m, _n: lambda s1, s2, **kw: smiles_tfidf_similarity(s1, s2, ngram_range=(_m, _n), **kw))(m, n),
            "description": f"TF-IDF cosine similarity with chemical tokenization (ngram ({m},{n}))",
            "params": {"ngram_range": (m, n)},
            "requires": "sklearn",
        }
        for m in range(1, 7)
        for n in range(m, 7)
    },
    "tok-smiles_tfidf": {
        "function": smiles_tfidf_similarity,
        "description": "TF-IDF cosine similarity with chemical tokenization (ngram (1,2))",
        "params": {"ngram_range": (1, 2)},
        "requires": "sklearn",
    },
    **{
        f"tok-schwaller_tfidf{m}{n}": {
            "function": (lambda _m, _n: lambda s1, s2, **kw: schwaller_tfidf_similarity(s1, s2, ngram_range=(_m, _n), **kw))(m, n),
            "description": f"TF-IDF cosine similarity with Schwaller atom-level tokenization (ngram ({m},{n}))",
            "params": {"ngram_range": (m, n)},
            "requires": "sklearn",
        }
        for m in range(1, 7)
        for n in range(m, 7)
    },
    "tok-schwaller_tfidf": {
        "function": schwaller_tfidf_similarity,
        "description": "TF-IDF cosine similarity with Schwaller atom-level tokenization (ngram (1,2))",
        "params": {"ngram_range": (1, 2)},
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
    **{
        f"tok-selfies_tfidf{m}{n}": {
            "function": (lambda _m, _n: lambda s1, s2, **kw: selfies_tfidf_similarity(s1, s2, ngram_range=(_m, _n), **kw))(m, n),
            "description": f"TF-IDF cosine similarity on SELFIES tokens (ngram ({m},{n}))",
            "params": {"ngram_range": (m, n)},
            "requires": "sklearn",
        }
        for m in range(1, 7)
        for n in range(m, 7)
    },
    "tok-selfies_tfidf": {
        "function": selfies_tfidf_similarity,
        "description": "TF-IDF cosine similarity on SELFIES tokens (ngram (1,2))",
        "params": {"ngram_range": (1, 2)},
        "requires": "sklearn",
    },
    **{
        f"tok-bpe_tfidf{m}{n}": {
            "function": (lambda _m, _n: lambda s1, s2, **kw: bpe_tfidf_similarity(s1, s2, ngram_range=(_m, _n), **kw))(m, n),
            "description": f"TF-IDF cosine similarity with BPE tokenization trained on ChEMBL (ngram ({m},{n}))",
            "params": {"ngram_range": (m, n)},
            "requires": "sklearn",
        }
        for m in range(1, 7)
        for n in range(m, 7)
    },
    "tok-bpe_tfidf": {
        "function": bpe_tfidf_similarity,
        "description": "TF-IDF cosine similarity with BPE tokenization trained on ChEMBL (ngram (1,2))",
        "params": {"ngram_range": (1, 2)},
        "requires": "sklearn",
    },
    # Fixed-merge-count BPE families: tok-bpe{k}_tfidf{m}{n}
    # Each uses only the first k merges from the vocabulary file, allowing
    # comparison of tokenization granularities from a single large JSON.
    **{
        f"tok-bpe{_k}_tfidf{m}{n}": {
            "function": (lambda _k, _m, _n: lambda s1, s2, **kw: bpe_tfidf_similarity(s1, s2, ngram_range=(_m, _n), num_merges=_k, **kw))(
                _k, m, n
            ),
            "description": f"TF-IDF cosine similarity with BPE tokenization ({_k} merges, ngram ({m},{n}))",
            "params": {"ngram_range": (m, n), "num_merges": _k},
            "requires": "sklearn",
        }
        for _k in (16, 32, 64, 256, 512, 1024)
        for m in range(1, 7)
        for n in range(m, 7)
    },
    **{
        f"tok-bpe{_k}_tfidf": {
            "function": (lambda _k: lambda s1, s2, **kw: bpe_tfidf_similarity(s1, s2, ngram_range=(1, 2), num_merges=_k, **kw))(_k),
            "description": f"TF-IDF cosine similarity with BPE tokenization ({_k} merges, ngram (1,2))",
            "params": {"ngram_range": (1, 2), "num_merges": _k},
            "requires": "sklearn",
        }
        for _k in (16, 32, 64, 256, 512, 1024)
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
        if req == "selfies" and not SELFIES_AVAILABLE:
            raise ImportError(f"Method '{method}' requires selfies (pip install selfies)")

    return method_info["function"]


# ============================================================================
# Batch Processing & Similarity Matrix Generation
# ============================================================================


def _build_batch_kwargs(sim_func, method: str, corpus: List[str], kwargs: dict) -> dict:
    """
    Prepare kwargs for batch similarity calls:
    - filter to parameters the function actually accepts
    - preprocess the corpus once and set preprocess=False
    - for TF-IDF methods, fit a single vectorizer on the full corpus
    """
    import inspect

    try:
        params = inspect.signature(sim_func).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        filtered = kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        filtered = {}
        params = {}

    # Preprocess each string once rather than once per pair.
    if filtered.get("preprocess", True) and "preprocess" in params:
        corpus[:] = [preprocess_smiles(s) for s in corpus]
        filtered = {**filtered, "preprocess": False}

    # For TF-IDF methods, fit one vectorizer on the full corpus so IDF weights
    # reflect the whole dataset rather than each individual pair.
    if "tfidf" in method and "vectorizer" not in filtered:
        _tfidf_funcs = {
            "smiles": smiles_tfidf_similarity,
            "schwaller": schwaller_tfidf_similarity,
            "bpe": bpe_tfidf_similarity,
            "selfies": selfies_tfidf_similarity,
            "lingo": lingo_tfidf_similarity,
        }
        underlying = next((fn for key, fn in _tfidf_funcs.items() if key in method), None)
        if underlying is not None and SKLEARN_AVAILABLE:
            extra = {k: v for k, v in filtered.items() if k not in ("vectorizer", "corpus", "preprocess")}
            try:
                # Fit by passing the full corpus; discard the returned score.
                underlying(corpus[0], corpus[0], corpus=corpus, **extra)
                # The fitted vectorizer lives inside the closure — we need it directly.
                # Build it the same way the similarity function does.
                from sklearn.feature_extraction.text import TfidfVectorizer as _TV

                tok_map = {
                    "smiles": SMILESTokenizer,
                    "schwaller": SMILESTokenizerSchwaller,
                    "selfies": SELFIESTokenizer,
                }
                if "bpe" in method:
                    num_merges = filtered.get("num_merges", None)
                    tokenizer = SMILESTokenizerBPE(num_merges=num_merges)
                elif "lingo" in method:
                    tokenizer = None  # LingoVectorizer has its own fit path
                else:
                    tok_cls = next((cls for key, cls in tok_map.items() if key in method), None)
                    tokenizer = tok_cls() if tok_cls else None

                if "lingo" in method:
                    q = filtered.get("q", 4)
                    vec = LingoVectorizer(q=q, use_idf=True)
                    vec.fit(corpus)
                elif tokenizer is not None:
                    ngram_range = filtered.get("ngram_range", (1, 2))
                    vec = _TV(
                        tokenizer=tokenizer,
                        analyzer="word",
                        lowercase=False,
                        token_pattern=None,
                        ngram_range=ngram_range,
                        min_df=1,
                        sublinear_tf=True,
                    )
                    vec.fit(corpus)
                else:
                    vec = None
                if vec is not None:
                    filtered = {**filtered, "vectorizer": vec}
            except Exception:
                pass  # Fall back to per-pair fitting if anything goes wrong.

    return filtered, corpus


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
        Additional arguments for the similarity function.  ``preprocess``
        is passed through only to functions whose signature accepts it;
        unknown kwargs are silently ignored.  Set ``preprocess=False``
        when the inputs have already been transformed (e.g. to InChI)
        so that SMILES-oriented character substitution does not corrupt
        them.

    Returns
    -------
    np.ndarray
        n x n similarity matrix
    """
    n = len(smiles_list)
    sim_matrix = np.zeros((n, n))

    sim_func = get_similarity_function(method)
    smiles_list = list(smiles_list)
    filtered_kwargs, smiles_list = _build_batch_kwargs(sim_func, method, smiles_list, kwargs)

    for i in range(n):
        sim_matrix[i, i] = 1.0  # Self-similarity
        for j in range(i + 1, n):
            sim = sim_func(smiles_list[i], smiles_list[j], **filtered_kwargs)
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
        Additional arguments for the similarity function.  ``preprocess``
        is passed through only to functions whose signature accepts it;
        unknown kwargs are silently ignored.  Set ``preprocess=False``
        when the inputs have already been transformed (e.g. to InChI)
        so that SMILES-oriented character substitution does not corrupt
        them.

    Returns
    -------
    np.ndarray
        len(library) x len(templates) similarity matrix
    """
    n_lib = len(library)
    n_templates = len(templates)
    sim_matrix = np.zeros((n_lib, n_templates))

    sim_func = get_similarity_function(method)
    corpus = list(templates) + list(library)
    filtered_kwargs, corpus = _build_batch_kwargs(sim_func, method, corpus, kwargs)
    templates = corpus[:n_templates]
    library = corpus[n_templates:]

    for i, lib_smiles in enumerate(library):
        for j, template_smiles in enumerate(templates):
            sim = sim_func(lib_smiles, template_smiles, **filtered_kwargs)
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
  python smiles_similarity_kernels.py --templates templates/ --database library/ --output output.csv --method lingo

  # Calculate all methods (creates multiple output files)
  python smiles_similarity_kernels.py --templates templates/ --database library/ --output output.csv --all-methods

  # Use files instead of directories
  python smiles_similarity_kernels.py --templates templates.smi --database library.smi --output output.csv --method lingo

  # Use CSV files with column specification
  python smiles_similarity_kernels.py --templates templates.csv --database database.csv --output output.csv \\
      --method lingo --templates-smiles-col SMILES --templates-name-col ID \\
      --database-smiles-col SMILES --database-name-col MolID

  # Mix directory and file inputs
  python smiles_similarity_kernels.py --templates templates/ --database library.smi --output output.csv --method edit

  # Use edit distance similarity
  python smiles_similarity_kernels.py --templates templates/ --database library/ --output output.csv --method edit

  # List available methods
  python smiles_similarity_kernels.py --list-methods

  # Convert to SELFIES before comparison (requires selfies); disable SMILES preprocessing
  python smiles_similarity_kernels.py --templates templates.smi --database library.smi --output output.csv --method edit --selfies --no-preprocess

  # Use SELFIES-aware TF-IDF similarity
  python smiles_similarity_kernels.py --templates templates.smi --database library.smi --output output.csv --method selfies_tfidf --selfies --no-preprocess

  # Convert to InChI; disable SMILES preprocessing so InChI strings are not corrupted
  python smiles_similarity_kernels.py --templates templates.smi --database library.smi --output output.csv --method lingo --inchi --no-preprocess

  # Shuffle characters after conversion (negative control — destroys chemistry)
  python smiles_similarity_kernels.py --templates templates.smi --database library.smi --output output.csv --method lingo --shuffle

  # Reproducible shuffle with fixed seed
  python smiles_similarity_kernels.py --templates templates.smi --database library.smi --output output.csv --method lingo --shuffle --shuffle-seed 42

  # Benchmarking: compare raw SMILES strings without any normalization
  python smiles_similarity_kernels.py --templates templates.smi --database library.smi --output output.csv --method lingo --no-preprocess

  # Run demo with example molecules
  python smiles_similarity_kernels.py --demo

Input formats:
  - Directory: Reads all .smi files from the directory
  - .smi/.smiles file: Space/tab-separated, no header (SMILES Name)
  - .csv file: Comma-separated with header
  - .tsv file: Tab-separated with header

Available methods: edit, nlcs, clcs, substring, smifp_cbd, smifp_tanimoto,
                   smifp38_cbd, smifp38_tanimoto, lingo, lingo3, lingo5,
                   lingo_tversky, lingo_tversky_sym, lingo_dice,
                   spectrum, spectrum3, spectrum5, spectrum_cosine,
                   mismatch, mismatch3, mismatch5, lcs_substring,
                   tok-smiles_tfidf, tok-smiles_tfidf{m}{n} (m=1..6, n=m..6, e.g. tok-smiles_tfidf44),
                   tok-schwaller_tfidf, tok-schwaller_tfidf{m}{n} (m=1..6, n=m..6, e.g. tok-schwaller_tfidf44),
                   tok-bpe_tfidf, tok-bpe_tfidf{m}{n} (m=1..6, n=m..6, e.g. tok-bpe_tfidf44),
                   tok-selfies_tfidf, tok-selfies_tfidf{m}{n} (m=1..6, n=m..6, e.g. tok-selfies_tfidf44),
                   damerau_levenshtein, jaro, jaro_winkler, hamming, ncd
        """,
    )

    parser.add_argument("--templates", "-t", type=str, default=None, help="Directory or file containing template molecules (.smi, .csv, .tsv)")
    parser.add_argument(
        "--database", "-d", type=str, default=None, help="Directory or file containing database/library molecules (.smi, .csv, .tsv)"
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV file path")

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

    # ── CONVERT ────────────────────────────────────────────────────────────────
    convert_group = parser.add_argument_group(
        "Convert (stage 2)",
        "Select the string representation used for similarity. "
        "Input is always read as SMILES; one of these flags converts it before comparison. "
        "Default: keep as SMILES.",
    )
    convert_ex = convert_group.add_mutually_exclusive_group()
    convert_ex.add_argument(
        "--inchi",
        action="store_true",
        help="Convert SMILES → InChI (requires rdkit). " "Strips the 'InChI=1S/' prefix; use --inchi-layer to select a subset of layers.",
    )
    convert_ex.add_argument(
        "--selfies",
        action="store_true",
        help="Convert SMILES → SELFIES (requires selfies). " "All string-similarity methods apply directly to SELFIES bracket tokens.",
    )

    convert_group.add_argument(
        "--inchi-layer",
        type=str,
        default="all",
        metavar="LAYER[,LAYER,...]",
        help="When --inchi is used, restrict to selected InChI layer(s). "
        "Comma-separated. Supported layers: formula, connections, hydrogens, "
        "charge, protons, stereo_db, stereo_tet, stereo_parity, stereo_type, "
        "isotope, fixedH, reconnected. Default: 'all' (full InChI minus version tag).",
    )

    # ── NORMALIZE ──────────────────────────────────────────────────────────────
    norm_group = parser.add_argument_group(
        "Normalize (stage 3)",
        "Normalization applied after conversion. "
        "ELEMENT_REPLACEMENTS substitution (preprocess) is on by default for SMILES "
        "and automatically disabled for InChI and SELFIES.",
    )
    norm_group.add_argument(
        "--canonicalize",
        action="store_true",
        help="Canonicalize SMILES with RDKit before comparison (requires rdkit, SMILES only). "
        'Ensures "CCO" and "OCC" are treated as the same molecule.',
    )
    norm_group.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable SMILES ELEMENT_REPLACEMENTS character substitution (preprocess=False). "
        "Only relevant when string type is SMILES; ignored otherwise. "
        "Useful for benchmarking raw SMILES strings without normalization.",
    )

    # ── AUGMENT ────────────────────────────────────────────────────────────────
    aug_group = parser.add_argument_group(
        "Augment (stage 4)",
        "Applied after normalization, to the final string representation.",
    )
    aug_group.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle characters in each string (type-agnostic negative control). "
        "Destroys chemical meaning while preserving length and character composition.",
    )
    aug_group.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for --shuffle (default: None = non-reproducible).",
    )
    aug_group.add_argument(
        "--sort",
        action="store_true",
        help="Sort characters of each string alphabetically (deterministic negative control). "
        "Destroys chemical meaning while preserving length and character composition.",
    )

    # ── FINGERPRINT ────────────────────────────────────────────────────────────
    fp_group = parser.add_argument_group(
        "Fingerprint mode",
        "Instead of computing pairwise similarities, compute a fixed-length "
        "fingerprint for each molecule in --database and write one row per "
        "molecule to --output.  --templates is not required in this mode.",
    )
    fp_group.add_argument(
        "--fingerprint",
        type=str,
        default=None,
        metavar="TYPE",
        choices=list(AVAILABLE_FINGERPRINTS.keys()),
        help=(
            "Compute fingerprints instead of similarities. "
            "TYPE is one of: " + ", ".join(AVAILABLE_FINGERPRINTS.keys())
        ),
    )
    fp_group.add_argument(
        "--list-fingerprints",
        action="store_true",
        help="List available fingerprint types and exit.",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress information")
    parser.add_argument("--timing-log", default=None, metavar="FILE", help="Append per-method timing rows (CSV) to FILE")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files. Without this flag, existing files are skipped with a warning."
    )

    parser.add_argument("--demo", action="store_true", help="Run a demonstration with example molecules and exit")

    return parser, parser.parse_args()


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
    parser, args = parse_args()

    # Run demo if requested
    if args.demo:
        demo()
        return

    # List methods if requested
    if args.list_methods:
        print("\nAvailable similarity methods:")
        print("-" * 60)
        for name, info in AVAILABLE_METHODS.items():
            req = f" (requires {info.get('requires', 'nothing')})" if "requires" in info else ""
            print(f"  {name:20s} - {info['description']}{req}")
        print()
        return

    # List fingerprint types if requested
    if args.list_fingerprints:
        print("\nAvailable fingerprint types:")
        print("-" * 60)
        for name, info in AVAILABLE_FINGERPRINTS.items():
            length = info.get("length")
            length_str = f"{length}D" if length is not None else "variable-length"
            req = f" (requires {info.get('requires', 'nothing')})" if "requires" in info else ""
            print(f"  {name:25s} [{length_str:>14s}] - {info['description']}{req}")
        print()
        return

    # ── FINGERPRINT MODE ──────────────────────────────────────────────────────
    if args.fingerprint is not None:
        if not args.database or not args.output:
            print("Error: --fingerprint requires --database and --output", file=sys.stderr)
            sys.exit(1)

        database_smiles_col = _parse_col_arg(args.database_smiles_col)
        database_name_col = _parse_col_arg(args.database_name_col)
        library = read_molecules_from_source(
            args.database,
            smiles_col=database_smiles_col,
            name_col=database_name_col,
            delimiter=args.database_delimiter,
            header=None if not args.database_no_header else False,
        )
        if not library:
            print(f"Error: No molecules found in database source: {args.database}", file=sys.stderr)
            sys.exit(1)

        lib_names = list(library.keys())
        lib_smiles = [library[n] for n in lib_names]

        # Apply the same convert / normalize / augment pipeline as similarity mode.
        string_type = "smiles"
        if args.inchi:
            if not RDKIT_AVAILABLE:
                print("Error: --inchi requires rdkit.", file=sys.stderr)
                sys.exit(1)
            layers_arg = [s.strip() for s in args.inchi_layer.split(",") if s.strip()]
            layers_for_convert = "all" if layers_arg == ["all"] else layers_arg
            lib_smiles = [smiles_to_inchi_layers(s, layers_for_convert) or s for s in lib_smiles]
            string_type = "inchi"
        elif args.selfies:
            if not SELFIES_AVAILABLE:
                print("Error: --selfies requires selfies.", file=sys.stderr)
                sys.exit(1)
            lib_smiles = [smiles_to_selfies(s) or s for s in lib_smiles]
            string_type = "selfies"

        if args.canonicalize:
            if string_type == "smiles":
                if not RDKIT_AVAILABLE:
                    print("Error: --canonicalize requires rdkit.", file=sys.stderr)
                    sys.exit(1)
                lib_smiles = [canonicalize_smiles(s) for s in lib_smiles]

        if args.shuffle:
            lib_smiles = [shuffle_smiles(s, seed=args.shuffle_seed) for s in lib_smiles]
        if args.sort:
            lib_smiles = [sort_string(s) for s in lib_smiles]

        if Path(args.output).exists() and not args.overwrite:
            print(f"[skip] {args.output}: file exists (use --overwrite to replace)", file=sys.stderr)
            sys.exit(0)

        if args.verbose:
            print(f"Computing {args.fingerprint} fingerprints for {len(lib_smiles):,} molecules …")

        try:
            fp_func = get_fingerprint_function(args.fingerprint)
            fps = [fp_func(smi) for smi in lib_smiles]
        except (ImportError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        matrix = np.vstack(fps)
        n_bits = matrix.shape[1]
        feature_names = [f"bit_{i}" for i in range(n_bits)]
        write_fingerprint_csv(args.output, lib_names, matrix, feature_names, args.fingerprint)

        if args.verbose:
            print(f"Wrote {n_bits}-bit fingerprints for {len(lib_names):,} molecules to {args.output}")
        return

    # Check required arguments
    if not args.templates or not args.database or not args.output:
        parser.print_help()
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
    template_strings = [templates[n] for n in template_names]
    library_names = list(library.keys())
    library_strings = [library[n] for n in library_names]

    # ── CONVERT (stage 2) ─────────────────────────────────────────────────────
    # Input is always SMILES. Convert to the requested representation.
    string_type = "smiles"  # tracks current type through the pipeline

    if args.inchi:
        if not RDKIT_AVAILABLE:
            print("Error: --inchi requires rdkit. Install with: pip install rdkit", file=sys.stderr)
            sys.exit(1)
        layers_arg = [s.strip() for s in args.inchi_layer.split(",") if s.strip()]
        if len(layers_arg) == 1 and layers_arg[0] == "all":
            layers_for_convert: Union[str, List[str]] = "all"
        else:
            for _l in layers_arg:
                if _l != "all" and _l not in INCHI_LAYERS:
                    print(f"Error: unknown InChI layer '{_l}'. Available: {list(INCHI_LAYERS.keys())}", file=sys.stderr)
                    sys.exit(1)
            layers_for_convert = layers_arg
        if args.verbose:
            layer_desc = "all layers" if layers_for_convert == "all" else f"layers: {layers_for_convert}"
            print(f"[convert] SMILES → InChI ({layer_desc})")
        template_strings = [smiles_to_inchi_layers(s, layers_for_convert) or s for s in template_strings]
        library_strings = [smiles_to_inchi_layers(s, layers_for_convert) or s for s in library_strings]
        string_type = "inchi"

    elif args.selfies:
        if not SELFIES_AVAILABLE:
            print("Error: --selfies requires selfies. Install with: pip install selfies", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print("[convert] SMILES → SELFIES")
        template_strings = [smiles_to_selfies(s) or s for s in template_strings]
        library_strings = [smiles_to_selfies(s) or s for s in library_strings]
        string_type = "selfies"

    # ── NORMALIZE (stage 3) ───────────────────────────────────────────────────
    # --canonicalize: SMILES-only, applied before ELEMENT_REPLACEMENTS
    if args.canonicalize:
        if string_type != "smiles":
            print(f"Warning: --canonicalize ignored for string type '{string_type}' (SMILES only)", file=sys.stderr)
        elif not RDKIT_AVAILABLE:
            print("Error: --canonicalize requires rdkit. Install with: pip install rdkit", file=sys.stderr)
            sys.exit(1)
        else:
            if args.verbose:
                print("[normalize] canonicalizing SMILES")
            template_strings = [canonicalize_smiles(s) for s in template_strings]
            library_strings = [canonicalize_smiles(s) for s in library_strings]

    # ELEMENT_REPLACEMENTS (preprocess): on by default for SMILES, always off for others
    if string_type == "smiles" and not args.no_preprocess:
        preprocess = True
    else:
        preprocess = False
    if args.verbose and string_type == "smiles":
        state = "on" if preprocess else "off (--no-preprocess)"
        print(f"[normalize] ELEMENT_REPLACEMENTS: {state}")

    # ── AUGMENT (stage 4) ─────────────────────────────────────────────────────
    if args.shuffle:
        if args.verbose:
            seed_msg = f"seed={args.shuffle_seed}" if args.shuffle_seed is not None else "no seed"
            print(f"[augment] shuffling strings ({seed_msg}) — negative control")
        template_strings = [shuffle_smiles(s, seed=args.shuffle_seed) for s in template_strings]
        library_strings = [shuffle_smiles(s, seed=args.shuffle_seed) for s in library_strings]

    if args.sort:
        if args.verbose:
            print("[augment] sorting strings alphabetically — deterministic negative control")
        template_strings = [sort_string(s) for s in template_strings]
        library_strings = [sort_string(s) for s in library_strings]

    # ── SIMILARITY (stage 5) ──────────────────────────────────────────────────
    if args.verbose:
        print(
            f"\nString type: {string_type} | preprocess: {preprocess} | strings: {len(template_strings)} templates, {len(library_strings)} library"
        )

    if args.all_methods:
        methods_to_run = list(AVAILABLE_METHODS.keys())
    else:
        methods_to_run = [args.method]

    for method in methods_to_run:
        if args.all_methods:
            output_path = Path(args.output)
            # stem is e.g. "smiles__replaced" or "inchi_all__" (trailing __ when no mods)
            # final name: "{repr}__{mods}__{method}.csv"
            stem = output_path.stem.rstrip("_")
            method_output = output_path.parent / f"{stem}__{method}.csv"
        else:
            method_output = args.output

        if args.verbose:
            if args.all_methods:
                print(f"\nProcessing method: {method}")
            print("Calculating similarities...")
            total_comparisons = len(library_strings) * len(template_strings)
            print(f"  Total comparisons: {total_comparisons:,}")

        # Check overwrite before doing any work
        if Path(method_output).exists():
            if not args.overwrite:
                print(f"  [skip] {method_output}: file exists (use --overwrite to replace)", file=sys.stderr)
                if args.timing_log:
                    with open(args.timing_log, "a") as _f:
                        _f.write(f"{method},skip_exists,\n")
                continue

        extra_kwargs = {"preprocess": preprocess}

        try:
            _t0 = time.perf_counter()
            sim_matrix = compute_cross_similarity_matrix(template_strings, library_strings, method=method, **extra_kwargs)
            _elapsed = time.perf_counter() - _t0
        except ImportError as exc:
            if args.all_methods:
                print(f"  [skip] {method}: {exc}", file=sys.stderr)
                if args.timing_log:
                    with open(args.timing_log, "a") as _f:
                        _f.write(f"{method},skip,\n")
                continue
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        # Write output
        if args.verbose:
            print(f"Writing output to: {method_output}")

        write_similarity_csv(method_output, library_names, template_names, sim_matrix)

        if args.timing_log:
            with open(args.timing_log, "a") as _f:
                _f.write(f"{method},ok,{_elapsed:.6f}\n")

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
    main()
