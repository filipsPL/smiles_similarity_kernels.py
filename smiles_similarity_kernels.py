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

---

PhaSMIfp — Pharmacophoric SMILES Fingerprint
=============================================

PhaSMIfp is a 78-dimensional pharmacophoric fingerprint derived entirely from
the SMILES string.  It is corpus-free and deterministic: no training data or
fitting step is needed.

Pharmacophoric alphabet (12 classes, fixed order):

  Symbol  Class                 Detection rule
  ------  -------------------   --------------------------------------------------
  D       H-bond donor          bare N, O (counted unconditionally — the token alone
                                doesn't reveal substitution count, so a fully-substituted
                                bare N/O with no actual H, e.g. tertiary amine N or ether
                                O, is still flagged); bracket atoms with explicit H:
                                [NH2], [OH], [nH], [NH2+], etc.
  A       H-bond acceptor       bare N, O, n, o, F; bracket [N]/[O]/[F] w/o '+'
  R       Aromatic atom         bare c, n, o, s, p tokens
  T       Sp3 carbon            bare C token (Cl handled as one token by tokenizer)
  L       Lipophilic run        count of contiguous C/c token runs (not atom count)
  P       Positive ionizable    bracket N/n with explicit '+': [NH2+], [nH+], [N+]
  M       Negative ionizable    bracket O/N/S with explicit '-': [O-], [NH-], [S-]
  Q       Quaternary N+         bracket [N+] without any H: permanently charged N
  E       Carbonyl              '=O' bond where the other atom is C/c (C=O or O=C)
  X       Halogen               bare F, Cl, Br, I tokens
  S       Sulfur (any)          bare S, s, or bracket atom starting with S/s
  G       Ring closure          count of ring-closure tokens (digits 1-9, %NN)

Detection uses the Schwaller tokenizer so multi-character atoms (Cl, Br, [nH])
are indivisible units.  Canonicalization via RDKit is applied before detection
when RDKit is available (strongly recommended for consistent results).

78D vector layout:
  [0:12]   12D — per-class integer counts (one dimension per class above)
  [12:78]  66D — pairwise min co-occurrence: min(count_i, count_j)
                 for all C(12,2) = 66 unique pairs i < j

Three output modes (``output`` parameter / registry key suffix):
  'count'      — raw integer counts (default); use for distance-based methods
  'binary'     — presence/absence (0/1); use for Tanimoto similarity
  'normalized' — divide by sum of 12D counts; use for angle-based comparison

PhaSMIfp is order-agnostic by design: all features are derived from token
counts and pairwise co-occurrence, not from token sequence position.  This
means two valid SMILES for the same molecule — when canonicalized — produce
identical vectors, even if the atom ordering differs.

Usage example:
    from smiles_similarity_kernels import pharmacophoric_fingerprint
    fp = pharmacophoric_fingerprint("CC(=O)Nc1ccc(O)cc1")       # 78D count
    fp_bin = pharmacophoric_fingerprint("CCO", output='binary')  # 78D binary

CLI usage:
    python smiles_similarity_kernels.py --fingerprint phasmifp \\
        --canonicalize --database molecules.smi --output fp.csv
"""

import re
import sys
import json
import time
import random
import heapq
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
    # Single-character element symbols (W, V, U) mapped to unique Unicode purely
    # for completeness of the element table and a canonical preprocessed form.
    # NOTE: unlike the multi-character entries above, these do NOT affect any
    # similarity score — a 1:1 single-char relabeling is invariant for every
    # metric here (edit distance, LCS, q-gram multisets, character counts), and
    # W/V/U are already atomic tokens.  Kept for consistency, not correctness.
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

    Note: this assumes a flat, non-nested layer structure — true for every
    InChI produced by :func:`smiles_to_inchi` (a plain ``MolToInchi(mol)``
    call, standard InChI, no 'FixedH' option).  A 'fixedH' layer ('f...')
    from an externally-generated InChI can itself contain a nested '/h'
    sub-layer; because segments are matched purely by first-letter prefix,
    such a nested sub-layer would be indistinguishable from the top-level
    'hydrogens' layer and could overwrite it. Not a concern for InChI
    strings generated by this module; pass externally-sourced 'FixedH'
    InChI through with caution.

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
    '11cccccc'
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
        warnings.warn(
            f"clcs_similarity: weights w1={w1}, w2={w2}, w3={w3} sum to {w1+w2+w3:.6g}, not 1. Scores will be off-scale.", stacklevel=2
        )

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

    # Both-empty is a degenerate "equally empty" case, handled the same way as
    # every other similarity function in this module (e.g. nlcs_similarity,
    # longest_common_substring_similarity) rather than falling through to the
    # k11==0/k22==0 branch below, which returns 0.0 for genuinely dissimilar
    # (non-empty vs. too-short-for-min_length) inputs.
    if not smiles1 and not smiles2:
        return 1.0

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
# NOTE: the fingerprint counts characters of the (usually preprocessed) string with
# Counter, whose keys are single characters.  The '@@' chirality token therefore has
# to be represented by its post-preprocess single-character sentinel — counting the
# literal two-character "@@" always yields 0 (Counter never has multi-char keys, and
# preprocess_smiles has already rewritten '@@' -> the sentinel by the time counting
# happens).  Using the sentinel makes the chirality dimension live under the default
# preprocess=True path.
SMIFP_CHARS_38.extend(["/", "\\", ELEMENT_REPLACEMENTS["@@"]])


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


def lingo_ruzicka_similarity(smiles1: str, smiles2: str, q: int = 4, preprocess: bool = True) -> float:
    """
    Ruzicka similarity (weighted Jaccard) on LINGO (q-gram) count multisets.

        S(A, B) = Σ_i min(N(A,i), N(B,i)) / Σ_i max(N(A,i), N(B,i))

    where N(A,i)/N(B,i) are the multiplicities of q-gram *i* in the two
    molecules.  Ruzicka is the count-aware generalisation of the Jaccard
    index: on binary presence/absence vectors it reduces to Jaccard, and on
    counts it weights each shared q-gram by how many times it actually
    co-occurs.  Because max = min + |difference|, this is algebraically
    identical to the multiset Tversky index with ``alpha = beta = 1``, so it
    is implemented by delegating to :func:`lingo_tversky_similarity` — no new
    formula to maintain.

    It is **distinct** from the vector/cosine-Tanimoto used by
    :func:`spectrum_kernel_similarity` (``dot / (||A||² + ||B||² − dot)``),
    which is not the same coefficient on count vectors, and from the Dice
    coefficient (:func:`lingo_dice_similarity`, alpha=beta=0.5), which weights
    shared q-grams more heavily.  Ruzicka is symmetric and lies in [0, 1]
    (1 iff the q-gram multisets are identical, 0 iff they are disjoint).

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
        Ruzicka (weighted Jaccard) similarity in [0, 1].
    """
    return lingo_tversky_similarity(smiles1, smiles2, q=q, alpha=1.0, beta=1.0, preprocess=preprocess)


def lingo_jaccard_binary_similarity(smiles1: str, smiles2: str, q: int = 4, preprocess: bool = True) -> float:
    """
    Binary (set) Jaccard/Tanimoto similarity on LINGO (q-gram) presence/absence.

        S(A, B) = |A ∩ B| / |A ∪ B|

    where A, B are the *sets* of distinct q-grams appearing in each SMILES
    string — multiplicity is discarded entirely.  This is a different
    encoding from :func:`lingo_ruzicka_similarity` (the count-aware/multiset
    Jaccard already in this module): a q-gram repeated many times (e.g. a
    long alkyl chain's "CCCC" motif, or a fused-ring system) contributes
    only once here, whereas Ruzicka's min/max multiset arithmetic lets
    repetition inflate the intersection. Comparing the two empirically
    answers a concrete question — does q-gram *repetition count* carry
    useful chemical signal, or is presence/absence enough?

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
        Binary Jaccard similarity in [0, 1].
    """
    set1 = set(get_lingos(smiles1, q, normalize_rings=True, preprocess=preprocess))
    set2 = set(get_lingos(smiles2, q, normalize_rings=True, preprocess=preprocess))
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0


def lingo_dice_binary_similarity(smiles1: str, smiles2: str, q: int = 4, preprocess: bool = True) -> float:
    """
    Binary (set) Sørensen-Dice coefficient on LINGO (q-gram) presence/absence.

    Registered alongside :func:`lingo_jaccard_binary_similarity` to complete
    the binary-vs-count x Dice-vs-Jaccard 2x2 grid (the count/multiset half,
    :func:`lingo_dice_similarity` and :func:`lingo_ruzicka_similarity`, was
    already implemented).

    Note for benchmarking: binary Dice and binary Jaccard are related by the
    monotonic transform ``DSC = 2J / (1 + J)`` (computed directly below,
    rather than re-deriving the set arithmetic) — the same relationship
    holds between the pre-existing multiset Dice and Ruzicka/weighted-Jaccard.
    Because the transform is monotonic, Dice and Jaccard rank molecule pairs
    *identically*; they differ only in how spread out the scores are, not in
    which pairs come out on top. Treat them as one independent signal (binary
    q-gram overlap), not two, when assessing how much new information a
    results grid actually contains.

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
        Binary Dice similarity in [0, 1].
    """
    jaccard = lingo_jaccard_binary_similarity(smiles1, smiles2, q=q, preprocess=preprocess)
    return 2.0 * jaccard / (1.0 + jaccard)


# ============================================================================
# 7. Spectrum Kernel (fixed k, no ring normalization)
# ============================================================================


def spectrum_kernel_similarity(
    smiles1: str,
    smiles2: str,
    k: int = 4,
    coefficient: str = "tanimoto",
    alpha: float = 0.5,
    beta: float = 0.5,
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
    coefficient : {'tanimoto', 'dice', 'cosine', 'tversky', 'overlap'}
        Normalisation of the kernel inner product.  ``'tanimoto'``,
        ``'dice'`` and ``'cosine'`` are the classical inner-product-based
        (vector-space) coefficients computed from squared k-mer count
        norms.  ``'tversky'`` and ``'overlap'`` instead use multiset
        intersection arithmetic (min/max of counts per k-mer, the same
        mechanism as :func:`lingo_tversky_similarity`) rather than a dot
        product — a mathematically distinct normalisation, not merely a
        rename of the existing ``'dice'``.
    alpha, beta : float
        Asymmetric weights for ``coefficient='tversky'`` applied to the
        k-mers unique to *smiles1* and *smiles2* respectively (default
        0.5/0.5, i.e. symmetric multiset Dice/Sørensen). Ignored for
        other coefficients.
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

    coef = coefficient.lower()

    if coef in ("tversky", "overlap"):
        intersection = only1 = only2 = 0
        for kmer in set(counts1) | set(counts2):
            a = counts1.get(kmer, 0)
            b = counts2.get(kmer, 0)
            intersection += min(a, b)
            only1 += max(a - b, 0)
            only2 += max(b - a, 0)
        if coef == "overlap":
            denominator = min(sum(counts1.values()), sum(counts2.values()))
            return intersection / denominator if denominator > 0 else 0.0
        denominator = intersection + alpha * only1 + beta * only2
        return intersection / denominator if denominator > 0 else 0.0

    # Inner product, self-inner-products
    dot = 0.0
    for kmer, c in counts1.items():
        if kmer in counts2:
            dot += c * counts2[kmer]
    norm1 = sum(c * c for c in counts1.values())
    norm2 = sum(c * c for c in counts2.values())
    if norm1 == 0 or norm2 == 0:
        return 0.0

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
    raise ValueError(f"Unknown coefficient: '{coefficient}'. " "Supported: 'tanimoto', 'dice', 'cosine', 'tversky', 'overlap'.")


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
# 9b. Gap-weighted Subsequence String Kernel (SSK)
# ============================================================================


def _subsequence_kernel_raw(s: str, t: str, n: int, lam: float) -> float:
    """
    Raw (un-normalised) gap-weighted subsequence kernel value K_n(s, t).

    Implements the O(n·|s|·|t|) dynamic program of Lodhi et al. (2002).  The
    kernel sums over every length-``n`` subsequence ``u`` shared by ``s`` and
    ``t``, weighting each occurrence by ``lam`` raised to the *span* it covers
    (last index − first index + 1, so interior gaps are penalised):

        K_n(s, t) = Σ_u ( Σ_{i: s[i]=u} lam^span(i) ) ( Σ_{j: t[j]=u} lam^span(j) )

    Verified bit-for-bit against a brute-force enumeration of this definition.
    Returns 0.0 when either string is shorter than ``n``.
    """
    ls, lt = len(s), len(t)
    if ls < n or lt < n:
        return 0.0

    # Kp = K'_{i-1}(s[:a], t[:b]); K'_0 = 1 for all prefixes (incl. empty ones).
    Kp = [[1.0] * (lt + 1) for _ in range(ls + 1)]

    for _ in range(1, n):
        curr = [[0.0] * (lt + 1) for _ in range(ls + 1)]
        for a in range(1, ls + 1):
            sa = s[a - 1]
            kpp = 0.0  # K''_i(a, b), rolled over b
            curr_a = curr[a]
            curr_am1 = curr[a - 1]
            kp_am1 = Kp[a - 1]
            for b in range(1, lt + 1):
                # K''_i(a,b) = lam*(K''_i(a,b-1) + lam*K'_{i-1}(a-1,b-1)·[s_a==t_b])
                if sa == t[b - 1]:
                    kpp = lam * (kpp + lam * kp_am1[b - 1])
                else:
                    kpp = lam * kpp
                # K'_i(a,b) = lam*K'_i(a-1,b) + K''_i(a,b)
                curr_a[b] = lam * curr_am1[b] + kpp
        Kp = curr

    # Final level: K_n(s,t) = Σ_{a,b: s_a==t_b} lam² · K'_{n-1}(a-1, b-1)
    lam2 = lam * lam
    total = 0.0
    for a in range(1, ls + 1):
        sa = s[a - 1]
        kp_am1 = Kp[a - 1]
        for b in range(1, lt + 1):
            if sa == t[b - 1]:
                total += lam2 * kp_am1[b - 1]
    return total


def subsequence_kernel_similarity(
    smiles1: str,
    smiles2: str,
    n: int = 3,
    lam: float = 0.5,
    normalized: bool = True,
    preprocess: bool = True,
) -> float:
    """
    Gap-weighted subsequence string kernel similarity (Lodhi et al. 2002).

    Unlike the contiguous :func:`substring_kernel_similarity`, this kernel
    matches subsequences whose characters need **not** be adjacent, but
    discounts them by ``lam`` per unit of span so that tightly-packed matches
    count for more than gappy ones.  This captures conserved scaffolds that are
    interrupted by substituents in the SMILES string.

        sim(S1, S2) = K_n(S1, S2) / sqrt(K_n(S1, S1) · K_n(S2, S2))

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings to compare.
    n : int
        Subsequence length (default 3).
    lam : float
        Decay factor in (0, 1] penalising gaps; smaller ``lam`` penalises gaps
        more strongly.  Default 0.5.
    normalized : bool
        If True (default) return the cosine-normalised kernel in [0, 1]; if
        False return the raw kernel value.
    preprocess : bool
        Whether to apply SMILES multi-character preprocessing.

    Returns
    -------
    float
        Normalised similarity in [0, 1] (or the raw kernel if ``normalized`` is
        False).  When both strings are shorter than ``n`` the result is 1.0 iff
        they are identical, else 0.0 (matching :func:`spectrum_kernel_similarity`).

    References
    ----------
    Lodhi H., Saunders C., Shawe-Taylor J., Cristianini N., Watkins C.
    "Text classification using string kernels." JMLR 2 (2002) 419–444.
    """
    if preprocess:
        smiles1 = preprocess_smiles(smiles1)
        smiles2 = preprocess_smiles(smiles2)

    if len(smiles1) < n and len(smiles2) < n:
        return 1.0 if smiles1 == smiles2 else 0.0
    if len(smiles1) < n or len(smiles2) < n:
        return 0.0

    k12 = _subsequence_kernel_raw(smiles1, smiles2, n, lam)
    if not normalized:
        return k12

    k11 = _subsequence_kernel_raw(smiles1, smiles1, n, lam)
    k22 = _subsequence_kernel_raw(smiles2, smiles2, n, lam)
    if k11 <= 0.0 or k22 <= 0.0:
        return 0.0
    return k12 / np.sqrt(k11 * k22)


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
# 10a. SMILES TF-IDF Cosine Similarity (chemical tokenization)
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


def _tfidf_cosine_similarity(s1, s2, make_tokenizer, func_name, corpus, ngram_range, vectorizer):
    """
    Shared implementation for the tokenizer-backed TF-IDF cosine methods.

    The four public ``*_tfidf_similarity`` functions differ only in their
    tokenizer, so they all delegate here — the fit / transform / cosine logic
    lives in exactly one place.  ``make_tokenizer`` is a zero-argument callable
    returning a fresh tokenizer instance; ``func_name`` is used only for the
    ImportError message so it names the caller.

    Behaviour is identical to the original per-method bodies: when no fitted
    ``vectorizer`` is supplied one is fit on ``corpus`` (defaulting to the two
    inputs), an empty-vocabulary ``ValueError`` returns 0.0, and any tokenizer
    construction error (e.g. a missing BPE vocab file) propagates unchanged.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(f"sklearn is required for {func_name}")

    if vectorizer is None:
        if corpus is None:
            corpus = [s1, s2]
        vectorizer = TfidfVectorizer(
            tokenizer=make_tokenizer(),
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

    vec1 = vectorizer.transform([s1])
    vec2 = vectorizer.transform([s2])
    return float(sklearn_cosine_similarity(vec1, vec2)[0, 0])


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
    return _tfidf_cosine_similarity(smiles1, smiles2, SMILESTokenizer, "smiles_tfidf_similarity", corpus, ngram_range, vectorizer)


# ============================================================================
# 10b. Schwaller TF-IDF Cosine Similarity
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
    return _tfidf_cosine_similarity(smiles1, smiles2, SMILESTokenizerSchwaller, "schwaller_tfidf_similarity", corpus, ngram_range, vectorizer)


# ============================================================================
# 10c. BPE TF-IDF Cosine Similarity
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

    Assumes ``_merges`` has no duplicate ``(a, b)`` pairs (true of every
    vocabulary written by ``train_bpe_tokenizer.py``, which never repeats a
    rule): ``self._rank`` keeps only the *last* index for a repeated pair, so
    a hand-crafted merge table with duplicates could tokenize differently
    than the original per-rule-rescan algorithm would. Verified against real
    vocabulary files (see ``smiles_bpe_vocab.json``), which have none.

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

    @property
    def _merges(self) -> list:
        return self.__merges

    @_merges.setter
    def _merges(self, value) -> None:
        # Rebuild the rank lookup whenever _merges is (re)assigned — including
        # direct overrides like ``tok._merges = [...]`` (used in tests) — so
        # tokenize() never runs against a rank table stale relative to _merges.
        self.__merges = list(value)
        self._rank: dict = {pair: i for i, pair in enumerate(self.__merges)}

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string using BPE merges."""
        return self._apply_merges(self._BASE_RE.findall(smiles))

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """
        Apply ``self._merges`` to an arbitrary token sequence.

        Equivalent to applying each rule in ``self._merges`` in order — for
        each rule, merging every non-overlapping left-to-right occurrence of
        that exact adjacent pair before moving to the next rule — but
        implemented as a priority-queue merge over a doubly-linked token list
        instead of rescanning the whole token sequence once per rule
        regardless of whether that rule matches anything. That naive rescan
        is O(num_merges * token_count) per string; with thousands of merges
        in the default vocabulary this dominated real-dataset runtimes (see
        git history).

        Each live token tracks the rank of the merge that created its current
        value (``created_rank``; ``None`` for an original, never-merged base
        token). A candidate merge at rank r is only ever considered if both
        its operand tokens were created strictly before r (or are still base
        tokens) — without this, a token created by a *later*-ranked merge
        could get eagerly consumed by an *earlier*-ranked rule that, in the
        one-pass-per-rank naive algorithm, would already have finished its
        single pass long before that later merge ever ran and so would never
        have seen it. This is what makes "always merge the lowest-rank pair
        currently present" equivalent to "apply each rule fully, in rank
        order" for *any* merge table, not just ones shaped like the output of
        standard BPE training. Cross-checked by exhaustive random fuzzing
        against the original per-rule-rescan algorithm, including on
        deliberately adversarial (not realistically trained) merge tables —
        see TestSMILESTokenizerBPEFuzz.

        Factored out from :meth:`tokenize` so tests can fuzz arbitrary token
        sequences directly, independent of what the SMILES regex can produce.
        """
        n = len(tokens)
        if n < 2 or not self._rank:
            return list(tokens)

        rank = self._rank
        tok = list(tokens)
        nxt = list(range(1, n)) + [-1]
        prv = [-1] + list(range(n - 1))
        alive = [True] * n
        created_rank: list = [None] * n  # None = still an original base token

        heap: list = []

        def push_pair(i: int) -> None:
            j = nxt[i]
            if j == -1:
                return
            r = rank.get((tok[i], tok[j]))
            if r is None:
                return
            ci, cj = created_rank[i], created_rank[j]
            if (ci is not None and ci >= r) or (cj is not None and cj >= r):
                return  # an operand was itself created at/after rank r: unreachable in rank-order execution
            heapq.heappush(heap, (r, i))

        for i in range(n - 1):
            push_pair(i)

        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if rank.get((tok[i], tok[j])) != r:
                continue  # stale entry: tok[i] or tok[j] changed since this was pushed
            tok[i] = tok[i] + tok[j]
            created_rank[i] = r
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prv[k] = i
            p = prv[i]
            if p != -1:
                push_pair(p)
            push_pair(i)

        out = []
        i = 0
        while i != -1:
            if alive[i]:
                out.append(tok[i])
            i = nxt[i]
        return out

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
    return _tfidf_cosine_similarity(
        smiles1,
        smiles2,
        lambda: SMILESTokenizerBPE(vocab_path=vocab_path, num_merges=num_merges),
        "bpe_tfidf_similarity",
        corpus,
        ngram_range,
        vectorizer,
    )


# ============================================================================
# 10d. SELFIES TF-IDF Cosine Similarity
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
    return _tfidf_cosine_similarity(selfies1, selfies2, SELFIESTokenizer, "selfies_tfidf_similarity", corpus, ngram_range, vectorizer)


# ============================================================================
# 10e. Token-level Edit Distance Similarity
# ============================================================================


def token_edit_similarity(smiles1: str, smiles2: str, tokenizer=None, preprocess: bool = False) -> float:
    """
    Levenshtein edit similarity over chemically-meaningful *tokens* instead of
    raw characters.

    Standard :func:`edit_similarity` operates on characters (after multi-character
    atoms are collapsed to single characters by ``preprocess_smiles``).  This
    variant first splits each SMILES into atom-level tokens with a chemical
    tokenizer — so a bracket atom such as ``[nH+]`` or ``[C@@H]`` is a single
    unit — then computes the Levenshtein distance over the *token sequences*.  A
    one-atom change therefore costs exactly one edit, which is more chemically
    interpretable than the several character edits the same change incurs at the
    character level (e.g. ``[nH+]`` → ``[nH]`` is 1 token edit but 2 char edits).

        sim(S1, S2) = 1 - editdistance(tok(S1), tok(S2)) / max(|tok(S1)|, |tok(S2)|)

    where ``tok`` is the tokenizer and each token insert/delete/substitute costs 1.
    The score lies in [0, 1]: 1.0 for identical token sequences (edit distance 0),
    0.0 when the sequences share no alignment.

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings to compare.
    tokenizer : callable, optional
        A callable ``str -> List[str]`` (e.g. an instance of
        :class:`SMILESTokenizerSchwaller`, :class:`SMILESTokenizer`, or
        :class:`SMILESTokenizerBPE`).  Defaults to
        :class:`SMILESTokenizerSchwaller` (atom-level, the de-facto standard).
    preprocess : bool
        Accepted for API compatibility but **ignored**: the tokenizer already
        treats multi-character atoms (``Cl``, ``Br``, ``[nH+]``, …) as indivisible
        tokens, so character-level substitution is neither needed nor applied.
        (Mirrors the tokenizer-backed TF-IDF methods.)

    Returns
    -------
    float
        Similarity in [0, 1].

    Examples
    --------
    >>> token_edit_similarity("c1ccccc1", "c1ccccc1")
    1.0
    """
    if tokenizer is None:
        tokenizer = SMILESTokenizerSchwaller()

    toks1 = tokenizer(smiles1)
    toks2 = tokenizer(smiles2)

    if len(toks1) == 0 and len(toks2) == 0:
        return 1.0

    ed = edit_distance(toks1, toks2)
    max_len = max(len(toks1), len(toks2))
    return 1.0 - ed / max_len


# ============================================================================
# 10f. Monge-Elkan Token Similarity
# ============================================================================


def monge_elkan_similarity(
    smiles1: str,
    smiles2: str,
    tokenizer=None,
    token_similarity: Optional[Callable[[str, str], float]] = None,
    bidirectional: bool = False,
    preprocess: bool = False,
) -> float:
    """
    Monge-Elkan token-level similarity (Monge & Elkan, 1996).

        ME(A, B) = (1 / |A|) * sum_{a in A} max_{b in B} sim(a, b)

    where A, B are the token sequences of *smiles1*/*smiles2* and ``sim``
    is a secondary, per-token similarity function.  Unlike
    :func:`token_edit_similarity` (which aligns the *whole* token sequence
    via a single global edit distance), Monge-Elkan matches each token of
    A independently against its single best partner anywhere in B — so a
    substituent moved to a different position in the SMILES string still
    contributes full credit, and near-miss tokens (e.g. ``[nH+]`` vs
    ``[NH+]``, or ``Cl`` vs ``Br``) contribute *partial* credit via
    ``sim`` instead of an all-or-nothing token match.

    Default ``sim`` is character-level Levenshtein similarity
    (:func:`edit_similarity` with ``preprocess=False``, since tokens are
    already atomic); pass a different ``token_similarity`` callable to use
    something else (e.g. exact match, or a chemically-weighted
    substitution score).

    Monge-Elkan is **asymmetric by construction**: it averages over the
    tokens of *smiles1*, so ``sim(A, B) != sim(B, A)`` in general (a short
    query matched against a long candidate scores differently than the
    reverse — the classic use case is validating a short "query" record
    against a longer reference). Set ``bidirectional=True`` to average
    both directions into a proper symmetric similarity (registered
    separately as ``monge_elkan_sym``).

    Parameters
    ----------
    smiles1, smiles2 : str
        SMILES strings to compare. *smiles1* drives the average when
        ``bidirectional=False`` (the default).
    tokenizer : callable, optional
        A callable ``str -> List[str]``. Defaults to
        :class:`SMILESTokenizerSchwaller` (atom-level), matching
        :func:`token_edit_similarity`'s convention.
    token_similarity : callable, optional
        A callable ``(str, str) -> float`` in [0, 1] scoring a pair of
        tokens. Defaults to character-level Levenshtein similarity.
    bidirectional : bool
        If True, return the average of ME(A, B) and ME(B, A) instead of
        just ME(A, B). Default False (the original, asymmetric definition).
    preprocess : bool
        Accepted for API compatibility but **ignored**, like
        :func:`token_edit_similarity`: the tokenizer already treats
        multi-character atoms as indivisible tokens.

    Returns
    -------
    float
        Similarity in [0, 1].

    References
    ----------
    Monge A., Elkan C. "The field matching problem: Algorithms and
    applications." KDD 1996, 267–270.
    """
    if tokenizer is None:
        tokenizer = SMILESTokenizerSchwaller()
    if token_similarity is None:
        token_similarity = lambda t1, t2: edit_similarity(t1, t2, preprocess=False)

    toks1 = tokenizer(smiles1)
    toks2 = tokenizer(smiles2)

    def _directional(a: List[str], b: List[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return sum(max(token_similarity(tok, other) for other in b) for tok in a) / len(a)

    forward = _directional(toks1, toks2)
    if not bidirectional:
        return forward
    backward = _directional(toks2, toks1)
    return (forward + backward) / 2.0


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
# 12. Normalized Compression Distance (NCD) similarity
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
    # One tokenizer instance owns both the merge table (→ feature dimensions)
    # and the tokenization, so the merge logic is not duplicated here.  It also
    # raises FileNotFoundError when the vocabulary is missing.
    tokenizer = SMILESTokenizerBPE(vocab_path=vocab_path, num_merges=num_merges)

    # One fingerprint dimension per *merged* token (base single-char tokens excluded).
    merged_tokens = [a + b for a, b in tokenizer._merges]

    token_counts = Counter(tokenizer.tokenize(smiles))
    fp = np.array([float(token_counts.get(tok, 0)) for tok in merged_tokens], dtype=float)
    if binary:
        fp = (fp > 0).astype(float)
    return fp


# ============================================================================
# PhaSMIfp — Pharmacophoric SMILES Fingerprint
# ============================================================================

# Fixed ordered alphabet — indices must never change (they determine feature positions).
PHASMIFP_CLASSES = ["D", "A", "R", "T", "L", "P", "M", "Q", "E", "X", "S", "G"]

# Reuse the Schwaller tokenizer regex for token-level detection.
_PHARM_TOKEN_RE = SMILESTokenizerSchwaller._TOKEN_RE


def _is_carbon_token(tok: str) -> bool:
    """Return True if *tok* represents a carbon atom (bare C/c or bracket [C...]/ [c...])."""
    if tok in ("C", "c"):
        return True
    if tok.startswith("[") and tok[1:2] in ("C", "c"):
        return True
    return False


def _is_ring_closure_token(tok: str) -> bool:
    """Return True if *tok* is a ring-closure bond token: a single digit or '%NN'."""
    return (len(tok) == 1 and tok.isdigit()) or tok.startswith("%")


def _skip_back_to_atom(tokens: List[str], j: int) -> int:
    """
    Walk backward from index *j* over ring-closure tokens (single digits,
    ``%NN``) and fully-closed branch groups, stopping at the first atom
    token (or running off the start of the sequence).

    Used by the carbonyl (E) detector to find "the atom before this
    position" — an atom can be followed by a ring-closure digit before a
    branch opens (``C1(=O)...``) or before a bare ``=`` (``C1=O``), and
    that digit is not itself the atom.

    Returns the resulting index, or -1 if no atom is found.
    """
    while j >= 0:
        tok = tokens[j]
        if tok == ")":
            depth = 0
            while j >= 0:
                if tokens[j] == ")":
                    depth += 1
                elif tokens[j] == "(":
                    depth -= 1
                    if depth == 0:
                        j -= 1
                        break
                j -= 1
            continue
        if _is_ring_closure_token(tok):
            j -= 1
            continue
        break
    return j


def _compute_pharmacophore_counts(smiles: str) -> np.ndarray:
    """
    Return the 12D pharmacophoric class count vector for *smiles*.

    Detection is performed on Schwaller tokens so that multi-character atoms
    (Cl, Br, [nH], %10) are treated as single indivisible units, eliminating
    substring matching artefacts.

    Classes (in PHASMIFP_CLASSES order):
      D  H-bond donor       — bare N/O counted unconditionally (a single token doesn't
                              reveal substitution count, so a fully-substituted bare
                              N/O with no actual H — e.g. tertiary amine N, ether O —
                              is still flagged) + bracket N/O/n/o with explicit H
                              ([NH2], [OH], [nH], [NH2+], etc.)
      A  H-bond acceptor    — bare N, O, n, o, F; bracket [N]/[O]/[F] without '+'
      R  Aromatic atom      — bare c, n, o, s, p tokens
      T  Sp3 carbon         — bare C token (tokenizer gives Cl/C as separate tokens)
      L  Lipophilic run     — count of contiguous C/c runs in token stream
      P  Positive ionizable — bracket atoms with N/n and explicit '+'
      M  Negative ionizable — bracket atoms with O/N/S and explicit '-'
      Q  Quaternary N+      — bracket [N+] without any H
      E  Carbonyl           — '=O' where the bonded atom is C/c (scan back past closed
                              branches and ring-closure digits via _skip_back_to_atom)
      X  Halogen            — bare F, Cl, Br, I tokens
      S  Sulfur (any)       — bare S, s, or bracket atom starting with S
      G  Ring closure       — single-digit (1-9) and %NN ring closure tokens

    Note: D/A/R/T/... are per-token heuristics, not a full valence/connectivity
    model — see the D rule above for the clearest example of the tradeoff.
    """
    if not smiles:
        return np.zeros(12, dtype=float)

    try:
        tokens = _PHARM_TOKEN_RE.findall(smiles)
    except Exception:
        return np.zeros(12, dtype=float)

    counts = np.zeros(12, dtype=float)
    n_tokens = len(tokens)
    in_lipophilic_run = False

    for i, tok in enumerate(tokens):
        is_bracket = tok.startswith("[")

        # D — H-bond donor:
        #   • bare N or O — implicit H by SMILES valence rules (N has 3 bonds, O has 2)
        #   • bracket atoms with explicit H: [NH2], [OH], [nH], [NH2+], [NH-], etc.
        if tok in ("N", "O"):
            counts[0] += 1  # D (bare, has implicit H unless charge/over-valenced)
        elif is_bracket:
            inner = tok[1:-1]
            if inner and inner[0] in ("N", "O", "n", "o") and "H" in inner:
                counts[0] += 1  # D (explicit H in bracket)

        # A — H-bond acceptor: bare N, O, n, o, F; bracket [N]/[O]/[F] without '+'
        if tok in ("N", "O", "n", "o", "F"):
            counts[1] += 1  # A
        elif is_bracket:
            inner = tok[1:-1]
            if inner and inner[0] in ("N", "O", "F", "n", "o", "f") and "+" not in inner:
                counts[1] += 1  # A

        # R — aromatic bare atom
        if tok in ("c", "n", "o", "s", "p"):
            counts[2] += 1  # R

        # T — sp3 carbon: bare 'C' (tokenizer gives Cl as one token, so bare C is safe)
        if tok == "C":
            counts[3] += 1  # T

        # L — lipophilic run: count transitions into a C/c run
        is_carbon = tok in ("C", "c")
        if is_carbon and not in_lipophilic_run:
            counts[4] += 1  # L — new run starts
            in_lipophilic_run = True
        elif not is_carbon:
            in_lipophilic_run = False

        # P — positive ionizable: bracket N/n with explicit '+'
        if is_bracket:
            inner = tok[1:-1]
            if inner and inner[0] in ("N", "n") and "+" in inner:
                counts[5] += 1  # P

        # M — negative ionizable: bracket O/N/S/s with explicit '-'
        if is_bracket:
            inner = tok[1:-1]
            if inner and inner[0] in ("O", "N", "S", "o", "n", "s") and "-" in inner:
                counts[6] += 1  # M

        # Q — quaternary N+: bracket [N+] without any H
        if is_bracket:
            inner = tok[1:-1]
            if inner.startswith("N+") and "H" not in inner:
                counts[7] += 1  # Q

        # E — carbonyl: any '=O' bond where the other atom is C/c.
        # Detect at the '=' token to handle all orderings:
        #   C=O  (C before =)  and  O=C  (O before =, aldehyde/ketone head form)
        # Look both one step back and one step forward from '='.
        # To avoid double-counting a single =O group, only count at '='.
        if tok == "=" and i + 1 < n_tokens and i > 0:
            next_tok = tokens[i + 1]
            prev_tok_raw = tokens[i - 1]

            # resolve branch-open: if '=' is right after '(' the parent atom is further back.
            # Either way, a ring-closure digit can sit directly between the atom and this
            # position (e.g. 'C1(=O)...' or 'C1=O') and is not itself the atom, so
            # _skip_back_to_atom skips both closed branches and ring-closure tokens.
            start = i - 2 if prev_tok_raw == "(" else i - 1
            j = _skip_back_to_atom(tokens, start)
            prev_atom = tokens[j] if j >= 0 else ""

            # pattern 1: C=O  (carbon before '=', oxygen after)
            if next_tok == "O" and _is_carbon_token(prev_atom):
                counts[8] += 1  # E
            # pattern 2: O=C  (oxygen before '=', carbon after)
            elif prev_atom == "O" and _is_carbon_token(next_tok):
                counts[8] += 1  # E

        # X — halogen: bare F, Cl, Br, I (tokenizer gives Cl/Br as single tokens)
        if tok in ("F", "Cl", "Br", "I"):
            counts[9] += 1  # X

        # S — sulfur: bare S/s, or bracket atom starting with S
        if tok in ("S", "s"):
            counts[10] += 1  # S
        elif is_bracket and tok[1:2] in ("S", "s"):
            counts[10] += 1  # S

        # G — ring closure: single digit 1-9, or %NN token
        if (len(tok) == 1 and tok.isdigit() and tok != "0") or tok.startswith("%"):
            counts[11] += 1  # G

    return counts


def get_pharmacophoric_feature_names() -> List[str]:
    """
    Return the 78 human-readable feature names for PhaSMIfp.

    Layout:
      [0:12]  pharm_D … pharm_G   — per-class counts
      [12:78] pharm_DA, pharm_DR, … pharm_SG — pairwise min co-occurrence
    """
    names = [f"pharm_{c}" for c in PHASMIFP_CLASSES]
    for i in range(12):
        for j in range(i + 1, 12):
            names.append(f"pharm_{PHASMIFP_CLASSES[i]}{PHASMIFP_CLASSES[j]}")
    return names


def pharmacophoric_fingerprint(
    smiles: str,
    output: str = "count",
    canonicalize: bool = True,
) -> np.ndarray:
    """
    PhaSMIfp: 78D pharmacophoric SMILES fingerprint.

    Computes a 12D pharmacophoric class count vector and a 66D pairwise
    co-occurrence vector, concatenated into a single 78D hologram.

    Layers:
        [0:12]   12D — per-class counts (D, A, R, T, L, P, M, Q, E, X, S, G)
        [12:78]  66D — pairwise min co-occurrence: min(count_i, count_j)
                       for all unique pairs i < j (upper triangle, no diagonal)

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    output : str
        'count'      — raw integer counts (default)
        'binary'     — presence/absence (clip counts to 0/1)
        'normalized' — divide by sum of 12D count vector (float, sums to 1
                       for the first 12 dimensions; pairwise scaled accordingly)
    canonicalize : bool
        Whether to RDKit-canonicalize *smiles* before counting (default
        True), when RDKit is available. Set False to count on the input
        string as given — e.g. for shuffled/negative-control inputs, or
        non-SMILES representations, where canonicalization is meaningless
        or would defeat the purpose of the input transform.

    Returns
    -------
    np.ndarray
        1-D float64 array of length 78.
    """
    if not smiles:
        return np.zeros(78, dtype=float)

    smi = canonicalize_smiles(smiles) if (canonicalize and RDKIT_AVAILABLE) else smiles
    if not smi:
        smi = smiles

    try:
        counts = _compute_pharmacophore_counts(smi)
    except Exception:
        return np.zeros(78, dtype=float)

    pairs = np.array(
        [min(counts[i], counts[j]) for i in range(12) for j in range(i + 1, 12)],
        dtype=float,
    )

    fp_78 = np.concatenate([counts, pairs]).astype(float)

    if output == "binary":
        return (fp_78 > 0).astype(float)
    elif output == "normalized":
        total = counts.sum()
        return fp_78 / total if total > 0 else fp_78
    else:
        return fp_78


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
    # ── PhaSMIfp ─────────────────────────────────────────────────────────────
    "phasmifp": {
        "function": pharmacophoric_fingerprint,
        "description": "PhaSMIfp 78D pharmacophoric hologram (12D counts + 66D pairwise co-occurrence, count)",
        "length": 78,
        "params": {"output": "count"},
    },
    "phasmifp_binary": {
        "function": lambda smi, **kw: pharmacophoric_fingerprint(smi, output="binary", **kw),
        "description": "PhaSMIfp 78D pharmacophoric hologram (binary presence/absence)",
        "length": 78,
        "params": {"output": "binary"},
    },
    "phasmifp_normalized": {
        "function": lambda smi, **kw: pharmacophoric_fingerprint(smi, output="normalized", **kw),
        "description": "PhaSMIfp 78D pharmacophoric hologram (normalized float)",
        "length": 78,
        "params": {"output": "normalized"},
    },
    "phasmifp12": {
        "function": lambda smi, canonicalize=True, **kw: _compute_pharmacophore_counts(
            (canonicalize_smiles(smi) if (canonicalize and RDKIT_AVAILABLE) else smi) or smi
        ),
        "description": "PhaSMIfp 12D pharmacophoric class count vector only (no pairwise layer)",
        "length": 12,
        "params": {},
    },
    "phasmifp12_binary": {
        "function": lambda smi, canonicalize=True, **kw: (
            _compute_pharmacophore_counts((canonicalize_smiles(smi) if (canonicalize and RDKIT_AVAILABLE) else smi) or smi) > 0
        ).astype(float),
        "description": "PhaSMIfp 12D pharmacophoric class binary vector only (no pairwise layer)",
        "length": 12,
        "params": {},
    },
}


def get_fingerprint_function(fp_type: str):
    """Return the fingerprint function for *fp_type*, checking availability."""
    if fp_type not in AVAILABLE_FINGERPRINTS:
        raise ValueError(f"Unknown fingerprint type: '{fp_type}'. " f"Available: {list(AVAILABLE_FINGERPRINTS.keys())}")
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
    # Use integer format for count/binary fingerprints; float for normalized ones.
    is_integer = matrix.size > 0 and np.all(matrix == np.floor(matrix))
    fmt = "%.0f" if is_integer else "%.6f"
    df.to_csv(output_path, index=False, float_format=fmt)


# ============================================================================
# Available Methods Registry
# ============================================================================

AVAILABLE_METHODS = {
    "edit": {"function": edit_similarity, "description": "Edit distance similarity", "params": {}},
    "nlcs": {"function": nlcs_similarity, "description": "Normalized Longest Common Subsequence", "params": {}},
    "clcs": {"function": clcs_similarity, "description": "Combined LCS models", "params": {}},
    "substring": {
        "function": lambda s1, s2, **kw: substring_kernel_similarity(s1, s2, **{**{"normalized": True}, **kw}),
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
        "function": lambda s1, s2, **kw: smifp_similarity_cityblock(s1, s2, **{**{"chars": SMIFP_CHARS_38}, **kw}),
        "description": "SMILES fingerprint 38D with City Block Distance (Manhattan)",
        "params": {},
        "requires": "scipy",
    },
    "smifp38_tanimoto": {
        "function": lambda s1, s2, **kw: smifp_similarity_tanimoto(s1, s2, **{**{"chars": SMIFP_CHARS_38}, **kw}),
        "description": "SMILES fingerprint 38D with Tanimoto",
        "params": {},
    },
    "lingo": {"function": lingo_similarity, "description": "LINGO similarity (q=4)", "params": {"q": 4}},
    "lingo3": {
        "function": lambda s1, s2, **kw: lingo_similarity(s1, s2, **{**{"q": 3}, **kw}),
        "description": "LINGO similarity (q=3)",
        "params": {"q": 3},
    },
    "lingo5": {
        "function": lambda s1, s2, **kw: lingo_similarity(s1, s2, **{**{"q": 5}, **kw}),
        "description": "LINGO similarity (q=5)",
        "params": {"q": 5},
    },
    "lingo_tversky": {
        "function": lingo_tversky_similarity,
        "description": "Asymmetric Tversky on LINGO q-grams (q=4, alpha=0.9, beta=0.1) — query-weighted",
        "params": {"q": 4, "alpha": 0.9, "beta": 0.1},
    },
    "lingo_tversky_sym": {
        "function": lambda s1, s2, **kw: lingo_tversky_similarity(s1, s2, **{**{"q": 4, "alpha": 0.5, "beta": 0.5}, **kw}),
        "description": "Symmetric Tversky (alpha=beta=0.5, equivalent to Dice) on LINGO q-grams",
        "params": {"q": 4, "alpha": 0.5, "beta": 0.5},
    },
    "lingo_dice": {
        "function": lingo_dice_similarity,
        "description": "Sørensen-Dice coefficient on LINGO q-gram counts (q=4)",
        "params": {"q": 4},
    },
    "lingo_ruzicka": {
        "function": lingo_ruzicka_similarity,
        "description": "Ruzicka (weighted Jaccard) on LINGO q-gram counts (q=4) — Σmin/Σmax, i.e. Tversky(α=β=1)",
        "params": {"q": 4},
    },
    "lingo_jaccard_binary": {
        "function": lingo_jaccard_binary_similarity,
        "description": "Binary Jaccard/Tanimoto on LINGO q-gram presence/absence (q=4, multiplicity discarded)",
        "params": {"q": 4},
    },
    "lingo_dice_binary": {
        "function": lingo_dice_binary_similarity,
        "description": "Binary Sørensen-Dice on LINGO q-gram presence/absence (q=4) — monotonic transform of lingo_jaccard_binary",
        "params": {"q": 4},
    },
    "spectrum": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(s1, s2, **{**{"k": 4, "coefficient": "tanimoto"}, **kw}),
        "description": "Spectrum kernel (k=4, Tanimoto) — classical fixed-k string kernel",
        "params": {"k": 4, "coefficient": "tanimoto"},
    },
    "spectrum3": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(s1, s2, **{**{"k": 3, "coefficient": "tanimoto"}, **kw}),
        "description": "Spectrum kernel (k=3, Tanimoto)",
        "params": {"k": 3, "coefficient": "tanimoto"},
    },
    "spectrum5": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(s1, s2, **{**{"k": 5, "coefficient": "tanimoto"}, **kw}),
        "description": "Spectrum kernel (k=5, Tanimoto)",
        "params": {"k": 5, "coefficient": "tanimoto"},
    },
    "spectrum_cosine": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(s1, s2, **{**{"k": 4, "coefficient": "cosine"}, **kw}),
        "description": "Spectrum kernel (k=4, cosine normalisation)",
        "params": {"k": 4, "coefficient": "cosine"},
    },
    "spectrum_tversky": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(
            s1, s2, **{**{"k": 4, "coefficient": "tversky", "alpha": 0.9, "beta": 0.1}, **kw}
        ),
        "description": "Asymmetric Tversky on spectrum k-mers (k=4, alpha=0.9, beta=0.1) — query-weighted",
        "params": {"k": 4, "coefficient": "tversky", "alpha": 0.9, "beta": 0.1},
    },
    "spectrum_tversky_sym": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(
            s1, s2, **{**{"k": 4, "coefficient": "tversky", "alpha": 0.5, "beta": 0.5}, **kw}
        ),
        "description": "Symmetric Tversky (alpha=beta=0.5, equivalent to multiset Dice) on spectrum k-mers (k=4)",
        "params": {"k": 4, "coefficient": "tversky", "alpha": 0.5, "beta": 0.5},
    },
    "spectrum_overlap": {
        "function": lambda s1, s2, **kw: spectrum_kernel_similarity(s1, s2, **{**{"k": 4, "coefficient": "overlap"}, **kw}),
        "description": "Overlap coefficient (intersection / min) on spectrum k-mers (k=4) — robust to size-mismatched pairs",
        "params": {"k": 4, "coefficient": "overlap"},
    },
    "mismatch": {
        "function": lambda s1, s2, **kw: mismatch_kernel_similarity(
            s1, s2, **{**{"k": 4, "m": 1, "coefficient": "tanimoto"}, **kw}
        ),
        "description": "Mismatch (spectrum-(k,m)) kernel (k=4, m=1, Tanimoto) — tolerates 1 atom swap",
        "params": {"k": 4, "m": 1, "coefficient": "tanimoto"},
    },
    "mismatch3": {
        "function": lambda s1, s2, **kw: mismatch_kernel_similarity(
            s1, s2, **{**{"k": 3, "m": 1, "coefficient": "tanimoto"}, **kw}
        ),
        "description": "Mismatch kernel (k=3, m=1, Tanimoto)",
        "params": {"k": 3, "m": 1, "coefficient": "tanimoto"},
    },
    "mismatch5": {
        "function": lambda s1, s2, **kw: mismatch_kernel_similarity(
            s1, s2, **{**{"k": 5, "m": 1, "coefficient": "tanimoto"}, **kw}
        ),
        "description": "Mismatch kernel (k=5, m=1, Tanimoto)",
        "params": {"k": 5, "m": 1, "coefficient": "tanimoto"},
    },
    "lcs_substring": {
        "function": longest_common_substring_similarity,
        "description": "Normalised Longest Common Substring (contiguous) — LCSubstr²/(len1×len2)",
        "params": {},
    },
    "token_edit": {
        "function": token_edit_similarity,
        "description": "Levenshtein edit distance over Schwaller atom-level tokens (chemically-aware edit distance)",
        "params": {},
    },
    "monge_elkan": {
        "function": monge_elkan_similarity,
        "description": "Monge-Elkan token similarity (Schwaller tokens, char-edit secondary metric) — asymmetric, smiles1 drives the average",
        "params": {},
    },
    "monge_elkan_sym": {
        "function": lambda s1, s2, **kw: monge_elkan_similarity(s1, s2, **{**{"bidirectional": True}, **kw}),
        "description": "Symmetric Monge-Elkan (average of both directions)",
        "params": {"bidirectional": True},
    },
    "subsequence": {
        "function": subsequence_kernel_similarity,
        "description": "Gap-weighted subsequence string kernel (Lodhi et al. 2002; n=3, lambda=0.5)",
        "params": {"n": 3, "lam": 0.5},
    },
    "subsequence2": {
        "function": lambda s1, s2, **kw: subsequence_kernel_similarity(s1, s2, **{**{"n": 2, "lam": 0.5}, **kw}),
        "description": "Gap-weighted subsequence string kernel (n=2, lambda=0.5)",
        "params": {"n": 2, "lam": 0.5},
    },
    "subsequence4": {
        "function": lambda s1, s2, **kw: subsequence_kernel_similarity(s1, s2, **{**{"n": 4, "lam": 0.5}, **kw}),
        "description": "Gap-weighted subsequence string kernel (n=4, lambda=0.5)",
        "params": {"n": 4, "lam": 0.5},
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


# Methods whose default parameters make them asymmetric: sim(a, b) != sim(b, a).
# Used by compute_similarity_matrix to decide whether the upper triangle may be
# mirrored into the lower triangle.  lingo_tversky and spectrum_tversky default to
# alpha=0.9, beta=0.1 (query-weighted); monge_elkan is asymmetric by construction
# (averages over smiles1's tokens). Their _sym / dice / overlap variants are
# symmetric and are not listed.
ASYMMETRIC_METHODS = {"lingo_tversky", "spectrum_tversky", "monge_elkan"}


def is_symmetric_method(method: str, kwargs: Optional[dict] = None) -> bool:
    """
    Best-effort test for whether ``method`` yields a symmetric similarity, i.e.
    ``sim(a, b) == sim(b, a)`` for all inputs.

    A method is treated as asymmetric if it is listed in :data:`ASYMMETRIC_METHODS`.
    Explicit Tversky weights passed via ``kwargs`` override that default: equal
    ``alpha``/``beta`` are symmetric, unequal are not. Likewise an explicit
    ``bidirectional`` kwarg (Monge-Elkan) overrides the default directly.
    """
    kwargs = kwargs or {}
    if "alpha" in kwargs and "beta" in kwargs:
        return kwargs["alpha"] == kwargs["beta"]
    if "bidirectional" in kwargs:
        return bool(kwargs["bidirectional"])
    return method not in ASYMMETRIC_METHODS


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


# TF-IDF method names follow ``tok-<family>[<merges>]_tfidf[<m><n>]``
# (e.g. tok-smiles_tfidf44, tok-schwaller_tfidf, tok-bpe512_tfidf12).  This anchored
# pattern is the single source of truth for the tokenizer family, replacing fragile
# ``"<family>" in method`` substring tests.
_TFIDF_FAMILY_RE = re.compile(r"tok-([a-z]+?)\d*_tfidf")


def _tfidf_family(method: str) -> Optional[str]:
    """Return the tokenizer family ('smiles'/'schwaller'/'bpe'/'selfies') of a
    TF-IDF method name, or ``None`` if *method* is not a TF-IDF method."""
    match = _TFIDF_FAMILY_RE.match(method)
    return match.group(1) if match else None


def _make_batch_tfidf_vectorizer(family: str, params: dict):
    """
    Build a single **unfitted** vectorizer for a TF-IDF method family.

    ``params`` supplies ``ngram_range`` / ``num_merges`` / ``q`` — these are baked
    into each method's registry entry (and its function lambda), so they must be
    read from the registry ``params``, not from the batch ``kwargs`` (which only
    carry ``preprocess``).  Returns ``None`` for an unknown family.
    """
    if family == "lingo":
        return LingoVectorizer(q=params.get("q", 4), use_idf=True)
    _tokenizer_factories = {
        "smiles": SMILESTokenizer,
        "schwaller": SMILESTokenizerSchwaller,
        "selfies": SELFIESTokenizer,
        "bpe": lambda: SMILESTokenizerBPE(num_merges=params.get("num_merges")),
    }
    factory = _tokenizer_factories.get(family)
    if factory is None:
        return None
    return TfidfVectorizer(
        tokenizer=factory(),
        analyzer="word",
        lowercase=False,
        token_pattern=None,
        ngram_range=params.get("ngram_range", (1, 2)),
        min_df=1,
        sublinear_tf=True,
    )


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

    # Preprocess each string once rather than once per pair.  Rebind rather than
    # mutate in place: the caller uses the returned list, so an in-place side
    # effect on the argument would be surprising and is not relied upon.
    if filtered.get("preprocess", True) and "preprocess" in params:
        corpus = [preprocess_smiles(s) for s in corpus]
        filtered = {**filtered, "preprocess": False}

    # For TF-IDF methods, fit ONE vectorizer on the full corpus so IDF weights
    # reflect the whole dataset rather than each individual pair.  The vectorizer
    # is built directly (no throwaway warm-up call) using the method's registered
    # ngram_range / num_merges, then passed to every pairwise call via vectorizer=.
    family = _tfidf_family(method)
    if family is not None and "vectorizer" not in filtered and SKLEARN_AVAILABLE:
        registry_params = dict(AVAILABLE_METHODS.get(method, {}).get("params", {}))
        for key in ("ngram_range", "num_merges", "q"):
            if key in kwargs:  # explicit user override wins over the registry default
                registry_params[key] = kwargs[key]
        try:
            vec = _make_batch_tfidf_vectorizer(family, registry_params)
            if vec is not None:
                vec.fit(corpus)
                filtered = {**filtered, "vectorizer": vec}
        except Exception:
            pass  # Fall back to per-pair fitting if anything goes wrong.

    return filtered, corpus


# ============================================================================
# Featurize-once batch acceleration
# ============================================================================
#
# The standalone similarity functions (e.g. ``lingo_similarity``) re-derive the
# per-string representation of BOTH arguments on every call.  In a matrix that
# re-featurizes each string O(N) or O(M) times.  The featurizers below split a
# feature-based method into two steps:
#
#     featurize(string) -> repr          # compute once per unique string
#     combine(repr, repr) -> float       # cheap pairwise reduction
#
# The batch helpers call ``featurize`` once per string and then only ``combine``
# per pair, which removes the redundant work while producing numerically
# identical results to the per-pair path.
#
# Featurizers assume the input string is already in its final normalized form:
# ``_build_batch_kwargs`` applies ``preprocess_smiles`` to the corpus once (and
# sets ``preprocess=False``), so featurizers must NOT preprocess again.  Methods
# not registered here (edit/nlcs/clcs DP, mismatch, jellyfish, TF-IDF) fall back
# to the per-pair path unchanged.


def _combine_lingo_sim(c1: Counter, c2: Counter) -> float:
    """Averaged per-q-gram agreement (mirrors :func:`lingo_similarity`)."""
    all_keys = set(c1) | set(c2)
    if not all_keys:
        return 1.0
    total = 0.0
    for lg in all_keys:
        n1 = c1.get(lg, 0)
        n2 = c2.get(lg, 0)
        denom = n1 + n2
        if denom > 0:
            total += 1.0 - abs(n1 - n2) / denom
    return total / len(all_keys)


def _make_combine_tversky(alpha: float, beta: float) -> Callable:
    """Multiset Tversky on q-gram counts (mirrors :func:`lingo_tversky_similarity`)."""

    def _combine(c1: Counter, c2: Counter) -> float:
        if not c1 and not c2:
            return 1.0
        if not c1 or not c2:
            return 0.0
        intersection = only1 = only2 = 0
        for k in set(c1) | set(c2):
            a = c1.get(k, 0)
            b = c2.get(k, 0)
            intersection += min(a, b)
            only1 += max(a - b, 0)
            only2 += max(b - a, 0)
        denom = intersection + alpha * only1 + beta * only2
        return intersection / denom if denom else 0.0

    return _combine


def _make_lingo_batch(params: dict) -> Tuple[Callable, Callable]:
    q = params.get("q", 4)

    def featurize(s: str) -> Counter:
        return get_lingos(s, q=q, normalize_rings=True, preprocess=False)

    return featurize, _combine_lingo_sim


def _make_lingo_binary_batch(params: dict, metric: str) -> Tuple[Callable, Callable]:
    """Binary (set) Jaccard/Dice on LINGO q-grams; mirrors :func:`lingo_jaccard_binary_similarity`/`lingo_dice_binary_similarity`."""
    q = params.get("q", 4)

    def featurize(s: str) -> frozenset:
        return frozenset(get_lingos(s, q=q, normalize_rings=True, preprocess=False))

    def combine(set1: frozenset, set2: frozenset) -> float:
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union else 0.0
        if metric == "jaccard":
            return jaccard
        return 2.0 * jaccard / (1.0 + jaccard)

    return featurize, combine


def _make_tversky_batch(params: dict, alpha: Optional[float] = None, beta: Optional[float] = None) -> Tuple[Callable, Callable]:
    q = params.get("q", 4)
    a = alpha if alpha is not None else params.get("alpha", 0.9)
    b = beta if beta is not None else params.get("beta", 0.1)

    def featurize(s: str) -> Counter:
        return get_lingos(s, q=q, normalize_rings=True, preprocess=False)

    return featurize, _make_combine_tversky(a, b)


def _make_spectrum_batch(params: dict) -> Tuple[Callable, Callable]:
    k = params.get("k", 4)
    coef = params.get("coefficient", "tanimoto").lower()
    alpha = params.get("alpha", 0.5)
    beta = params.get("beta", 0.5)

    def featurize(s: str):
        # Returns (string, kmer_counts_or_None, self_norm, total_count).  A
        # string shorter than k has no k-mers; the string is kept so the
        # degenerate branch can compare equality, exactly like
        # spectrum_kernel_similarity.  total_count (sum of raw counts, as
        # opposed to self_norm's sum of squares) is only needed by the
        # multiset-based 'overlap' coefficient.
        if len(s) < k:
            return (s, None, 0.0, 0)
        counts = Counter(s[i : i + k] for i in range(len(s) - k + 1))
        norm = sum(c * c for c in counts.values())
        total = sum(counts.values())
        return (s, counts, norm, total)

    def combine(f1, f2) -> float:
        s1, c1, n1, t1 = f1
        s2, c2, n2, t2 = f2
        if c1 is None and c2 is None:
            return 1.0 if s1 == s2 else 0.0
        if c1 is None or c2 is None:
            return 0.0

        if coef in ("tversky", "overlap"):
            intersection = only1 = only2 = 0
            for kmer in set(c1) | set(c2):
                a = c1.get(kmer, 0)
                b = c2.get(kmer, 0)
                intersection += min(a, b)
                only1 += max(a - b, 0)
                only2 += max(b - a, 0)
            if coef == "overlap":
                denom = min(t1, t2)
                return intersection / denom if denom > 0 else 0.0
            denom = intersection + alpha * only1 + beta * only2
            return intersection / denom if denom > 0 else 0.0

        # Inner product; iterate the smaller counter for speed (dot is exact).
        small, large = (c1, c2) if len(c1) <= len(c2) else (c2, c1)
        dot = 0.0
        for kmer, cc in small.items():
            other = large.get(kmer)
            if other:
                dot += cc * other
        if n1 == 0 or n2 == 0:
            return 0.0
        if coef == "cosine":
            return dot / (np.sqrt(n1) * np.sqrt(n2))
        if coef == "tanimoto":
            denom = n1 + n2 - dot
            return dot / denom if denom > 0 else 0.0
        if coef == "dice":
            denom = n1 + n2
            return 2.0 * dot / denom if denom > 0 else 0.0
        raise ValueError(f"Unknown coefficient: '{coef}'. Supported: 'tanimoto', 'dice', 'cosine', 'tversky', 'overlap'.")

    return featurize, combine


def _make_substring_batch(params: dict) -> Tuple[Callable, Callable]:
    min_length = params.get("min_length", 2)

    def featurize(s: str):
        freq = get_all_substrings(s, min_length)
        kself = sum(v * v for v in freq.values())
        return (freq, kself, len(s) == 0)

    def combine(f1, f2) -> float:
        freq1, k11, empty1 = f1
        freq2, k22, empty2 = f2
        if empty1 and empty2:
            return 1.0
        if k11 == 0 or k22 == 0:
            return 0.0
        small, large = (freq1, freq2) if len(freq1) <= len(freq2) else (freq2, freq1)
        k12 = 0
        for sub, cc in small.items():
            other = large.get(sub)
            if other:
                k12 += cc * other
        return k12 / np.sqrt(k11 * k22)

    return featurize, combine


def _make_smifp_batch(chars: List[str], metric: str) -> Tuple[Callable, Callable]:
    def featurize(s: str) -> np.ndarray:
        return smiles_to_fingerprint(s, chars)

    if metric == "tanimoto":

        def combine(fp1: np.ndarray, fp2: np.ndarray) -> float:
            dot = float(np.dot(fp1, fp2))
            n1 = float(np.dot(fp1, fp1))
            n2 = float(np.dot(fp2, fp2))
            denom = n1 + n2 - dot
            if denom == 0:
                return 1.0 if n1 == 0 and n2 == 0 else 0.0
            return dot / denom

    else:  # cityblock (Manhattan L1); identical to scipy.spatial.distance.cityblock

        def combine(fp1: np.ndarray, fp2: np.ndarray) -> float:
            cbd = float(np.abs(fp1 - fp2).sum())
            return 1.0 / (1.0 + cbd)

    return featurize, combine


def _make_ncd_batch(params: dict) -> Tuple[Callable, Callable]:
    sep = b"|"

    def featurize(s: str):
        data = s.encode("utf-8")
        return (s, data, _compress_bytes(data))

    def combine(f1, f2) -> float:
        s1, a, c_a = f1
        s2, b, c_b = f2
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0
        c_ab = _compress_bytes(a + sep + b)
        c_ba = _compress_bytes(b + sep + a)
        c_xy = min(c_ab, c_ba)
        denom = max(c_a, c_b)
        if denom == 0:
            return 1.0
        ncd = (c_xy - min(c_a, c_b)) / denom
        return max(0.0, min(1.0, 1.0 - ncd))

    return featurize, combine


def _make_subsequence_batch(params: dict) -> Tuple[Callable, Callable]:
    n = params.get("n", 3)
    lam = params.get("lam", 0.5)

    def featurize(s: str):
        # Cache the self-kernel K_n(s,s) once per string; the cross-kernel is
        # still computed per pair.  None marks a string shorter than n.
        if len(s) < n:
            return (s, None)
        return (s, _subsequence_kernel_raw(s, s, n, lam))

    def combine(f1, f2) -> float:
        s1, kself1 = f1
        s2, kself2 = f2
        if kself1 is None and kself2 is None:
            return 1.0 if s1 == s2 else 0.0
        if kself1 is None or kself2 is None:
            return 0.0
        if kself1 <= 0.0 or kself2 <= 0.0:
            return 0.0
        k12 = _subsequence_kernel_raw(s1, s2, n, lam)
        return k12 / np.sqrt(kself1 * kself2)

    return featurize, combine


# method name -> builder(params) -> (featurize, combine)
BATCH_FEATURIZERS: Dict[str, Callable[[dict], Tuple[Callable, Callable]]] = {
    "lingo": _make_lingo_batch,
    "lingo3": _make_lingo_batch,
    "lingo5": _make_lingo_batch,
    "lingo_tversky": _make_tversky_batch,
    "lingo_tversky_sym": lambda p: _make_tversky_batch(p, alpha=0.5, beta=0.5),
    "lingo_dice": lambda p: _make_tversky_batch(p, alpha=0.5, beta=0.5),
    "lingo_ruzicka": lambda p: _make_tversky_batch(p, alpha=1.0, beta=1.0),
    "lingo_jaccard_binary": lambda p: _make_lingo_binary_batch(p, metric="jaccard"),
    "lingo_dice_binary": lambda p: _make_lingo_binary_batch(p, metric="dice"),
    "spectrum": _make_spectrum_batch,
    "spectrum3": _make_spectrum_batch,
    "spectrum5": _make_spectrum_batch,
    "spectrum_cosine": _make_spectrum_batch,
    "spectrum_tversky": _make_spectrum_batch,
    "spectrum_tversky_sym": _make_spectrum_batch,
    "spectrum_overlap": _make_spectrum_batch,
    "substring": _make_substring_batch,
    "subsequence": _make_subsequence_batch,
    "subsequence2": _make_subsequence_batch,
    "subsequence4": _make_subsequence_batch,
    "smifp_tanimoto": lambda p: _make_smifp_batch(SMIFP_CHARS_34, "tanimoto"),
    "smifp38_tanimoto": lambda p: _make_smifp_batch(SMIFP_CHARS_38, "tanimoto"),
    "smifp_cbd": lambda p: _make_smifp_batch(SMIFP_CHARS_34, "cityblock"),
    "smifp38_cbd": lambda p: _make_smifp_batch(SMIFP_CHARS_38, "cityblock"),
    "ncd": _make_ncd_batch,
}

# User kwargs that legitimately override a method's feature parameters.
_BATCH_PARAM_KEYS = ("q", "k", "coefficient", "alpha", "beta", "min_length", "n", "lam")


def _resolve_batch_featurizer(method: str, kwargs: dict) -> Optional[Tuple[Callable, Callable]]:
    """
    Return ``(featurize, combine)`` for *method* if it supports the
    featurize-once fast path, else ``None``.

    Feature parameters come from the method's registry ``params`` entry,
    overlaid with any matching user kwargs so that explicit overrides
    (e.g. ``q=5``) are honoured exactly as the per-pair path would.
    """
    builder = BATCH_FEATURIZERS.get(method)
    if builder is None:
        return None
    params = dict(AVAILABLE_METHODS.get(method, {}).get("params", {}))
    for key in _BATCH_PARAM_KEYS:
        if key in kwargs:
            params[key] = kwargs[key]
    return builder(params)


def compute_similarity_matrix(smiles_list: List[str], method: str = "lingo", symmetric: Optional[bool] = None, **kwargs) -> np.ndarray:
    """
    Compute pairwise similarity matrix for a list of SMILES.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings
    method : str
        Similarity method name
    symmetric : bool or None
        Whether ``sim(a, b) == sim(b, a)`` for this method.  When ``None``
        (default) it is inferred with :func:`is_symmetric_method`.  For a
        symmetric method only the upper triangle is computed and mirrored;
        for an asymmetric method (e.g. query-weighted ``lingo_tversky``)
        both off-diagonal cells are computed independently.
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

    if symmetric is None:
        symmetric = is_symmetric_method(method, kwargs)

    sim_func = get_similarity_function(method)  # also enforces optional-dependency checks
    smiles_list = list(smiles_list)

    # Fast path: featurize each string once, then only combine per pair.  The
    # featurizer owns its normalization (preprocess each string once here), so
    # it does not depend on _build_batch_kwargs mutating the corpus.
    batch = _resolve_batch_featurizer(method, kwargs)
    if batch is not None:
        featurize, combine = batch
        prep = preprocess_smiles if kwargs.get("preprocess", True) else (lambda x: x)
        feats = [featurize(prep(s)) for s in smiles_list]
        for i in range(n):
            sim_matrix[i, i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                sim_matrix[i, j] = combine(feats[i], feats[j])
                sim_matrix[j, i] = sim_matrix[i, j] if symmetric else combine(feats[j], feats[i])
        return sim_matrix

    # General path: per-pair evaluation, corpus preprocessed once up front.
    filtered_kwargs, smiles_list = _build_batch_kwargs(sim_func, method, smiles_list, kwargs)

    # TF-IDF fast path: when _build_batch_kwargs fit one shared vectorizer (all
    # tok-*_tfidf* methods, including tok-bpe*), transform the whole corpus in
    # one batched call and get the full similarity matrix via a single
    # cosine_similarity matrix multiply, instead of the per-pair loop below
    # calling sim_func() -> vectorizer.transform([s]) once per pair each string
    # appears in.  Each document's TF-IDF vector depends only on itself and the
    # already-fitted IDF weights, so results are numerically identical to the
    # per-pair path; this only removes the redundant retokenization (for BPE
    # tokenization in particular, retokenizing on every pair turned into
    # multi-day runtimes on real-sized datasets).
    vectorizer = filtered_kwargs.get("vectorizer")
    if vectorizer is not None and _tfidf_family(method) is not None:
        vecs = vectorizer.transform(smiles_list)
        result = sklearn_cosine_similarity(vecs, vecs)
        # The per-pair path force-sets self-similarity to 1.0 unconditionally
        # (see the loop below); match that for short strings whose n-gram range
        # yields an all-zero TF-IDF vector (e.g. ngram_range=(4,4) on a 1-token
        # molecule), where cosine(0-vector, 0-vector) is otherwise 0.0, not 1.0.
        np.fill_diagonal(result, 1.0)
        return result

    for i in range(n):
        sim_matrix[i, i] = 1.0  # Self-similarity
        for j in range(i + 1, n):
            sim_matrix[i, j] = sim_func(smiles_list[i], smiles_list[j], **filtered_kwargs)
            if symmetric:
                sim_matrix[j, i] = sim_matrix[i, j]
            else:
                sim_matrix[j, i] = sim_func(smiles_list[j], smiles_list[i], **filtered_kwargs)

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

    sim_func = get_similarity_function(method)  # also enforces optional-dependency checks
    templates = list(templates)
    library = list(library)

    # Fast path: featurize each template and library string exactly once (the
    # whole point — avoid re-deriving library features once per template and
    # vice versa).  The featurizer owns its normalization here.
    batch = _resolve_batch_featurizer(method, kwargs)
    if batch is not None:
        featurize, combine = batch
        prep = preprocess_smiles if kwargs.get("preprocess", True) else (lambda x: x)
        tfeats = [featurize(prep(t)) for t in templates]
        lfeats = [featurize(prep(lib)) for lib in library]
        for i in range(n_lib):
            lf = lfeats[i]
            for j in range(n_templates):
                # Argument order matches sim_func(template, lib): for asymmetric
                # methods (query-weighted Tversky) the template is the query, per
                # lingo_tversky_similarity's own docstring and the paper it cites.
                sim_matrix[i, j] = combine(tfeats[j], lf)
        return sim_matrix

    # General path: per-pair evaluation, corpus preprocessed once up front.
    corpus = templates + library
    filtered_kwargs, corpus = _build_batch_kwargs(sim_func, method, corpus, kwargs)
    templates = corpus[:n_templates]
    library = corpus[n_templates:]

    # TF-IDF fast path: see the matching block in compute_similarity_matrix for
    # the full rationale. Transform templates and library in two batched calls
    # and get the whole n_lib x n_templates grid via one cosine_similarity
    # matrix multiply, instead of the O(n_lib * n_templates) per-pair loop below
    # retokenizing each string once per pair it appears in.
    vectorizer = filtered_kwargs.get("vectorizer")
    if vectorizer is not None and _tfidf_family(method) is not None:
        template_vecs = vectorizer.transform(templates)
        library_vecs = vectorizer.transform(library)
        return sklearn_cosine_similarity(library_vecs, template_vecs)

    for i, lib_smiles in enumerate(library):
        for j, template_smiles in enumerate(templates):
            # template is the query (first arg) for asymmetric methods; see the
            # matching comment in the fast path above.
            sim = sim_func(template_smiles, lib_smiles, **filtered_kwargs)
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
    # .smi/.smiles files carry no header; column defaults are the same either way.
    if ext in [".smi", ".smiles"]:
        header = False
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
    Write similarity matrix to CSV or CSV.gz file.

    Output format:
    Name,Similarity_{template1},Similarity_{template2},...

    Parameters
    ----------
    output_path : str
        Output file path. If it ends with '.gz', written as gzip-compressed CSV.
    library_names : List[str]
        Names of library molecules (rows)
    template_names : List[str]
        Names of template molecules (columns)
    sim_matrix : np.ndarray
        Similarity matrix (library x templates)
    """
    data = {"Name": library_names}
    for j, template_name in enumerate(template_names):
        data[f"Similarity_{template_name}"] = sim_matrix[:, j]

    df = pd.DataFrame(data)
    compression = "gzip" if str(output_path).endswith(".gz") else None
    df.to_csv(output_path, index=False, float_format="%.5f", compression=compression)


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
                   lingo_tversky, lingo_tversky_sym, lingo_dice, lingo_ruzicka,
                   lingo_jaccard_binary, lingo_dice_binary,
                   spectrum, spectrum3, spectrum5, spectrum_cosine,
                   spectrum_tversky, spectrum_tversky_sym, spectrum_overlap,
                   mismatch, mismatch3, mismatch5, lcs_substring, token_edit,
                   monge_elkan, monge_elkan_sym,
                   subsequence, subsequence2, subsequence4,
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
        "--methods",
        nargs="+",
        choices=list(AVAILABLE_METHODS.keys()),
        metavar="METHOD",
        help="Run a specific subset of methods (same output naming as --all-methods). Useful when some methods are invalid for the current representation.",
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
        help=("Compute fingerprints instead of similarities. " "TYPE is one of: " + ", ".join(AVAILABLE_FINGERPRINTS.keys())),
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
        multi_output = True
    elif args.methods:
        methods_to_run = args.methods
        multi_output = True
    else:
        methods_to_run = [args.method]
        multi_output = False

    for method in methods_to_run:
        if multi_output:
            output_path = Path(args.output)
            # stem is e.g. "smiles__replaced" or "inchi_all__" (trailing __ when no mods)
            # Strip both .csv.gz and .csv suffixes to get the bare variant stem.
            stem = output_path.name
            for suffix in (".csv.gz", ".csv"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            stem = stem.rstrip("_")
            method_output = output_path.parent / f"{stem}__{method}.csv.gz"
        else:
            method_output = args.output

        if args.verbose:
            if multi_output:
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
            if multi_output:
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
        if multi_output:
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
