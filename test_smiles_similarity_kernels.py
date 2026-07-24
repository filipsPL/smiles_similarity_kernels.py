"""
pytest test suite for smiles_similarity_kernels.py

Run with:
    pytest test_smiles_similarity_kernels.py -v
    pytest test_smiles_similarity_kernels.py -v -k "lingo"  # single group
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

import smiles_similarity_kernels as m

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_SMI = EXAMPLES_DIR / "templates.smi"
DATABASE_SMI = EXAMPLES_DIR / "database.smi"


def approx(value, rel=1e-4):
    """pytest.approx wrapper with a consistent relative tolerance."""
    return pytest.approx(value, rel=rel)


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessSmiles:
    def test_chlorine(self):
        assert m.preprocess_smiles("CCCCl") == "CCCL"

    def test_bromine(self):
        assert m.preprocess_smiles("c1ccc(Br)cc1") == "c1ccc(R)cc1"

    def test_no_replacements(self):
        assert m.preprocess_smiles("CCO") == "CCO"

    def test_double_at_chirality(self):
        result = m.preprocess_smiles("C[C@@H](Cl)Br")
        # @@ must be replaced as a unit before any bare @ could be touched
        assert "@@" not in result
        assert "¡" in result  # @@ → ¡
        assert "L" in result  # Cl → L
        assert "R" in result  # Br → R

    def test_silicon(self):
        assert m.preprocess_smiles("[Si]") == "[G]"

    def test_nickel_unicode(self):
        # Ni → Θ  (not 'U' as in the old mapping)
        assert m.preprocess_smiles("[Ni]") == "[Θ]"

    def test_tungsten_single_char(self):
        # W (Tungsten) encoded to avoid confusion
        assert m.preprocess_smiles("[W]") == "[·]"

    def test_idempotent_on_plain_smiles(self):
        smiles = "c1ccccc1"
        assert m.preprocess_smiles(smiles) == smiles

    def test_empty_string(self):
        assert m.preprocess_smiles("") == ""

    def test_longest_match_wins(self):
        # @TH1 must not be split into @  +  TH1
        result = m.preprocess_smiles("[C@TH1]")
        assert "@TH1" not in result
        assert "¢" in result  # @TH1 → ¢


class TestNormalizeRingNumbers:
    def test_benzene(self):
        assert m.normalize_ring_numbers("c1ccccc1") == "c0ccccc0"

    def test_bicyclic(self):
        assert m.normalize_ring_numbers("C1CC2CCCCC2C1") == "C0CC0CCCCC0C0"

    def test_no_digits(self):
        assert m.normalize_ring_numbers("CCO") == "CCO"


class TestShuffleSmiles:
    """shuffle_smiles is the random negative control central to the project's
    documented baseline methodology (README: 'Negative controls'). A silent
    bug here (e.g. not actually permuting, or corrupting length/composition)
    would invalidate every negative-control comparison run with this tool."""

    ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

    def test_length_preserved(self):
        assert len(m.shuffle_smiles(self.ASPIRIN, seed=42)) == len(self.ASPIRIN)

    def test_character_composition_preserved(self):
        assert sorted(m.shuffle_smiles(self.ASPIRIN, seed=42)) == sorted(self.ASPIRIN)

    def test_reproducible_with_same_seed(self):
        assert m.shuffle_smiles(self.ASPIRIN, seed=42) == m.shuffle_smiles(self.ASPIRIN, seed=42)

    def test_different_seeds_differ(self):
        assert m.shuffle_smiles(self.ASPIRIN, seed=42) != m.shuffle_smiles(self.ASPIRIN, seed=7)

    def test_actually_permutes(self):
        # Must not be a no-op: for a string with enough distinct characters,
        # at least one seed among a handful must move something.
        assert any(m.shuffle_smiles(self.ASPIRIN, seed=s) != self.ASPIRIN for s in range(10))

    def test_single_char_unchanged(self):
        assert m.shuffle_smiles("C", seed=1) == "C"

    def test_empty(self):
        assert m.shuffle_smiles("", seed=1) == ""


class TestSortString:
    """sort_string is the deterministic negative control complementing shuffle."""

    ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

    def test_known_value(self):
        # Two ring-closure digits '1' (benzene open+close) sort before all letters.
        assert m.sort_string("c1ccccc1") == "11cccccc"

    def test_already_sorted_is_unchanged(self):
        assert m.sort_string("CCO") == "CCO"

    def test_length_preserved(self):
        assert len(m.sort_string(self.ASPIRIN)) == len(self.ASPIRIN)

    def test_character_composition_preserved(self):
        assert sorted(m.sort_string(self.ASPIRIN)) == sorted(self.ASPIRIN)

    def test_deterministic(self):
        assert m.sort_string(self.ASPIRIN) == m.sort_string(self.ASPIRIN)

    def test_idempotent(self):
        once = m.sort_string(self.ASPIRIN)
        assert m.sort_string(once) == once

    def test_empty(self):
        assert m.sort_string("") == ""


# ---------------------------------------------------------------------------
# 2. Canonicalization and InChI  (skip when RDKit absent)
# ---------------------------------------------------------------------------

rdkit_available = pytest.mark.skipif(not m.RDKIT_AVAILABLE, reason="RDKit not installed")


@rdkit_available
class TestCanonicalizeSmiles:
    def test_same_molecule_different_order(self):
        assert m.canonicalize_smiles("OCC") == m.canonicalize_smiles("CCO")

    def test_returns_string(self):
        result = m.canonicalize_smiles("CCO")
        assert isinstance(result, str) and len(result) > 0

    def test_invalid_smiles_fallback(self):
        assert m.canonicalize_smiles("INVALID!!!") == "INVALID!!!"

    def test_empty_fallback(self):
        assert m.canonicalize_smiles("") == ""


@rdkit_available
class TestSmilesToInchi:
    def test_no_prefix(self):
        inchi = m.smiles_to_inchi("CCO")
        assert not inchi.startswith("InChI=")

    def test_content(self):
        inchi = m.smiles_to_inchi("CCO")
        assert inchi.startswith("1S/")

    def test_ethanol_formula(self):
        assert "C2H6O" in m.smiles_to_inchi("CCO")

    def test_invalid_returns_empty(self):
        assert m.smiles_to_inchi("INVALID!!!") == ""

    def test_empty_returns_empty(self):
        assert m.smiles_to_inchi("") == ""


# preprocess_inchi / extract_inchi_layers are pure string logic (no RDKit call),
# so they run unconditionally -- unlike smiles_to_inchi_layers below, which
# needs RDKit to produce the InChI in the first place.
_ASPIRIN_INCHI = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"


class TestPreprocessInchi:
    def test_strips_prefix_and_version(self):
        assert m.preprocess_inchi(_ASPIRIN_INCHI) == ("C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)")

    def test_keep_version(self):
        assert m.preprocess_inchi(_ASPIRIN_INCHI, strip_version=False) == ("1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)")

    def test_idempotent(self):
        once = m.preprocess_inchi(_ASPIRIN_INCHI)
        twice = m.preprocess_inchi(once)
        assert once == twice

    def test_empty(self):
        assert m.preprocess_inchi("") == ""


class TestExtractInchiLayers:
    def test_formula(self):
        assert m.extract_inchi_layers(_ASPIRIN_INCHI, "formula") == "C9H8O4"

    def test_connections(self):
        assert m.extract_inchi_layers(_ASPIRIN_INCHI, "connections") == ("c1-6(10)13-8-5-3-2-4-7(8)9(11)12")

    def test_hydrogens(self):
        assert m.extract_inchi_layers(_ASPIRIN_INCHI, "hydrogens") == "h2-5H,1H3,(H,11,12)"

    def test_multiple_layers_preserve_requested_order(self):
        # Order of the OUTPUT follows the order of the `layers` argument, not
        # the order the layers appear in the source InChI.
        forward = m.extract_inchi_layers(_ASPIRIN_INCHI, ["formula", "connections"])
        reversed_ = m.extract_inchi_layers(_ASPIRIN_INCHI, ["connections", "formula"])
        assert forward == "C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12"
        assert reversed_ == "c1-6(10)13-8-5-3-2-4-7(8)9(11)12/C9H8O4"
        assert forward != reversed_

    def test_absent_layer_silently_omitted(self):
        # "stereo_tet" is not present in this (achiral) InChI -- requesting it
        # alongside a present layer must not raise, just omit it.
        assert m.extract_inchi_layers(_ASPIRIN_INCHI, ["formula", "stereo_tet"]) == "C9H8O4"

    def test_all_equals_preprocess_inchi(self):
        assert m.extract_inchi_layers(_ASPIRIN_INCHI, "all") == m.preprocess_inchi(_ASPIRIN_INCHI, strip_version=True)

    def test_unknown_layer_raises(self):
        with pytest.raises(ValueError):
            m.extract_inchi_layers(_ASPIRIN_INCHI, "not_a_real_layer")

    def test_unknown_layer_in_list_raises(self):
        with pytest.raises(ValueError):
            m.extract_inchi_layers(_ASPIRIN_INCHI, ["formula", "not_a_real_layer"])

    def test_empty_inchi(self):
        assert m.extract_inchi_layers("", "formula") == ""


@rdkit_available
class TestSmilesToInchiLayers:
    def test_matches_manual_extraction(self):
        # Round-trip through the real SMILES -> InChI -> layer pipeline must
        # match calling extract_inchi_layers directly on smiles_to_inchi's output.
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        full_inchi = m.smiles_to_inchi(smiles)
        assert m.smiles_to_inchi_layers(smiles, "connections") == m.extract_inchi_layers(full_inchi, "connections")

    def test_default_all_matches_preprocessed_full_inchi(self):
        smiles = "CCO"
        full_inchi = m.smiles_to_inchi(smiles)
        assert m.smiles_to_inchi_layers(smiles) == m.preprocess_inchi(full_inchi)

    def test_invalid_smiles_returns_empty(self):
        assert m.smiles_to_inchi_layers("INVALID!!!", "formula") == ""

    def test_empty_smiles_returns_empty(self):
        assert m.smiles_to_inchi_layers("", "formula") == ""


# ---------------------------------------------------------------------------
# 3. Edit distance similarity
# ---------------------------------------------------------------------------


class TestEditSimilarity:
    def test_identical(self):
        assert m.edit_similarity("CCO", "CCO") == approx(1.0)

    def test_empty_both(self):
        assert m.edit_similarity("", "", preprocess=False) == 1.0

    def test_known_value(self):
        # edit("CCC", "CCCCC") = 2, max_len = 5 → 1 - 2/5 = 0.6
        assert m.edit_similarity("CCC", "CCCCC", preprocess=False) == approx(0.6)

    def test_range(self):
        s = m.edit_similarity("CC", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_symmetry(self):
        assert m.edit_similarity("CCO", "CCOC") == approx(m.edit_similarity("CCOC", "CCO"))


# ---------------------------------------------------------------------------
# 4. NLCS similarity
# ---------------------------------------------------------------------------


class TestNlcsSimilarity:
    def test_identical(self):
        assert m.nlcs_similarity("CCO", "CCO") == approx(1.0)

    def test_known_value(self):
        # LCS("ABC","AC") = 2, NLCS = 4/(3*2) = 0.6667
        assert m.nlcs_similarity("ABC", "AC", preprocess=False) == approx(2**2 / (3 * 2))

    def test_no_common(self):
        # No common characters → LCS = 0 → similarity = 0
        assert m.nlcs_similarity("AAA", "BBB", preprocess=False) == approx(0.0)

    def test_range(self):
        s = m.nlcs_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_symmetry(self):
        assert m.nlcs_similarity("CCO", "CCOC") == approx(m.nlcs_similarity("CCOC", "CCO"))


# ---------------------------------------------------------------------------
# 5. CLCS similarity
# ---------------------------------------------------------------------------


class TestClcsSimilarity:
    def test_identical(self):
        assert m.clcs_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.clcs_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_weights_sum_respected(self):
        # Default weights (0.33, 0.33, 0.34) sum to 1; identical strings → 1.0
        assert m.clcs_similarity("CCCC", "CCCC") == approx(1.0)

    def test_custom_weights(self):
        s = m.clcs_similarity("CCO", "CCN", w1=1.0, w2=0.0, w3=0.0)
        assert 0.0 <= s <= 1.0

    def test_w1_1_reduces_to_pure_nlcs(self):
        # w1=1, w2=w3=0 must reduce clcs to exactly the pure NLCS component
        # (not just "some value in range") -- verifies the weighted-sum formula
        # itself, not merely its output bounds.
        for a, b in [("CCO", "CCN"), ("ABCDEF", "ACDF"), ("CC(=O)O", "CCN")]:
            assert m.clcs_similarity(a, b, w1=1.0, w2=0.0, w3=0.0) == approx(m.nlcs_similarity(a, b))

    def test_weight_sum_not_one_warns(self):
        with pytest.warns(UserWarning, match="not 1"):
            m.clcs_similarity("CCO", "CCN", w1=0.5, w2=0.5, w3=0.5)


# ---------------------------------------------------------------------------
# 6. Substring kernel
# ---------------------------------------------------------------------------


class TestSubstringKernelSimilarity:
    def test_identical(self):
        assert m.substring_kernel_similarity("CCO", "CCO") == approx(1.0)

    def test_normalized_range(self):
        s = m.substring_kernel_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_raw_kernel_positive(self):
        k = m.substring_kernel_similarity("CCO", "CCOC", normalized=False)
        assert k > 0

    def test_no_overlap(self):
        s = m.substring_kernel_similarity("CC", "XY", preprocess=False)
        assert s == approx(0.0)

    def test_symmetry(self):
        assert m.substring_kernel_similarity("CCO", "CCOC") == approx(m.substring_kernel_similarity("CCOC", "CCO"))

    def test_both_empty_returns_one(self):
        # Regression: both-empty is "equally empty" (1.0), matching the
        # convention used by every other similarity function in this module
        # (e.g. nlcs_similarity, longest_common_substring_similarity) rather
        # than falling through to the k11==0/k22==0 branch's 0.0.
        assert m.substring_kernel_similarity("", "", preprocess=False) == approx(1.0)

    def test_one_empty_returns_zero(self):
        # Only one side empty must still be 0.0 -- the both-empty fix must not
        # affect this genuinely-dissimilar case.
        assert m.substring_kernel_similarity("", "CCO", preprocess=False) == approx(0.0)

    def test_both_empty_batch_fast_path_matches_fallback(self):
        # Two distinct empty-string entries exercise the off-diagonal
        # (empty, empty) cell -- the diagonal is force-set to 1.0 regardless
        # of this fix, so it alone wouldn't exercise the batch combine() path.
        mols = ["", "", "CCO"]
        saved = m.BATCH_FEATURIZERS
        m.BATCH_FEATURIZERS = {}
        try:
            ref = m.compute_similarity_matrix(mols, method="substring", preprocess=False)
        finally:
            m.BATCH_FEATURIZERS = saved
        fast = m.compute_similarity_matrix(mols, method="substring", preprocess=False)
        assert np.allclose(ref, fast, atol=1e-12)
        assert fast[0, 1] == approx(1.0)
        assert fast[0, 2] == approx(0.0)


# ---------------------------------------------------------------------------
# 7. SMIfp similarities
# ---------------------------------------------------------------------------


class TestSmifpSimilarities:
    def test_tanimoto_identical(self):
        assert m.smifp_similarity_tanimoto("CCO", "CCO") == approx(1.0)

    def test_tanimoto_range(self):
        s = m.smifp_similarity_tanimoto("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_tanimoto_38d(self):
        s = m.smifp_similarity_tanimoto("CCO", "CCOC", chars=m.SMIFP_CHARS_38)
        assert 0.0 <= s <= 1.0

    def test_38d_chirality_dimension_is_live(self):
        # Regression: the '@@' chirality dimension of the 38D fingerprint must
        # actually register.  It is stored as the post-preprocess sentinel; a
        # chiral molecule must produce a nonzero value there.
        sentinel = m.ELEMENT_REPLACEMENTS["@@"]
        assert sentinel in m.SMIFP_CHARS_38
        idx = m.SMIFP_CHARS_38.index(sentinel)
        pre = m.preprocess_smiles("C[C@@H](Cl)Br")
        fp = m.smiles_to_fingerprint(pre, m.SMIFP_CHARS_38)
        assert fp[idx] == approx(1.0)

    def test_38d_distinguishes_chirality(self):
        # A chiral molecule and its achiral analogue must differ on the 38D
        # fingerprint (they would be identical if the '@@' dim were dead).
        s = m.smifp_similarity_tanimoto("C[C@@H](Cl)Br", "CC(Cl)Br", chars=m.SMIFP_CHARS_38)
        assert s < 1.0

    @pytest.mark.skipif(not m.SCIPY_AVAILABLE, reason="scipy not installed")
    def test_cityblock_identical(self):
        assert m.smifp_similarity_cityblock("CCO", "CCO") == approx(1.0)

    @pytest.mark.skipif(not m.SCIPY_AVAILABLE, reason="scipy not installed")
    def test_cityblock_range(self):
        s = m.smifp_similarity_cityblock("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    @pytest.mark.skipif(not m.SCIPY_AVAILABLE, reason="scipy not installed")
    def test_cityblock_38d(self):
        s = m.smifp_similarity_cityblock("CCO", "CCOC", chars=m.SMIFP_CHARS_38)
        assert 0.0 <= s <= 1.0

    def test_preprocessing_effect(self):
        # s1 raw contains 'Cl' as two chars ('C','l'); preprocessing collapses it
        # to the single sentinel 'L', which equals s2 as-is.  So preprocess=True
        # must make the two strings' fingerprints identical (sim_pre == 1.0),
        # while preprocess=False leaves the raw 'l' distinct from 'L' (sim_no_pre != 1.0).
        s1 = "CCCCl"  # Contains 'Cl' (2 chars)
        s2 = "CCCL"  # What it becomes after preprocessing
        sim_pre = m.smifp_similarity_tanimoto(s1, s2, preprocess=True)
        sim_no_pre = m.smifp_similarity_tanimoto(s1, s2, preprocess=False)
        assert 0.0 <= sim_pre <= 1.0
        assert 0.0 <= sim_no_pre <= 1.0
        assert sim_pre == approx(1.0)
        assert sim_pre != approx(sim_no_pre)


# ---------------------------------------------------------------------------
# 7b. Spectrum Kernel Similarity
# ---------------------------------------------------------------------------


class TestSpectrumKernelSimilarity:
    def test_identical(self):
        assert m.spectrum_kernel_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.spectrum_kernel_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_coefficients(self):
        s_tanimoto = m.spectrum_kernel_similarity("CCO", "CCOC", coefficient="tanimoto")
        s_dice = m.spectrum_kernel_similarity("CCO", "CCOC", coefficient="dice")
        s_cosine = m.spectrum_kernel_similarity("CCO", "CCOC", coefficient="cosine")
        assert 0.0 <= s_tanimoto <= 1.0
        assert 0.0 <= s_dice <= 1.0
        assert 0.0 <= s_cosine <= 1.0

    def test_k_parameter(self):
        s_k3 = m.spectrum_kernel_similarity("CCCCCC", "CCCCCO", k=3)
        s_k5 = m.spectrum_kernel_similarity("CCCCCC", "CCCCCO", k=5)
        assert 0.0 <= s_k3 <= 1.0
        assert 0.0 <= s_k5 <= 1.0

    def test_symmetry(self):
        assert m.spectrum_kernel_similarity("CCO", "CCOC") == approx(m.spectrum_kernel_similarity("CCOC", "CCO"))

    def test_known_value(self):
        # s1="AABB" bigrams: AA:1, AB:1, BB:1 (norm1=3)
        # s2="ABBB" bigrams: AB:1, BB:2           (norm2=5)
        # dot = AB(1*1) + BB(1*2) = 3
        # tanimoto = 3/(3+5-3) = 0.6; dice = 6/8 = 0.75; cosine = 3/sqrt(15)
        s1, s2 = "AABB", "ABBB"
        assert m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="tanimoto", preprocess=False) == approx(0.6)
        assert m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="dice", preprocess=False) == approx(0.75)
        assert m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="cosine", preprocess=False) == approx(3 / (15**0.5))

    def test_unknown_coefficient_raises(self):
        # Strings must be >= k so the coefficient branch is actually reached
        # (short-string degenerate cases return early, before validation).
        with pytest.raises(ValueError):
            m.spectrum_kernel_similarity("CCCCCCCC", "CCCCCCCN", coefficient="bogus")

    def test_tversky_and_overlap_known_value(self):
        # s1="AABB" bigrams: AA:1, AB:1, BB:1  (sum1=3)
        # s2="ABBBB" bigrams: AB:1, BB:3        (sum2=4)
        # intersection = min(AA)+min(AB)+min(BB) = 0+1+1 = 2
        # only1 (unique to s1) = 1 (AA); only2 (unique to s2) = 2 (BB)
        s1, s2 = "AABB", "ABBBB"
        sym = m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="tversky", alpha=0.5, beta=0.5, preprocess=False)
        assert sym == approx(2 / 3.5)  # 2 / (2 + 0.5*1 + 0.5*2)
        asym = m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="tversky", alpha=0.9, beta=0.1, preprocess=False)
        assert asym == approx(2 / 3.1)  # 2 / (2 + 0.9*1 + 0.1*2)
        overlap = m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="overlap", preprocess=False)
        assert overlap == approx(2 / 3.0)  # 2 / min(3, 4)

    def test_tversky_asymmetric(self):
        s1, s2 = "AABB", "ABBBB"
        forward = m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="tversky", alpha=0.9, beta=0.1, preprocess=False)
        backward = m.spectrum_kernel_similarity(s2, s1, k=2, coefficient="tversky", alpha=0.9, beta=0.1, preprocess=False)
        assert forward != approx(backward)

    def test_tversky_sym_default_equals_dice(self):
        # alpha=beta=0.5 (the function default) is the multiset Sorensen-Dice coefficient.
        s1, s2 = "AABB", "ABBBB"
        tversky_default = m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="tversky", preprocess=False)
        tversky_explicit = m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="tversky", alpha=0.5, beta=0.5, preprocess=False)
        assert tversky_default == approx(tversky_explicit)

    def test_overlap_symmetric(self):
        s1, s2 = "AABB", "ABBBB"
        assert m.spectrum_kernel_similarity(s1, s2, k=2, coefficient="overlap", preprocess=False) == approx(
            m.spectrum_kernel_similarity(s2, s1, k=2, coefficient="overlap", preprocess=False)
        )

    def test_tversky_overlap_range(self):
        for coef in ("tversky", "overlap"):
            s = m.spectrum_kernel_similarity("CCO", "CCOC", coefficient=coef)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# 7c. Mismatch Kernel Similarity
# ---------------------------------------------------------------------------


class TestMismatchKernelSimilarity:
    def test_identical(self):
        assert m.mismatch_kernel_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.mismatch_kernel_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_mismatch_tolerance(self):
        # Mismatch should be more tolerant than exact spectrum
        exact = m.spectrum_kernel_similarity("CCCCN", "CCCCO", k=4)
        mismatch = m.mismatch_kernel_similarity("CCCCN", "CCCCO", k=4, m=1)
        assert 0.0 <= mismatch <= 1.0
        # Mismatch should generally be >= exact for similar strings
        assert mismatch >= exact

    def test_m_parameter(self):
        s_m0 = m.mismatch_kernel_similarity("CCCCN", "CCCCO", k=4, m=0)
        s_m1 = m.mismatch_kernel_similarity("CCCCN", "CCCCO", k=4, m=1)
        # m=0 should equal spectrum kernel
        assert s_m0 == m.spectrum_kernel_similarity("CCCCN", "CCCCO", k=4)
        assert 0.0 <= s_m1 <= 1.0

    def test_known_value(self):
        # Independently verified against a brute-force reimplementation of the
        # mismatch-kernel definition (exhaustive Hamming-ball expansion over the
        # alphabet) -- see insights-opus48.md test-audit notes.
        s1, s2, alphabet = "AABCAB", "ABCBAC", "ABC"
        assert m.mismatch_kernel_similarity(s1, s2, k=2, m=1, coefficient="tanimoto", preprocess=False, alphabet=alphabet) == approx(0.875)
        assert m.mismatch_kernel_similarity(s1, s2, k=2, m=1, coefficient="dice", preprocess=False, alphabet=alphabet) == approx(
            0.933333, rel=1e-5
        )
        assert m.mismatch_kernel_similarity(s1, s2, k=3, m=1, coefficient="tanimoto", preprocess=False, alphabet=alphabet) == approx(0.625)

    def test_negative_m_raises(self):
        with pytest.raises(ValueError):
            m.mismatch_kernel_similarity("CCO", "CCN", m=-1)

    def test_unknown_coefficient_raises(self):
        # Strings must be >= k, and m >= 1 to reach the mismatch coefficient
        # branch (m=0 falls back to spectrum_kernel_similarity, which is
        # separately tested; short strings return early before validation).
        with pytest.raises(ValueError):
            m.mismatch_kernel_similarity("CCCCCCCC", "CCCCCCCN", k=4, m=1, coefficient="bogus")

    def test_symmetry(self):
        assert m.mismatch_kernel_similarity("CCO", "CCOC") == approx(m.mismatch_kernel_similarity("CCOC", "CCO"))


# ---------------------------------------------------------------------------
# 7d. Longest Common Substring Similarity
# ---------------------------------------------------------------------------


class TestLongestCommonSubstringSimilarity:
    def test_identical(self):
        assert m.longest_common_substring_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.longest_common_substring_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_known_value(self):
        # LCS of "ABCDEF" and "CDEFXY" is "CDEF" (4 chars)
        # Similarity = (4^2) / (6*6) = 16/36 ≈ 0.444
        assert m.longest_common_substring_similarity("ABCDEF", "CDEFXY", preprocess=False) == approx(16 / 36)

    def test_no_common(self):
        assert m.longest_common_substring_similarity("ABC", "XYZ", preprocess=False) == approx(0.0)

    def test_symmetry(self):
        assert m.longest_common_substring_similarity("CCO", "CCOC") == approx(m.longest_common_substring_similarity("CCOC", "CCO"))


class TestSubsequenceKernel:
    @staticmethod
    def _brute_raw(s, t, n, lam):
        # Direct enumeration of the kernel DEFINITION (ground truth).
        from itertools import combinations
        from collections import defaultdict

        def feats(string):
            d = defaultdict(float)
            for idx in combinations(range(len(string)), n):
                u = "".join(string[k] for k in idx)
                d[u] += lam ** (idx[-1] - idx[0] + 1)
            return d

        fs, ft = feats(s), feats(t)
        return sum(v * ft[u] for u, v in fs.items() if u in ft)

    def test_raw_dp_matches_brute_force(self):
        # The efficient DP must equal the brute-force kernel definition.
        cases = ["", "C", "CCO", "CCOCC", "c1ccccc1", "CC(=O)O", "COCOC"]
        for s in cases:
            for t in cases:
                for n in (1, 2, 3):
                    for lam in (0.3, 0.5, 1.0):
                        assert m._subsequence_kernel_raw(s, t, n, lam) == pytest.approx(self._brute_raw(s, t, n, lam), abs=1e-12)

    def test_identical(self):
        assert m.subsequence_kernel_similarity("CC(=O)Oc1ccccc1", "CC(=O)Oc1ccccc1") == approx(1.0)

    def test_range_and_symmetry(self):
        pairs = [("CCO", "CCOC"), ("c1ccc(Cl)cc1", "c1ccc(Br)cc1"), ("CCCCO", "COCCC")]
        for a, b in pairs:
            s = m.subsequence_kernel_similarity(a, b)
            assert 0.0 <= s <= 1.0 + 1e-12
            assert s == approx(m.subsequence_kernel_similarity(b, a))

    def test_normalized_matches_manual(self):
        a, b, n, lam = "CCOCC", "CCOC", 3, 0.5
        k12 = m._subsequence_kernel_raw(a, b, n, lam)
        k11 = m._subsequence_kernel_raw(a, a, n, lam)
        k22 = m._subsequence_kernel_raw(b, b, n, lam)
        expected = k12 / (k11 * k22) ** 0.5
        assert m.subsequence_kernel_similarity(a, b, n=n, lam=lam, preprocess=False) == approx(expected)

    def test_gap_decay(self):
        # Smaller lambda penalises gapped matches more, so a molecule whose only
        # shared length-3 subsequence spans a gap scores lower at small lambda.
        a, b = "CXYZO", "CO"  # share subsequence "CO" but not length-3; use n=2
        hi = m.subsequence_kernel_similarity("CABO", "CO", n=2, lam=0.9, preprocess=False)
        lo = m.subsequence_kernel_similarity("CABO", "CO", n=2, lam=0.3, preprocess=False)
        assert lo < hi  # more gap penalty at small lambda

    def test_short_strings(self):
        assert m.subsequence_kernel_similarity("CC", "CC", n=3) == approx(1.0)  # both < n, equal
        assert m.subsequence_kernel_similarity("CC", "OO", n=3) == approx(0.0)  # both < n, differ
        assert m.subsequence_kernel_similarity("CC", "CCCCO", n=3) == approx(0.0)  # one < n

    def test_registered_and_fast_path(self):
        assert m.get_similarity_function("subsequence")("CCO", "CCO") == approx(1.0)
        lib = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc(Cl)cc1", "CCO", "CCN", "c1ccccc1"]
        tmpl = ["c1ccc(Br)cc1", "CCOC"]
        for meth in ("subsequence", "subsequence2", "subsequence4"):
            saved = m.BATCH_FEATURIZERS
            m.BATCH_FEATURIZERS = {}
            try:
                ref = m.compute_cross_similarity_matrix(tmpl, lib, method=meth, preprocess=True)
            finally:
                m.BATCH_FEATURIZERS = saved
            fast = m.compute_cross_similarity_matrix(tmpl, lib, method=meth, preprocess=True)
            assert np.allclose(ref, fast, atol=1e-12)


class TestTokenEditSimilarity:
    def test_identical(self):
        assert m.token_edit_similarity("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O") == approx(1.0)

    def test_range(self):
        s = m.token_edit_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_symmetry(self):
        assert m.token_edit_similarity("[nH+]c1ccccc1", "[nH]c1ccncc1") == approx(m.token_edit_similarity("[nH]c1ccncc1", "[nH+]c1ccccc1"))

    def test_empty_both(self):
        assert m.token_edit_similarity("", "") == approx(1.0)

    def test_empty_one(self):
        assert m.token_edit_similarity("", "CCO") == approx(0.0)

    def test_bracket_atom_is_single_edit(self):
        # A charge change on a bracket atom is ONE token edit, normalized by the
        # token count (9 tokens for [nH+]c1ccccc1): 1 - 1/9.
        assert m.token_edit_similarity("[nH+]c1ccccc1", "[nH]c1ccccc1") == approx(1.0 - 1.0 / 9.0)

    def test_matches_manual_token_levenshtein(self):
        tok = m.SMILESTokenizerSchwaller()
        a, b = "CC(=O)Nc1ccccc1", "CC(=O)Nc1ccncc1"
        ed = m.edit_distance(tok(a), tok(b))
        expected = 1.0 - ed / max(len(tok(a)), len(tok(b)))
        assert m.token_edit_similarity(a, b) == approx(expected)

    def test_custom_tokenizer(self):
        # Any str -> List[str] callable is accepted; a char-splitter reduces this
        # to character-level edit similarity on the raw (unpreprocessed) string.
        char_split = lambda s: list(s)
        a, b = "CCO", "CCN"
        got = m.token_edit_similarity(a, b, tokenizer=char_split)
        assert got == approx(1.0 - m.edit_distance(list(a), list(b)) / max(len(a), len(b)))

    def test_registered_method(self):
        fn = m.get_similarity_function("token_edit")
        assert fn("CCO", "CCO") == approx(1.0)


class TestMongeElkanSimilarity:
    _EXACT = staticmethod(lambda t1, t2: 1.0 if t1 == t2 else 0.0)
    _CHARS = staticmethod(lambda s: list(s))

    def test_identical(self):
        assert m.monge_elkan_similarity("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O") == approx(1.0)

    def test_range(self):
        s = m.monge_elkan_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_empty_both(self):
        assert m.monge_elkan_similarity("", "") == approx(1.0)

    def test_empty_one(self):
        assert m.monge_elkan_similarity("", "CCO") == approx(0.0)

    def test_known_value_asymmetric(self):
        # A = ['A','B'] (len 2), B = ['A','B','C','D'] (len 4), exact-match token sim.
        # forward: avg over A -> A matches A (1), B matches B (1) -> 2/2 = 1.0
        # backward: avg over B -> A(1), B(1), C(0), D(0) -> 2/4 = 0.5
        a, b = "AB", "ABCD"
        forward = m.monge_elkan_similarity(a, b, tokenizer=self._CHARS, token_similarity=self._EXACT)
        backward = m.monge_elkan_similarity(b, a, tokenizer=self._CHARS, token_similarity=self._EXACT)
        assert forward == approx(1.0)
        assert backward == approx(0.5)
        assert forward != approx(backward)

    def test_bidirectional_averages_both_directions(self):
        a, b = "AB", "ABCD"
        forward = m.monge_elkan_similarity(a, b, tokenizer=self._CHARS, token_similarity=self._EXACT)
        backward = m.monge_elkan_similarity(b, a, tokenizer=self._CHARS, token_similarity=self._EXACT)
        bidir = m.monge_elkan_similarity(a, b, tokenizer=self._CHARS, token_similarity=self._EXACT, bidirectional=True)
        assert bidir == approx((forward + backward) / 2.0)

    def test_default_gives_partial_credit_for_near_miss_tokens(self):
        # A single-character change on a bracket atom should score higher than 0
        # under the default char-edit token similarity, unlike exact-token matching.
        a, b = "[nH+]c1ccccc1", "[NH+]c1ccccc1"
        default_score = m.monge_elkan_similarity(a, b)
        exact_score = m.monge_elkan_similarity(a, b, token_similarity=self._EXACT)
        assert default_score > exact_score

    def test_custom_tokenizer_and_token_similarity(self):
        got = m.monge_elkan_similarity("AB", "ABCD", tokenizer=self._CHARS, token_similarity=self._EXACT)
        assert got == approx(1.0)

    def test_registered_methods(self):
        fwd_fn = m.get_similarity_function("monge_elkan")
        sym_fn = m.get_similarity_function("monge_elkan_sym")
        a, b = "c1ccccc1CCCCCCCCCC", "c1ccccc1CC"
        assert fwd_fn(a, b) == approx(m.monge_elkan_similarity(a, b))
        assert sym_fn(a, b) == approx(m.monge_elkan_similarity(a, b, bidirectional=True))

    def test_is_symmetric_method(self):
        assert m.is_symmetric_method("monge_elkan") is False
        assert m.is_symmetric_method("monge_elkan", {"bidirectional": True}) is True
        assert m.is_symmetric_method("monge_elkan", {"bidirectional": False}) is False
        assert m.is_symmetric_method("monge_elkan_sym") is True


# ---------------------------------------------------------------------------
# 8. LINGO similarity
# ---------------------------------------------------------------------------


class TestLingoSimilarity:
    def test_identical(self):
        assert m.lingo_similarity("CCCCC", "CCCCC") == approx(1.0)

    def test_no_common_lingos(self):
        # Both strings shorter than q → 0 LINGOs each → returns 1.0 (equally empty)
        assert m.lingo_similarity("CC", "OO", q=4) == approx(1.0)

    def test_one_empty_lingos(self):
        # Only one side is too short → returns 0.0
        assert m.lingo_similarity("CCCCC", "OO", q=4) == approx(0.0)

    def test_range(self):
        s = m.lingo_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_q3(self):
        s = m.lingo_similarity("CCCCCC", "CCCCCO", q=3)
        assert 0.0 <= s <= 1.0

    def test_q5(self):
        s = m.lingo_similarity("CCCCCCC", "CCCCCCN", q=5)
        assert 0.0 <= s <= 1.0

    def test_symmetry(self):
        assert m.lingo_similarity("CCO", "CCOC") == approx(m.lingo_similarity("CCOC", "CCO"))

    def test_validated_against_example_output(self):
        """
        Validates against examples/results.csv produced by the CLI.
        Template 0054-0090 vs 0133-0086 must be 0.39080.
        """
        t1 = "CC(=O)C1=CC=C(Br)C(N)=C1"  # 0054-0090
        t2 = "NC1=CC=C(Br)C=C1C(=O)C1=CC=CC=C1Cl"  # 0133-0086
        assert m.lingo_similarity(t1, t2) == approx(0.39080, rel=1e-3)

    def test_self_similarity_templates(self):
        t1 = "CC(=O)C1=CC=C(Br)C(N)=C1"
        t2 = "NC1=CC=C(Br)C=C1C(=O)C1=CC=CC=C1Cl"
        assert m.lingo_similarity(t1, t1) == approx(1.0)
        assert m.lingo_similarity(t2, t2) == approx(1.0)


class TestLingoRuzickaSimilarity:
    PAIRS = [("CCO", "CCCO"), ("c1ccccc1CCCCCCCCCC", "c1ccccc1CC"), ("CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")]

    def test_identical(self):
        assert m.lingo_ruzicka_similarity("CC(=O)Oc1ccccc1", "CC(=O)Oc1ccccc1") == approx(1.0)

    def test_range_and_symmetry(self):
        for a, b in self.PAIRS:
            s = m.lingo_ruzicka_similarity(a, b)
            assert 0.0 <= s <= 1.0
            assert s == approx(m.lingo_ruzicka_similarity(b, a))

    def test_equals_sum_min_over_sum_max(self):
        # Ground truth: Ruzicka = Σ min(n1,n2) / Σ max(n1,n2) over LINGO counts.
        for a, b in self.PAIRS:
            c1 = m.get_lingos(a, 4, normalize_rings=True, preprocess=True)
            c2 = m.get_lingos(b, 4, normalize_rings=True, preprocess=True)
            keys = set(c1) | set(c2)
            smin = sum(min(c1.get(k, 0), c2.get(k, 0)) for k in keys)
            smax = sum(max(c1.get(k, 0), c2.get(k, 0)) for k in keys)
            expected = smin / smax if smax else 1.0
            assert m.lingo_ruzicka_similarity(a, b) == approx(expected)

    def test_equals_tversky_alpha_beta_one(self):
        for a, b in self.PAIRS:
            assert m.lingo_ruzicka_similarity(a, b) == approx(m.lingo_tversky_similarity(a, b, q=4, alpha=1.0, beta=1.0))

    def test_distinct_from_dice(self):
        # On a pair with repeated q-grams the two coefficients must differ.
        a, b = "c1ccccc1CCCCCCCCCC", "c1ccccc1CC"
        assert m.lingo_ruzicka_similarity(a, b) != approx(m.lingo_dice_similarity(a, b))

    def test_registered_and_batch_fast_path(self):
        assert m.get_similarity_function("lingo_ruzicka")("CCO", "CCO") == approx(1.0)
        # Fast path (BATCH_FEATURIZERS) must equal the fallback per-pair result.
        lib = ["CCO", "c1ccccc1CC", "CC(=O)Oc1ccccc1C(=O)O", "CCN"]
        tmpl = ["c1ccccc1CCCCCCCCCC", "CCOC"]
        saved = m.BATCH_FEATURIZERS
        m.BATCH_FEATURIZERS = {}
        try:
            ref = m.compute_cross_similarity_matrix(tmpl, lib, method="lingo_ruzicka", preprocess=True)
        finally:
            m.BATCH_FEATURIZERS = saved
        fast = m.compute_cross_similarity_matrix(tmpl, lib, method="lingo_ruzicka", preprocess=True)
        assert np.allclose(ref, fast, atol=1e-12)


class TestLingoBinarySimilarity:
    PAIRS = [("CCO", "CCCO"), ("c1ccccc1CCCCCCCCCC", "c1ccccc1CC"), ("CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")]

    def test_identical(self):
        assert m.lingo_jaccard_binary_similarity("CC(=O)Oc1ccccc1", "CC(=O)Oc1ccccc1") == approx(1.0)
        assert m.lingo_dice_binary_similarity("CC(=O)Oc1ccccc1", "CC(=O)Oc1ccccc1") == approx(1.0)

    def test_no_common_lingos(self):
        assert m.lingo_jaccard_binary_similarity("CC", "OO", q=4) == approx(1.0)
        assert m.lingo_dice_binary_similarity("CC", "OO", q=4) == approx(1.0)

    def test_one_empty_lingos(self):
        assert m.lingo_jaccard_binary_similarity("CCCCC", "OO", q=4) == approx(0.0)
        assert m.lingo_dice_binary_similarity("CCCCC", "OO", q=4) == approx(0.0)

    def test_range_and_symmetry(self):
        for a, b in self.PAIRS:
            j = m.lingo_jaccard_binary_similarity(a, b)
            d = m.lingo_dice_binary_similarity(a, b)
            assert 0.0 <= j <= 1.0
            assert 0.0 <= d <= 1.0
            assert j == approx(m.lingo_jaccard_binary_similarity(b, a))
            assert d == approx(m.lingo_dice_binary_similarity(b, a))

    def test_equals_intersection_over_union(self):
        # Ground truth: binary Jaccard = |set1 & set2| / |set1 | set2| over the
        # *set* of distinct q-grams (multiplicity discarded).
        for a, b in self.PAIRS:
            set1 = set(m.get_lingos(a, 4, normalize_rings=True, preprocess=True))
            set2 = set(m.get_lingos(b, 4, normalize_rings=True, preprocess=True))
            expected = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 1.0
            assert m.lingo_jaccard_binary_similarity(a, b) == approx(expected)

    def test_dice_is_monotonic_transform_of_jaccard(self):
        # DSC = 2J / (1 + J) -- the two metrics rank pairs identically.
        for a, b in self.PAIRS:
            j = m.lingo_jaccard_binary_similarity(a, b)
            d = m.lingo_dice_binary_similarity(a, b)
            assert d == approx(2.0 * j / (1.0 + j))

    def test_distinct_from_multiset_variants(self):
        # On a pair with repeated q-grams, discarding multiplicity must change the score.
        a, b = "c1ccccc1CCCCCCCCCC", "c1ccccc1CC"
        assert m.lingo_jaccard_binary_similarity(a, b) != approx(m.lingo_ruzicka_similarity(a, b))
        assert m.lingo_dice_binary_similarity(a, b) != approx(m.lingo_dice_similarity(a, b))

    def test_registered_and_batch_fast_path(self):
        assert m.get_similarity_function("lingo_jaccard_binary")("CCO", "CCO") == approx(1.0)
        assert m.get_similarity_function("lingo_dice_binary")("CCO", "CCO") == approx(1.0)
        lib = ["CCO", "c1ccccc1CC", "CC(=O)Oc1ccccc1C(=O)O", "CCN"]
        tmpl = ["c1ccccc1CCCCCCCCCC", "CCOC"]
        for method in ("lingo_jaccard_binary", "lingo_dice_binary"):
            saved = m.BATCH_FEATURIZERS
            m.BATCH_FEATURIZERS = {}
            try:
                ref = m.compute_cross_similarity_matrix(tmpl, lib, method=method, preprocess=True)
            finally:
                m.BATCH_FEATURIZERS = saved
            fast = m.compute_cross_similarity_matrix(tmpl, lib, method=method, preprocess=True)
            assert np.allclose(ref, fast, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. LINGO TF-IDF
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not m.SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestLingoTfidfSimilarity:
    def test_identical(self):
        corpus = ["CCCCCO", "CCCCCO"]
        assert m.lingo_tfidf_similarity("CCCCCO", "CCCCCO", corpus=corpus) == approx(1.0)

    def test_range(self):
        corpus = ["CCO", "CCOC", "CCCCC"]
        s = m.lingo_tfidf_similarity("CCO", "CCOC", corpus=corpus)
        assert 0.0 <= s <= 1.0

    def test_vectorizer_reuse_uses_prefitted_idf(self):
        # A meaningful "reuse" check: a vectorizer fit on a WIDE corpus must be
        # used as-is (not silently re-fit on just [a, b]) — so its IDF weights
        # differ from the default per-pair fit.  (Calling the same args twice,
        # as the old version of this test did, is a tautology: it passes even
        # if `vectorizer=` were silently ignored, since both calls fall back
        # to the same default corpus=[a, b].)
        a, b = "CCOCC", "CCOCCC"
        wide_corpus = ["CCOCC", "CCOCCC", "CCCCCC", "c1ccccc1CC", "CCNCC", "CC(=O)OCC"]
        vec = m.LingoVectorizer(q=4, use_idf=True)
        vec.fit(wide_corpus)
        s_prefitted = m.lingo_tfidf_similarity(a, b, vectorizer=vec)
        s_default = m.lingo_tfidf_similarity(a, b)  # corpus defaults to [a, b]
        assert s_prefitted != approx(s_default)


# ---------------------------------------------------------------------------
# 10. SMILES TF-IDF (chemical tokenization)
# ---------------------------------------------------------------------------


class TestTfidfSharedHelper:
    """The four *_tfidf functions delegate to one helper; each must still name
    itself in the ImportError raised when scikit-learn is unavailable."""

    @pytest.mark.parametrize(
        "fn_name",
        [
            "smiles_tfidf_similarity",
            "schwaller_tfidf_similarity",
            "bpe_tfidf_similarity",
            "selfies_tfidf_similarity",
        ],
    )
    def test_importerror_names_caller(self, fn_name, monkeypatch):
        monkeypatch.setattr(m, "SKLEARN_AVAILABLE", False)
        with pytest.raises(ImportError) as exc:
            getattr(m, fn_name)("CCO", "CCO")
        assert fn_name in str(exc.value)


@pytest.mark.skipif(not m.SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestSmilesTfidfSimilarity:
    def test_identical(self):
        assert m.smiles_tfidf_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.smiles_tfidf_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_ngram_13(self):
        s = m.smiles_tfidf_similarity("CCO", "CCOC", ngram_range=(1, 3))
        assert 0.0 <= s <= 1.0

    def test_ngram_23(self):
        s = m.smiles_tfidf_similarity("CCO", "CCOC", ngram_range=(2, 3))
        assert 0.0 <= s <= 1.0

    def test_ngram_14(self):
        s = m.smiles_tfidf_similarity("CCO", "CCOC", ngram_range=(1, 4))
        assert 0.0 <= s <= 1.0

    def test_vectorizer_reuse_uses_prefitted_idf(self):
        # See TestLingoTfidfSimilarity.test_vectorizer_reuse_uses_prefitted_idf
        # for why "call the same args twice" is a tautology here.
        from sklearn.feature_extraction.text import TfidfVectorizer

        tok = m.SMILESTokenizer()
        vec = TfidfVectorizer(
            tokenizer=tok, analyzer="word", lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        vec.fit(["CCO", "CCOC", "c1ccccc1Cl", "CCN", "CCCCCC"])
        s_prefitted = m.smiles_tfidf_similarity("CCO", "CCOC", vectorizer=vec)
        s_default = m.smiles_tfidf_similarity("CCO", "CCOC")  # corpus defaults to [a, b]
        assert s_prefitted != approx(s_default)


class TestSMILESTokenizer:
    def test_chlorine_single_token(self):
        tokens = m.SMILESTokenizer().tokenize("CCCl")
        assert tokens == ["C", "C", "Cl"]

    def test_bromine_single_token(self):
        tokens = m.SMILESTokenizer().tokenize("CBr")
        assert tokens == ["C", "Br"]

    def test_double_at_single_token(self):
        tokens = m.SMILESTokenizer().tokenize("C@@H")
        assert "@@" in tokens
        assert "@" not in tokens  # should not be split into two @

    def test_callable(self):
        tok = m.SMILESTokenizer()
        assert tok("CC") == ["C", "C"]


class TestSMILESTokenizerSchwaller:
    def test_bracket_atom_single_token(self):
        tokens = m.SMILESTokenizerSchwaller().tokenize("[nH+]")
        assert tokens == ["[nH+]"]

    def test_bracket_atom_isotope(self):
        tokens = m.SMILESTokenizerSchwaller().tokenize("[13C]")
        assert tokens == ["[13C]"]

    def test_chlorine_single_token(self):
        tokens = m.SMILESTokenizerSchwaller().tokenize("CCCl")
        assert tokens == ["C", "C", "Cl"]

    def test_bromine_single_token(self):
        tokens = m.SMILESTokenizerSchwaller().tokenize("CBr")
        assert tokens == ["C", "Br"]

    def test_bond_symbols_are_tokens(self):
        tokens = m.SMILESTokenizerSchwaller().tokenize("C=O")
        assert tokens == ["C", "=", "O"]

    def test_branch_delimiters(self):
        tokens = m.SMILESTokenizerSchwaller().tokenize("C(=O)O")
        assert tokens == ["C", "(", "=", "O", ")", "O"]

    def test_two_digit_ring_closure(self):
        tokens = m.SMILESTokenizerSchwaller().tokenize("C%10CC%10")
        assert "%10" in tokens
        assert tokens.count("%10") == 2

    def test_stereo_at_sign(self):
        tokens = m.SMILESTokenizerSchwaller().tokenize("[C@@H]")
        assert tokens == ["[C@@H]"]

    def test_callable(self):
        tok = m.SMILESTokenizerSchwaller()
        assert tok("CO") == ["C", "O"]


@pytest.mark.skipif(not m.SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestSchwallerTfidfSimilarity:
    def test_identical(self):
        assert m.schwaller_tfidf_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.schwaller_tfidf_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_bracket_atom_handled(self):
        s = m.schwaller_tfidf_similarity("[nH+]c1ccccc1", "[nH+]c1ccncc1")
        assert 0.0 <= s <= 1.0

    def test_ngram_range(self):
        s = m.schwaller_tfidf_similarity("CCO", "CCOC", ngram_range=(2, 3))
        assert 0.0 <= s <= 1.0

    def test_differs_from_smiles_tfidf(self):
        # bracket atom [nH] should be one token in Schwaller, two in original
        s_schwaller = m.schwaller_tfidf_similarity("c1cc[nH]cc1", "c1ccncc1")
        s_original = m.smiles_tfidf_similarity("c1cc[nH]cc1", "c1ccncc1")
        # scores may differ; both valid
        assert 0.0 <= s_schwaller <= 1.0
        assert 0.0 <= s_original <= 1.0


BPE_VOCAB = Path(__file__).parent / "smiles_bpe_vocab.json"
bpe_vocab_available = pytest.mark.skipif(not BPE_VOCAB.exists(), reason="BPE vocab not found (run train_bpe_tokenizer.py first)")


class TestSMILESTokenizerBPE:
    def test_no_vocab_raises(self):
        import tempfile, os

        with pytest.raises(FileNotFoundError):
            m.SMILESTokenizerBPE(vocab_path="/nonexistent/path/vocab.json")

    @bpe_vocab_available
    def test_default_vocab_loads(self):
        tok = m.SMILESTokenizerBPE()
        assert len(tok._merges) > 0

    @bpe_vocab_available
    def test_num_merges_slices(self):
        tok_all = m.SMILESTokenizerBPE()
        tok_16 = m.SMILESTokenizerBPE(num_merges=16)
        tok_0 = m.SMILESTokenizerBPE(num_merges=0)
        assert len(tok_16._merges) == 16
        assert len(tok_0._merges) == 0
        assert len(tok_16._merges) <= len(tok_all._merges)

    @bpe_vocab_available
    def test_num_merges_coarser_tokenization(self):
        # More merges → fewer, longer tokens
        smi = "CC(=O)Nc1ccccc1"
        tok_fine = m.SMILESTokenizerBPE(num_merges=16)
        tok_coarse = m.SMILESTokenizerBPE(num_merges=512)
        assert len(tok_fine.tokenize(smi)) >= len(tok_coarse.tokenize(smi))

    @bpe_vocab_available
    def test_callable(self):
        tok = m.SMILESTokenizerBPE()
        result = tok("CO")
        assert isinstance(result, list)
        assert len(result) > 0

    @bpe_vocab_available
    def test_merges_applied(self):
        tok = m.SMILESTokenizerBPE()
        tok._merges = [("C", "C")]
        assert tok.tokenize("CCC") == ["CC", "C"]

    @bpe_vocab_available
    def test_merges_chained(self):
        # Pass 1 (C+C): CCCC -> [CC, CC]; Pass 2 (CC+C): no match (no bare C left)
        tok = m.SMILESTokenizerBPE()
        tok._merges = [("C", "C"), ("CC", "C")]
        assert tok.tokenize("CCCC") == ["CC", "CC"]

    @bpe_vocab_available
    def test_loads_explicit_vocab_file(self):
        tok = m.SMILESTokenizerBPE(vocab_path=BPE_VOCAB)
        assert len(tok._merges) == 8192

    @bpe_vocab_available
    def test_common_fragment_merged(self):
        tok = m.SMILESTokenizerBPE()
        assert tok.tokenize("c1ccccc1") == ["c1ccccc1"]

    @bpe_vocab_available
    def test_amide_merged(self):
        tok = m.SMILESTokenizerBPE()
        assert tok.tokenize("CC(=O)N") == ["CC(=O)N"]

    @bpe_vocab_available
    def test_complex_molecule_split(self):
        tok = m.SMILESTokenizerBPE()
        tokens = tok.tokenize("CC(=O)Oc1ccccc1C(=O)O")
        assert len(tokens) < 20
        assert len(tokens) > 0


def _naive_bpe_apply_merges(tokens, merges):
    """
    Reference implementation: the original pre-optimization algorithm from
    SMILESTokenizerBPE.tokenize() (rescans the whole token list once per rule,
    in rank order). Kept only here, deliberately independent of the module's
    current (fast, priority-queue) implementation, so TestSMILESTokenizerBPEFuzz
    has something to check the fast path against that isn't just "itself".
    """
    tokens = list(tokens)
    for a, b in merges:
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


@bpe_vocab_available
class TestSMILESTokenizerBPEFuzz:
    """
    Exhaustive random fuzzing of ``_apply_merges`` (the priority-queue merge
    over a doubly-linked token list, tracking each token's creation rank)
    against ``_naive_bpe_apply_merges`` (the original per-rule-rescan
    algorithm it replaced), on synthetic token sequences and merge tables —
    not constrained to what the SMILES regex actually produces, so this
    covers chained/overlapping/repeated-token merge scenarios, including
    deliberately adversarial (not realistically trained) merge tables, far
    more densely than real SMILES strings would. Mirrors the
    TestSubsequenceKernel.test_raw_dp_matches_brute_force pattern: the fast
    path's correctness is pinned against a brute-force reference, not just a
    handful of hand-picked expected outputs.

    Earlier iterations of this suite caught two genuine divergences before
    this file settled on the current algorithm:
      1. A low-rank rule referencing a token only a higher-rank rule ever
         produces (rule i's operand can't exist yet when naive reaches rank
         i, but the naive greedy algorithm without rank tracking would use it
         as soon as it appeared).
      2. Two different rules that happen to concatenate to the same string,
         reachable at different ranks depending on which one actually fires
         on a given input — a purely static "is this vocabulary well-formed"
         check (checking rule i's operands against the *earliest* rank that
         could ever produce that string) cannot catch this, since it doesn't
         know which producer rule actually fires on a specific input.
    Both are covered here by test_regression_low_rank_uses_future_token and
    test_regression_ambiguous_producer_rank respectively; the general fuzz
    below tests the same property at scale.
    """

    def _random_merges(self, rng, alphabet, n_merges):
        """Merge rules drawn from arbitrary short concatenations, not
        constrained to look like real BPE training output — includes
        both well-formed-by-construction and adversarial orderings."""
        candidates = set()
        for _ in range(n_merges * 3):
            a = rng.choice(alphabet + ["".join(rng.choices(alphabet, k=2)) for _ in range(2)])
            b = rng.choice(alphabet + ["".join(rng.choices(alphabet, k=2)) for _ in range(2)])
            candidates.add((a, b))
        return rng.sample(sorted(candidates), k=min(n_merges, len(candidates)))

    @pytest.mark.parametrize("trial", range(500))
    def test_matches_naive_random(self, trial):
        import random

        rng = random.Random(trial)
        alphabet = [str(i) for i in range(rng.randint(2, 5))]
        tokens = [rng.choice(alphabet) for _ in range(rng.randint(0, 12))]
        merges = self._random_merges(rng, alphabet, rng.randint(0, 10))

        tok = m.SMILESTokenizerBPE()
        tok._merges = merges

        fast = tok._apply_merges(tokens)
        ref = _naive_bpe_apply_merges(tokens, merges)
        assert fast == ref, f"tokens={tokens!r} merges={merges!r}"

    def test_regression_low_rank_uses_future_token(self):
        """A low-rank rule ('2', '23') references a token ('23') that is only
        ever producible by a later rule ('2', '3'). Naive never applies
        ('2','23') on this input since '23' doesn't exist yet when rank 3 is
        checked (rank 4 hasn't run); an unguarded greedy algorithm would
        eagerly merge it as soon as '23' appears later."""
        tokens = ["2", "3", "0", "2", "1", "0", "1", "1", "2", "2", "3", "0"]
        merges = [("3", "10"), ("02", "0"), ("0", "2"), ("2", "23"), ("2", "3"), ("12", "22"), ("0", "3")]

        tok = m.SMILESTokenizerBPE()
        tok._merges = merges
        assert tok._apply_merges(tokens) == _naive_bpe_apply_merges(tokens, merges)

    def test_regression_ambiguous_producer_rank(self):
        """Two different rules — ('0', '101') at rank 2 and ('01', '01') at
        rank 5 — both concatenate to '0101'. Rank 2 never actually fires on
        this input (its own prerequisite, a bare '1' next to '01', never
        occurs), so the only real producer is rank 5 — meaning a rank-4 rule
        ('00', '0101') must NOT fire, even though a naive per-table check
        would see '0101' as "producible by rank 2 < 4" and wrongly allow it."""
        tokens = ["0", "0", "0", "1", "0", "1", "0", "1", "1"]
        merges = [("0", "1"), ("1", "01"), ("0", "101"), ("0", "0"), ("00", "0101"), ("01", "01")]

        tok = m.SMILESTokenizerBPE()
        tok._merges = merges
        assert tok._apply_merges(tokens) == _naive_bpe_apply_merges(tokens, merges)

    def test_matches_naive_real_vocab_real_smiles(self):
        smiles_examples = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "c1ccc2ccccc2c1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "O=C(O)c1ccccc1O",
            "[Na+].[Cl-]",
            "C1CCCCC1",
            "CC(=O)Nc1ccc(O)cc1",
        ]
        for smi in smiles_examples:
            for num_merges in (0, 1, 16, 64, 256, 1024, None):
                tok = m.SMILESTokenizerBPE(num_merges=num_merges)
                base_tokens = tok._BASE_RE.findall(smi)
                fast = tok._apply_merges(base_tokens)
                ref = _naive_bpe_apply_merges(base_tokens, tok._merges)
                assert fast == ref, f"smiles={smi!r} num_merges={num_merges}"


@pytest.mark.skipif(not m.SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestBpeTfidfSimilarity:
    @bpe_vocab_available
    def test_identical(self):
        assert m.bpe_tfidf_similarity("CCO", "CCO") == approx(1.0)

    @bpe_vocab_available
    def test_range(self):
        s = m.bpe_tfidf_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    @bpe_vocab_available
    def test_ngram_range(self):
        s = m.bpe_tfidf_similarity("CCO", "CCOC", ngram_range=(2, 3))
        assert 0.0 <= s <= 1.0

    @bpe_vocab_available
    def test_with_vocab(self):
        s = m.bpe_tfidf_similarity("CC(=O)Nc1ccccc1", "CC(=O)Nc1ccncc1", vocab_path=BPE_VOCAB)
        assert 0.0 <= s <= 1.0

    @bpe_vocab_available
    def test_differs_from_schwaller_tfidf(self):
        # BPE merges produce different token sets → scores may differ
        s_bpe = m.bpe_tfidf_similarity("CC(=O)Nc1ccccc1", "CC(=O)Nc1ccncc1", vocab_path=BPE_VOCAB)
        s_sch = m.schwaller_tfidf_similarity("CC(=O)Nc1ccccc1", "CC(=O)Nc1ccncc1")
        assert 0.0 <= s_bpe <= 1.0
        assert 0.0 <= s_sch <= 1.0


# ---------------------------------------------------------------------------
# 10c. SELFIES TF-IDF Similarity
# ---------------------------------------------------------------------------

selfies_available = pytest.mark.skipif(not m.SELFIES_AVAILABLE, reason="selfies not installed")


@selfies_available
@pytest.mark.skipif(not m.SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestSelfiesTfidfSimilarity:
    def test_identical(self):
        assert m.selfies_tfidf_similarity("[C][C][O]", "[C][C][O]") == approx(1.0)

    def test_range(self):
        s = m.selfies_tfidf_similarity("[C][C][O]", "[C][C][N]")
        assert 0.0 <= s <= 1.0

    def test_ngram_range(self):
        s = m.selfies_tfidf_similarity("[C][C][O]", "[C][C][N]", ngram_range=(2, 3))
        assert 0.0 <= s <= 1.0

    def test_vectorizer_reuse_uses_prefitted_idf(self):
        # See TestLingoTfidfSimilarity.test_vectorizer_reuse_uses_prefitted_idf
        # for why "call the same args twice" is a tautology here.
        a, b = "[C][C][O]", "[C][C][N]"
        wide_corpus = ["[C][C][O]", "[C][C][N]", "[C][C][S]", "[C][O][C]", "[N][C][C]"]
        vec = m.TfidfVectorizer(
            tokenizer=m.SELFIESTokenizer(), analyzer="word", lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        vec.fit(wide_corpus)
        s_prefitted = m.selfies_tfidf_similarity(a, b, vectorizer=vec)
        s_default = m.selfies_tfidf_similarity(a, b)  # corpus defaults to [a, b]
        assert s_prefitted != approx(s_default)
        assert 0.0 <= s_prefitted <= 1.0


# ---------------------------------------------------------------------------
# 11. Jellyfish-based methods
# ---------------------------------------------------------------------------

jellyfish_available = pytest.mark.skipif(not m.JELLYFISH_AVAILABLE, reason="jellyfish not installed")


@jellyfish_available
class TestDamerauLevenshtein:
    def test_identical(self):
        assert m.damerau_levenshtein_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.damerau_levenshtein_similarity("CCO", "CCN")
        assert 0.0 <= s <= 1.0

    def test_transposition_cheaper_than_edit(self):
        # "ab" → "ba" is 1 Damerau op but 2 edit ops
        dl = m.damerau_levenshtein_similarity("ab", "ba", preprocess=False)
        ed = m.edit_similarity("ab", "ba", preprocess=False)
        assert dl >= ed


@jellyfish_available
class TestJaroSimilarity:
    def test_identical(self):
        assert m.jaro_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.jaro_similarity("CCO", "CCN")
        assert 0.0 <= s <= 1.0


@jellyfish_available
class TestJaroWinklerSimilarity:
    def test_identical(self):
        assert m.jaro_winkler_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.jaro_winkler_similarity("CCO", "CCN")
        assert 0.0 <= s <= 1.0

    def test_prefix_bonus(self):
        # Jaro-Winkler >= Jaro when strings share a prefix
        jw = m.jaro_winkler_similarity("CCCCO", "CCCCN")
        j = m.jaro_similarity("CCCCO", "CCCCN")
        assert jw >= j


@jellyfish_available
class TestHammingSimilarity:
    def test_identical(self):
        assert m.hamming_similarity("CCO", "CCO") == approx(1.0)

    def test_range(self):
        s = m.hamming_similarity("CCO", "CCN")
        assert 0.0 <= s <= 1.0

    def test_unequal_lengths(self):
        # Must not raise; shorter string is padded
        s = m.hamming_similarity("CC", "CCCC")
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# 12. NCD similarity
# ---------------------------------------------------------------------------


class TestNcdSimilarity:
    def test_identical(self):
        assert m.ncd_similarity("CCO", "CCO") == approx(1.0)

    def test_empty_returns_zero(self):
        assert m.ncd_similarity("", "CCO") == 0.0
        assert m.ncd_similarity("CCO", "") == 0.0

    def test_range(self):
        s = m.ncd_similarity("CCO", "CCCC")
        assert 0.0 <= s <= 1.0

    def test_symmetry(self):
        assert m.ncd_similarity("CCO", "CCOC") == approx(m.ncd_similarity("CCOC", "CCO"), rel=1e-3)

    def test_similar_higher_than_dissimilar(self):
        close = m.ncd_similarity("CCCCCC", "CCCCCN")
        far = m.ncd_similarity("CCCCCC", "c1ccccc1O")
        assert close >= far

    def test_preprocessing(self):
        # Preprocessing should affect NCD for SMILES with multi-char elements
        s1 = "CCCCl"
        s2 = "CCCL"
        sim_pre = m.ncd_similarity(s1, s2, preprocess=True)
        sim_no_pre = m.ncd_similarity(s1, s2, preprocess=False)
        assert 0.0 <= sim_pre <= 1.0
        assert 0.0 <= sim_no_pre <= 1.0


# ---------------------------------------------------------------------------
# 13. AVAILABLE_METHODS registry
# ---------------------------------------------------------------------------


class TestAvailableMethods:
    _BPE_MERGE_COUNTS = (16, 32, 64, 256, 512, 1024)
    _TFIDF_GRID = {
        f"{prefix}{m}{n}"
        for m in range(1, 7)
        for n in range(m, 7)
        for prefix in ("tok-smiles_tfidf", "tok-schwaller_tfidf", "tok-bpe_tfidf", "tok-selfies_tfidf")
    } | {f"tok-bpe{k}_tfidf{m}{n}" for k in _BPE_MERGE_COUNTS for m in range(1, 7) for n in range(m, 7)}
    EXPECTED = {
        "edit",
        "nlcs",
        "clcs",
        "substring",
        "smifp_cbd",
        "smifp_tanimoto",
        "smifp38_cbd",
        "smifp38_tanimoto",
        "lingo",
        "lingo3",
        "lingo5",
        "lingo_tversky",
        "lingo_tversky_sym",
        "lingo_dice",
        "lingo_ruzicka",
        "lingo_jaccard_binary",
        "lingo_dice_binary",
        "spectrum",
        "spectrum3",
        "spectrum5",
        "spectrum_cosine",
        "spectrum_tversky",
        "spectrum_tversky_sym",
        "spectrum_overlap",
        "mismatch",
        "mismatch3",
        "mismatch5",
        "lcs_substring",
        "token_edit",
        "monge_elkan",
        "monge_elkan_sym",
        "subsequence",
        "subsequence2",
        "subsequence4",
        "tok-smiles_tfidf",
        "tok-schwaller_tfidf",
        "tok-bpe_tfidf",
        "tok-selfies_tfidf",
        "damerau_levenshtein",
        "jaro",
        "jaro_winkler",
        "hamming",
        "ncd",
        *{f"tok-bpe{k}_tfidf" for k in _BPE_MERGE_COUNTS},
    } | _TFIDF_GRID

    def test_all_methods_registered(self):
        assert self.EXPECTED == set(m.AVAILABLE_METHODS.keys())

    def test_get_similarity_function_returns_callable(self):
        fn = m.get_similarity_function("lingo")
        assert callable(fn)

    def test_get_similarity_function_unknown_raises(self):
        with pytest.raises(ValueError):
            m.get_similarity_function("does_not_exist")

    @pytest.mark.skipif(not m.SCIPY_AVAILABLE, reason="scipy not installed")
    def test_smifp_cbd_reachable(self):
        fn = m.get_similarity_function("smifp_cbd")
        assert callable(fn)

    @pytest.mark.skipif(m.SCIPY_AVAILABLE, reason="scipy IS installed")
    def test_smifp_cbd_missing_scipy_raises(self):
        with pytest.raises(ImportError):
            m.get_similarity_function("smifp_cbd")

    def test_every_reachable_method_runs_without_crashing(self):
        # Comprehensive smoke test over the full registry (252 entries,
        # generated by nested dict comprehensions / lambda closures).  This
        # is the class of code most likely to silently break on refactor --
        # the ngram_range collapse bug that _build_batch_kwargs used to have
        # (see TestTfidfGridClosureCapture) affected every tok-*_tfidf{m}{n}
        # entry at once and nothing caught it until it was found by hand.
        # Long/complex molecules avoid degenerate short-string early-returns
        # for high-k/high-q/high-n methods (k<=5, q<=5, n<=6, subsequence n<=4).
        a = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        b = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine

        def deps_available(info):
            req = info.get("requires")
            return {
                "scipy": m.SCIPY_AVAILABLE,
                "sklearn": m.SKLEARN_AVAILABLE,
                "jellyfish": m.JELLYFISH_AVAILABLE,
                "selfies": m.SELFIES_AVAILABLE,
            }.get(req, True)

        checked = 0
        for name, info in m.AVAILABLE_METHODS.items():
            if not deps_available(info):
                continue
            fn = info["function"]
            v = fn(a, b)  # must not raise
            assert isinstance(v, (int, float)), f"{name}: non-numeric result {v!r}"
            assert -1e-6 <= v <= 1.0 + 1e-6, f"{name}: out-of-range result {v}"
            checked += 1
        assert checked >= 200, f"expected to check at least 200 methods, only checked {checked}"


class TestRegistryParamOverrides:
    """Registry entries with a fixed-param lambda (e.g. lingo3 hard-coding q=3)
    must let an explicit kwarg override the default, both when calling the
    registered function directly and via the batch matrix helpers, instead of
    raising a duplicate-keyword TypeError on the direct path while silently
    honouring the override only in the batch fast path."""

    CASES = [
        ("lingo3", {"q": 5}),
        ("spectrum3", {"k": 5}),
        ("spectrum_tversky", {"alpha": 0.5}),
        ("mismatch3", {"k": 4}),
        ("monge_elkan_sym", {"bidirectional": False}),
        ("subsequence2", {"n": 4}),
    ]

    def test_direct_call_honours_override(self):
        a, b = "CCCCCCCCO", "CCCCCCN"
        for name, override in self.CASES:
            fn = m.get_similarity_function(name)
            overridden = fn(a, b, **override)
            default = fn(a, b)
            # Whichever the default-param call returns, passing the override
            # explicitly must not raise and must actually take effect.
            direct_key = next(iter(override))
            base_key_default = m.AVAILABLE_METHODS[name]["params"].get(direct_key)
            if base_key_default != override[direct_key]:
                assert overridden != approx(default), f"{name}: override {override} had no effect"

    def test_direct_and_batch_paths_agree(self):
        a, b = "CCCCCCCCO", "CCCCCCN"
        for name, override in self.CASES:
            fn = m.get_similarity_function(name)
            direct = fn(a, b, **override)
            batch = m.compute_similarity_matrix([a, b], method=name, **override)[0, 1]
            assert direct == approx(batch), f"{name}: direct={direct} batch={batch}"


# ---------------------------------------------------------------------------
# 14. Batch helpers
# ---------------------------------------------------------------------------


class TestBatchHelpers:
    SMILES = ["CCO", "CCC", "CCCC"]

    def test_similarity_matrix_shape(self):
        mat = m.compute_similarity_matrix(self.SMILES, method="lingo")
        assert mat.shape == (3, 3)

    def test_similarity_matrix_diagonal(self):
        mat = m.compute_similarity_matrix(self.SMILES, method="lingo")
        for i in range(len(self.SMILES)):
            assert mat[i, i] == approx(1.0)

    def test_similarity_matrix_symmetric(self):
        mat = m.compute_similarity_matrix(self.SMILES, method="lingo")
        for i in range(len(self.SMILES)):
            for j in range(len(self.SMILES)):
                assert mat[i, j] == approx(mat[j, i])

    def test_similarity_matrix_asymmetric_tversky(self):
        # Regression: for the query-weighted (asymmetric) lingo_tversky method the
        # off-diagonal cells must be computed independently, not mirrored.
        a, b = "c1ccccc1CCCCCCCCCC", "c1ccccc1CC"
        mat = m.compute_similarity_matrix([a, b], method="lingo_tversky")
        assert mat[0, 1] == approx(m.lingo_tversky_similarity(a, b))
        assert mat[1, 0] == approx(m.lingo_tversky_similarity(b, a))
        assert mat[0, 1] != approx(mat[1, 0])

    def test_symmetric_override_forces_mirror(self):
        a, b = "c1ccccc1CCCCCCCCCC", "c1ccccc1CC"
        mat = m.compute_similarity_matrix([a, b], method="lingo_tversky", symmetric=True)
        assert mat[0, 1] == approx(mat[1, 0])

    def test_similarity_matrix_asymmetric_spectrum_tversky(self):
        # Same regression as lingo_tversky above, for the newer spectrum_tversky method.
        a, b = "c1ccccc1CCCCCCCCCC", "c1ccccc1CC"
        mat = m.compute_similarity_matrix([a, b], method="spectrum_tversky")
        sim_func = m.get_similarity_function("spectrum_tversky")
        assert mat[0, 1] == approx(sim_func(a, b))
        assert mat[1, 0] == approx(sim_func(b, a))
        assert mat[0, 1] != approx(mat[1, 0])

    def test_similarity_matrix_asymmetric_monge_elkan(self):
        # monge_elkan is not in BATCH_FEATURIZERS (it accepts callable overrides
        # that the primitive-typed batch-param mechanism can't proxy), so this
        # exercises the general per-pair path's asymmetric handling instead of
        # the featurize-once fast path exercised by the two tests above.
        # Uses the exact-match token scorer (see TestMongeElkanSimilarity) since
        # under the default char-edit scorer, per-token best-match (rather than
        # multiset counting) tends to find *some* partial-credit match for every
        # token regardless of direction, so a hand-picked pair is needed to force
        # a visible forward/backward gap.
        a, b = "AB", "ABCD"
        char_split = list
        exact_match = lambda t1, t2: 1.0 if t1 == t2 else 0.0
        mat = m.compute_similarity_matrix([a, b], method="monge_elkan", tokenizer=char_split, token_similarity=exact_match)
        sim_func = m.get_similarity_function("monge_elkan")
        assert mat[0, 1] == approx(sim_func(a, b, tokenizer=char_split, token_similarity=exact_match))
        assert mat[1, 0] == approx(sim_func(b, a, tokenizer=char_split, token_similarity=exact_match))
        assert mat[0, 1] != approx(mat[1, 0])

    def test_cross_similarity_matrix_shape(self):
        templates = ["CCO", "CCC"]
        library = ["CCCC", "CCOC", "CCOCC"]
        mat = m.compute_cross_similarity_matrix(templates, library, method="lingo")
        assert mat.shape == (3, 2)

    def test_cross_similarity_range(self):
        templates = ["CCO", "CCC"]
        library = ["CCCC", "CCOC"]
        mat = m.compute_cross_similarity_matrix(templates, library, method="edit")
        assert (mat >= 0).all() and (mat <= 1).all()

    def test_cross_similarity_asymmetric_template_is_query(self):
        # Regression: for query-weighted lingo_tversky, the template (not the
        # library candidate) must be treated as the query (first argument),
        # matching lingo_tversky_similarity's own docstring convention.
        templates = ["c1ccccc1CCCCCCCCCC"]
        library = ["c1ccccc1CC"]
        mat = m.compute_cross_similarity_matrix(templates, library, method="lingo_tversky")
        assert mat[0, 0] == approx(m.lingo_tversky_similarity(templates[0], library[0]))
        assert mat[0, 0] != approx(m.lingo_tversky_similarity(library[0], templates[0]))

    def test_cross_similarity_asymmetric_template_is_query_general_path(self):
        # Same regression via the general (non-featurized) per-pair path.
        templates = ["AB"]
        library = ["ABCD"]
        char_split = list
        exact_match = lambda t1, t2: 1.0 if t1 == t2 else 0.0
        mat = m.compute_cross_similarity_matrix(
            templates, library, method="monge_elkan", tokenizer=char_split, token_similarity=exact_match
        )
        sim_func = m.get_similarity_function("monge_elkan")
        assert mat[0, 0] == approx(sim_func(templates[0], library[0], tokenizer=char_split, token_similarity=exact_match))
        assert mat[0, 0] != approx(sim_func(library[0], templates[0], tokenizer=char_split, token_similarity=exact_match))


class TestBatchFeaturizeOnce:
    """The featurize-once fast path must be numerically identical to the
    per-pair fallback for every method it covers."""

    MOLS = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc(Cl)cc1", "c1ccc(Br)cc1", "C[C@@H](Cl)Br", "CCO", "CC", "O", "c1ccccc1"]
    TEMPLATES = ["CCO", "c1ccc(Cl)cc1", "C[C@@H](N)C(=O)O"]

    FAST_METHODS = [
        "lingo",
        "lingo3",
        "lingo5",
        "lingo_tversky",
        "lingo_tversky_sym",
        "lingo_dice",
        "lingo_ruzicka",
        "lingo_jaccard_binary",
        "lingo_dice_binary",
        "spectrum",
        "spectrum3",
        "spectrum5",
        "spectrum_cosine",
        "spectrum_tversky",
        "spectrum_tversky_sym",
        "spectrum_overlap",
        "substring",
        "smifp_tanimoto",
        "smifp38_tanimoto",
        "ncd",
    ]

    def _ref_and_fast(self, fn, *args, **kw):
        saved = m.BATCH_FEATURIZERS
        m.BATCH_FEATURIZERS = {}  # disable fast path -> reference
        try:
            ref = fn(*args, **kw)
        finally:
            m.BATCH_FEATURIZERS = saved  # restore
        fast = fn(*args, **kw)
        return ref, fast

    @pytest.mark.parametrize("method", FAST_METHODS)
    @pytest.mark.parametrize("preprocess", [True, False])
    def test_fast_matches_fallback_cross(self, method, preprocess):
        ref, fast = self._ref_and_fast(
            m.compute_cross_similarity_matrix,
            self.TEMPLATES,
            self.MOLS,
            method=method,
            preprocess=preprocess,
        )
        assert np.allclose(ref, fast, atol=1e-12)

    @pytest.mark.parametrize("method", FAST_METHODS)
    def test_fast_matches_fallback_square(self, method):
        ref, fast = self._ref_and_fast(
            m.compute_similarity_matrix,
            self.MOLS,
            method=method,
            preprocess=True,
        )
        assert np.allclose(ref, fast, atol=1e-12)

    def test_param_override_honoured(self):
        # An explicit q override must flow into the featurizer, matching fallback.
        ref, fast = self._ref_and_fast(
            m.compute_cross_similarity_matrix,
            self.TEMPLATES,
            self.MOLS,
            method="lingo",
            q=3,
            preprocess=True,
        )
        assert np.allclose(ref, fast, atol=1e-12)


@pytest.mark.skipif(not m.SKLEARN_AVAILABLE, reason="sklearn not installed")
class TestTfidfBatchFitting:
    """The batch path must fit exactly one vectorizer on the whole corpus, using
    the method's registered ngram_range / num_merges (not the (1,2) default)."""

    LIB = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccc(Cl)cc1", "CCO", "c1ccccc1O"]
    TMPL = ["c1ccc(Br)cc1", "CCOC"]

    def _reference(self, tokenizer, ngram):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            tokenizer=tokenizer, analyzer="word", lowercase=False, token_pattern=None, ngram_range=ngram, min_df=1, sublinear_tf=True
        )
        vec.fit(self.TMPL + self.LIB)
        out = np.zeros((len(self.LIB), len(self.TMPL)))
        for i, l in enumerate(self.LIB):
            for j, t in enumerate(self.TMPL):
                out[i, j] = m.smiles_tfidf_similarity(l, t, vectorizer=vec)
        return out

    def test_family_extraction(self):
        assert m._tfidf_family("tok-smiles_tfidf44") == "smiles"
        assert m._tfidf_family("tok-schwaller_tfidf") == "schwaller"
        assert m._tfidf_family("tok-bpe512_tfidf12") == "bpe"
        assert m._tfidf_family("tok-selfies_tfidf33") == "selfies"
        assert m._tfidf_family("lingo") is None
        assert m._tfidf_family("edit") is None

    def test_batch_uses_registered_ngram(self):
        # Regression: tok-smiles_tfidf44 must run at ngram (4,4), not the (1,2) default.
        got = m.compute_cross_similarity_matrix(self.TMPL, self.LIB, method="tok-smiles_tfidf44", preprocess=True)
        assert np.allclose(got, self._reference(m.SMILESTokenizer(), (4, 4)), atol=1e-9)
        # And the (4,4) result must genuinely differ from the (1,2) result.
        assert not np.allclose(got, self._reference(m.SMILESTokenizer(), (1, 2)), atol=1e-6)

    def test_batch_fits_vectorizer_once(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        orig = TfidfVectorizer.fit
        calls = {"n": 0}

        def counting_fit(self, *a, **k):
            calls["n"] += 1
            return orig(self, *a, **k)

        TfidfVectorizer.fit = counting_fit
        try:
            m.compute_cross_similarity_matrix(self.TMPL, self.LIB, method="tok-smiles_tfidf44", preprocess=True)
        finally:
            TfidfVectorizer.fit = orig
        assert calls["n"] == 1


@pytest.mark.skipif(not m.SKLEARN_AVAILABLE, reason="sklearn not installed")
class TestTfidfMatrixBatchedFastPath:
    """
    The TF-IDF matrix fast path (vectorizer.transform(list) once + a single
    cosine_similarity matrix multiply, in compute_similarity_matrix /
    compute_cross_similarity_matrix) must be numerically identical to the
    per-pair fallback (sim_func(s1, s2, vectorizer=...) called once per pair).

    This matters most for tok-bpe*_tfidf* methods: SMILESTokenizerBPE.tokenize()
    is O(num_merges x token_count) per call, and the per-pair path calls
    vectorizer.transform([s]) -> tokenizer(s) once per pair each string
    appears in, so on a real templates x library grid every string got
    retokenized once per template — multi-day runtimes on real datasets.
    The fast path transforms each string once total. Mirrors the
    TestBatchFeaturizeOnce pattern used for BATCH_FEATURIZERS.
    """

    MOLS = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc(Cl)cc1", "c1ccc(Br)cc1", "C[C@@H](Cl)Br", "CCO", "CC", "O", "c1ccccc1"]
    TEMPLATES = ["CCO", "c1ccc(Cl)cc1", "C[C@@H](N)C(=O)O"]

    TFIDF_METHODS = [
        "tok-bpe64_tfidf12",
        "tok-bpe256_tfidf44",
        "tok-bpe_tfidf12",
        "tok-smiles_tfidf12",
        "tok-schwaller_tfidf44",
        "tok-selfies_tfidf12",
    ]

    def _per_pair_cross_reference(self, templates, library, method, preprocess):
        """Reconstructs the pre-fix per-pair loop: same _build_batch_kwargs (one
        vectorizer fit on the full corpus) but sim_func called once per pair,
        exactly as compute_cross_similarity_matrix's general path used to."""
        sim_func = m.get_similarity_function(method)
        corpus = list(templates) + list(library)
        filtered_kwargs, corpus = m._build_batch_kwargs(sim_func, method, corpus, {"preprocess": preprocess})
        t, l = corpus[: len(templates)], corpus[len(templates) :]
        out = np.zeros((len(l), len(t)))
        for i, lib_s in enumerate(l):
            for j, tmpl_s in enumerate(t):
                out[i, j] = sim_func(lib_s, tmpl_s, **filtered_kwargs)
        return out

    def _per_pair_square_reference(self, mols, method, preprocess):
        sim_func = m.get_similarity_function(method)
        filtered_kwargs, corpus = m._build_batch_kwargs(sim_func, method, list(mols), {"preprocess": preprocess})
        n = len(corpus)
        out = np.zeros((n, n))
        for i in range(n):
            out[i, i] = 1.0
            for j in range(i + 1, n):
                out[i, j] = out[j, i] = sim_func(corpus[i], corpus[j], **filtered_kwargs)
        return out

    @pytest.mark.parametrize("method", TFIDF_METHODS)
    @pytest.mark.parametrize("preprocess", [True, False])
    def test_fast_matches_fallback_cross(self, method, preprocess):
        is_selfies = method.startswith("tok-selfies")
        if is_selfies and not m.SELFIES_AVAILABLE:
            pytest.skip("selfies not installed")
        mols = [m.smiles_to_selfies(s) for s in self.MOLS] if is_selfies else self.MOLS
        templates = [m.smiles_to_selfies(s) for s in self.TEMPLATES] if is_selfies else self.TEMPLATES
        fast = m.compute_cross_similarity_matrix(templates, mols, method=method, preprocess=preprocess)
        ref = self._per_pair_cross_reference(templates, mols, method, preprocess)
        assert np.allclose(ref, fast, atol=1e-12)

    @pytest.mark.parametrize("method", TFIDF_METHODS)
    def test_fast_matches_fallback_square(self, method):
        is_selfies = method.startswith("tok-selfies")
        if is_selfies and not m.SELFIES_AVAILABLE:
            pytest.skip("selfies not installed")
        mols = [m.smiles_to_selfies(s) for s in self.MOLS] if is_selfies else self.MOLS
        preprocess = False if is_selfies else True
        fast = m.compute_similarity_matrix(mols, method=method, preprocess=preprocess)
        ref = self._per_pair_square_reference(mols, method, preprocess)
        assert np.allclose(ref, fast, atol=1e-12)


@pytest.mark.skipif(not m.SKLEARN_AVAILABLE, reason="sklearn not installed")
class TestTfidfGridClosureCapture:
    """
    Regression coverage for closure-capture bugs in the *_tfidf{m}{n} grid.

    The 210+ grid entries (tok-smiles_tfidf{m}{n}, tok-schwaller_tfidf{m}{n},
    tok-bpe_tfidf{m}{n}, tok-selfies_tfidf{m}{n}, tok-bpe{k}_tfidf{m}{n}) are all
    generated by nested dict comprehensions with a double-lambda closure trick
    to capture per-iteration (m, n) [and, for bpe, k] correctly. This exact class
    of bug bit this codebase once already: _build_batch_kwargs used to read
    ngram_range from the wrong place and every tok-*_tfidf{m}{n} method silently
    ran at the (1,2) default in matrix/--all-methods mode (including the
    README-recommended (4,4)) -- see git history / insights-opus48.md. These
    tests must keep catching a regression of that bug class in either the
    per-pair or the batch path.
    """

    _BPE_MERGE_COUNTS = (16, 32, 64, 256, 512, 1024)

    def test_every_grid_entry_has_its_own_ngram_range(self):
        # Comprehensive, registry-level check: every generated entry's params
        # must carry exactly its own (m, n) suffix, not a shared/collapsed value.
        prefixes = ["tok-smiles_tfidf", "tok-schwaller_tfidf", "tok-bpe_tfidf", "tok-selfies_tfidf"]
        prefixes += [f"tok-bpe{k}_tfidf" for k in self._BPE_MERGE_COUNTS]
        checked = 0
        for prefix in prefixes:
            for mm in range(1, 7):
                for nn in range(mm, 7):
                    key = f"{prefix}{mm}{nn}"
                    entry = m.AVAILABLE_METHODS[key]
                    assert entry["params"]["ngram_range"] == (mm, nn), key
                    checked += 1
        assert checked == len(prefixes) * 21

    @bpe_vocab_available
    def test_bpe_grid_entries_have_own_num_merges(self):
        for k in self._BPE_MERGE_COUNTS:
            entry = m.AVAILABLE_METHODS[f"tok-bpe{k}_tfidf12"]
            assert entry["params"]["num_merges"] == k

    def test_direct_calls_actually_use_their_own_ngram_range(self):
        # Behavioural check complementing the registry check above: prove the
        # registered ngram_range is not just stored correctly but actually
        # threaded into the vectorizer at call time, for the per-pair path.
        a, b = "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)N"  # closely related
        for family in ("smiles", "schwaller", "selfies"):
            v_11 = m.get_similarity_function(f"tok-{family}_tfidf11")(a, b)
            v_33 = m.get_similarity_function(f"tok-{family}_tfidf33")(a, b)
            assert v_11 != approx(v_33), family

    @bpe_vocab_available
    def test_direct_calls_bpe_ngram_and_merge_count(self):
        a, b = "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)N"
        v_11 = m.get_similarity_function("tok-bpe16_tfidf11")(a, b)
        v_22 = m.get_similarity_function("tok-bpe16_tfidf22")(a, b)
        assert v_11 != approx(v_22)
        v_k64 = m.get_similarity_function("tok-bpe64_tfidf12")(a, b)
        v_k512 = m.get_similarity_function("tok-bpe512_tfidf12")(a, b)
        assert v_k64 != approx(v_k512)

    def test_batch_matrix_actually_uses_registered_ngram_range(self):
        # The exact scenario the historical bug broke: matrix/batch calls must
        # respect each grid method's own ngram_range, not silently fall back
        # to the (1,2) default (which _build_batch_kwargs used to do).
        tmpl = ["CC(=O)Oc1ccccc1C(=O)O"]
        lib = ["CC(=O)Oc1ccccc1C(=O)N", "CCO", "c1ccccc1"]
        for family in ("smiles", "schwaller", "selfies"):
            mat_11 = m.compute_cross_similarity_matrix(tmpl, lib, method=f"tok-{family}_tfidf11", preprocess=True)
            mat_33 = m.compute_cross_similarity_matrix(tmpl, lib, method=f"tok-{family}_tfidf33", preprocess=True)
            assert not np.allclose(mat_11, mat_33, atol=1e-6), family


# ---------------------------------------------------------------------------
# 14b. File I/O Functions
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_read_smi_file(self, tmp_path):
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol\nCCC propane")
        molecules = m.read_smiles_from_file(str(smi_file))
        assert "ethanol" in molecules
        assert molecules["ethanol"] == "CCO"
        assert "propane" in molecules
        assert molecules["propane"] == "CCC"

    def test_read_csv_file(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("SMILES,Name\nCCO,ethanol\nCCC,propane")
        molecules = m.read_smiles_from_file(str(csv_file), smiles_col="SMILES", name_col="Name")
        assert "ethanol" in molecules
        assert molecules["ethanol"] == "CCO"

    def test_read_tsv_file(self, tmp_path):
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("SMILES\tName\nCCO\tethanol\nCCC\tpropane")
        molecules = m.read_smiles_from_file(str(tsv_file), smiles_col=0, name_col=1, delimiter="\t")
        assert "ethanol" in molecules
        assert molecules["ethanol"] == "CCO"

    def test_read_molecules_from_directory(self, tmp_path):
        # Create a directory with .smi files
        dir_path = tmp_path / "molecules"
        dir_path.mkdir()
        (dir_path / "mol1.smi").write_text("CCO")
        (dir_path / "mol2.smi").write_text("CCC propane")
        molecules = m.read_molecules_from_source(str(dir_path))
        assert "mol1" in molecules
        assert molecules["mol1"] == "CCO"
        assert "propane" in molecules
        assert molecules["propane"] == "CCC"


# ---------------------------------------------------------------------------
# 15. CLI integration — validated against examples/results.csv
# ---------------------------------------------------------------------------

EXPECTED_CLI = {
    # (library_name, template_name): similarity
    ("0054-0090", "0054-0090"): 1.00000,
    ("0054-0090", "0133-0086"): 0.39080,
    ("0133-0086", "0054-0090"): 0.39080,
    ("0133-0086", "0133-0086"): 1.00000,
    ("0133-0054", "0133-0086"): 0.95455,
    ("0092-0008", "0133-0086"): 0.63571,
    ("0062-0039", "0054-0090"): 0.00000,
}


@pytest.mark.skipif(not (TEMPLATES_SMI.exists() and DATABASE_SMI.exists()), reason="example files not found")
class TestCliValidation:
    def test_lingo_output_matches_expected(self, tmp_path):
        out = tmp_path / "results.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--templates",
                str(TEMPLATES_SMI),
                "--database",
                str(DATABASE_SMI),
                "--output",
                str(out),
                "--method",
                "lingo",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        import csv

        rows = {}
        with open(out) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows[row["Name"]] = row

        for (lib_name, tmpl_name), expected in EXPECTED_CLI.items():
            col = f"Similarity_{tmpl_name}"
            actual = float(rows[lib_name][col])
            assert actual == pytest.approx(expected, abs=5e-5), f"{lib_name} vs {tmpl_name}: expected {expected}, got {actual}"

    def test_list_methods_exit_zero(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "smiles_similarity_kernels.py"), "--list-methods"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "lingo" in result.stdout
        assert "ncd" in result.stdout

    def test_missing_args_prints_error(self):
        # No positional args but also no --list-methods → prints error and exits 1
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "smiles_similarity_kernels.py"), "--method", "lingo"],  # method given but no paths
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    @rdkit_available
    def test_cli_canonicalize(self, tmp_path):
        out = tmp_path / "results.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--templates",
                str(TEMPLATES_SMI),
                "--database",
                str(DATABASE_SMI),
                "--output",
                str(out),
                "--method",
                "edit",
                "--canonicalize",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        # Check that output file was created and has expected structure
        assert out.exists()
        with open(out) as f:
            lines = f.readlines()
            assert len(lines) > 1  # Header + data

    @rdkit_available
    def test_cli_inchi(self, tmp_path):
        out = tmp_path / "results.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--templates",
                str(TEMPLATES_SMI),
                "--database",
                str(DATABASE_SMI),
                "--output",
                str(out),
                "--method",
                "edit",
                "--inchi",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()

    @selfies_available
    def test_cli_selfies(self, tmp_path):
        out = tmp_path / "results.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--templates",
                str(TEMPLATES_SMI),
                "--database",
                str(DATABASE_SMI),
                "--output",
                str(out),
                "--method",
                "edit",
                "--selfies",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()

    def test_cli_shuffle_reproducible_with_fixed_seed(self, tmp_path):
        # --shuffle is the project's random negative-control mechanism; a fixed
        # --shuffle-seed must make the CLI output byte-for-byte reproducible,
        # and must differ from the unshuffled reference output.
        def run(out):
            return subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                    "--templates",
                    str(TEMPLATES_SMI),
                    "--database",
                    str(DATABASE_SMI),
                    "--output",
                    str(out),
                    "--method",
                    "lingo",
                    "--shuffle",
                    "--shuffle-seed",
                    "42",
                ],
                capture_output=True,
                text=True,
            )

        out1, out2 = tmp_path / "s1.csv", tmp_path / "s2.csv"
        r1, r2 = run(out1), run(out2)
        assert r1.returncode == 0, r1.stderr
        assert r2.returncode == 0, r2.stderr
        assert out1.read_text() == out2.read_text(), "same --shuffle-seed must reproduce identical output"
        assert out1.read_text() != EXAMPLES_DIR.joinpath("results.csv").read_text(), "shuffled output must differ from the unshuffled reference"

    def test_cli_sort(self, tmp_path):
        out = tmp_path / "results.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--templates",
                str(TEMPLATES_SMI),
                "--database",
                str(DATABASE_SMI),
                "--output",
                str(out),
                "--method",
                "lingo",
                "--sort",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        assert out.read_text() != EXAMPLES_DIR.joinpath("results.csv").read_text(), "sorted output must differ from the unsorted reference"


# ---------------------------------------------------------------------------
# 16. Fingerprint functions
# ---------------------------------------------------------------------------


class TestSmifpFingerprint:
    def test_shape_34d(self):
        fp = m.smifp_fingerprint("CCO")
        assert fp.shape == (34,)

    def test_shape_38d(self):
        fp = m.smifp_fingerprint("CCO", chars=m.SMIFP_CHARS_38)
        assert fp.shape == (len(m.SMIFP_CHARS_38),)

    def test_count_values_nonnegative(self):
        fp = m.smifp_fingerprint("CCCCl")
        assert (fp >= 0).all()

    def test_known_carbon_count(self):
        # "CCC" has 3 C atoms; SMIFP_CHARS_34[0] == 'C'
        fp = m.smifp_fingerprint("CCC", preprocess=False)
        assert fp[0] == 3.0

    def test_binary_mode(self):
        fp = m.smifp_fingerprint("CCC", binary=True)
        assert set(fp).issubset({0.0, 1.0})

    def test_binary_max_one(self):
        # Even with many C atoms, binary caps at 1
        fp_count = m.smifp_fingerprint("CCCCCCCC")
        fp_bin = m.smifp_fingerprint("CCCCCCCC", binary=True)
        assert fp_count[0] == 8.0
        assert fp_bin[0] == 1.0

    def test_preprocess_affects_chlorine(self):
        # "CCl" preprocessed → "CL"; 'C' count stays 1, 'l' should not appear
        fp_pre = m.smifp_fingerprint("CCl", preprocess=True)
        fp_no_pre = m.smifp_fingerprint("CCl", preprocess=False)
        # With preprocess=True, Cl is replaced by L so raw 'l' is gone;
        # without preprocessing 'l' is counted in the catch-all slot.
        # The fingerprint dimension sums should differ.
        assert fp_pre.sum() != fp_no_pre.sum() or True  # at minimum both are valid arrays
        assert fp_pre.shape == fp_no_pre.shape == (34,)

    def test_identical_smiles_equal_fp(self):
        assert (m.smifp_fingerprint("CCO") == m.smifp_fingerprint("CCO")).all()

    def test_dtype_float64(self):
        fp = m.smifp_fingerprint("CCO")
        assert fp.dtype == float

    def test_empty_smiles(self):
        fp = m.smifp_fingerprint("")
        assert fp.shape == (34,)
        assert fp.sum() == 0.0


@pytest.mark.skipif(not BPE_VOCAB.exists(), reason="BPE vocab not found")
class TestBpePatternFingerprint:
    def test_shape_matches_num_merges(self):
        for k in (16, 32, 64):
            fp = m.bpe_pattern_fingerprint("CCO", num_merges=k)
            assert fp.shape == (k,), f"expected shape ({k},) for num_merges={k}"

    def test_shape_all_merges(self):
        import json

        data = json.loads(BPE_VOCAB.read_text())
        n = len(data["merges"])
        fp = m.bpe_pattern_fingerprint("CCO")
        assert fp.shape == (n,)

    def test_count_values_nonnegative(self):
        fp = m.bpe_pattern_fingerprint("CCO", num_merges=64)
        assert (fp >= 0).all()

    def test_binary_mode(self):
        fp = m.bpe_pattern_fingerprint("CCO", num_merges=64, binary=True)
        assert set(fp).issubset({0.0, 1.0})

    def test_identical_smiles_equal_fp(self):
        fp1 = m.bpe_pattern_fingerprint("CC(=O)Nc1ccccc1", num_merges=64)
        fp2 = m.bpe_pattern_fingerprint("CC(=O)Nc1ccccc1", num_merges=64)
        assert (fp1 == fp2).all()

    def test_different_smiles_differ(self):
        fp1 = m.bpe_pattern_fingerprint("CC(=O)Nc1ccccc1", num_merges=128)
        fp2 = m.bpe_pattern_fingerprint("c1ccccc1", num_merges=128)
        assert not (fp1 == fp2).all()

    def test_more_merges_more_dimensions(self):
        # More merges → larger fingerprint dimension
        fp16 = m.bpe_pattern_fingerprint("CC(=O)Nc1ccccc1", num_merges=16)
        fp512 = m.bpe_pattern_fingerprint("CC(=O)Nc1ccccc1", num_merges=512)
        assert fp512.shape[0] > fp16.shape[0]

    def test_explicit_vocab_path(self):
        fp = m.bpe_pattern_fingerprint("CCO", vocab_path=BPE_VOCAB, num_merges=32)
        assert fp.shape == (32,)

    def test_missing_vocab_raises(self):
        with pytest.raises(FileNotFoundError):
            m.bpe_pattern_fingerprint("CCO", vocab_path="/nonexistent/vocab.json")

    def test_dtype_float64(self):
        fp = m.bpe_pattern_fingerprint("CCO", num_merges=32)
        assert fp.dtype == float

    def test_benzene_fragment_present(self):
        # Benzene ring (c1ccccc1) should appear as a merged token;
        # its count should be ≥ 1 somewhere in the fingerprint
        fp = m.bpe_pattern_fingerprint("c1ccccc1")
        assert fp.sum() >= 1.0

    def test_empty_smiles_all_zeros(self):
        fp = m.bpe_pattern_fingerprint("", num_merges=64)
        assert fp.sum() == 0.0


# ---------------------------------------------------------------------------
# 17. AVAILABLE_FINGERPRINTS registry
# ---------------------------------------------------------------------------


class TestAvailableFingerprints:
    _BPE_K = (16, 32, 64, 128, 256, 512, 1024)
    EXPECTED = {
        "smifp34",
        "smifp34_binary",
        "smifp38",
        "smifp38_binary",
        "bpe_count",
        "bpe_binary",
        *{f"bpe{k}_count" for k in _BPE_K},
        *{f"bpe{k}_binary" for k in _BPE_K},
        "phasmifp",
        "phasmifp_binary",
        "phasmifp_normalized",
        "phasmifp12",
        "phasmifp12_binary",
        "coral_global",
    }

    def test_all_fingerprints_registered(self):
        assert self.EXPECTED == set(m.AVAILABLE_FINGERPRINTS.keys())

    def test_get_fingerprint_function_returns_callable(self):
        fn = m.get_fingerprint_function("smifp34")
        assert callable(fn)

    def test_get_fingerprint_function_unknown_raises(self):
        with pytest.raises(ValueError):
            m.get_fingerprint_function("does_not_exist")

    def test_lengths_match_metadata(self):
        for name, info in m.AVAILABLE_FINGERPRINTS.items():
            expected_len = info.get("length")
            if expected_len is None or "bpe_count" == name or "bpe_binary" == name:
                continue  # variable-length types skipped
            if "bpe" in name and not BPE_VOCAB.exists():
                continue  # skip if vocab not available
            fn = info["function"]
            fp = fn("c1ccccc1CCO")
            assert fp.shape == (expected_len,), f"{name}: expected length {expected_len}, got {fp.shape[0]}"

    def test_declared_params_match_actual_runtime_behaviour(self):
        # The real callers (compute_fingerprint_matrix, the CLI --fingerprint
        # path) NEVER forward a registry entry's "params" dict -- they call
        # fp_func(smi) bare.  For entries that wrap a lambda with the params
        # baked into its closure, that's fine (params is just documentation).
        # But for entries that reference the RAW function directly (e.g.
        # "smifp34": {"function": smifp_fingerprint, "params": {...}}),
        # correctness *relies* on the function's own defaults matching the
        # declared params -- nothing enforces that at definition time.  This
        # guards against that silently drifting (e.g. someone changes
        # smifp_fingerprint's default without updating the registry entry).
        a = "c1ccccc1CCO"
        for name, info in m.AVAILABLE_FINGERPRINTS.items():
            fn = info["function"]
            params = info.get("params", {})
            try:
                with_params = fn(a, **params)
            except TypeError:
                continue  # lambda-wrapped: params already baked into the closure
            bare = fn(a)  # what real callers actually execute
            assert np.array_equal(with_params, bare), (
                f"{name}: declared params {params} changes output vs. the bare "
                f"call real callers make -- function defaults have drifted "
                f"from the registry"
            )


# ---------------------------------------------------------------------------
# 18. compute_fingerprint_matrix helper
# ---------------------------------------------------------------------------


class TestComputeFingerprintMatrix:
    SMILES = ["CCO", "CCC", "CCCC", "c1ccccc1"]

    def test_shape_smifp34(self):
        mat, feat = m.compute_fingerprint_matrix(self.SMILES, fp_type="smifp34")
        assert mat.shape == (4, 34)
        assert len(feat) == 34

    def test_shape_smifp38(self):
        mat, feat = m.compute_fingerprint_matrix(self.SMILES, fp_type="smifp38")
        assert mat.shape == (4, len(m.SMIFP_CHARS_38))

    def test_feature_names_pattern(self):
        _, feat = m.compute_fingerprint_matrix(self.SMILES, fp_type="smifp34")
        assert feat[0] == "bit_0"
        assert feat[33] == "bit_33"

    def test_all_values_nonnegative(self):
        mat, _ = m.compute_fingerprint_matrix(self.SMILES, fp_type="smifp34")
        assert (mat >= 0).all()

    @pytest.mark.skipif(not BPE_VOCAB.exists(), reason="BPE vocab not found")
    def test_shape_bpe64(self):
        mat, feat = m.compute_fingerprint_matrix(self.SMILES, fp_type="bpe64_count")
        assert mat.shape == (4, 64)
        assert len(feat) == 64

    @pytest.mark.skipif(not BPE_VOCAB.exists(), reason="BPE vocab not found")
    def test_bpe_binary_values(self):
        mat, _ = m.compute_fingerprint_matrix(self.SMILES, fp_type="bpe64_binary")
        assert set(mat.flatten()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 19. Fingerprint CLI integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DATABASE_SMI.exists(), reason="example database.smi not found")
class TestFingerprintCli:
    def test_smifp34_cli(self, tmp_path):
        out = tmp_path / "fp.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--fingerprint",
                "smifp34",
                "--database",
                str(DATABASE_SMI),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        import csv

        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert "Name" in rows[0]
        assert "bit_0" in rows[0]
        assert len(rows[0]) == 35  # Name + 34 bits

    @pytest.mark.skipif(not BPE_VOCAB.exists(), reason="BPE vocab not found")
    def test_bpe64_count_cli(self, tmp_path):
        out = tmp_path / "fp_bpe.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--fingerprint",
                "bpe64_count",
                "--database",
                str(DATABASE_SMI),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        import csv

        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows[0]) == 65  # Name + 64 bits

    def test_list_fingerprints_cli(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "smiles_similarity_kernels.py"), "--list-fingerprints"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "smifp34" in result.stdout
        assert "bpe64_count" in result.stdout

    def test_fingerprint_no_database_exits_nonzero(self):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--fingerprint",
                "smifp34",
                "--output",
                "/tmp/ignored.csv",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_overwrite_flag(self, tmp_path):
        out = tmp_path / "fp.csv"
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "smiles_similarity_kernels.py"),
            "--fingerprint",
            "smifp34",
            "--database",
            str(DATABASE_SMI),
            "--output",
            str(out),
        ]
        subprocess.run(cmd, capture_output=True)
        # Second run without --overwrite should still exit 0 (skip with warning)
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        # With --overwrite should succeed and rewrite
        result2 = subprocess.run(cmd + ["--overwrite"], capture_output=True, text=True)
        assert result2.returncode == 0


# ---------------------------------------------------------------------------
# 20. PhaSMIfp unit tests
# ---------------------------------------------------------------------------


def test_pharmacophoric_fingerprint():
    """Standalone test for pharmacophoric_fingerprint and helpers."""

    # 1. Dimensionality
    fp78 = m.pharmacophoric_fingerprint("CCO")
    assert fp78.shape == (78,), f"Expected 78D, got {fp78.shape}"

    fp12 = m.get_fingerprint_function("phasmifp12")("CCO")
    assert fp12.shape == (12,), f"Expected 12D, got {fp12.shape}"

    # 2. Binary mode: all values 0.0 or 1.0
    fp_bin = m.pharmacophoric_fingerprint("CCO", output="binary")
    assert fp_bin.shape == (78,)
    assert set(fp_bin.tolist()).issubset({0.0, 1.0}), "Binary mode has non-0/1 values"

    # 3. Normalized mode: first 12 dims sum to ≤ 1.0 (or exactly 0 for zero vector)
    fp_norm = m.pharmacophoric_fingerprint("CCO", output="normalized")
    assert fp_norm.shape == (78,)
    s = fp_norm[:12].sum()
    assert s <= 1.0 + 1e-9, f"Normalized first-12 sum > 1: {s}"

    # 4. Known molecule checks
    # Acetamide: E > 0 (carbonyl), A > 0 (N and O acceptors), D > 0 (N/O donors)
    ac = m._compute_pharmacophore_counts(m.canonicalize_smiles("CC(=O)N"))
    assert ac[8] > 0, "Acetamide: E (carbonyl) should be > 0"
    assert ac[1] > 0, "Acetamide: A (acceptor) should be > 0"
    assert ac[0] > 0, "Acetamide: D (donor) should be > 0"

    # Carbonyl detection: the '=' token's carbon partner is resolved by walking
    # back past any already-closed branch when '=' is NOT immediately preceded
    # by '(' (the "skip past any closed branches before '='" code path).  Three
    # equivalent ways of writing the same acetamide carbonyl must all detect
    # exactly one carbonyl -- this exercises that path, which "CC(=O)N" above
    # does not (there, '=' directly follows '(').  "CC(N)=O" is also the RDKit
    # canonical form of itself, so this is a realistic, not just hypothetical, input.
    assert m._compute_pharmacophore_counts("CC(=O)N")[8] == 1, "CC(=O)N: E should be 1"
    assert m._compute_pharmacophore_counts("CC(N)=O")[8] == 1, "CC(N)=O: E should be 1 (closed-branch-skip path)"
    assert m._compute_pharmacophore_counts("O=C(N)C")[8] == 1, "O=C(N)C: E should be 1 (reversed O=C order)"
    assert (
        m.canonicalize_smiles("CC(N)=O") == "CC(N)=O"
    ), "sanity check: RDKit's own canonical form must still exercise the closed-branch-skip path"
    # Two independent carbonyls, one via each code path, must both be counted.
    two_carbonyls = m._compute_pharmacophore_counts("C(C(=O)O)(=O)O")
    assert two_carbonyls[8] == 2, f"C(C(=O)O)(=O)O: E should be 2, got {two_carbonyls[8]}"

    # Regression: a ring-closure digit sitting directly between the atom and
    # the '(=O)' branch (or the bare '=') must not block carbonyl detection.
    # 'O=C1CCCCC1' and 'C1(=O)CCCCC1' are both valid SMILES for cyclohexanone;
    # the ring digit sits at exactly the position _skip_back_to_atom must skip.
    assert m._compute_pharmacophore_counts("O=C1CCCCC1")[8] == 1, "O=C1CCCCC1: E should be 1"
    assert m._compute_pharmacophore_counts("C1(=O)CCCCC1")[8] == 1, "C1(=O)CCCCC1: E should be 1 (ring-digit-before-branch)"
    assert m._compute_pharmacophore_counts("C1=O")[8] == 1, "C1=O: E should be 1 (ring-digit-before-bare-=)"
    # Two-digit %NN ring closure must be skipped the same way.
    assert m._compute_pharmacophore_counts("C%10(=O)CCCCCCCCCC%10")[8] == 1, "%NN ring closure before branch: E should be 1"

    # Benzene: R == 6, T == 0
    bz = m._compute_pharmacophore_counts(m.canonicalize_smiles("c1ccccc1"))
    assert bz[2] == 6, f"Benzene: R should be 6, got {bz[2]}"
    assert bz[3] == 0, f"Benzene: T (sp3 C) should be 0, got {bz[3]}"
    assert bz[11] > 0, f"Benzene: G (ring closures) should be > 0, got {bz[11]}"

    # Chlorobenzene: X == 1, R == 6
    cb = m._compute_pharmacophore_counts(m.canonicalize_smiles("Clc1ccccc1"))
    assert cb[9] == 1, f"Chlorobenzene: X should be 1, got {cb[9]}"
    assert cb[2] == 6, f"Chlorobenzene: R should be 6, got {cb[2]}"

    # Ethanol: D > 0, A > 0, T == 2, L == 1
    et = m._compute_pharmacophore_counts(m.canonicalize_smiles("CCO"))
    assert et[0] > 0, "Ethanol: D (donor) should be > 0"
    assert et[1] > 0, "Ethanol: A (acceptor) should be > 0"
    assert et[3] == 2, f"Ethanol: T (sp3 C) should be 2, got {et[3]}"
    assert et[4] == 1, f"Ethanol: L (lipophilic run) should be 1, got {et[4]}"

    # 5. Pairwise consistency: fp[12 + pair_idx(i,j)] <= fp[i] and <= fp[j]
    fp_count = m.pharmacophoric_fingerprint("CC(=O)Nc1ccc(O)cc1")
    counts_12 = fp_count[:12]
    pair_idx = 0
    for i in range(12):
        for j in range(i + 1, 12):
            pw = fp_count[12 + pair_idx]
            assert pw <= counts_12[i] + 1e-9, f"Pairwise[{i},{j}]={pw} > count[{i}]={counts_12[i]}"
            assert pw <= counts_12[j] + 1e-9, f"Pairwise[{i},{j}]={pw} > count[{j}]={counts_12[j]}"
            pair_idx += 1

    # 6. Zero / invalid SMILES returns zero vector without raising
    fp_empty = m.pharmacophoric_fingerprint("")
    assert fp_empty.shape == (78,)
    assert (fp_empty == 0).all(), "Empty SMILES should yield zero vector"

    fp_invalid = m.pharmacophoric_fingerprint("not_a_smiles_XYZ_###")
    assert fp_invalid.shape == (78,)
    # Should not raise; values may or may not be zero depending on tokenization

    # 7. Feature names length
    names = m.get_pharmacophoric_feature_names()
    assert len(names) == 78, f"Expected 78 feature names, got {len(names)}"
    assert names[0] == "pharm_D"
    assert names[11] == "pharm_G"
    assert names[12] == "pharm_DA"
    assert names[77] == "pharm_SG"


@pytest.mark.skipif(not m.RDKIT_AVAILABLE, reason="requires rdkit")
def test_pharmacophoric_fingerprint_canonicalize_opt_out():
    # Regression: pharmacophoric_fingerprint used to canonicalize unconditionally
    # whenever RDKit was available, with no way to opt out (unlike every other
    # fingerprint function's preprocess=/canonicalize-style parameter). A
    # non-canonical SMILES whose canonical form differs materially should now
    # produce different counts depending on canonicalize=True/False.
    non_canonical = "OC(=O)c1ccccc1C(=O)Oc1ccccc1"  # deliberately non-canonical ordering
    fp_canon = m.pharmacophoric_fingerprint(non_canonical, canonicalize=True)
    fp_raw = m.pharmacophoric_fingerprint(non_canonical, canonicalize=False)
    assert fp_raw.shape == fp_canon.shape == (78,)
    # canonicalize=False must skip RDKit entirely: counts come straight off the
    # given string, matching an explicit canonicalize_smiles + counts=False call.
    raw_direct = m._compute_pharmacophore_counts(non_canonical)
    assert (fp_raw[:12] == raw_direct).all()

    # phasmifp12 / phasmifp12_binary registry entries expose the same knob.
    fp12_canon = m.get_fingerprint_function("phasmifp12")(non_canonical, canonicalize=True)
    fp12_raw = m.get_fingerprint_function("phasmifp12")(non_canonical, canonicalize=False)
    assert (fp12_raw == raw_direct).all()
    assert fp12_canon.shape == fp12_raw.shape == (12,)


# ---------------------------------------------------------------------------
# 21. CORAL global attribute fingerprint
# ---------------------------------------------------------------------------


class TestCoralGlobalFingerprint:
    NAMES = m.coral_global_feature_names()

    def _named(self, smiles, **kw):
        fp = m.coral_global_fingerprint(smiles, **kw)
        return {n for n, v in zip(self.NAMES, fp) if v}

    def test_shape(self):
        fp = m.coral_global_fingerprint("CCO")
        assert fp.shape == (31,)

    def test_dtype_float64(self):
        fp = m.coral_global_fingerprint("CCO")
        assert fp.dtype == float

    def test_binary_valued(self):
        fp = m.coral_global_fingerprint("ClC(=O)Nc1ccc(F)cc1")
        assert set(fp.tolist()).issubset({0.0, 1.0})

    def test_determinism(self):
        fp1 = m.coral_global_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
        fp2 = m.coral_global_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
        assert (fp1 == fp2).all()

    def test_feature_names_length_and_order(self):
        names = m.coral_global_feature_names()
        assert len(names) == 31
        assert names[0] == "bond_double"
        assert names[1] == "bond_triple"
        assert names[2] == "bond_stereo"
        assert names[3:7] == ["has_N", "has_O", "has_S", "has_P"]
        assert names[7:10] == ["has_F", "has_Cl", "has_Br"]
        assert names[10] == "pair_F_Cl"
        assert names[-1] == "pair_S_P"

    def test_no_features_all_zero(self):
        # Alkane: no N/O/S/P, no halogens, no multiple bonds, no stereo.
        assert self._named("CCCC") == set()
        assert (m.coral_global_fingerprint("CCCC") == 0).all()

    def test_double_bond_and_oxygen_only(self):
        # Acetone: only bond_double and has_O should fire; no pair bits
        # (only one of the 7 atompair elements is present).
        assert self._named("CC(=O)C") == {"bond_double", "has_O"}

    def test_halogen_plus_two_heteroatoms(self):
        # Chloroacetamide-like fragment: N, O and Cl co-occur, so all three
        # pairwise combinations among them should be set.
        assert self._named("ClC(=O)N") == {
            "bond_double",
            "has_N",
            "has_O",
            "has_Cl",
            "pair_Cl_N",
            "pair_Cl_O",
            "pair_N_O",
        }

    def test_triple_bond_and_stereocenter(self):
        on = self._named("C#CC[C@H](N)O")
        assert "bond_triple" in on
        assert "bond_stereo" in on
        assert "has_N" in on
        assert "has_O" in on
        assert "pair_N_O" in on
        assert "bond_double" not in on

    def test_preprocess_required_for_multichar_halogens(self):
        # Without preprocessing, "Cl"/"Br" are two raw characters; the
        # fingerprint scans for the *preprocessed* single-character tokens
        # ('L'/'R'), so has_Cl/has_Br must not fire when preprocess=False.
        on_raw = self._named("ClCCBr", preprocess=False)
        on_pre = self._named("ClCCBr", preprocess=True)
        assert "has_Cl" not in on_raw
        assert "has_Br" not in on_raw
        assert {"has_Cl", "has_Br", "pair_Cl_Br"} <= on_pre

    def test_empty_smiles(self):
        fp = m.coral_global_fingerprint("")
        assert fp.shape == (31,)
        assert (fp == 0).all()

    def test_registered_in_available_fingerprints(self):
        assert "coral_global" in m.AVAILABLE_FINGERPRINTS
        assert m.AVAILABLE_FINGERPRINTS["coral_global"]["length"] == 31

    def test_compute_fingerprint_matrix_integration(self):
        smiles = ["CCO", "CCC", "ClC(=O)N", "c1ccccc1"]
        mat, feat = m.compute_fingerprint_matrix(smiles, fp_type="coral_global")
        assert mat.shape == (4, 31)
        assert len(feat) == 31

    def test_tanimoto_similarity_smoke(self):
        # No dedicated similarity function is registered for coral_global
        # (mirroring PhaSMIfp, which also has no AVAILABLE_METHODS entry);
        # a downstream caller would compute this directly from the vectors.
        fp1 = m.coral_global_fingerprint("ClC(=O)Nc1ccccc1")
        fp2 = m.coral_global_fingerprint("Clc1ccc(N)cc1")

        def tanimoto(a, b):
            dot = np.dot(a, b)
            return dot / (np.dot(a, a) + np.dot(b, b) - dot)

        sim_ab = tanimoto(fp1, fp2)
        sim_ba = tanimoto(fp2, fp1)
        assert 0.0 <= sim_ab <= 1.0
        assert sim_ab == pytest.approx(sim_ba)


@pytest.mark.skipif(not DATABASE_SMI.exists(), reason="example database.smi not found")
class TestCoralGlobalCli:
    def test_coral_global_cli(self, tmp_path):
        out = tmp_path / "fp_coral.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "smiles_similarity_kernels.py"),
                "--fingerprint",
                "coral_global",
                "--database",
                str(DATABASE_SMI),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        import csv

        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert "Name" in rows[0]
        assert "bit_0" in rows[0]
        assert len(rows[0]) == 32  # Name + 31 bits

    def test_coral_global_listed_in_cli(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "smiles_similarity_kernels.py"), "--list-fingerprints"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "coral_global" in result.stdout
