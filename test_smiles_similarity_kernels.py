"""
pytest test suite for smiles_similarity_kernels.py

Run with:
    pytest test_smiles_similarity_kernels.py -v
    pytest test_smiles_similarity_kernels.py -v -k "lingo"  # single group
"""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import smiles_similarity_kernels as m

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent / "examples"
TEMPLATES_SMI = EXAMPLES_DIR / "templates.smi"
DATABASE_SMI  = EXAMPLES_DIR / "database.smi"

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
        assert "¡" in result   # @@ → ¡
        assert "L" in result   # Cl → L
        assert "R" in result   # Br → R

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
        assert "¢" in result   # @TH1 → ¢


class TestNormalizeRingNumbers:
    def test_benzene(self):
        assert m.normalize_ring_numbers("c1ccccc1") == "c0ccccc0"

    def test_bicyclic(self):
        assert m.normalize_ring_numbers("C1CC2CCCCC2C1") == "C0CC0CCCCC0C0"

    def test_no_digits(self):
        assert m.normalize_ring_numbers("CCO") == "CCO"


# ---------------------------------------------------------------------------
# 2. Canonicalization and InChI  (skip when RDKit absent)
# ---------------------------------------------------------------------------

rdkit_available = pytest.mark.skipif(
    not m.RDKIT_AVAILABLE, reason="RDKit not installed"
)


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
        assert m.edit_similarity("CCO", "CCOC") == approx(
            m.edit_similarity("CCOC", "CCO")
        )


# ---------------------------------------------------------------------------
# 4. NLCS similarity
# ---------------------------------------------------------------------------

class TestNlcsSimilarity:
    def test_identical(self):
        assert m.nlcs_similarity("CCO", "CCO") == approx(1.0)

    def test_known_value(self):
        # LCS("ABC","AC") = 2, NLCS = 4/(3*2) = 0.6667
        assert m.nlcs_similarity("ABC", "AC", preprocess=False) == approx(2**2 / (3*2))

    def test_no_common(self):
        # No common characters → LCS = 0 → similarity = 0
        assert m.nlcs_similarity("AAA", "BBB", preprocess=False) == approx(0.0)

    def test_range(self):
        s = m.nlcs_similarity("CCO", "CCOC")
        assert 0.0 <= s <= 1.0

    def test_symmetry(self):
        assert m.nlcs_similarity("CCO", "CCOC") == approx(
            m.nlcs_similarity("CCOC", "CCO")
        )


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
        assert m.substring_kernel_similarity("CCO", "CCOC") == approx(
            m.substring_kernel_similarity("CCOC", "CCO")
        )


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
        assert m.lingo_similarity("CCO", "CCOC") == approx(
            m.lingo_similarity("CCOC", "CCO")
        )

    def test_validated_against_example_output(self):
        """
        Validates against examples/results.csv produced by the CLI.
        Template 0054-0090 vs 0133-0086 must be 0.39080.
        """
        t1 = "CC(=O)C1=CC=C(Br)C(N)=C1"           # 0054-0090
        t2 = "NC1=CC=C(Br)C=C1C(=O)C1=CC=CC=C1Cl"  # 0133-0086
        assert m.lingo_similarity(t1, t2) == approx(0.39080, rel=1e-3)

    def test_self_similarity_templates(self):
        t1 = "CC(=O)C1=CC=C(Br)C(N)=C1"
        t2 = "NC1=CC=C(Br)C=C1C(=O)C1=CC=CC=C1Cl"
        assert m.lingo_similarity(t1, t1) == approx(1.0)
        assert m.lingo_similarity(t2, t2) == approx(1.0)


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

    def test_vectorizer_reuse(self):
        corpus = ["CCO", "CCOC", "CCCCC"]
        vec = m.LingoVectorizer(q=4, use_idf=True)
        vec.fit(corpus)
        s1 = m.lingo_tfidf_similarity("CCO", "CCOC", vectorizer=vec)
        s2 = m.lingo_tfidf_similarity("CCO", "CCOC", vectorizer=vec)
        assert s1 == approx(s2)


# ---------------------------------------------------------------------------
# 10. SMILES TF-IDF (chemical tokenization)
# ---------------------------------------------------------------------------

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

    def test_vectorizer_reuse(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        tok = m.SMILESTokenizer()
        vec = TfidfVectorizer(tokenizer=tok, analyzer="word", lowercase=False,
                              token_pattern=None, ngram_range=(1, 2), min_df=1,
                              sublinear_tf=True)
        vec.fit(["CCO", "CCOC", "c1ccccc1Cl"])
        s1 = m.smiles_tfidf_similarity("CCO", "CCOC", vectorizer=vec)
        s2 = m.smiles_tfidf_similarity("CCO", "CCOC", vectorizer=vec)
        assert s1 == approx(s2)


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


# ---------------------------------------------------------------------------
# 11. Jellyfish-based methods
# ---------------------------------------------------------------------------

jellyfish_available = pytest.mark.skipif(
    not m.JELLYFISH_AVAILABLE, reason="jellyfish not installed"
)


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
        j  = m.jaro_similarity("CCCCO", "CCCCN")
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
        assert m.ncd_similarity("CCO", "CCOC") == approx(
            m.ncd_similarity("CCOC", "CCO"), rel=1e-3
        )

    def test_similar_higher_than_dissimilar(self):
        close = m.ncd_similarity("CCCCCC", "CCCCCN")
        far   = m.ncd_similarity("CCCCCC", "c1ccccc1O")
        assert close >= far


# ---------------------------------------------------------------------------
# 13. AVAILABLE_METHODS registry
# ---------------------------------------------------------------------------

class TestAvailableMethods:
    EXPECTED = {
        "edit", "nlcs", "clcs", "substring",
        "smifp_cbd", "smifp_tanimoto", "smifp38_cbd", "smifp38_tanimoto",
        "lingo", "lingo3", "lingo5",
        "smiles_tfidf", "smiles_tfidf13",
        "damerau_levenshtein", "jaro", "jaro_winkler", "hamming",
        "ncd",
    }

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

    def test_cross_similarity_matrix_shape(self):
        templates = ["CCO", "CCC"]
        library   = ["CCCC", "CCOC", "CCOCC"]
        mat = m.compute_cross_similarity_matrix(templates, library, method="lingo")
        assert mat.shape == (3, 2)

    def test_cross_similarity_range(self):
        templates = ["CCO", "CCC"]
        library   = ["CCCC", "CCOC"]
        mat = m.compute_cross_similarity_matrix(templates, library, method="edit")
        assert (mat >= 0).all() and (mat <= 1).all()


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


@pytest.mark.skipif(
    not (TEMPLATES_SMI.exists() and DATABASE_SMI.exists()),
    reason="example files not found"
)
class TestCliValidation:
    def test_lingo_output_matches_expected(self, tmp_path):
        out = tmp_path / "results.csv"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "smiles_similarity_kernels.py"),
             str(TEMPLATES_SMI), str(DATABASE_SMI), str(out), "--method", "lingo"],
            capture_output=True, text=True
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
            assert actual == pytest.approx(expected, abs=5e-5), (
                f"{lib_name} vs {tmpl_name}: expected {expected}, got {actual}"
            )

    def test_list_methods_exit_zero(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "smiles_similarity_kernels.py"),
             "--list-methods"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "lingo" in result.stdout
        assert "ncd" in result.stdout

    def test_missing_args_prints_error(self):
        # No positional args but also no --list-methods → prints error and exits 1
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "smiles_similarity_kernels.py"),
             "--method", "lingo"],   # method given but no paths
            capture_output=True, text=True
        )
        assert result.returncode != 0
