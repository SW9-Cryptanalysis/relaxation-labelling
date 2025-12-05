import pytest
import numpy as np

from classes.statistical_profile import StatisticalProfile

from classes.cipher import Cipher


@pytest.fixture
def simple_profile():
	"""
	Creates a simple 2x2 profile for math testing.
	Matrix:
	. A  B
	A 2  2
	B 4  0
	"""
	p_raw = np.array([[2, 2], [4, 0]], dtype=float)
	symbols = ["A", "B"]
	return StatisticalProfile(p_raw, symbols)


@pytest.fixture
def mock_pandas_data():
	"""Returns a mock pandas DataFrame-like object."""

	class MockIndex:
		def to_list(self):
			return ["A", "B"]

	class MockPD:
		index = MockIndex()

	matrix = np.eye(2)
	return MockPD(), matrix


class TestStatisticalProfile:
	def test_initialization(self, simple_profile):
		"""Test basic attribute assignment."""
		assert simple_profile.size == 2
		assert simple_profile.symbols == ["A", "B"]
		assert simple_profile.symbol_to_idx == {"A": 0, "B": 1}

		assert simple_profile._p_row_normalized is None
		assert simple_profile._p_col_normalized is None

	def test_row_normalization(self, simple_profile):
		"""
		Matrix: [[2, 2], [4, 0]]
		Row Sums: [4, 4]
		Normalized: [[0.5, 0.5], [1.0, 0.0]]
		"""
		p_norm = simple_profile.p_row_normalized

		expected = np.array([[0.5, 0.5], [1.0, 0.0]])
		np.testing.assert_array_almost_equal(p_norm, expected)

		assert simple_profile._p_row_normalized is not None

	def test_col_normalization(self, simple_profile):
		"""
		Matrix: [[2, 2], [4, 0]]
		Col Sums: [6, 2]
		Normalized: [[2/6, 2/2], [4/6, 0/2]] -> [[0.333, 1.0], [0.666, 0.0]]
		"""
		p_norm = simple_profile.p_col_normalized

		expected = np.array([[2 / 6, 1.0], [4 / 6, 0.0]])
		np.testing.assert_array_almost_equal(p_norm, expected)

	def test_normalization_handles_zeros(self):
		"""Test that normalization handles zero-sum rows/cols without div-by-zero error."""
		p_raw = np.zeros((2, 2))
		profile = StatisticalProfile(p_raw, ["A", "B"])

		assert np.all(profile.p_row_normalized == 0)
		assert np.all(profile.p_col_normalized == 0)

	def test_unigram_frequencies(self, simple_profile):
		"""
		Matrix: [[2, 2], [4, 0]]
		Row Sums: [4, 4] -> Total 8
		Frequencies: [4/8, 4/8] -> [0.5, 0.5]
		"""
		freqs = simple_profile.unigram_frequencies
		expected = np.array([0.5, 0.5])

		np.testing.assert_array_almost_equal(freqs, expected)

	def test_from_text_construction(self):
		"""Test constructing a matrix from a text sequence."""
		text = ["A", "B", "A", "A"]
		# Pairs: (A,B), (B,A), (A,A)
		# Matrix:
		#   A B
		# A 1 1  (AA, AB)
		# B 1 0  (BA)

		symbols = ["A", "B"]
		profile = StatisticalProfile.from_text(text, symbols)

		expected = np.array([[1, 1], [1, 0]])
		np.testing.assert_array_equal(profile.p_raw, expected)

	def test_from_matrix_file(self, mocker, mock_pandas_data):
		"""Test loading from an external matrix file."""
		mock_load = mocker.patch(
			"classes.statistical_profile.load_matrix", return_value=mock_pandas_data
		)

		profile = StatisticalProfile.from_matrix_file("dummy.csv")

		assert profile.size == 2
		assert profile.symbols == ["A", "B"]
		np.testing.assert_array_equal(profile.p_raw, np.eye(2))
		mock_load.assert_called_once_with("dummy.csv")

	def test_from_cipher_cache_hit(self, mocker, mock_pandas_data):
		"""Test loading cipher profile from existing cache file."""
		mocker.patch("classes.statistical_profile.matrix_exists", return_value=True)

		mocker.patch(
			"classes.statistical_profile.load_matrix", return_value=mock_pandas_data
		)

		cipher = mocker.Mock(spec=Cipher)
		cipher.name = "test_cipher"

		profile = StatisticalProfile.from_cipher(cipher)

		assert profile.size == 2

		assert profile.symbols == ["A", "B"]

	def test_from_cipher_build_new(self, mocker):
		"""Test building cipher profile from scratch (cache miss)."""
		mocker.patch("classes.statistical_profile.matrix_exists", return_value=False)
		mock_save = mocker.patch("classes.statistical_profile.save_matrix")

		cipher = mocker.Mock(spec=Cipher)
		cipher.name = "test_cipher"
		cipher.get_symbols.return_value = ["X", "Y"]
		cipher.ciphertext = ["X", "Y", "X"]

		profile = StatisticalProfile.from_cipher(cipher, save=True)

		assert profile.symbols == ["X", "Y"]

		expected = np.array([[0, 1], [1, 0]])
		np.testing.assert_array_equal(profile.p_raw, expected)

		mock_save.assert_called_once()
		assert mock_save.call_args[0][3] == "test_cipher-matrix.csv"

	def test_from_english_corpus_cache_miss(self, mocker):
		"""Test building English profile from NLTK when cache file missing."""

		mocker.patch("classes.statistical_profile.matrix_exists", return_value=False)

		mock_nltk = mocker.patch("classes.statistical_profile.nltk")

		mock_nltk.corpus.brown.words.return_value = ["abc", "def"]

		mock_save = mocker.patch("classes.statistical_profile.save_matrix")

		profile = StatisticalProfile.from_english_corpus(save=True)

		assert len(profile.symbols) == 26
		assert "a" in profile.symbols

		mock_save.assert_called_once()

	def test_json_serialization(self, simple_profile):
		"""Test __json__ output format."""
		j_data = simple_profile.__json__()

		assert j_data["symbols"] == ["A", "B"]

		assert j_data["p_raw"] == [[2.0, 2.0], [4.0, 0.0]]

	def test_from_json(self):
		"""Test __from_json__ reconstruction."""
		json_data = {"p_raw": [[1, 0], [0, 1]], "symbols": ["X", "Y"]}

		profile = StatisticalProfile.__from_json__(json_data)

		assert profile.symbols == ["X", "Y"]
		assert isinstance(profile.p_raw, np.ndarray)
		np.testing.assert_array_equal(profile.p_raw, np.eye(2))
