import pytest
import math

from classes.ngram_scorer import NGramScorer

class TestNGramScorer:

	@pytest.fixture
	def mock_file_content(self) -> str:
		"""Returns dummy n-gram file content."""
		return "ABCD 10\nEFGH 30"

	@pytest.fixture
	def scorer(self, mocker, mock_file_content) -> NGramScorer:
		"""Initializes a scorer with mocked file I/O."""
		mocker.patch("classes.ngram_scorer.NGRAM_PATH", "dummy_path")

		mocker.patch("builtins.open", mocker.mock_open(read_data=mock_file_content))

		return NGramScorer(n=4, file_name="test_ngrams.txt")

	def test_init_loads_and_parses_data(self, scorer):
		"""Test that data is read, lowercased, and converted to log probs."""
		assert scorer.total_count == 40
		assert scorer.n == 4

		assert "abcd" in scorer.n_grams
		assert "efgh" in scorer.n_grams

		expected_abcd = math.log(10 / 40)
		expected_efgh = math.log(30 / 40)

		assert scorer.n_grams["abcd"] == pytest.approx(expected_abcd)
		assert scorer.n_grams["efgh"] == pytest.approx(expected_efgh)

	def test_floor_score_calculation(self, scorer):
		"""Test the lazy loading and calculation of the floor score."""
		expected_floor = math.log(0.00001 / 40)

		assert scorer.floor_score == pytest.approx(expected_floor)

		assert scorer._floor_score == pytest.approx(expected_floor)

	def test_score_exact_match(self, scorer):
		text = "abcdefgh"

		expected_score = (
			scorer.n_grams["abcd"] +
			scorer.floor_score +
			scorer.floor_score +
			scorer.floor_score +
			scorer.n_grams["efgh"]
		)

		actual_score = scorer.score(text)
		assert actual_score == pytest.approx(expected_score)

	def test_score_short_text(self, scorer):
		"""Test scoring text shorter than N returns 0."""
		text = "ABC"
		assert scorer.score(text) == 0.0

	def test_score_case_insensitivity(self, scorer):
		"""Test that input text is treated case-insensitively via dictionary keys."""
		text = "abcd"
		expected = scorer.n_grams["abcd"]

		assert scorer.score(text) == pytest.approx(expected)

	def test_json_serialization(self, scorer):
		"""Test conversion to JSON-serializable dict."""
		json_data = scorer.__json__()

		assert json_data["n"] == 4
		assert json_data["total_count"] == 40
		assert json_data["floor_score"] == scorer.floor_score

	def test_from_json(self, mocker, mock_file_content):
		"""
		Test deserialization.
		Note: __from_json__ hardcodes 'english_quadgrams.txt', so we must
		mock open() again to prevent FileNotFoundError.
		"""
		mocker.patch("classes.ngram_scorer.NGramScorer._load_n_grams")

		json_input = {
			"n": 4,
			"total_count": 1000,
			"floor_score": -15.5
		}

		restored_scorer = NGramScorer.__from_json__(json_input)

		assert restored_scorer.n == 4
		assert restored_scorer.total_count == 1000
		assert restored_scorer.floor_score == -15.5

		assert restored_scorer._floor_score == -15.5
