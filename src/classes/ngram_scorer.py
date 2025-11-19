import math
from utils.constants import NGRAM_PATH
from typing import Any


class NGramScorer:
	"""Scores texts using n-grams.

	Attributes:
		n (int): The n in n-gram
		n_grams (dict[str, float]): The n-grams and their scores
		total_count (int): The total count of n-grams
		_floor_score (float, optional): The floor score of the n-gram scorer.
			Defaults to None.

	Methods:
		score(text: str) -> float: Score a text using the n-gram scorer.
		__json__() -> dict[str, Any]: Convert the NGramScorer to a JSON object.
		__from_json__(json: dict[str, Any]) -> NGramScorer: Load a NGramScorer from a
			JSON object.

	"""

	def __init__(self, n: int, file_name: str) -> None:
		"""Initialize an n-gram scorer.

		Args:
			n (int): The n in n-gram
			file_name (str): The file name of the n-gram file

		Returns:
			None

		"""
		self.n_grams = {}
		self.total_count = 0
		self._floor_score = None
		self.n = n
		self._load_n_grams(NGRAM_PATH + file_name)

	@property
	def floor_score(self) -> float:
		"""Get the floor score of the n-gram scorer.

		Returns:
			float: The floor score of the n-gram scorer

		"""
		if self._floor_score is None:
			self._floor_score = math.log(0.00001 / self.total_count)
		return self._floor_score

	def _load_n_grams(self, file_path: str) -> None:
		"""Load n-grams from a file.

		Args:
			file_path (str): The path to the file to load n-grams from

		Returns:
			None

		"""
		with open(file_path) as f:
			for line in f:
				ngram, count = line.split()
				count = int(count)
				self.n_grams[ngram.lower()] = count
				self.total_count += count

		for ngram, count in self.n_grams.items():
			self.n_grams[ngram] = math.log(count / self.total_count)

	def score(self, text: str) -> float:
		"""Score a text using the n-gram scorer.

		Args:
			text (str): The text to score

		Returns:
			float: The score of the text

		"""
		score = 0.0
		for i in range(len(text) - (self.n - 1)):
			ngram = text[i : i + self.n]
			score += self.n_grams.get(ngram, self.floor_score)
		return score

	def __json__(self) -> dict[str, Any]:
		"""Convert the NGramScorer to a JSON object.

		Returns:
			dict[str, Any]: The JSON object

		"""
		return {
			"n": self.n,
			"total_count": self.total_count,
			"floor_score": self.floor_score,
		}

	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "NGramScorer":
		"""Load a NGramScorer from a JSON object.

		Args:
			json (dict[str, Any]): The JSON object to load the NGramScorer from

		Returns:
			NGramScorer: The NGramScorer object

		"""
		scorer = NGramScorer(json["n"], json["n_grams"])
		scorer.total_count = json["total_count"]
		scorer._floor_score = json["floor_score"]
		return scorer
