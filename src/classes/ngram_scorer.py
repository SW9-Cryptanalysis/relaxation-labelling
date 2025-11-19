import math
from utils.constants import NGRAM_PATH
from typing import Any


class NGramScorer:
	def __init__(self, n: int, file_name: str):
		self.n_grams = {}
		self.total_count = 0
		self._floor_score = None
		self.n = n
		self._load_n_grams(NGRAM_PATH + file_name)
  
	@property
	def floor_score(self) -> float:
		if self._floor_score is None:
			self._floor_score = math.log(0.00001 / self.total_count)
		return self._floor_score

	def _load_n_grams(self, file_path: str) -> None:
		with open(file_path, "r") as f:
			for line in f:
				ngram, count = line.split()
				count = int(count)
				self.n_grams[ngram.lower()] = count
				self.total_count += count

		for ngram, count in self.n_grams.items():
			self.n_grams[ngram] = math.log(count / self.total_count)

	def score(self, text: str) -> float:
		score = 0.0
		for i in range(len(text) - (self.n - 1)):
			ngram = text[i : i + self.n]
			score += self.n_grams.get(ngram, self.floor_score)
		return score

	def __json__(self) -> dict[str, Any]:
		return {
			"n": self.n,
			"total_count": self.total_count,
			"floor_score": self.floor_score,
		}

	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "NGramScorer":
		scorer = NGramScorer(json["n"], json["n_grams"])
		scorer.total_count = json["total_count"]
		scorer._floor_score = json["floor_score"]
		return scorer
