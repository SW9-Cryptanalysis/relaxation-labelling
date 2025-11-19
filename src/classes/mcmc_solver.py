import numpy as np
import numpy.typing as npt
import math
import random
from .statistical_profile import StatisticalProfile
from .cipher import Cipher
from .ngram_scorer import NGramScorer
from .solver import Solver
from typing import Any
from utils.logging import get_colored_logger

log = get_colored_logger("MCMC Solver")

ENGLISH_FREQUENCIES = [0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758, 0.00978, 0.02360, 0.00150, 0.01974, 0.00074]


class McmcSolver(Solver):
	"""
	Solves a homophonic substitution cipher using a
	Markov Chain Monte Carlo (MCMC) search, guided by
	an n-gram fitness function (Simulated Annealing).
	"""

	ALPHA: float = 0.9998

	def __init__(
		self,
		eng_profile: StatisticalProfile,
		cip_profile: StatisticalProfile,
		cipher: Cipher,
		ngram_scorer: NGramScorer
	):
		super().__init__(eng_profile, cip_profile, cipher)

		self.scorer = ngram_scorer

		self.idx_to_letter: list[str] = self.eng_profile.symbols
		self.sym_to_idx: dict[str, int] = self.cip_profile.symbol_to_idx

		self.decoded: str | None = None
		self.best_score: float | None = None
		self.expected_counts: npt.NDArray[np.int_] = np.array(ENGLISH_FREQUENCIES) * self.cip_size

	def decode(self, guesses: npt.NDArray[np.int_] | None = None) -> str:
		"""Fast decoding using a given key (guesses array)."""
		if guesses is None:
			guesses = self.guesses
			if guesses is None:
				log.error("Cannot decode: Guesses not set. Run .run() first.")
				return ""

		decoded_chars = []
		for sym in self.cipher.ciphertext:
			sym_str = str(sym).strip()
			row = self.sym_to_idx[sym_str]
			best_letter_idx = guesses[row]
			decoded_chars.append(self.idx_to_letter[best_letter_idx])
		return "".join(decoded_chars)

	def _calculate_combined_score(self, text: str, key: npt.NDArray[np.int_]) -> float:
		"""
		Combines n-gram score with a penalty for bad letter distribution.
		This guides the solver away from the high-frequency 'e, s, t' trap.
		"""

		ngram_score = self.scorer.score(text)
		actual_counts = np.bincount(key, minlength=self.eng_size)

		diffs = (actual_counts - self.expected_counts) ** 2

		penalty = np.sum(diffs / (self.expected_counts + 1e-6))

		penalty_weight = 30.0

		return ngram_score - (penalty * penalty_weight)

	def run(self, max_iters: int = 50000, max_swaps: int = 6, initial_key: npt.NDArray[np.int_] | None = None) -> None:
		"""
		Runs the MCMC search (Simulated Annealing) to find the best key.
		"""
		self.initial_temp: float = 50.0


		if initial_key is not None:
			self.guesses = np.copy(initial_key)

		else:
			self.guesses = np.array(random.choices(range(self.eng_size), weights=ENGLISH_FREQUENCIES, k=self.cip_size))

		current_key = np.copy(self.guesses)
		current_decoded = self.decode(current_key)
		current_score = self.scorer.score(current_decoded)

		best_key = np.copy(current_key)
		best_score = current_score

		self.temperature = self.initial_temp

		log.info(f"Starting MCMC. Initial score: {current_score:.2f}")

		for i in range(max_iters):
			try:
				best_key = self._iteration(max_swaps, current_key, current_score, i)
			except KeyboardInterrupt:
				break

		self.guesses = best_key
		self.decoded = self.decode(self.guesses)

	def _iteration(self, max_swaps, current_key, current_score, i):
		self.temperature *= self.ALPHA

		temp_ratio = self.temperature / self.initial_temp
		num_swaps = max(1, int(max_swaps * temp_ratio))

		new_key = np.copy(current_key)

		best_key = np.copy(current_key)

		for _ in range(num_swaps):
			new_key = self._make_swap(current_key,)

		new_decoded = self.decode(new_key)
		new_score = self.scorer.score(new_decoded)

		if new_score > current_score:
			current_key = new_key
			current_score = new_score
		else:
			delta = new_score - current_score
			acceptance_prob = math.exp(delta / (self.temperature + 1e-6))

			if random.random() < acceptance_prob:
				current_key = new_key
				current_score = new_score

		if i % 2500 == 0:
			log.info(f"Iter {i}: Score {current_score:.2f} (Best: {self.best_score:.2f})")

		if current_score > self.best_score if self.best_score else 0:
			self.best_score = current_score
			best_key = np.copy(current_key)

		return best_key


	def _make_swap(self, current_key: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
		"""Creates a copy of the key with one swap."""
		new_key = np.copy(current_key)
		idx1, idx2 = np.random.randint(0, self.cip_size, 2)
		# --- End of New Mutation Logic ---
		if idx1 == idx2:
			return new_key

		new_key[idx1] = current_key[idx2]
		new_key[idx2] = current_key[idx1]
		return new_key

	def __json__(self) -> dict[str, Any]:
		return {
			"eng_profile": self.eng_profile.__json__(),
			"cip_profile": self.cip_profile.__json__(),
			"cipher": self.cipher.__json__(),
			"scorer": self.scorer.__json__(),
			"guesses": self.guesses.tolist() if self.guesses is not None else None,
			"decoded": self.decoded,
			"time": self.time,
		}

	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "McmcSolver":
		from classes.solver_config import SolverConfig
		from classes.cipher import Cipher
		from classes.statistical_profile import StatisticalProfile
		eng_profile = StatisticalProfile.__from_json__(json["eng_profile"])
		cip_profile = StatisticalProfile.__from_json__(json["cip_profile"])
		cipher = Cipher.__from_json__(json["cipher"])
		config = SolverConfig.__from_json__(json["config"])
		guesses = np.array(json["guesses"]) if json["guesses"] is not None else None
		decoded = json["decoded"]
		scorer = NGramScorer.__from_json__(json["scorer"])

		solver = McmcSolver(eng_profile, cip_profile, cipher, scorer)
		solver.guesses = guesses
		solver.decoded = decoded
		solver.time = json["time"]

		return solver

