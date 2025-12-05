from classes.mcmc_solver import McmcSolver
from classes.relaxation_solver import RelaxationSolver
from classes.ngram_scorer import NGramScorer
from classes.solver import Solver
from utils.logging import get_colored_logger
import numpy as np
from typing import Any
import json
import os

log = get_colored_logger("Relaxation Solver")


class SolverAnalytics:
	"""Analyzes a solver and cipher object and stores the results.

	Attributes:
		solver (Solver): The solver object to analyze
		cipher (Cipher): The cipher object to analyze
		eng_profile (StatisticalProfile): The English statistical profile
		cip_profile (StatisticalProfile): The cipher statistical profile
		valid_key (dict[int, int], optional): The {cip_idx: eng_idx} validation
			key. Defaults to None.
		_mer (float, optional): The Mapping Error Rate (MER). Defaults to None.
		_ser (float, optional): The Symbol Error Rate (SER). Defaults to None.

	Methods:
		save(path: str) -> None: Save the solver analytics to a file.
		__str__() -> str: Convert the SolverAnalytics to a string.
		__json__() -> dict[str, Any]: Convert the SolverAnalytics to a JSON object.
		__from_json__(json: dict[str, Any]) -> SolverAnalytics: Load a
			SolverAnalytics from a JSON object.

	"""

	def __init__(self, solver: Solver) -> None:
		"""Initialize a SolverAnalytics object.

		Args:
			solver (Solver): The solver object to analyze

		Returns:
			None

		"""
		self.solver = solver
		self.eng_profile = solver.eng_profile
		self.cip_profile = solver.cip_profile
		self._scorer = NGramScorer(4, "english_quadgrams.txt")

		self._valid_key: dict[int, int] | None = None
		self._mer: float | None = None
		self._ser: float | None = None
		self._score: float | None = None

	@property
	def score(self) -> float:
		"""Return the n-gram score of the cipher.

		Returns:
			float: The n-gram score of the cipher

		"""
		if self.solver.decoded is None:
			log.warning("Cannot calculate score. Decoded not yet calculated.")
			return np.nan
		if self._score is None:
			self._score = self._scorer.score(self.solver.decoded)
		return self._score

	@property
	def valid_key(self) -> dict[int, int]:
		"""Return the {cip_idx: eng_idx} validation key.

		Lazily builds the {cip_idx: eng_idx} validation key	from the cipher's
		{letter: [symbols]} (homophonic) key.

		Returns:
			dict[int, int]: The {cip_idx: eng_idx} validation key

		"""
		if self._valid_key is None:
			self._valid_key = {}
			cip_map = self.cip_profile.symbol_to_idx
			eng_map = self.eng_profile.symbol_to_idx

			for letter, symbol_list in self.solver.cipher.key.items():
				# Ensure letter is a valid English profile symbol
				letter_str = str(letter).strip().lower()
				if letter_str not in eng_map:
					continue

				eng_idx = eng_map[letter_str]

				self._traverse_symbol_list(cip_map, symbol_list, eng_idx)

			if not self._valid_key:
				log.warning(
					"Warning: Could not build valid_key from cipher.key. "
					"Check for symbol mismatches.",
				)

		return self._valid_key

	def _traverse_symbol_list(
		self,
		cip_map: dict[str, int],
		symbol_list: list[str],
		eng_idx: int,
	) -> None:
		if not self._valid_key:
			self._valid_key = {}
		for symbol in symbol_list:
			symbol_str = str(
				symbol,
			).strip()
			if symbol_str in cip_map:
				cip_idx = cip_map[symbol_str]
				self._valid_key[cip_idx] = eng_idx

	@property
	def mer(self) -> float:
		"""Return the Mapping Error Rate (MER).

		Mapping Error Rate (MER): Ratio of incorrect key mappings.

		Returns:
			float: The Mapping Error Rate (MER)

		"""
		if self._mer is not None:
			return self._mer

		if self.solver.guesses is None:
			log.warning("MER: Guesses not set. Run .run() first.")
			return np.nan

		correct = 0
		total_symbols = len(self.valid_key)
		if total_symbols == 0:
			log.warning("MER: No valid key loaded. Cannot calculate.")
			return np.nan

		for cipher_idx, actual_letter_idx in self.valid_key.items():
			if self.solver.guesses[cipher_idx] == actual_letter_idx:
				correct += 1
		self._mer = 1.0 - (correct / total_symbols)
		return self._mer

	@property
	def ser(self) -> float:
		"""Return the Symbol Error Rate (SER).

		Symbol Error Rate (SER): Ratio of incorrect characters in the decoded text.
		Returns np.nan if not calculated.

		Returns:
			float: The Symbol Error Rate (SER)

		"""
		if not self.solver.decoded:
			log.warning("SER not yet calculated. Run .decode() first.")
			return np.nan
		if not self.solver.cipher.plaintext or len(self.solver.decoded) != len(
			self.solver.cipher.plaintext,
		):
			log.warning("Plaintext mismatch or missing. Cannot calculate SER.")
			log.warning(f"Decoded length: {len(self.solver.decoded)}")
			log.warning(f"Plaintext length: {len(self.solver.cipher.plaintext)}")
			return np.nan

		correct = sum(
			1
			for d, a in zip(
				self.solver.decoded, self.solver.cipher.plaintext, strict=True,
			)
			if d == a
		)
		self._ser = 1.0 - (correct / len(self.solver.decoded))
		return self._ser

	def __str__(self) -> str:
		"""Convert the SolverAnalytics to a string.

		Returns:
			str: The string representation of the SolverAnalytics

		"""
		return f"SolverAnalytics {{\n  solver: {self.solver.__str__()}\n}}"

	def __json__(self) -> dict[str, Any]:
		"""Convert the SolverAnalytics to a JSON object.

		Returns:
			dict[str, Any]: The JSON object

		"""
		return {
			"mer": self.mer,
			"ser": self.ser,
			"time": self.solver.time,
			"cipher": self.solver.cipher.name,
			"decoded": self.solver.decoded,
		}

	@staticmethod
	def __from_json__(json: dict[str, Any], solver_type: str) -> "SolverAnalytics":
		"""Load a SolverAnalytics from a JSON object.

		Args:
			json (dict[str, Any]): The JSON object to load the SolverAnalytics from
			solver_type (str): The type of solver to load (e.g., "McmcSolver")

		Returns:
			SolverAnalytics: The SolverAnalytics object

		"""
		if "mcmcsolver" in solver_type.lower():
			solver = McmcSolver.__from_json__(json["solver"])
		elif "relaxationsolver" in solver_type.lower():
			solver = RelaxationSolver.__from_json__(json["solver"])
		else:
			raise ValueError("Unknown solver type: " + solver_type)
		analytics = SolverAnalytics(solver)

		return analytics

	def save(self, path: str) -> None:
		"""Save the solver analytics to a file.

		Args:
			path (str): The path to save the solver analytics to

		Returns:
			None

		"""
		if not path.endswith(".json"):
			path += ".json"

		path_dir = os.path.dirname(os.path.abspath(path))
		if not os.path.exists(path_dir):
			os.makedirs(path_dir)
		with open(path, "w") as f:
			json.dump(self.__json__(), f, indent=4)
