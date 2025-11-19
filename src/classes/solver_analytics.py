from classes import RelaxationSolver, Cipher
from classes.solver import Solver
from utils.logging import get_colored_logger
import numpy as np
from typing import Any
import json
import os

log = get_colored_logger("Relaxation Solver")

class SolverAnalytics:
	def __init__(self, solver: Solver, cipher: Cipher):
		self.solver = solver
		self.cipher = cipher
		self.eng_profile = solver.eng_profile
		self.cip_profile = solver.cip_profile
		
		self._valid_key: dict[int, int] | None = None
		self._mer: float | None = None
		self._ser: float | None = None
		
  
	@property
	def valid_key(self) -> dict[int, int]:
		"""
		Lazily builds the {cip_idx: eng_idx} validation key 
		from the cipher's {letter: [symbols]} (homophonic) key.
		"""
		if self._valid_key is None:
			self._valid_key = {}
			cip_map = self.cip_profile.symbol_to_idx
			eng_map = self.eng_profile.symbol_to_idx

			for letter, symbol_list in self.cipher.key.items():
				# Ensure letter is a valid English profile symbol
				letter_str = str(letter).strip().lower()
				if letter_str not in eng_map:
					continue
				
				eng_idx = eng_map[letter_str]

				self._traverse_symbol_list(cip_map, symbol_list, eng_idx)
			
			if not self._valid_key:
				log.warning("Warning: Could not build valid_key from cipher.key. Check for symbol mismatches.")

		return self._valid_key

	def _traverse_symbol_list(self, cip_map, symbol_list, eng_idx):
		if not self._valid_key:
			self._valid_key = {}
		for symbol in symbol_list:
			symbol_str = str(symbol).strip() # Ensure symbol is a valid cipher profile symbol
			if symbol_str in cip_map:
				cip_idx = cip_map[symbol_str]
				self._valid_key[cip_idx] = eng_idx
	
	
	@property
	def mer(self) -> float:
		"""
		Mapping Error Rate (MER): Percentage of incorrect key mappings.
		Returns np.nan if not calculated.
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
		"""
		Symbol Error Rate (SER): Percentage of incorrect characters in the decoded text.
		Returns np.nan if not calculated.
		"""
		if not self.solver.decoded:
			log.warning("SER not yet calculated. Run .decode() first.")
			return np.nan
		if not self.cipher.plaintext or len(self.solver.decoded) != len(self.cipher.plaintext):
			log.warning("Plaintext mismatch or missing. Cannot calculate SER.")
			log.warning(f"Decoded length: {len(self.solver.decoded)}")
			log.warning(f"Plaintext length: {len(self.cipher.plaintext)}")
			return np.nan

		correct = sum(1 for d, a in zip(self.solver.decoded, self.cipher.plaintext) if d == a)
		self._ser = 1.0 - (correct / len(self.solver.decoded))
		return self._ser

	def __str__(self) -> str:
		return (
			f"SolverAnalytics {{\n  solver: {self.solver.__str__()}\n  cipher: {self.cipher.__str__()}\n}}"
		)

	def __json__(self) -> dict[str, Any]:
		return {
			"solver": self.solver.__json__(),
			"cipher": self.cipher.__json__(),
		}
  
	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "SolverAnalytics":
		solver = Solver.__from_json__(json["solver"])
		cipher = Cipher.__from_json__(json["cipher"])

		analytics = SolverAnalytics(solver, cipher)

		return analytics

	def save(self, path: str) -> None:
		"""
		Saves the solver to a file.
		"""
		if not path.endswith(".json"):
			path += ".json"

		path_dir = os.path.dirname(path)
		if not os.path.exists(path_dir):
			os.makedirs(path_dir)
		with open(path, "w") as f:
			json.dump(self.__json__(), f, indent=4)