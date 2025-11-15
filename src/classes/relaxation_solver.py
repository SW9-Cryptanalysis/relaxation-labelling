import numpy as np
import logging
from utils.logging import get_colored_logger
from classes.solver_config import SolverConfig
import time
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from classes import StatisticalProfile, Cipher

log = get_colored_logger("Relaxation Solver")


class RelaxationSolver:
	"""
	Solves substitution ciphers using a relaxation labeling algorithm based on
	bigram statistics.
	"""

	def __init__(
		self,
		eng_profile: "StatisticalProfile",
		cip_profile: "StatisticalProfile",
		cipher: "Cipher",
		config: SolverConfig = SolverConfig(),
	):
		self.eng_profile = eng_profile
		self.cip_profile = cip_profile
		self.cipher = cipher
		self.config = config

		self.time = 0.0

		self.cip_size = cip_profile.size
		self.eng_size = eng_profile.size

		# Pre-calculate the transposed English bigram matrix for efficiency
		self._eng_p_raw_T = self.eng_profile.p_raw.T

		# Initialize the probability map
		self._initialize_map()

		# Stores indices of mappings that are "locked"
		self.locked_mappings = np.full(self.cip_size, -1, dtype=int)

		# Pre-calculate the true key mapping (if available) for accuracy checks
		self._valid_key: dict[int, int] | None = None

		# --- Solver Results (initialized to None) ---
		self.p_map: np.ndarray  # The probability map
		self.guesses: np.ndarray | None = None

	def _initialize_map(self) -> None:
		"""Initializes the probability map p_map."""
		self.p_map = np.zeros((self.cip_size, self.eng_size))
		for i in range(self.cip_size):
			for j in range(self.eng_size):
				self.p_map[i, j] = (
					self.cip_profile.unigram_frequencies[i]
					* self.eng_profile.unigram_frequencies[j]
				)

		# Normalize rows to sum to 1
		self.p_map = self.p_map / (
			self.p_map.sum(axis=1, keepdims=True) + self.config.epsilon
		)

	def run(self, valid_key: dict[int, int] | None = None) -> None:
		"""
		Runs the relaxation algorithm for a given number of iterations.
		"""
		start_time = time.time()
		for i in range(self.config.max_iters):
			if i == self.config.lock_iteration:
				self._lock_mappings()

			# --- Calculate Support (Compatibility) ---
			# Support from successor symbols (c_i -> e_j implies c_k -> e_l)
			support_successor = (
				self.cip_profile.p_row_normalized @ self.p_map @ self._eng_p_raw_T
			)
			# Support from predecessor symbols
			support_predecessor = (
				self.cip_profile.p_col_normalized.T
				@ self.p_map
				@ self.eng_profile.p_raw
			)
			total_support = support_successor + support_predecessor

			# --- Update Step ---
			lr = (
				self.config.lr_phase1
				if i < self.config.lock_iteration
				else self.config.lr_phase2
			)
			p_map_new = self.p_map * (1 + lr * total_support)

			# --- Balance Step (to match unigram frequencies) ---
			balance_power = (
				self.config.balance_phase1
				if i < self.config.lock_iteration
				else self.config.balance_phase2
			)
			col_sums = p_map_new.sum(axis=0) + self.config.epsilon
			balance_factor = (
				self.eng_profile.unigram_frequencies * self.cip_size / col_sums
			) ** balance_power
			p_map_new = p_map_new * balance_factor

			# --- Normalize Step ---
			p_map_new = p_map_new / (
				p_map_new.sum(axis=1, keepdims=True) + self.config.epsilon
			)

			if i >= self.config.lock_iteration:
				self._restore_locked_mappings(p_map_new)

			self.p_map = p_map_new

			# --- Log Accuracy (if true key is known) ---
			if i % 50 == 0 and valid_key:
				temp_guesses = np.argmax(self.p_map, axis=1)
				correct = sum(1 for c, e in valid_key.items() if temp_guesses[c] == e)
				log.debug(f"Iter {i:3d}: Acc {100 * correct / len(valid_key):.1f}%")

		# --- Finalize results ---
		self.guesses = np.argmax(self.p_map, axis=1)  # This is the final guesses
		self.decode()

		self.time = time.time() - start_time

	def _lock_mappings(self) -> None:
		"""Locks high-confidence mappings."""
		confs = np.max(self.p_map, axis=1)
		winners = np.argmax(self.p_map, axis=1)

		for i in range(self.cip_size):
			if confs[i] < self.config.confidence_threshold:
				continue

			self.locked_mappings[i] = winners[i]
			self.p_map[i, :] = 0.0
			self.p_map[i, winners[i]] = 1.0

	def _restore_locked_mappings(self, p_map_new: np.ndarray) -> None:
		"""
		Enforces locked mappings on the new probability map.
		Note: This modifies p_map_new in place.
		"""
		for i in range(self.cip_size):
			locked_val = self.locked_mappings[i]
			if locked_val == -1:
				continue

			p_map_new[i, :] = 0.0
			p_map_new[i, locked_val] = 1.0

	def decode(self) -> str:
		"""
		Decodes the ciphertext using the final guessed key.
		"""
		if self.guesses is None:
			logging.error(
				"Cannot decode. Guesses have not been generated. Run .run() first."
			)
			return ""

		decoded_chars = []
		idx_to_letter = {v: k for k, v in self.eng_profile.symbol_to_idx.items()}
		sym_to_idx = self.cip_profile.symbol_to_idx

		for sym in self.cipher.ciphertext:
			# Assume symbols might be strings that need stripping
			sym = sym.strip()

			if sym in sym_to_idx:
				row = sym_to_idx[sym]  # No redundant str()
				best_letter_idx = self.guesses[row]
				decoded_chars.append(idx_to_letter[best_letter_idx])
			else:
				logging.warning(f"Unknown symbol encountered during decode: {sym}")
				decoded_chars.append("?")

		self.decoded = "".join(decoded_chars)
		return self.decoded

	def __str__(self) -> str:
		return (
			f"RelaxationSolver {{\n"
			f"  cipher: {self.cipher.name}\n"
			f"  config: {self.config.__str__()}\n"
			f"  guesses: {self.guesses}\n"
			f"  decoded: {self.decoded}\n"
			f"  time: {self.time:.4f}\n}}"
		)

	def __json__(self) -> dict[str, Any]:
		return {
			"eng_profile": self.eng_profile.__json__(),
			"cip_profile": self.cip_profile.__json__(),
			"cipher": self.cipher.__json__(),
			"config": self.config.__json__(),
			"guesses": self.guesses.tolist() if self.guesses is not None else None,
			"decoded": self.decoded,
			"time": self.time,
		}
  
	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "RelaxationSolver":
		from classes.solver_config import SolverConfig
		from classes.cipher import Cipher
		from classes.statistical_profile import StatisticalProfile
		eng_profile = StatisticalProfile.__from_json__(json["eng_profile"])
		cip_profile = StatisticalProfile.__from_json__(json["cip_profile"])
		cipher = Cipher.__from_json__(json["cipher"])
		config = SolverConfig.__from_json__(json["config"])
		guesses = np.array(json["guesses"]) if json["guesses"] is not None else None
		decoded = json["decoded"]

		solver = RelaxationSolver(eng_profile, cip_profile, cipher, config)
		solver.guesses = guesses
		solver.decoded = decoded
		solver.time = json["time"]

		return solver