from dataclasses import dataclass
from typing import Any

@dataclass
class SolverConfig:
	"""Configuration for a solver.

	Attributes:
		confidence_threshold (float, optional): The confidence threshold for
			filtering out mappings. Defaults to 0.95.
		lock_iteration (int, optional): The iteration to lock the mapping
			probabilities. Defaults to 200.
		lr_phase1 (float, optional): The learning rate phase 1 for the
			probabilistic update. Defaults to 1.0.
		lr_phase2 (float, optional): The learning rate phase 2 for the
			probabilistic update. Defaults to 3.0.
		balance_phase1 (float, optional): The balance phase 1 for the
			probabilistic update. Defaults to 1.0.
		balance_phase2 (float, optional): The balance phase 2 for the
			probabilistic update. Defaults to 0.8.
		epsilon (float, optional): The epsilon value for numerical stability.
			Defaults to 1e-12.
		max_iters (int, optional): The maximum number of iterations to run.
			Defaults to 3500.

	Methods:
		__str__() -> str: Convert the SolverConfig to a string.
		__json__() -> dict[str, Any]: Convert the SolverConfig to a JSON object.
		__from_json__(json: dict[str, Any]) -> SolverConfig: Load a SolverConfig
			from a JSON object.

	"""

	confidence_threshold: float = 0.95
	lock_iteration: int = 200
	lr_phase1: float = 1.0
	lr_phase2: float = 3.0
	balance_phase1: float = 1.0
	balance_phase2: float = 0.8
	epsilon: float = 1e-12
	max_iters: int = 3500

	def __str__(self) -> str:
		"""Convert the SolverConfig to a string.

		Returns:
			str: The string representation of the SolverConfig

		"""
		return (
			f"SolverConfig {{\n"
			f"  confidence_threshold: {self.confidence_threshold}\n"
			f"  lock_iteration: {self.lock_iteration}\n"
			f"  lr_phase1: {self.lr_phase1}\n"
			f"  lr_phase2: {self.lr_phase2}\n"
			f"  balance_phase1: {self.balance_phase1}\n"
			f"  balance_phase2: {self.balance_phase2}\n"
			f"  epsilon: {self.epsilon}\n"
			f"  max_iters: {self.max_iters}\n}}"
		)

	def __json__(self) -> dict[str, Any]:
		"""Convert the SolverConfig to a JSON object.

		Returns:
			dict[str, Any]: The JSON object

		"""
		return {
			"confidence_threshold": self.confidence_threshold,
			"lock_iteration": self.lock_iteration,
			"lr_phase1": self.lr_phase1,
			"lr_phase2": self.lr_phase2,
			"balance_phase1": self.balance_phase1,
			"balance_phase2": self.balance_phase2,
			"epsilon": self.epsilon,
			"max_iters": self.max_iters,
		}

	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "SolverConfig":
		"""Load a SolverConfig from a JSON object.

		Args:
			json (dict[str, Any]): The JSON object to load the SolverConfig from

		Returns:
			SolverConfig: The resulting SolverConfig object

		"""
		return SolverConfig(
			confidence_threshold=json["confidence_threshold"],
			lock_iteration=json["lock_iteration"],
			lr_phase1=json["lr_phase1"],
			lr_phase2=json["lr_phase2"],
			balance_phase1=json["balance_phase1"],
			balance_phase2=json["balance_phase2"],
			epsilon=json["epsilon"],
			max_iters=json["max_iters"],
		)
