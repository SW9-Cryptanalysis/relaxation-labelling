from dataclasses import dataclass
from typing import Any

@dataclass
class SolverConfig:
	confidence_threshold: float = 0.95
	lock_iteration: int = 200
	lr_phase1: float = 1.0
	lr_phase2: float = 3.0
	balance_phase1: float = 1.0
	balance_phase2: float = 0.8
	epsilon: float = 1e-12
	max_iters: int = 3500
	
	def __str__(self) -> str:
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