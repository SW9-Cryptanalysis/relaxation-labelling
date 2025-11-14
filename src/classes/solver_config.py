from dataclasses import dataclass

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