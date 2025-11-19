import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from classes.statistical_profile import StatisticalProfile
from classes.cipher import Cipher
from typing import Any

class Solver(ABC):
    """
    Abstract Base Class for a cipher solver.
    Defines the common interface that all solvers (Relaxation, MCMC, etc.)
    must implement to be compatible with the analytics tools.
    """
    
    def __init__(
        self, 
        eng_profile: StatisticalProfile, 
        cip_profile: StatisticalProfile, 
        cipher: Cipher
    ):
        self.eng_profile = eng_profile
        self.cip_profile = cip_profile
        self.cipher = cipher
        
        self.cip_size: int = cip_profile.size
        self.eng_size: int = eng_profile.size

        self.guesses: npt.NDArray[np.int_] | None = None
        self.decoded: str | None = None
        self.time: float = 0.0

    @abstractmethod
    def run(self) -> None:
        """
        Must implement the main solving method.
        """
        pass

    @abstractmethod
    def decode(self) -> str:
        """
        Must implement a method to decode the ciphertext
        using self.guesses and return the string.
        """
        pass
    
    @abstractmethod
    def __json__(self) -> dict[str, Any]:
        """
        Must implement a method to convert the solver to a JSON object.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def __from_json__(json: dict[str, Any]) -> "Solver":
        """
        Must implement a static method to load a solver from a JSON object.
        """
        pass
