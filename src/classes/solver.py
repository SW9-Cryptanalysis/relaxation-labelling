import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from classes.statistical_profile import StatisticalProfile
from classes.cipher import Cipher
from typing import Any

class Solver(ABC):
	"""Abstract Base Class for a cipher solver.

	Defines the common interface that all solvers (Relaxation, MCMC, etc.)
	must implement to be compatible with the analytics tools.

	Attributes:
		eng_profile (StatisticalProfile): The English statistical profile
		cip_profile (StatisticalProfile): The cipher statistical profile
		cipher (Cipher): The cipher object
		cip_size (int): The size of the cipher statistical profile
		eng_size (int): The size of the English statistical profile
		guesses (npt.NDArray[np.int_], optional): The guesses array. Defaults to None.
		decoded (str, optional): The decoded string. Defaults to None.
		time (float, optional): The time taken to run the solver. Defaults to 0.0.

	Methods:
		run() -> None: Run the solver.
		decode() -> str: Decode the ciphertext using self.guesses and return the string.
		__json__() -> dict[str, Any]: Convert the solver to a JSON object.
		__from_json__(json: dict[str, Any]) -> Solver: Create a solver from a JSON
			object.

	"""

	def __init__(
		self,
		eng_profile: StatisticalProfile,
		cip_profile: StatisticalProfile,
		cipher: Cipher,
	) -> None:
		"""Initialize a solver.

		Args:
			eng_profile (StatisticalProfile): The English statistical profile
			cip_profile (StatisticalProfile): The cipher statistical profile
			cipher (Cipher): The cipher object

		Returns:
			None

		"""
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
		"""Run the solver.

		Returns:
			None

		"""
		pass

	@abstractmethod
	def decode(self) -> str:
		"""Decode the ciphertext using self.guesses and return the string.

		Returns:
			str: The decoded string

		"""
		pass

	@abstractmethod
	def __json__(self) -> dict[str, Any]:
		"""Convert the solver to a JSON object.

		Returns:
			dict[str, Any]: The JSON object

		"""
		pass

	@staticmethod
	@abstractmethod
	def __from_json__(json: dict[str, Any]) -> "Solver":
		"""Create a solver from a JSON object.

		Args:
			json (dict[str, Any]): The JSON object to load the solver from

		Returns:
			Solver: The solver object

		"""
		pass
