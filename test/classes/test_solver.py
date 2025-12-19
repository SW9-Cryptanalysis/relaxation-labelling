import pytest
import numpy as np
from typing import Any
from abc import ABC


from classes.solver import Solver


class DummyProfile:
	def __init__(self, size: int):
		self.size = size


class DummyCipher:
	def __init__(self, name: str):
		self.name = name


class ConcreteTestSolver(Solver):
	"""A concrete implementation of Solver strictly for testing purposes."""

	def run(self) -> None:
		pass

	def decode(self) -> str:
		return "decoded_string"

	def __json__(self) -> dict[str, Any]:
		return {"test": "data"}

	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "ConcreteTestSolver":
		return ConcreteTestSolver(DummyProfile(5), DummyProfile(5), DummyCipher("test"))  # type: ignore


class TestSolverABC:
	@pytest.fixture
	def dependencies(self):
		"""Returns the dependencies required to initialize a Solver."""
		eng_profile = DummyProfile(size=26)
		cip_profile = DummyProfile(size=26)
		cipher = DummyCipher(name="test_cipher")
		return eng_profile, cip_profile, cipher

	@pytest.fixture
	def concrete_solver(self, dependencies):
		"""Returns an instance of the ConcreteTestSolver."""
		eng, cip, cipher = dependencies
		return ConcreteTestSolver(eng, cip, cipher)

	def test_abc_cannot_be_instantiated(self, dependencies):
		"""
		Test that the abstract Solver class cannot be instantiated directly.
		This ensures strict enforcement of the interface.
		"""
		eng, cip, cipher = dependencies

		with pytest.raises(TypeError) as excinfo:
			Solver(eng, cip, cipher)  # type: ignore

		msg = str(excinfo.value)
		assert "Can't instantiate abstract class Solver" in msg
		assert "run" in msg
		assert "decode" in msg
		assert "__json__" in msg

	def test_init_assigns_attributes_correctly(self, concrete_solver, dependencies):
		"""
		Test that the base __init__ correctly assigns the passed objects.
		"""
		eng, cip, cipher = dependencies

		assert concrete_solver.eng_profile is eng
		assert concrete_solver.cip_profile is cip
		assert concrete_solver.cipher is cipher

	def test_init_sets_sizes(self, concrete_solver):
		"""
		Test that cip_size and eng_size are extracted from the profiles.
		"""

		assert concrete_solver.eng_size == 26
		assert concrete_solver.cip_size == 26

	def test_init_sets_defaults(self, concrete_solver):
		"""
		Test that optional attributes are initialized to their default states.
		"""
		assert concrete_solver.guesses is None
		assert concrete_solver.decoded is None
		assert concrete_solver.time == 0.0

	def test_inheritance_structure(self, concrete_solver):
		"""
		Verify that the concrete implementation is recognized as a subclass
		of both Solver and ABC.
		"""
		assert isinstance(concrete_solver, Solver)
		assert isinstance(concrete_solver, ABC)

	def test_attribute_type_compatibility(self, concrete_solver):
		"""
		Verify that the attributes can hold the expected types (numpy arrays).
		"""

		concrete_solver.guesses = np.array([0, 1, 2])

		assert isinstance(concrete_solver.guesses, np.ndarray)
		assert concrete_solver.guesses.shape == (3,)
