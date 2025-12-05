import pytest
import numpy as np


from classes.relaxation_solver import RelaxationSolver, SolverConfig


class DummyProfile:
	"""A simple stub for StatisticalProfile to facilitate math testing."""

	def __init__(self, size: int, name: str = "dummy"):
		self.size = size
		self.name = name

		self.unigram_frequencies = np.full(size, 1.0 / size)

		self.p_raw = np.full((size, size), 1.0 / (size * size))

		self.p_row_normalized = np.eye(size)
		self.p_col_normalized = np.eye(size)

		self.symbol_to_idx = {chr(65 + i): i for i in range(size)}

	def __json__(self) -> dict:
		return {"name": self.name}


class DummyCipher:
	"""A simple stub for Cipher."""

	def __init__(self, text: list[str]):
		self.ciphertext = text
		self.name = "test_cipher"

	def __json__(self) -> dict:
		return {"name": self.name}


class TestRelaxationSolver:
	@pytest.fixture
	def mock_solver_config(self):
		"""Creates a deterministic configuration for testing."""
		config = SolverConfig()
		config.epsilon = 1e-10
		config.max_iters = 5
		config.lock_iteration = 2
		config.confidence_threshold = 0.8
		config.lr_phase1 = 0.1
		config.lr_phase2 = 0.2
		config.balance_phase1 = 1.0
		config.balance_phase2 = 1.0
		return config

	@pytest.fixture
	def profiles(self):
		"""Returns a tuple of (eng_profile, cip_profile) of size 3."""

		return DummyProfile(3, "eng"), DummyProfile(3, "cip")

	@pytest.fixture
	def cipher_obj(self):
		"""Returns a cipher object with simple tokens."""

		return DummyCipher(["A", " B ", "C"])

	@pytest.fixture
	def solver(self, profiles, cipher_obj, mock_solver_config):
		"""Initializes the solver with dependencies."""
		eng, cip = profiles

		np.random.seed(42)
		return RelaxationSolver(eng, cip, cipher_obj, config=mock_solver_config)

	def test_initialization_state(self, solver):
		"""Test that the solver initializes attributes correctly."""

		assert solver.eng_profile.size == 3
		assert solver.cip_profile.size == 3

		assert solver.time == 0.0

		assert np.all(solver.locked_mappings == -1)
		assert solver.locked_mappings.shape == (3,)

	def test_initialize_map_structure(self, solver):
		"""Test the probability map structure and noise generation."""

		assert solver.p_map.shape == (3, 3)

		row_sums = solver.p_map.sum(axis=1)
		np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

		assert np.all(solver.p_map > 0)

	def test_run_updates_guesses_and_time(self, solver, mocker):
		"""Test that run() iterates, updates time, and generates guesses."""

		mocker.patch("time.time", side_effect=[100.0, 100.5])

		solver.run()

		assert solver.time == 0.5

		assert isinstance(solver.guesses, np.ndarray)
		assert solver.guesses.shape == (3,)

		assert solver.decoded is not None

	def test_locking_logic(self, solver, mock_solver_config):
		"""Test that high confidence mappings get locked."""

		solver.p_map[0] = [0.95, 0.025, 0.025]

		solver.p_map[1] = [0.33, 0.33, 0.34]

		solver.p_map[2] = [0.05, 0.05, 0.90]

		solver._lock_mappings()

		assert solver.locked_mappings[0] == 0
		assert solver.p_map[0, 0] == 1.0
		assert solver.p_map[0, 1] == 0.0

		assert solver.locked_mappings[1] == -1

		assert solver.locked_mappings[2] == 2
		assert solver.p_map[2, 2] == 1.0

	def test_restore_locked_mappings(self, solver):
		"""Test that locked mappings are enforced on a new map."""

		solver.locked_mappings[0] = 0

		new_p_map = np.array([[0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6]])

		solver._restore_locked_mappings(new_p_map)

		assert new_p_map[0, 0] == 1.0
		assert new_p_map[0, 1] == 0.0

		assert new_p_map[1, 1] == 0.3

	def test_decode_functionality(self, solver):
		"""Test decoding logic with mocked guesses."""

		solver.guesses = np.array([2, 0, 1])

		decoded = solver.decode()

		assert decoded == "CAB"
		assert solver.decoded == "CAB"

	def test_decode_without_guesses_logs_error(self, solver, caplog):
		"""Test that decoding without running the solver logs an error."""
		solver.guesses = None

		decoded = solver.decode()

		assert decoded == ""
		assert "Cannot decode. Guesses have not been generated" in caplog.text

	def test_decode_unknown_symbol(self, solver, caplog):
		"""Test decoding behavior when ciphertext contains unknown symbols."""
		solver.guesses = np.array([0, 1, 2])

		solver.cipher.ciphertext = ["A", "Z"]

		decoded = solver.decode()

		assert decoded == "A?"
		assert "Unknown symbol encountered" in caplog.text

	def test_serialization_methods(self, solver):
		"""Test __str__ and __json__ methods."""
		solver.guesses = np.array([0, 1, 2])
		solver.decoded = "ABC"

		s_rep = str(solver)
		assert "RelaxationSolver" in s_rep
		assert "decoded: ABC" in s_rep

		j_rep = solver.__json__()
		assert isinstance(j_rep, dict)
		assert j_rep["decoded"] == "ABC"
		assert j_rep["cipher"]["name"] == "test_cipher"
		assert isinstance(j_rep["guesses"], list)

	def test_from_json(self, mocker, mock_solver_config):
		"""Test loading from JSON, mocking the internal static imports."""

		json_data = {
			"eng_profile": {},
			"cip_profile": {},
			"cipher": {},
			"config": {},
			"guesses": [0, 1, 2],
			"decoded": "ABC",
			"time": 1.23,
		}

		mocker.patch(
			"classes.solver_config.SolverConfig.__from_json__",
			return_value=mock_solver_config,
		)

		d_eng = DummyProfile(3)
		d_cip = DummyProfile(3)
		d_ciph_obj = DummyCipher(["A"])

		mocker.patch(
			"classes.statistical_profile.StatisticalProfile.__from_json__",
			side_effect=[d_eng, d_cip],
		)
		mocker.patch("classes.cipher.Cipher.__from_json__", return_value=d_ciph_obj)

		solver = RelaxationSolver.__from_json__(json_data)

		assert isinstance(solver, RelaxationSolver)
		assert solver.decoded == "ABC"
		assert solver.time == 1.23
		assert solver.guesses is not None
		assert np.array_equal(solver.guesses, np.array([0, 1, 2]))
