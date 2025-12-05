import pytest
import numpy as np


from classes.mcmc_solver import McmcSolver, ENGLISH_FREQUENCIES


class DummyProfile:
	def __init__(self, size: int = 26):
		self.size = size

		self.symbols = [chr(97 + i) for i in range(size)]
		self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}

	def __json__(self) -> dict:
		return {"type": "dummy_profile", "size": self.size}


class DummyCipher:
	def __init__(self, ciphertext: list[str]):
		self.ciphertext = ciphertext
		self.name = "dummy_cipher"

	def __json__(self) -> dict:
		return {"name": self.name}


class DummyScorer:
	def __init__(self):
		self.call_count = 0

	def score(self, text: str) -> float:
		self.call_count += 1

		return len(text) * 10.0

	def __json__(self) -> dict:
		return {"type": "dummy_scorer"}


class TestMcmcSolver:
	@pytest.fixture
	def dependencies(self):
		"""Returns instances of the dummy classes."""

		eng = DummyProfile(26)
		cip = DummyProfile(26)

		cipher = DummyCipher(["a", "b", "c"])
		scorer = DummyScorer()
		return eng, cip, cipher, scorer

	@pytest.fixture
	def solver(self, dependencies):
		"""Initializes the McmcSolver with dummy dependencies."""
		eng, cip, cipher, scorer = dependencies
		return McmcSolver(eng, cip, cipher, scorer)

	def test_initialization(self, solver):
		"""Test attributes are set correctly on init."""
		assert solver.decoded is None
		assert solver.best_score is None

		expected = np.array(ENGLISH_FREQUENCIES) * 26
		np.testing.assert_array_equal(solver.expected_counts, expected)

	def test_decode_logic(self, solver):
		"""Test decoding using a specific key."""

		key = np.arange(26, dtype=int)

		decoded = solver.decode(guesses=key)
		assert decoded == "abc"

	def test_decode_no_guesses_error(self, solver, mocker):
		"""Test that calling decode without guesses logs an error."""

		mock_log = mocker.patch("classes.mcmc_solver.log")

		solver.guesses = None

		decoded = solver.decode()

		assert decoded == ""

		assert mock_log.error.call_count == 1

		mock_log.error.assert_called_with(
			"Cannot decode: Guesses not set. Run .run() first."
		)

	def test_calculate_combined_score(self, solver):
		"""
		Test the custom scoring logic (NGram score - Penalty).
		"""
		key = np.arange(26, dtype=int)
		text = "abc"

		score = solver._calculate_combined_score(text, key)

		assert score < 30.0
		assert isinstance(score, float)

	def test_make_swap_logic(self, solver, mocker):
		"""Test that _make_swap actually swaps two elements."""

		mocker.patch("numpy.random.randint", return_value=[0, 5])

		current_key = np.arange(26, dtype=int)

		new_key = solver._make_swap(current_key)

		assert new_key[0] == 5
		assert new_key[5] == 0

		assert new_key[1] == 1

	def test_make_swap_identical_indices(self, solver, mocker):
		"""Test that _make_swap changes nothing if random indices are identical."""
		mocker.patch("numpy.random.randint", return_value=[2, 2])

		current_key = np.arange(26, dtype=int)
		new_key = solver._make_swap(current_key)

		np.testing.assert_array_equal(current_key, new_key)

	def test_iteration_accepts_better_score(self, solver, mocker):
		"""
		Test iteration logic: If new score > old score, update key.
		"""
		solver.initial_temp = 100.0
		solver.temperature = 100.0

		solver.best_score = 10.0

		better_key = np.zeros(26, dtype=int)
		mocker.patch.object(solver, "_make_swap", return_value=better_key)

		mocker.patch.object(solver.scorer, "score", return_value=50.0)

		current_key = np.ones(26, dtype=int)
		current_score = 10.0

		result_key = solver._iteration(
			max_swaps=1, current_key=current_key, current_score=current_score, i=0
		)

		np.testing.assert_array_equal(result_key, better_key)

		assert solver.best_score == 50.0

	def test_iteration_probabilistic_acceptance(self, solver, mocker):
		"""
		Test iteration logic: New score is worse, but random chance accepts it.
		Strategy: Set 'best_score' very low so we can verify the accepted key
		becomes the new 'best' and is returned.
		"""
		solver.initial_temp = 100.0
		solver.temperature = 100.0

		solver.best_score = 30.0

		worse_key = np.zeros(26, dtype=int)
		current_key = np.ones(26, dtype=int)

		mocker.patch.object(solver, "_make_swap", return_value=worse_key)
		mocker.patch.object(solver.scorer, "score", return_value=40.0)

		mocker.patch("random.random", return_value=0.0)

		current_score = 50.0

		result_key = solver._iteration(1, current_key, current_score, 0)

		np.testing.assert_array_equal(result_key, worse_key)

		assert solver.best_score == 40.0

	def test_run_orchestration(self, solver, mocker, caplog):
		"""
		Test the main run loop.
		We set max_iters=1 to keep it fast.
		"""
		import logging

		caplog.set_level(logging.INFO)

		fake_initial = np.arange(26, dtype=int)
		mocker.patch("random.choices", return_value=fake_initial)

		solver.run(max_iters=1)

		assert solver.guesses is not None
		assert solver.decoded is not None
		assert solver.best_score is not None
		assert solver.time == 0.0

		np.testing.assert_array_equal(solver.guesses, fake_initial)

	def test_run_with_logging(self, solver, mocker):
		mock_log = mocker.patch("classes.mcmc_solver.log")

		solver.config = mocker.Mock()
		solver.config.max_iters = 1

		current_key = np.arange(26, dtype=int)

		solver.initial_temp = 50.0
		solver.temperature = 50.0
		solver.best_score = 0.0

		solver._iteration(
			max_swaps=1, current_key=current_key, current_score=10.0, i=2500
		)

		mock_log.info.assert_called_once_with("Iter 2500: Score 30.00 (Best: 0.00)")

	def test_json_serialization(self, solver):
		"""Test standard JSON serialization."""
		solver.guesses = np.array([0, 1, 2])
		solver.decoded = "abc"
		solver.time = 5.5

		json_data = solver.__json__()

		assert json_data["guesses"] == [0, 1, 2]
		assert json_data["decoded"] == "abc"
		assert json_data["scorer"]["type"] == "dummy_scorer"

	def test_from_json(self, mocker):
		"""Test deserialization."""

		d_eng = DummyProfile(26)
		d_cip = DummyProfile(26)
		d_ciph = DummyCipher(["a"])
		d_scor = DummyScorer()

		mocker.patch(
			"classes.statistical_profile.StatisticalProfile.__from_json__",
			side_effect=[d_eng, d_cip],
		)
		mocker.patch("classes.cipher.Cipher.__from_json__", return_value=d_ciph)
		mocker.patch(
			"classes.ngram_scorer.NGramScorer.__from_json__", return_value=d_scor
		)

		data = {
			"eng_profile": {},
			"cip_profile": {},
			"cipher": {},
			"scorer": {},
			"guesses": [0, 1, 2],
			"decoded": "abc",
			"time": 1.23,
		}

		new_solver = McmcSolver.__from_json__(data)

		assert isinstance(new_solver, McmcSolver)
		assert new_solver.decoded == "abc"
		np.testing.assert_array_equal(new_solver.guesses, np.array([0, 1, 2]))

	def test_run_with_initial_key(self, solver):
		initial_key = np.arange(26, dtype=int)

		solver.run(max_iters=0, initial_key=initial_key)

		np.testing.assert_array_equal(solver.guesses, initial_key)

	def test_run_keyboard_interrupt(self, solver, mocker):
		mocker.patch.object(solver, "_iteration", side_effect=KeyboardInterrupt)

		solver.run(max_iters=100)

		assert solver.decoded is not None
		assert solver.guesses is not None
