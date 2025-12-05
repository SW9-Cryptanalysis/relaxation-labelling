import pytest


from main import solve_cipher


class DummyCipher:
	def __init__(self, name: str):
		self.name = name
		self.data_loaded = False

	def load_data(self):
		self.data_loaded = True


class DummyProfile:
	def __init__(self):
		self.name = "dummy_profile"


class DummySolver:
	"""Replaces RelaxationSolver for the orchestration test."""

	def __init__(self, eng_profile, cip_profile, cipher, config):
		self.eng_profile = eng_profile
		self.cip_profile = cip_profile
		self.cipher = cipher
		self.config = config
		self.run_called_count = 0
		self.decode_called_count = 0

	def run(self):
		self.run_called_count += 1

	def decode(self):
		self.decode_called_count += 1


class DummyAnalytics:
	"""Replaces SolverAnalytics."""

	def __init__(self, solver):
		self.solver = solver
		self.id = "analytics_obj"


@pytest.fixture
def mock_dependencies(mocker):
	"""
	Patches the classes imported inside the solve_cipher function.
	Instead of MagicMocks, we return our Dummy classes.
	"""

	mocker.patch("main.Cipher", side_effect=DummyCipher)

	mocker.patch("main.StatisticalProfile.from_cipher", return_value=DummyProfile())
	mocker.patch(
		"main.StatisticalProfile.from_english_corpus", return_value=DummyProfile()
	)

	mocker.patch("main.RelaxationSolver", side_effect=DummySolver)

	mocker.patch("main.SolverAnalytics", side_effect=DummyAnalytics)

	mocker.patch("main.SolverConfig", return_value="dummy_config")


def test_solve_cipher_happy_path(mock_dependencies):
	"""Test that the function initializes everything and returns analytics."""

	results = solve_cipher(cipher_name="test_cipher", restarts=1)

	assert len(results) == 1
	assert isinstance(results[0], DummyAnalytics)

	solver_instance = results[0].solver
	assert isinstance(solver_instance, DummySolver)
	assert solver_instance.run_called_count == 1
	assert solver_instance.decode_called_count == 1
	assert solver_instance.cipher.name == "test_cipher"
	assert solver_instance.cipher.data_loaded is True # type: ignore


def test_solve_cipher_multiple_restarts(mock_dependencies):
	"""Test that the loop runs correctly for multiple restarts."""

	results = solve_cipher(cipher_name="test_cipher", restarts=3)

	assert len(results) == 3

	assert results[0].solver is not results[1].solver

	for res in results:
		assert res.solver.run_called_count == 1 # type: ignore
		assert res.solver.decode_called_count == 1 # type: ignore


def test_solve_cipher_progress_bar_update(mock_dependencies, mocker):
	"""Test that the tqdm progress bar is updated."""

	mock_pbar = mocker.Mock()

	solve_cipher(cipher_name="test_cipher", restarts=2, p_bar=mock_pbar)

	assert mock_pbar.update.call_count == 2
	mock_pbar.update.assert_called_with(1)


def test_solve_cipher_failure_exit(mock_dependencies, mocker, caplog):
	"""
	Test the case where analytics is empty (e.g., restarts=0).
	Expected: Log error and sys.exit(1).
	"""

	mock_exit = mocker.patch("builtins.exit")
	mock_log = mocker.patch("main.log.error")

	solve_cipher(cipher_name="test_cipher", restarts=0)

	assert mock_log.call_count == 1
	assert mock_log.call_args[0][0] == "Could not decode ciphertext. Check for symbol mismatches."
	mock_exit.assert_called_once_with(1)
