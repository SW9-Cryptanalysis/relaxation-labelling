import pytest
import numpy as np
import json
import os
from typing import Any

from classes.solver_analytics import SolverAnalytics
from classes.solver import Solver


class DummyProfile:
	def __init__(self):
		self.symbol_to_idx = {"a": 0, "b": 1}
		self.size = 2


class DummyCipher:
	def __init__(self):
		self.name = "test_cipher"
		self.plaintext = "ab"
		self.key = {"a": ["0"], "b": ["1"]}


class DummySolver(Solver):
	"""Minimal concrete implementation of Solver."""

	def __init__(self):
		self.eng_profile = DummyProfile()
		self.cip_profile = DummyProfile()
		self.cip_profile.symbol_to_idx = {"0": 0, "1": 1}

		self.cipher = DummyCipher()
		self.decoded = "ab"
		self.guesses = np.array([0, 1])
		self.time = 1.0

	def run(self):
		pass

	def decode(self):
		return self.decoded or "?"

	def __json__(self):
		return {"type": "dummy"}

	@staticmethod
	def __from_json__(j):
		return DummySolver()


class DummyScorer:
	"""Replaces NGramScorer."""

	def __init__(self, n, path):
		pass

	def score(self, text):
		return 100.0 if text == "ab" else 0.0


class TestSolverAnalytics:
	@pytest.fixture
	def analytics(self, mocker):
		"""Creates an Analytics instance with a DummySolver and DummyScorer."""
		mocker.patch("classes.solver_analytics.NGramScorer", side_effect=DummyScorer)
		solver = DummySolver()
		return SolverAnalytics(solver)

	def test_initialization(self, analytics):
		assert analytics.solver is not None
		assert analytics._mer is None
		assert analytics._ser is None

	def test_score_property(self, analytics):
		assert analytics.score == 100.0
		assert analytics.score == 100.0

	def test_score_returns_nan_if_no_decoded(self, analytics, mocker):
		"""Test handling of missing decoded text using mock_log."""
		mock_log = mocker.patch("classes.solver_analytics.log")

		analytics.solver.decoded = None

		assert np.isnan(analytics.score)
		mock_log.warning.assert_called_with(
			"Cannot calculate score. Decoded not yet calculated."
		)

	def test_valid_key_construction(self, analytics):
		v_key = analytics.valid_key

		assert len(v_key) == 2
		assert v_key[0] == 0
		assert v_key[1] == 1

	def test_mer_perfect_match(self, analytics):
		assert analytics.mer == 0.0

	def test_mer_calculation_errors(self, analytics):
		analytics.solver.guesses = np.array([1, 0])
		assert analytics.mer == 1.0

	def test_mer_returns_nan_missing_guesses(self, analytics, mocker):
		"""Test MER handles missing guesses using mock_log."""
		mock_log = mocker.patch("classes.solver_analytics.log")

		analytics.solver.guesses = None

		assert np.isnan(analytics.mer)
		mock_log.warning.assert_called_with("MER: Guesses not set. Run .run() first.")

	def test_ser_perfect_match(self, analytics):
		assert analytics.ser == 0.0

	def test_ser_calculation_errors(self, analytics):
		analytics.solver.decoded = "aa"
		assert analytics.ser == 0.5

	def test_ser_mismatched_lengths(self, analytics, mocker):
		"""Test SER returns NaN if lengths differ using mock_log."""
		mock_log = mocker.patch("classes.solver_analytics.log")

		analytics.solver.decoded = "abc"

		assert np.isnan(analytics.ser)

		mock_log.warning.assert_any_call(
			"Plaintext mismatch or missing. Cannot calculate SER."
		)
		mock_log.warning.assert_any_call("Decoded length: 3")
		mock_log.warning.assert_any_call("Plaintext length: 2")

	def test_json_serialization(self, analytics):
		j_data = analytics.__json__()

		assert j_data["mer"] == 0.0
		assert j_data["ser"] == 0.0
		assert j_data["time"] == 1.0
		assert j_data["cipher"] == "test_cipher"
		assert j_data["decoded"] == "ab"

	def test_save_to_file(self, analytics, mocker):
		mock_open = mocker.patch("builtins.open", mocker.mock_open())
		mocker.patch("os.makedirs")
		mocker.patch("os.path.exists", return_value=False)
		mocker.patch("json.dump")

		analytics.save("test_output")

		args, _ = mock_open.call_args
		assert args[0].endswith("test_output.json")
		assert args[1] == "w"

	def test_from_json_factory(self, mocker):
		mock_mcmc = mocker.MagicMock()
		mock_relax = mocker.MagicMock()

		mocker.patch(
			"classes.mcmc_solver.McmcSolver.__from_json__", return_value=mock_mcmc
		)
		mocker.patch(
			"classes.relaxation_solver.RelaxationSolver.__from_json__",
			return_value=mock_relax,
		)

		mocker.patch("classes.solver_analytics.NGramScorer")

		res_mcmc = SolverAnalytics.__from_json__({"solver": {}}, "mcmcsolver")
		assert res_mcmc.solver == mock_mcmc

		res_relax = SolverAnalytics.__from_json__({"solver": {}}, "relaxationsolver")
		assert res_relax.solver == mock_relax

		with pytest.raises(ValueError):
			SolverAnalytics.__from_json__({}, "unknown_type")

	def test_valid_key_skips_unknown_letters(self, analytics):
		analytics.solver.cipher.key = {"z": ["99"], "a": ["0"]}
		analytics.eng_profile.symbol_to_idx = {"a": 0}

		v_key = analytics.valid_key

		assert len(v_key) == 1
		assert 99 not in v_key

	def test_mer_caching(self, analytics):
		analytics._mer = 0.123

		result = analytics.mer

		assert result == 0.123

	def test_mer_no_valid_key(self, analytics, mocker):
		mock_log = mocker.patch("classes.solver_analytics.log")
		analytics.solver.cipher.key = {"z": ["99"]}
		analytics.eng_profile.symbol_to_idx = {"a": 0}

		assert np.isnan(analytics.mer)
		mock_log.warning.assert_called_with("MER: No valid key loaded. Cannot calculate.")

	def test_ser_not_decoded(self, analytics, mocker):
		mock_log = mocker.patch("classes.solver_analytics.log")
		analytics.solver.decoded = ""

		assert np.isnan(analytics.ser)
		mock_log.warning.assert_called_with("SER not yet calculated. Run .decode() first.")

	def test_str_representation(self, analytics, mocker):
		mocker.patch.object(analytics.solver, "__str__", return_value="MockedSolverString")

		s_rep = str(analytics)

		assert "SolverAnalytics" in s_rep
		assert "MockedSolverString" in s_rep
