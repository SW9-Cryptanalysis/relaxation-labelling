import pytest
import numpy as np
import pandas as pd

from utils.data import (
	load_matrix,
	load_cipher,
	save_matrix,
	matrix_exists,
	write_results_to_file,
	load_ciphers_list,
	sort_ciphers,
	results_cached,
	save_results,
	load_results,
)


class DummyCipher:
	def __init__(self, name="test_c", plaintext="hello"):
		self.name = name
		self.plaintext = plaintext


class DummySolver:
	def __init__(self, decoded="hullo"):
		self.decoded = decoded


class DummyAnalytics:
	def __init__(self, ser=0.1, mer=0.2):
		self.ser = ser
		self.mer = mer

	def __json__(self):
		return {"ser": self.ser, "mer": self.mer}

	@staticmethod
	def __from_json__(json_data, type_hint):
		return DummyAnalytics(json_data["ser"], json_data["mer"])


@pytest.fixture
def mock_paths(mocker):
	"""Patches the constants to safe dummy paths."""
	mocker.patch("utils.data.DATA_PATH", "dummy_data/")
	mocker.patch("utils.data.CIPHER_PATH", "dummy_cipher/")
	mocker.patch("utils.data.RESULT_PATH", "dummy_result/")


@pytest.fixture
def dummy_dataframe():
	data = {"A": [1, 2], "B": [3, 4]}
	return pd.DataFrame(data, index=["X", "Y"])


class TestUtilsData:
	def test_load_matrix(self, mocker, mock_paths, dummy_dataframe):
		mock_read = mocker.patch("pandas.read_csv", return_value=dummy_dataframe)

		df, values = load_matrix("test.csv")

		mock_read.assert_called_once_with("dummy_data/test.csv", index_col=0)

		pd.testing.assert_frame_equal(df, dummy_dataframe)
		assert isinstance(values, np.ndarray)
		assert values.dtype == float

	def test_load_cipher(self, mocker, mock_paths):
		mock_open = mocker.patch(
			"builtins.open", mocker.mock_open(read_data='{"key": "val"}')
		)

		result = load_cipher("my_cipher")

		mock_open.assert_called_once_with("dummy_cipher/my_cipher.json")
		assert result == {"key": "val"}

	def test_save_matrix(self, mocker, mock_paths):
		mocker.patch("os.path.exists", return_value=False)
		mock_makedirs = mocker.patch("os.makedirs")
		mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")
		mock_log = mocker.patch("utils.data.log")

		data = np.array([[1, 2], [3, 4]])
		save_matrix(data, ["r1", "r2"], ["c1", "c2"], "out.csv")

		mock_makedirs.assert_called_once_with("dummy_data/")

		assert mock_to_csv.called
		args, _ = mock_to_csv.call_args
		assert args[0].endswith("dummy_data/out.csv")

		mock_log.debug.assert_called_once()

	def test_matrix_exists(self, mocker, mock_paths):
		mock_exists = mocker.patch("os.path.exists", return_value=True)

		assert matrix_exists("test.csv") is True
		mock_exists.assert_called_once_with("dummy_data/test.csv")

	def test_write_results_to_file(self, mocker, mock_paths):
		mocker.patch("os.path.exists", return_value=False)
		mock_makedirs = mocker.patch("os.makedirs")
		mock_open = mocker.patch("builtins.open", mocker.mock_open())

		analytics = DummyAnalytics(ser=0.05, mer=0.10)
		solver = DummySolver(decoded="decrypted")
		cipher = DummyCipher(name="z340", plaintext="secret")

		write_results_to_file(analytics, solver, cipher)  # type: ignore

		mock_makedirs.assert_called_once_with("dummy_result/")
		mock_open.assert_called_once_with("dummy_result//z340.txt", "w")

		handle = mock_open()
		handle.write.assert_any_call("SER: 0.0500\n")
		handle.write.assert_any_call("Plaintext: secret\n")

	def test_sort_ciphers_logic(self):
		"""
		Tests the specific sorting logic:
		1. Filters for starting with 'c' and not containing 'mono'.
		2. Splits by '_'.
		3. Sorts by 2nd element (len) then 3rd element (diff).
		"""
		input_list = ["c_100_5", "c_50_5", "c_100_1", "mono_cipher", "other_file"]

		sorted_list = sort_ciphers(input_list)

		expected = ["c_50_5", "c_100_1", "c_100_5"]
		assert sorted_list == expected

	def test_load_ciphers_list(self, mocker, mock_paths):
		files = ["c_20_2.json", "c_10_1.json", "random.txt"]
		mocker.patch("os.listdir", return_value=files)

		result = load_ciphers_list()

		assert result[0] == "c_10_1"
		assert result[1] == "c_20_2"
		assert "random.txt" in result
		assert len(result) == 3

	def test_results_cached(self, mocker, mock_paths):
		mock_exists = mocker.patch("os.path.exists", return_value=True)

		assert results_cached() is True
		mock_exists.assert_called_with("dummy_result//results.json")

	def test_save_results(self, mocker, mock_paths):
		mock_makedirs = mocker.patch("os.makedirs")
		mock_open = mocker.patch("builtins.open", mocker.mock_open())
		mock_dump = mocker.patch("json.dump")

		results = [DummyAnalytics(), DummyAnalytics()]

		save_results(results)  # type: ignore

		mock_makedirs.assert_called_once_with("dummy_result/")
		mock_open.assert_called_once_with("dummy_result//results.json", "w")

		assert mock_dump.call_count == 1
		args, _ = mock_dump.call_args
		serialized_list = args[0]
		assert len(serialized_list) == 2
		assert serialized_list[0]["ser"] == 0.1

	def test_load_results(self, mocker, mock_paths):
		raw_json = [{"ser": 0.5, "mer": 0.6}, {"ser": 0.1, "mer": 0.2}]
		mocker.patch("builtins.open", mocker.mock_open())
		mocker.patch("json.load", return_value=raw_json)

		mocker.patch(
			"classes.solver_analytics.SolverAnalytics.__from_json__",
			side_effect=DummyAnalytics.__from_json__,
		)

		results = load_results()

		assert len(results) == 2
		assert isinstance(results[0], DummyAnalytics)
		assert results[0].ser == 0.5
		assert results[1].mer == 0.2
