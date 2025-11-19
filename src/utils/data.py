import numpy as np
import pandas as pd
from utils.constants import DATA_PATH, CIPHER_PATH, RESULT_PATH
import os
import json
from typing import Any
from utils.logging import get_colored_logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from classes.solver_analytics import SolverAnalytics
	from classes.relaxation_solver import RelaxationSolver
	from classes.cipher import Cipher

log = get_colored_logger("Relaxation Solver")


def load_matrix(path: str) -> tuple[pd.DataFrame, np.ndarray]:
	"""Load a matrix from a CSV file.

	Args:
		path (str): The path to the CSV file

	Returns:
		tuple[pd.DataFrame, np.ndarray]: The matrix dataframe and matrix data

	"""
	df = pd.read_csv(DATA_PATH + path, index_col=0)
	return df, df.values.astype(float)


def load_cipher(name: str) -> dict[str, Any]:
	"""Load a cipher from the cipher directory.

	Args:
		name (str): The name of the cipher

	Returns:
		dict[str, Any]: The cipher data as a dictionary

	"""
	with open(CIPHER_PATH + f"{name}.json") as f:
		cipher_json = json.load(f)

	return cipher_json


def save_matrix(
	data: np.ndarray, index: list[str], columns: list[str], path: str,
) -> None:
	"""Save a matrix to a CSV file.

	Args:
		data (np.ndarray): The matrix data
		index (list[str]): The index of the matrix
		columns (list[str]): The columns of the matrix
		path (str): The path to save the matrix to

	Returns:
		None

	"""
	if not os.path.exists(DATA_PATH):
		os.makedirs(DATA_PATH)

	df = pd.DataFrame(data, index=index, columns=columns)
	df.to_csv(DATA_PATH + path)
	log.debug(f"Saved {data.shape[0]}x{data.shape[1]} matrix to '{DATA_PATH + path}'")


def matrix_exists(path: str) -> bool:
	"""Check if a matrix exists in the data directory.

	Args:
		path (str): The path to the matrix

	Returns:
		bool: True if the matrix exists, False otherwise

	"""
	return os.path.exists(DATA_PATH + path)


def write_results_to_file(
	analytics: "SolverAnalytics",
	solver: "RelaxationSolver",
	cipher: "Cipher",
) -> None:
	"""Write results to a file.

	Args:
		analytics (SolverAnalytics): Solver analytics (results) object
		solver (RelaxationSolver): Relaxation solver with information about
			the solving process
		cipher (Cipher): Cipher object (the cipher being analyzed)

	Returns:
		None

	"""
	if not os.path.exists(RESULT_PATH):
		os.makedirs(RESULT_PATH)
	with open(f"{RESULT_PATH}/{cipher.name}.txt", "w") as f:
		f.write(f"SER: {analytics.ser:.4f}\n")
		f.write(f"MER: {analytics.mer:.4f}\n")
		f.write(f"Plaintext: {cipher.plaintext}\n")
		f.write(f"Decoded: {solver.decoded}\n")


def load_ciphers_list() -> list[str]:
	"""Load a list of cipher names from the cipher directory.

	Returns:
		list[str]: List of cipher names

	"""
	list = os.listdir(CIPHER_PATH)
	list = [cipher.replace(".json", "") for cipher in list]
	sorted_list = sort_ciphers(list)
	for cipher in list:
		if cipher not in sorted_list:
			sorted_list.append(cipher)
	return sorted_list


def sort_ciphers(list: list[str]) -> list[str]:
	"""Sort ciphers according to length & difficulty.

	Args:
		list (list[str]): String list of cipher names

	Returns:
		list[str]: Sorted list of cipher names

	"""
	cipher_info = []
	for cipher in list:
		if cipher[0] == "c" and "mono" not in cipher:
			cipher_info.append(cipher.split("_"))

	cipher_info.sort(key=lambda x: (int(x[1]), int(x[2])))
	return ["_".join(x) for x in cipher_info]


def results_cached() -> bool:
	"""Check if results already exist (are cached).

	Returns:
		bool: True if results exist, False otherwise

	"""
	return os.path.exists(f"{RESULT_PATH}/results.json")


def save_results(results: list["SolverAnalytics"]) -> None:
	"""Save results to a file.

	Args:
		results (list[SolverAnalytics]): A list of SolverAnalytics objects

	"""
	if not os.path.exists(RESULT_PATH):
		os.makedirs(RESULT_PATH)

	with open(f"{RESULT_PATH}/results.json", "w") as f:
		json.dump([result.__json__() for result in results], f, indent=4)


def load_results() -> list["SolverAnalytics"]:
	"""Load results from earlier run.

	Returns:
		list[SolverAnalytics]: A list of SolverAnalytics objects

	"""
	results = []
	with open(f"{RESULT_PATH}/results.json") as f:
		result_json = json.load(f)
	from classes.solver_analytics import SolverAnalytics

	for result in result_json:
		results.append(SolverAnalytics.__from_json__(result))

	return results
