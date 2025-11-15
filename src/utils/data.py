import numpy as np
import pandas as pd
from utils.constants import DATA_PATH, CIPHER_PATH, RESULT_PATH
import os
import json
from typing import Any
from utils.logging import get_colored_logger
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from classes.solver_analytics import SolverAnalytics
	from classes.relaxation_solver import RelaxationSolver
	from classes.cipher import Cipher

log = get_colored_logger("Relaxation Solver")


def load_matrix(path) -> tuple[pd.DataFrame, np.ndarray]:
	df = pd.read_csv(DATA_PATH + path, index_col=0)
	return df, df.values.astype(float)


def load_cipher(name: str) -> dict[str, Any]:
	with open(CIPHER_PATH + f"{name}.json", "r") as f:
		cipher_json = json.load(f)

	return cipher_json


def save_matrix(data, index, columns, path):
	if not os.path.exists(DATA_PATH):
		os.makedirs(DATA_PATH)

	df = pd.DataFrame(data, index=index, columns=columns)
	df.to_csv(DATA_PATH + path)
	log.debug(f"Saved {data.shape[0]}x{data.shape[1]} matrix to '{DATA_PATH + path}'")


def matrix_exists(path) -> bool:
	return os.path.exists(DATA_PATH + path)


def write_results_to_file(
	analytics: "SolverAnalytics", solver: "RelaxationSolver", cipher: "Cipher"
):
	if not os.path.exists(RESULT_PATH):
		os.makedirs(RESULT_PATH)
	with open(f"{RESULT_PATH}/{cipher.name}.txt", "w") as f:
		f.write(f"SER: {analytics.ser:.4f}\n")
		f.write(f"MER: {analytics.mer:.4f}\n")
		f.write(f"Plaintext: {cipher.plaintext}\n")
		f.write(f"Decoded: {solver.decoded}\n")


def load_ciphers_list() -> list[str]:
	list = os.listdir(CIPHER_PATH)
	sorted_list = sort_ciphers(list)
	for cipher in list:
		if cipher not in sorted_list:
			sorted_list.append(cipher)
	return sorted_list


def sort_ciphers(list: list[str]) -> list[str]:
	cipher_info = []
	for cipher in list:
		if cipher[0] == "c" and not "mono" in cipher:
			cipher_info.append(cipher.replace(".json", "").split("_"))

	cipher_info.sort(key=lambda x: (int(x[1]), int(x[2])))
	return ["_".join(x) for x in cipher_info]


def results_cached() -> bool:
	return os.path.exists(f"{RESULT_PATH}/results.json")


def save_results(results: list["SolverAnalytics"]) -> None:
	if not os.path.exists(RESULT_PATH):
		os.makedirs(RESULT_PATH)

	with open(f"{RESULT_PATH}/results.json", "w") as f:
		json.dump([result.__json__() for result in results], f, indent=4)


def load_results() -> list["SolverAnalytics"]:
	results = []
	with open(f"{RESULT_PATH}/results.json", "r") as f:
		result_json = json.load(f)
	from classes.solver_analytics import SolverAnalytics

	for result in result_json:
		results.append(SolverAnalytics.__from_json__(result))

	return results
