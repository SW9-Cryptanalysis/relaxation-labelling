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

def load_matrix(path) -> tuple[pd.DataFrame, np.ndarray]:
	df = pd.read_csv(DATA_PATH + path, index_col=0)
	return df, df.values.astype(float)

def load_cipher(name: str) -> dict[str, Any]:
	with open(CIPHER_PATH + f'{name}.json', 'r') as f:
		cipher_json = json.load(f)

	return cipher_json

def save_matrix(data, index, columns, path):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    df = pd.DataFrame(data, index=index, columns=columns)
    df.to_csv(DATA_PATH + path)
    log.debug(f"Saved {data.shape[0]}x{data.shape[1]} matrix to '{DATA_PATH +path}'")
    
def matrix_exists(path) -> bool:
    return os.path.exists(DATA_PATH + path)

def write_results_to_file (analytics: "SolverAnalytics", solver: "RelaxationSolver", cipher: "Cipher"):
	if not os.path.exists(RESULT_PATH):
		os.makedirs(RESULT_PATH)
	with open(f"{RESULT_PATH}/{cipher.name}.txt", "w") as f:
		f.write(f"SER: {analytics.ser:.4f}\n")
		f.write(f"MER: {analytics.mer:.4f}\n")
		f.write(f"Plaintext: {cipher.plaintext}\n")
		f.write(f"Decoded: {solver.decoded}\n")
  
def load_ciphers_list() -> list[str]:
	return os.listdir(CIPHER_PATH)
