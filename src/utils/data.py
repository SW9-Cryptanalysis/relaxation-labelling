import numpy as np
import pandas as pd
from utils.constants import DATA_PATH, CIPHER_PATH
import os
import json
from typing import Any
from utils.logging import get_colored_logger

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
