import os
import dotenv

dotenv.load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(os.getcwd()))
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def get_path(name: str, relative_dir: str) -> str:
	"""Get a path from an environment variable or a default path.

	Args:
		name (str): The name of the environment variable
		relative_dir (str): The relative path to the default directory

	Returns:
		str: The path from the environment variable or the default path

	"""
	if os.getenv(name) is not None:
		return os.getenv(name) # type: ignore
	else:
		return get_default_path(relative_dir)

def get_default_path(relative_dir: str) -> str:
	"""Create an absolute default path relative to the script's location."""
	return os.path.abspath(
		os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_dir),
	)

DATA_PATH = get_path("DATA_PATH", "../data/")
MODEL_PATH = get_path("MODEL_PATH", "../model/")
LOG_PATH = get_path("LOG_PATH", "../log/")
RESULT_PATH = get_path("RESULT_PATH", "../result/")
RESULT_PATH_MCMC = get_path("RESULT_PATH_MCMC", "../result/mcmc/")
CIPHER_PATH = get_path("CIPHER_PATH", "../cipher/")
NGRAM_PATH = get_path("NGRAM_PATH", "../ngram/")

RANDOM_RESTARTS = 1
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
