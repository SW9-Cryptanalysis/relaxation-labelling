import os
import dotenv

dotenv.load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(os.getcwd())) 
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") 

def get_default_path(relative_dir: str) -> str:
    """Creates an absolute default path relative to the script's location."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_dir))

if os.getenv("DATA_PATH"):
    DATA_PATH = os.getenv("DATA_PATH")
else:
    DATA_PATH = get_default_path("../data/")

if os.getenv("MODEL_PATH"):
    MODEL_PATH = os.getenv("MODEL_PATH")
else:
    MODEL_PATH = get_default_path("../model/")

if os.getenv("LOG_PATH"):
    LOG_PATH = os.getenv("LOG_PATH")
else:
    LOG_PATH = get_default_path("../log/")

if os.getenv("RESULT_PATH"):
    RESULT_PATH = os.getenv("RESULT_PATH")
else:
    RESULT_PATH = get_default_path("../result/")

if os.getenv("RESULT_PATH_MCMC"):
    RESULT_PATH_MCMC = os.getenv("RESULT_PATH_MCMC")
else:
    RESULT_PATH_MCMC = get_default_path("../result/mcmc/")

if os.getenv("CIPHER_PATH"):
    CIPHER_PATH = os.getenv("CIPHER_PATH")
else:
    CIPHER_PATH = get_default_path("../cipher/")

if os.getenv("NGRAM_PATH"):
    NGRAM_PATH = os.getenv("NGRAM_PATH")
else:
    NGRAM_PATH = get_default_path("../ngram/")


RANDOM_RESTARTS = 1
ALPHABET = "abcdefghijklmnopqrstuvwxyz"