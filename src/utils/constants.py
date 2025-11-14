import os
import dotenv

dotenv.load_dotenv()

if os.getenv('DATA_PATH'):
    DATA_PATH = os.getenv('DATA_PATH')
else:
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/')

if os.getenv('MODEL_PATH'):
    MODEL_PATH = os.getenv('MODEL_PATH')
else:
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model/')

if os.getenv('LOG_PATH'):
    LOG_PATH = os.getenv('LOG_PATH')
else:
    LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../log/')
    
if os.getenv('RESULT_PATH'):
    RESULT_PATH = os.getenv('RESULT_PATH')
else:
    RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../result/')
    
if os.getenv('CIPHER_PATH'):
    CIPHER_PATH = os.getenv('CIPHER_PATH')
else:
    CIPHER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cipher/')

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
