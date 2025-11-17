import numpy as np
from utils.data import load_matrix, matrix_exists, save_matrix
from classes.cipher import Cipher
from collections import Counter
import nltk
import string
from utils.logging import get_colored_logger
from typing import Any

log = get_colored_logger("Relaxation Solver")


class StatisticalProfile:
	def __init__(self, p_raw: np.ndarray, symbols: list[str]):
		self.p_raw = p_raw
		self.symbols = symbols
		self.symbol_to_idx = {str(sym): idx for idx, sym in enumerate(symbols)}
		self.size = len(symbols)

		self._p_row_normalized = None
		self._p_col_normalized = None
		self._unigram_frequencies = None

	@property
	def p_row_normalized(self) -> np.ndarray:
		if self._p_row_normalized is None:
			self._p_row_normalized = self._normalize_matrix(self.p_raw, axis=1)
		return self._p_row_normalized

	@property
	def p_col_normalized(self) -> np.ndarray:
		if self._p_col_normalized is None:
			self._p_col_normalized = self._normalize_matrix(self.p_raw, axis=0)
		return self._p_col_normalized

	@property
	def unigram_frequencies(self) -> np.ndarray:
		if self._unigram_frequencies is None:
			self._unigram_frequencies = self._get_frequencies()
		return self._unigram_frequencies

	def _get_frequencies(self) -> np.ndarray:
		p_sums = self.p_raw.sum(axis=1)
		return p_sums / p_sums.sum()

	def _normalize_matrix(self, P: np.ndarray, axis=0) -> np.ndarray:
		P = P / P.sum(axis=axis, keepdims=True)
		return P

	@staticmethod
	def from_matrix_file(matrix_file: str):
		pd, matrix = load_matrix(matrix_file)
		symbols = pd.index.to_list()
		return StatisticalProfile(matrix, symbols)

	@staticmethod
	def _get_bigrams(text: list[str]) -> Counter[tuple[str, str]]:
		"""Create a counter of adjacent symbol pairs and their occurences

		Args:
			text (list[str]): The text to create the counter from.

		Returns:
			Counter[tuple[str, str]]: The counter of adjacent symbol pairs and their occurences
		"""
		return Counter(zip(text, text[1:]))

	@staticmethod
	def _construct_matrix(
		rows: int, cols: int, data: Counter[tuple[str, str]], symbols
	) -> np.ndarray:
		"""Construct a matrix from a counter of adjacent symbol pairs and their occurences

		Args:
			rows (int): The number of rows in the matrix
			cols (int): The number of columns in the matrix
			data (Counter[tuple[str, str]]): The counter of adjacent symbol pairs and their occurences
			symbols (_type_): The symbols to use in the matrix

		Returns:
			np.ndarray: The constructed matrix as a numpy array
		"""
		matrix = np.zeros((rows, cols), dtype=int)

		sym_to_idx = {str(sym): idx for idx, sym in enumerate(symbols)}

		for (i, j), count in data.items():
			row = sym_to_idx[i]
			col = sym_to_idx[j]
			matrix[row, col] = count

		return matrix

	@staticmethod
	def from_text(text_sequence: list[str], symbols: list[str]) -> "StatisticalProfile":
		num_symbols = len(symbols)
		bigrams = StatisticalProfile._get_bigrams(text_sequence)
		matrix = StatisticalProfile._construct_matrix(
			num_symbols, num_symbols, bigrams, symbols
		)

		return StatisticalProfile(matrix, symbols)

	@staticmethod
	def from_cipher(
		cipher: Cipher, *, save: bool = False, no_cache: bool = False
	) -> "StatisticalProfile":
		"""Construct a StatisticalProfile from a Cipher object.

		Args:
			cipher (Cipher): The Cipher object to construct the StatisticalProfile from.
			save (bool, optional): Whether to save the matrix to a file. If overwrite is
				False, and the file exists, this will load the matrix from the file and
				override the save argument. Defaults to False.
			no_cache (bool, optional): Whether to skip the cache. If False, and the matrix
				exists in the cache, this will load the matrix from the cache and override
				the save argument. Defaults to False.

		Returns:
			StatisticalProfile: The constructed StatisticalProfile.
		"""

		if not no_cache and matrix_exists(cipher.name):
			try:
				log.debug(f"Cache hit. Loading {cipher.name}-matrix.csv...")
				return StatisticalProfile.from_matrix_file(f"{cipher.name}-matrix.csv")
			except Exception as e:
				log.debug(f"Cache load failed ({e}). Rebuilding...")

		all_symbols = []
		for symbol_list in cipher.key.values():
			for symbol in symbol_list:
				all_symbols.append(symbol)
		symbols = all_symbols

		profile = StatisticalProfile.from_text(cipher.ciphertext, symbols)

		if save:
			log.debug(f"Saving new {cipher.name} matrix to {cipher.name}-matrix.csv...")
			save_matrix(profile.p_raw, symbols, symbols, f"{cipher.name}-matrix.csv")

		return profile

	@staticmethod
	def from_english_corpus(
		*, save: bool = True, no_cache: bool = False
	) -> "StatisticalProfile":
		"""
		Constructs the English StatisticalProfile from the NLTK brown corpus.

		Handles caching: Tries to load "english_matrix.csv" first. If 'no_cache'
		is True or the file doesn't exist, it builds from the NLTK corpus
		and optionally saves the new matrix.
		"""
		matrix_file = "english-matrix.csv"

		if not no_cache and matrix_exists(matrix_file):
			try:
				log.debug(f"Cache hit. Loading {matrix_file}...")
				return StatisticalProfile.from_matrix_file(matrix_file)
			except Exception as e:
				log.debug(f"Cache load failed ({e}). Rebuilding...")

		# --- Cache miss or no_cache=True, run NLTK logic ---
		log.debug("Building English profile from NLTK brown corpus...")
		try:
			# Check if corpus exists
			nltk.data.find("corpora/brown")
		except LookupError:
			# Download if it doesn't
			log.debug("NLTK brown corpus not found. Downloading...")
			nltk.download("brown")

		text = nltk.corpus.brown.words()
		chars = [c.lower() for w in text for c in w if c.isalpha()]

		# Define the symbols (the 26-letter alphabet)
		symbols = list(string.ascii_lowercase)

		profile = StatisticalProfile.from_text(chars, symbols)

		if save:
			log.debug(f"Saving new English matrix to {matrix_file}...")
			save_matrix(profile.p_raw, symbols, symbols, matrix_file)

		return profile

	def __json__(self) -> dict[str, Any]:
		return {
			"p_raw": self.p_raw.tolist(),
			"symbols": self.symbols,
		}

	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "StatisticalProfile":
		p_raw = np.array(json["p_raw"])
		symbols = json["symbols"]

		profile = StatisticalProfile(p_raw, symbols)

		return profile
