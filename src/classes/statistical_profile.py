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
	"""Represents a statistical profile of symbols.

	Attributes:
		p_raw (np.ndarray): The raw probabilities of the StatisticalProfile
		symbols (list[str]): The symbols of the StatisticalProfile
		symbol_to_idx (dict[str, int]): A mapping of symbols to their indices
		size (int): The size of the StatisticalProfile
		p_row_normalized (np.ndarray, optional): The row-normalized probabilities
			of the StatisticalProfile.
		p_col_normalized (np.ndarray, optional): The column-normalized probabilities
			of the StatisticalProfile.
		unigram_frequencies (np.ndarray, optional): The unigram frequencies of the
			StatisticalProfile.

	Methods:
        from_matrix_file(matrix_file: str) -> StatisticalProfile: Load a
            StatisticalProfile from a matrix file.
        from_text(text_sequence: list[str], symbols: list[str]) -> StatisticalProfile:
            Construct a StatisticalProfile from a text sequence.
        from_cipher(cipher: Cipher, *, save: bool = False, no_cache: bool = False)
            -> StatisticalProfile: Construct a StatisticalProfile from a Cipher object.
        from_english_corpus(*, save: bool = True, no_cache: bool = False)
            -> StatisticalProfile: Construct the English StatisticalProfile from the
            NLTK brown corpus.
        __json__() -> dict[str, Any]: Convert the StatisticalProfile to a JSON object.
        __from_json__(json: dict[str, Any]) -> StatisticalProfile: Load a
            StatisticalProfile from a JSON object.

	"""

	def __init__(self, p_raw: np.ndarray, symbols: list[str]) -> None:
		"""Initialize a StatisticalProfile.

		Args:
			p_raw (np.ndarray): The raw probabilities of the StatisticalProfile
			symbols (list[str]): The symbols of the StatisticalProfile

		Returns:
			None

		"""
		self.p_raw = p_raw
		self.symbols = symbols
		self.symbol_to_idx = {str(sym): idx for idx, sym in enumerate(symbols)}
		self.size = len(symbols)

		self._p_row_normalized = None
		self._p_col_normalized = None
		self._unigram_frequencies = None

	@property
	def p_row_normalized(self) -> np.ndarray:
		"""Get the row-normalized probabilities of the StatisticalProfile.

		Returns:
			np.ndarray: The row-normalized probabilities of the StatisticalProfile

		"""
		if self._p_row_normalized is None:
			self._p_row_normalized = self._normalize_matrix(self.p_raw, axis=1)
		return self._p_row_normalized

	@property
	def p_col_normalized(self) -> np.ndarray:
		"""Get the column-normalized probabilities of the StatisticalProfile.

		Returns:
			np.ndarray: The column-normalized probabilities of the StatisticalProfile

		"""
		if self._p_col_normalized is None:
			self._p_col_normalized = self._normalize_matrix(self.p_raw, axis=0)
		return self._p_col_normalized

	@property
	def unigram_frequencies(self) -> np.ndarray:
		"""Get the unigram frequencies of the StatisticalProfile.

		Returns:
			np.ndarray: The unigram frequencies of the StatisticalProfile

		"""
		if self._unigram_frequencies is None:
			self._unigram_frequencies = self._get_frequencies()
		return self._unigram_frequencies

	def _get_frequencies(self) -> np.ndarray:
		p_sums = self.p_raw.sum(axis=1)
		return p_sums / p_sums.sum()

	def _normalize_matrix(self, p: np.ndarray, axis: int =0) -> np.ndarray:
		p = p / p.sum(axis=axis, keepdims=True)
		return p

	@staticmethod
	def from_matrix_file(matrix_file: str) -> "StatisticalProfile":
		"""Load a StatisticalProfile from a matrix file.

		Args:
			matrix_file (str): The path to the matrix file

		Returns:
			StatisticalProfile: The StatisticalProfile object

		"""
		pd, matrix = load_matrix(matrix_file)
		symbols = pd.index.to_list()
		return StatisticalProfile(matrix, symbols)

	@staticmethod
	def _get_bigrams(text: list[str]) -> Counter[tuple[str, str]]:
		"""Create a counter of adjacent symbol pairs and their occurences.

		Args:
			text (list[str]): The text to create the counter from.

		Returns:
			Counter[tuple[str, str]]: The counter of adjacent symbol pairs and their
				occurences

		"""
		return Counter(zip(text, text[1:], strict=False))

	@staticmethod
	def _construct_matrix(
		rows: int, cols: int, data: Counter[tuple[str, str]], symbols: list[str],
	) -> np.ndarray:
		"""Construct a matrix from a counter of adjacent symbol pairs.

		Args:
			rows (int): The number of rows in the matrix
			cols (int): The number of columns in the matrix
			data (Counter[tuple[str, str]]): The counter of adjacent symbol pairs and
				their occurences
			symbols (list[str]): The symbols to use in the matrix

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
		"""Construct a StatisticalProfile from a text sequence.

		Args:
			text_sequence (list[str]): The text sequence to construct the
				StatisticalProfile from
			symbols (list[str]): The symbols to use in the StatisticalProfile

		Returns:
			StatisticalProfile: The constructed StatisticalProfile

		"""
		num_symbols = len(symbols)
		bigrams = StatisticalProfile._get_bigrams(text_sequence)
		matrix = StatisticalProfile._construct_matrix(
			num_symbols, num_symbols, bigrams, symbols,
		)

		return StatisticalProfile(matrix, symbols)

	@staticmethod
	def from_cipher(
		cipher: Cipher, *, save: bool = False, no_cache: bool = False,
	) -> "StatisticalProfile":
		"""Construct a StatisticalProfile from a Cipher object.

		Args:
			cipher (Cipher): The Cipher object to construct the StatisticalProfile from.
			save (bool, optional): Whether to save the matrix to a file. If overwrite is
				False, and the file exists, this will load the matrix from the file and
				override the save argument. Defaults to False.
			no_cache (bool, optional): Whether to skip the cache. If False, and the
				matrix exists in the cache, this will load the matrix from the cache
				and override the save argument. Defaults to False.

		Returns:
			StatisticalProfile: The constructed StatisticalProfile.

		"""
		if not no_cache and matrix_exists(cipher.name):
			try:
				log.debug(f"Cache hit. Loading {cipher.name}-matrix.csv...")
				return StatisticalProfile.from_matrix_file(f"{cipher.name}-matrix.csv")
			except Exception as e:
				log.debug(f"Cache load failed ({e}). Rebuilding...")

		symbols = cipher.get_symbols()

		profile = StatisticalProfile.from_text(cipher.ciphertext, symbols)

		if save:
			log.debug(f"Saving new {cipher.name} matrix to {cipher.name}-matrix.csv...")
			save_matrix(profile.p_raw, symbols, symbols, f"{cipher.name}-matrix.csv")

		return profile

	@staticmethod
	def from_english_corpus(
		*, save: bool = True, no_cache: bool = False,
	) -> "StatisticalProfile":
		"""Construct the English StatisticalProfile from the NLTK brown corpus.

		Handles caching: Tries to load "english_matrix.csv" first. If 'no_cache'
		is True or the file doesn't exist, it builds from the NLTK corpus
		and optionally saves the new matrix.

		Args:
			save (bool, optional): Whether to save the matrix to a file. Defaults to
				True.
			no_cache (bool, optional): Whether to skip the cache. Defaults to False.

		Returns:
			StatisticalProfile: The constructed StatisticalProfile.

		"""
		matrix_file = "english-matrix.csv"

		if not no_cache and matrix_exists(matrix_file):
			try:
				log.debug(f"Cache hit. Loading {matrix_file}...")
				return StatisticalProfile.from_matrix_file(matrix_file)
			except Exception as e:
				log.debug(f"Cache load failed ({e}). Rebuilding...")

		log.debug("Building English profile from NLTK brown corpus...")
		try:
			nltk.data.find("corpora/brown")
		except LookupError:
			log.debug("NLTK brown corpus not found. Downloading...")
			nltk.download("brown")

		text = nltk.corpus.brown.words()
		chars = [c.lower() for w in text for c in w if c.isalpha()]

		symbols = list(string.ascii_lowercase)

		profile = StatisticalProfile.from_text(chars, symbols)

		if save:
			log.debug(f"Saving new English matrix to {matrix_file}...")
			save_matrix(profile.p_raw, symbols, symbols, matrix_file)

		return profile

	def __json__(self) -> dict[str, Any]:
		"""Convert the StatisticalProfile to a JSON object.

		Returns:
			dict[str, Any]: The JSON object representing the StatisticalProfile

		"""
		return {
			"p_raw": self.p_raw.tolist(),
			"symbols": self.symbols,
		}

	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "StatisticalProfile":
		"""Load a StatisticalProfile from a JSON object.

		Args:
			json (dict[str, Any]): The JSON object to load the StatisticalProfile from

		Returns:
			StatisticalProfile: The StatisticalProfile object

		"""
		p_raw = np.array(json["p_raw"])
		symbols = json["symbols"]

		profile = StatisticalProfile(p_raw, symbols)

		return profile
