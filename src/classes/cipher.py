from utils.data import load_cipher
from typing import Any


class Cipher:
	"""Represents a cipher.

	Attributes:
		name (str): The name of the cipher
		length (int): The length of the cipher
		difficulty (int): The difficulty of the cipher
		num_symbols (int): The number of symbols in the cipher
		key (dict[str, list[str]]): The key of the cipher
		plaintext (str): The plaintext of the cipher
		ciphertext (list[str]): The ciphertext of the cipher
		recurrence_encoding (str): The recurrence encoding of the cipher

	Methods:
		load_data(path: str | None = None) -> None: Load data from a file.
		get_symbols() -> list[str]: Get all symbols in the cipher.
		__str__() -> str: Convert the Cipher to a string.
		__json__() -> dict[str, Any]: Convert the Cipher to a JSON object.
		__from_json__(json: dict[str, Any]) -> Cipher: Load a Cipher from a JSON
			object.

	"""

	def __init__(self, name: str) -> None:
		"""Initialize a Cipher.

		Args:
			name (str): The name of the cipher

		Returns:
			None

		"""
		self.name = name

	def load_data(self, path: str | None = None) -> None:
		"""Load data from a file.

		Args:
			path (str, optional): The path to the file to load data from.
				Defaults to None.

		Returns:
			None

		"""
		if path is None:
			path = self.name
		cipher_json = load_cipher(path)
		self.ciphertext = cipher_json["ciphertext"].split()
		self.plaintext = cipher_json["plaintext"]
		self.key = cipher_json["key"]
		self.length = cipher_json["length"]
		self.num_symbols = cipher_json["num_symbols"]
		self.difficulty = cipher_json["difficulty"]
		self.recurrence_encoding = cipher_json["recurrence_encoding"]

	def get_symbols(self) -> list[str]:
		"""Get all symbols in the cipher.

		Returns:
			list[str]: All symbols in the cipher

		"""
		all_symbols = []
		for symbol_list in self.key.values():
			for symbol in symbol_list:
				all_symbols.append(symbol)
		return all_symbols

	def __str__(self) -> str:
		"""Convert the Cipher to a string.

		Returns:
			str: The string representation of the Cipher

		"""
		key_str = "    " + "\n    ".join(
			[f"{key}: {self.key[key]}" for key in self.key],
		)
		return (
			f"Cipher: {{\n  name: {self.name}\n  length: {self.length}\n"
			f"difficulty: {self.difficulty}\n  num_symbols: {self.num_symbols}\n  "
			f"key: {{ \n{key_str}\n  }}\n}}"
		)

	def __json__(self) -> dict[str, Any]:
		"""Convert the Cipher to a JSON object.

		Returns:
			dict[str, Any]: The JSON object

		"""
		return {
			"name": self.name,
			"length": self.length,
			"difficulty": self.difficulty,
			"num_symbols": self.num_symbols,
			"key": self.key,
			"plaintext": self.plaintext,
			"ciphertext": self.ciphertext,
			"recurrence_encoding": self.recurrence_encoding,
		}

	@staticmethod
	def __from_json__(json: dict[str, Any]) -> "Cipher":
		"""Load a Cipher from a JSON object.

		Args:
			json (dict[str, Any]): The JSON object to load the Cipher from

		Returns:
			Cipher: The Cipher object

		"""
		cipher = Cipher(json["name"])
		cipher.length = json["length"]
		cipher.difficulty = json["difficulty"]
		cipher.num_symbols = json["num_symbols"]
		cipher.key = json["key"]
		cipher.plaintext = json["plaintext"]
		cipher.ciphertext = json["ciphertext"]
		cipher.recurrence_encoding = json["recurrence_encoding"]

		return cipher
