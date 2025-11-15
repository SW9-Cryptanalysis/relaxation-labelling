from utils.data import load_cipher
from typing import Any


class Cipher:
	def __init__(self, name: str):
		self.name = name

	def load_data(self, path: str | None = None):
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

	def __str__(self) -> str:
		key_str = "    " + "\n    ".join(
			[f"{key}: {self.key[key]}" for key in self.key]
		)
		return (
			f"Cipher: {{\n  name: {self.name}\n  length: {self.length}\n"
			f"difficulty: {self.difficulty}\n  num_symbols: {self.num_symbols}\n  key: {{ \n{key_str}\n  }}\n}}"
		)

	def __json__(self) -> dict[str, Any]:
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
		cipher = Cipher(json["name"])
		cipher.length = json["length"]
		cipher.difficulty = json["difficulty"]
		cipher.num_symbols = json["num_symbols"]
		cipher.key = json["key"]
		cipher.plaintext = json["plaintext"]
		cipher.ciphertext = json["ciphertext"]
		cipher.recurrence_encoding = json["recurrence_encoding"]

		return cipher
