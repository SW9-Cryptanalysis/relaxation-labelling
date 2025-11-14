from utils.data import load_cipher


class Cipher:
	def __init__(self, name: str):
		self.name = name
		self._load_data()

	def _load_data(self):
		cipher_json = load_cipher(self.name)
		self.ciphertext = cipher_json["ciphertext"].split()
		self.plaintext = cipher_json["plaintext"]
		self.key = cipher_json["key"]
		self.length = cipher_json["length"]
		self.num_symbols = cipher_json["num_symbols"]
		self.difficulty = cipher_json["difficulty"]
		self.reccurence_encoding = cipher_json["recurrence_encoding"]

	def __str__(self) -> str:
		key_str = "    " + "\n    ".join(
			[f"{key}: {self.key[key]}" for key in self.key]
		)
		return (
			f"Cipher: {{\n  name: {self.name}\n  length: {self.length}\n"
			f"difficulty: {self.difficulty}\n  num_symbols: {self.num_symbols}\n  key: {{ \n{key_str}\n  }}\n}}"
		)
