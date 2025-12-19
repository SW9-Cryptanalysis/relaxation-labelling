import pytest
from typing import Any


from classes.cipher import Cipher


@pytest.fixture
def raw_cipher_data() -> dict[str, Any]:
	"""
	Returns a dictionary representing the raw JSON data typically
	returned by load_cipher().
	"""
	return {
		"ciphertext": "SYM1 SYM2 SYM3",
		"plaintext": "the",
		"key": {"t": ["SYM1"], "h": ["SYM2", "SYM2_VAR"], "e": ["SYM3"]},
		"length": 100,
		"num_symbols": 3,
		"difficulty": 5,
		"recurrence_encoding": "123123",
	}


@pytest.fixture
def populated_cipher(raw_cipher_data, mocker) -> Cipher:
	"""
	Returns a Cipher instance that has already loaded data.
	Useful for testing methods like get_symbols, __str__, etc.
	Without repeating the loading logic.
	"""

	mocker.patch("classes.cipher.load_cipher", return_value=raw_cipher_data)

	cipher = Cipher(name="test_z408")
	cipher.load_data()
	return cipher


class TestCipher:
	def test_init(self):
		"""Test that __init__ correctly assigns the name."""
		cipher = Cipher("zodiac_340")
		assert cipher.name == "zodiac_340"

	def test_load_data_with_default_path(self, mocker, raw_cipher_data):
		"""
		Test loading data using the cipher name as the default path.
		Also verifies the .split() logic on ciphertext.
		"""

		mock_loader = mocker.patch(
			"classes.cipher.load_cipher", return_value=raw_cipher_data
		)

		cipher = Cipher("zodiac_408")
		cipher.load_data()

		mock_loader.assert_called_once_with("zodiac_408")

		assert cipher.plaintext == "the"
		assert cipher.difficulty == 5
		assert cipher.key["h"] == ["SYM2", "SYM2_VAR"]

		assert isinstance(cipher.ciphertext, list)
		assert cipher.ciphertext == ["SYM1", "SYM2", "SYM3"]

	def test_load_data_with_custom_path(self, mocker, raw_cipher_data):
		"""Test loading data when a specific path is provided."""
		mock_loader = mocker.patch(
			"classes.cipher.load_cipher", return_value=raw_cipher_data
		)

		cipher = Cipher("zodiac_408")
		cipher.load_data(path="custom/path/to/file.json")

		mock_loader.assert_called_once_with("custom/path/to/file.json")

	def test_get_symbols(self, populated_cipher):
		"""
		Test that get_symbols correctly flattens the key dictionary lists.
		"""

		symbols = populated_cipher.get_symbols()

		assert len(symbols) == 4
		assert "SYM1" in symbols
		assert "SYM2" in symbols
		assert "SYM2_VAR" in symbols
		assert "SYM3" in symbols

	def test_str_representation(self, populated_cipher):
		"""
		Test the string representation of the class.
		We check for containment of key fields rather than exact string match
		to avoid brittleness with whitespace.
		"""
		s_rep = str(populated_cipher)

		assert "Cipher: {" in s_rep
		assert "name: test_z408" in s_rep
		assert "difficulty: 5" in s_rep

		assert "t: ['SYM1']" in s_rep

	def test_json_serialization(self, populated_cipher):
		"""Test conversion to JSON-serializable dict."""
		json_data = populated_cipher.__json__()

		assert isinstance(json_data, dict)
		assert json_data["name"] == "test_z408"
		assert json_data["ciphertext"] == ["SYM1", "SYM2", "SYM3"]
		assert json_data["key"] == populated_cipher.key
		assert json_data["recurrence_encoding"] == "123123"

	def test_from_json(self):
		"""
		Test the static factory method __from_json__.
		This verifies we can reconstruct a Cipher object from a dict.
		"""

		input_json = {
			"name": "restored_cipher",
			"length": 50,
			"difficulty": 2,
			"num_symbols": 10,
			"key": {"a": ["1"]},
			"plaintext": "a",
			"ciphertext": ["1"],
			"recurrence_encoding": "111",
		}

		cipher = Cipher.__from_json__(input_json)

		assert isinstance(cipher, Cipher)
		assert cipher.name == "restored_cipher"
		assert cipher.length == 50
		assert cipher.ciphertext == ["1"]
		assert cipher.key == {"a": ["1"]}
