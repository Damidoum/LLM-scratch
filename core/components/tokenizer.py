import torch


class Tokenizer:
    """Tokenizer class for encoding and decoding text data."""

    def __init__(self):
        self._encode_dict = {}
        self._decode_dict = {}

    def reset(self) -> None:
        """Reset the dictionaries."""
        self._encode_dict = {}
        self._decode_dict = {}

    def fit(self, text: str, reset: bool = True) -> None:
        """Fit the tokenizer on the text data.

        Args:
            text (str): Text data to fit the tokenizer on.
            reset (bool): Reset the dictionaries if True. Otherwise, update the dictionaries.
        """
        if reset:
            self.reset()  # reset the dictionaries if reset is True
        unique_char_list = list(set(text))
        unique_char_list.sort()  # not necessary
        for i, char in enumerate(unique_char_list):
            self._encode_dict[char] = i
            self._decode_dict[i] = char

    def encode(self, text: str) -> torch.Tensor:
        """Encode the text data.

        Args:
            text (str): text to encode

        Returns:
            torch.Tensor: encoded sequence
        """
        char_list = list(text)
        output = []
        for char in char_list:
            output.append(self._encode_dict[char])
        return torch.tensor(output)

    def decode(self, tensor: torch.Tensor) -> str:
        """Decode the tensor data.

        Args:
            tensor (torch.Tensor): sequence to decode

        Returns:
            str: decoded text
        """
        output = []
        tensor = tensor.ravel()
        for i in tensor:
            output.append(self._decode_dict[i.item()])
        return "".join(output)

    def __len__(self):
        return len(self._encode_dict)

    @property
    def encode_dict(self):
        return self._encode_dict

    @property
    def decode_dict(self):
        return self._decode_dict

    @property
    def vocab_list(self) -> list:
        return list(self._encode_dict.keys())
