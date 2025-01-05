import torch
from torch.utils.data import Dataset

from core.components.tokenizer import Tokenizer


class TokenizerDataset(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, seq_len: int = 10):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text = text

    def __len__(self):
        return len(self.text) - self.seq_len - 1

    def __getitem__(self, _):
        idx = torch.randint(0, len(self.text) - self.seq_len - 1, (1,)).item()
        encoded_sequence = self.tokenizer.encode(
            self.text[idx : idx + self.seq_len + 1]
        )
        return encoded_sequence[:-1], encoded_sequence[1:]


if __name__ == "__main__":
    # pylint: disable=all
    from pathlib import Path

    path_folder = Path(__file__).parent
    path = path_folder / "../../data/input.txt"

    with open(path.resolve(), "r") as f:
        txt = f.read()

    tokenizer = Tokenizer()
    tokenizer.fit(txt)

    dataset = TokenizerDataset(text=txt, tokenizer=tokenizer, seq_len=10)
    print(dataset[0])
