from dataclasses import dataclass, field

from torch.utils.data import DataLoader

from core.components.tokenizer import Tokenizer
from core.dataset.dataset import TokenizerDataset


@dataclass
class DataModule:
    train_dataset: TokenizerDataset = field(init=False)
    val_dataset: TokenizerDataset = field(init=False)

    tokenizer: Tokenizer
    validation_split: float = 0.1

    batch_size: int = 8
    num_workers: int = 1

    def load_datasets(self, text: str, seq_len: int = 10):
        train_size = int(len(text) * (1 - self.validation_split))
        self.tokenizer.fit(text[:train_size])  # Fit tokenizer on training data
        self.train_dataset = TokenizerDataset(
            text=text[:train_size], tokenizer=self.tokenizer, seq_len=seq_len
        )
        self.val_dataset = TokenizerDataset(
            text=text[train_size:], tokenizer=self.tokenizer, seq_len=seq_len
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    from pathlib import Path

    path_folder = Path(__file__).parent
    path = path_folder / "../../data/input.txt"

    with open(path.resolve(), "r") as f:
        txt = f.read()

    tokenizer = Tokenizer()
    data_module = DataModule(tokenizer)
    data_module.load_datasets(txt, seq_len=10)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    for batch in train_loader:
        print(batch[0].shape)
        break

    for batch in val_loader:
        print(batch)
        break
