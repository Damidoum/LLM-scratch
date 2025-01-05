from core.components.tokenizer import Tokenizer
from core.dataset.data_module import DataModule


def load_datamodule(path_to_data: str) -> DataModule:
    tokenizer = Tokenizer()
    data_module = DataModule(tokenizer)
    with open(path_to_data, "r") as f:
        txt = f.read()
    data_module.load_datasets(text=txt, seq_len=10)
    return data_module
