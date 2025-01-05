from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader


class CoreAgent(ABC):
    @abstractmethod
    def train(self, train_dataloader: DataLoader) -> None:
        pass

    @abstractmethod
    def generate(
        self, x: torch.Tensor, number_of_token_to_generate: int
    ) -> torch.Tensor:
        pass
