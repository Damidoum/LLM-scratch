import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.agents.agent_core import CoreAgent
from core.components.torch_models.bigram import Bigram


class BigramAgent(CoreAgent):
    def __init__(
        self, vocab_size: int, embedding_dim: int = 100, number_of_epochs: int = 2
    ):
        self.model = Bigram(vocab_size, embedding_dim)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.loss = CrossEntropyLoss()

        self.number_of_epochs = number_of_epochs

    def train(self, train_dataloader: DataLoader) -> None:
        for _ in tqdm(range(self.number_of_epochs)):
            with tqdm(len(train_dataloader), desc="Training", unit=" batch") as pbar:
                for _ in range(len(train_dataloader)):
                    batch = next(iter(train_dataloader))
                    self.optimizer.zero_grad()
                    x, y = batch
                    output = self.model(x)
                    output = output.view(-1, output.size(-1))
                    loss = self.loss(output, y.view(-1))
                    loss.backward()
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
                    self.optimizer.step()

    @torch.no_grad()
    def generate(self, x, number_of_token_to_generate):
        for _ in range(number_of_token_to_generate):
            output = self.model(x)[:, -1, :]  # take the last token
            token_next = output.argmax(dim=1).unsqueeze(
                1
            )  # get the next token (argmax), we could also sample from the distribution
            x = torch.cat((x, token_next), dim=1)  # append the token to the input
        return x
