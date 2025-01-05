import torch

from core.agents.agent_core import CoreAgent
from core.dataset.data_module import DataModule
from core.utility.agent import load_agent
from core.utility.data_module import load_datamodule


def train_model(data_module: DataModule, agent: CoreAgent) -> None:
    train_dataloader = data_module.train_dataloader()
    agent.train(train_dataloader)


def main():
    data_module = load_datamodule("data/input.txt")
    agent = load_agent("bigram", data_module.vocab_size)
    # train_model(data_module, agent)
    pred = agent.generate(torch.tensor([1, 2, 3]).unsqueeze(0), 10)
    print(pred)
    print(data_module.tokenizer.decode(pred))


if __name__ == "__main__":
    main()
