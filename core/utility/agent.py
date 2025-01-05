from core.agents.agent_core import CoreAgent
from core.agents.bigram import BigramAgent

_AGENT_NAME = {"bigram": BigramAgent}


def load_agent(
    agent_name: str, vocab_size: int, embedding_size: int = 100
) -> CoreAgent:
    """Load an agent based on the agent name.

    The args should be replaced by a class containing the good parameters for each agent.
    But for the sake of simplicity, we will keep it like this for now, since we only have one agent.

    Returns:
        CoreAgent
    """
    if agent_name not in _AGENT_NAME:
        raise ValueError(f"Invalid agent name: {agent_name}")
    agent = _AGENT_NAME[agent_name](vocab_size=vocab_size, embedding_dim=embedding_size)
    return agent
