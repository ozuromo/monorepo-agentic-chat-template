from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from shared.schema import AgentInfo

from backend.agents.example.react_agent import agent

DEFAULT_AGENT = "react_agent"

# Type alias to handle LangGraph's different agent patterns
AgentGraph = CompiledStateGraph


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "react_agent": Agent(description="A React-based agent", graph=agent),
}


def get_agent(agent_id: str) -> AgentGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()]
