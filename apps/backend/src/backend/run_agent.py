import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

load_dotenv()

from backend.agents import DEFAULT_AGENT, get_agent  # noqa: E402

agent = get_agent(DEFAULT_AGENT)


async def run_agent() -> None:
    inputs: MessagesState = {
        "messages": [HumanMessage("Find me a recipe for chocolate chip cookies")]
    }
    result = await agent.ainvoke(
        input=inputs,
        config=RunnableConfig(configurable={"thread_id": uuid4()}),
    )
    result["messages"][-1].pretty_print()


def main() -> None:
    asyncio.run(run_agent())
