from langchain.tools import tool
from langchain_core.messages.utils import trim_messages
from langgraph.prebuilt import create_react_agent
from shared.core import get_model, settings


def manage_agent_message_history(state: dict) -> dict:
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,
        max_tokens=10,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": messages}


@tool
def get_joke() -> str:
    """Uma tool que retorna uma piada engraçada."""
    return """
A professora pergunta para a turma:
- Se algum de vocês acha que é burro, fique de pé.
Todos ficam parados por alguns segundos, até que Joãozinho se levanta.
- Você se acha burro, Joãozinho?
- Não, mas fiquei com dó de ver a senhora em pé sozinha.
""".strip()


model = get_model(settings.DEFAULT_MODEL)

agent = create_react_agent(
    name="example-react-agent",
    model=model,
    tools=[get_joke],
    prompt="Responda as perguntas com bom humor e de forma amigável.",
    pre_model_hook=manage_agent_message_history,
)
