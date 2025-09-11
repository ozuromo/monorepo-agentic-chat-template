from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from shared.core import get_model, settings


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
    model=model,
    name="example-react-agent",
    tools=[get_joke],
    prompt="Responda as perguntas com bom humor e de forma amigável.",
)
