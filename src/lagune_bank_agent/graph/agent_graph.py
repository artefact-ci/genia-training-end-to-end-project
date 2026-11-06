"""Graphe de l'agent.

- `build_minimal_graph` : BRIQUE 0 — FOURNI. Un noeud LLM. Dès qu'un outil est
  implémenté (Briques 1-3), il est automatiquement proposé au LLM via une boucle
  outil générique ; sans outil, le graphe est un simple START -> llm -> END.
- `build_full_graph`    : BRIQUE 4 — à implémenter.
- `build_graph`         : FOURNI. Point d'entrée UNIQUE (CLI, Streamlit, tests, éval).
"""
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lagune_bank_agent.config import get_chat_model
from lagune_bank_agent.graph.state import AgentState
from lagune_bank_agent.prompts import SYSTEM_PROMPT
from lagune_bank_agent.stubs import is_stub
from lagune_bank_agent.tools import load_tools


def build_minimal_graph(checkpointer=None):
    """Graphe de la Brique 0 (un noeud LLM, boucle outil générique si des outils existent)."""
    tools = load_tools()
    model = get_chat_model()
    if tools:
        model = model.bind_tools(tools)

    def llm_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [model.invoke(messages)]}

    builder = StateGraph(AgentState)
    builder.add_node("llm", llm_node)
    builder.add_edge(START, "llm")
    if tools:
        builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        builder.add_conditional_edges("llm", tools_condition)
        builder.add_edge("tools", "llm")
    else:
        builder.add_edge("llm", END)
    return builder.compile(checkpointer=checkpointer)


def build_full_graph(checkpointer=None):
    """Construit et compile le graphe complet de l'agent (Brique 4).

    Entrée
    ------
    checkpointer : checkpointer LangGraph (ou None) à passer à la compilation.
                   Il est fourni par `build_graph` pour la mémoire de conversation
                   et pour une éventuelle interruption (Brique 3).

    Sortie
    ------
    Un graphe compilé, utilisable exactement comme celui de la Brique 0 :
      - entrée  : {"messages": [HumanMessage(...)]} avec un `thread_id` en config ;
      - sortie  : l'état final dont le dernier message est un AIMessage SANS tool_calls ;
      - les ToolMessage produits conservent leur `artifact` (l'UI en tire les sources).

    Comportement attendu
    --------------------
    - Selon l'intention du message courant, la demande est orientée vers :
      la recherche documentaire, le calcul de frais, une action sur le compte, ou une
      réponse directe sans outil. L'intention est stockée dans l'état.
    - L'agent peut enchaîner plusieurs appels d'outils pour un même message, mais la
      boucle est bornée : pas de cycle infini possible.
    - L'historique et les résultats d'outils sont portés par l'état et survivent d'un
      tour de conversation à l'autre dans un même thread.
    - Un outil non implémenté (cf. `load_tools`) est simplement indisponible : le
      graphe se construit et répond quand même.
    - Si un outil lève une interruption LangGraph (Brique 3), elle doit remonter
      jusqu'à l'appelant sans être avalée.

    Ne doit PAS
    -----------
    - coder en dur des réponses ni court-circuiter le LLM pour répondre ;
    - utiliser un autre client LLM que `get_chat_model()` ;
    - créer son propre checkpointer.
    """
    raise NotImplementedError("BRIQUE 4 — voir README")


def build_graph():
    """Point d'entrée unique : le graphe complet s'il est implémenté, sinon le minimal."""
    checkpointer = MemorySaver()
    if not is_stub(build_full_graph):
        return build_full_graph(checkpointer)
    return build_minimal_graph(checkpointer)
