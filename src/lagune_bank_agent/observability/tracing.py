"""Observabilité Langfuse.

- `invoke_traced` : FOURNI, FONCTIONNEL — chaque exécution du graphe est tracée
  (décorateur `@observe` + callback LangChain). Sans clés LANGFUSE_*, le tracing
  est simplement désactivé et l'agent fonctionne normalement.
- `record_tool_scores` : BRIQUE 6 — à implémenter.
"""
from __future__ import annotations

import logging
from typing import Any

from lagune_bank_agent.config import get_langfuse, settings
from lagune_bank_agent.schemas import ToolCallRecord

log = logging.getLogger(__name__)


def _observe(**kwargs):
    """`@observe` de Langfuse si le tracing est actif, sinon décorateur neutre."""
    if settings.langfuse_enabled:
        try:
            from langfuse import observe

            return observe(**kwargs)
        except Exception as exc:  # pragma: no cover
            log.warning("Langfuse indisponible (%r) : tracing désactivé.", exc)
    return lambda fn: fn


def _input_preview(graph_input: Any) -> str:
    if isinstance(graph_input, dict) and graph_input.get("messages"):
        last = graph_input["messages"][-1]
        return str(getattr(last, "content", last))
    return f"[reprise] {getattr(graph_input, 'resume', graph_input)}"


def _last_ai_text(graph, config) -> str:
    messages = graph.get_state(config).values.get("messages", [])
    return str(messages[-1].content) if messages else ""


@_observe(name="lagune-bank-agent", capture_input=False, capture_output=False)
def invoke_traced(graph, graph_input: Any, thread_id: str) -> str | None:
    """Exécute le graphe pour un tour de conversation et retourne le `trace_id`
    Langfuse (ou None si le tracing est désactivé).

    `graph_input` est soit {"messages": [...]}, soit une `Command(resume=...)`.
    L'état résultant se lit ensuite avec `graph.get_state(config)`.
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    client = get_langfuse()
    trace_id = None
    if client is not None:
        try:
            from langfuse.langchain import CallbackHandler

            config["callbacks"] = [CallbackHandler()]
            trace_id = client.get_current_trace_id()
            client.update_current_trace(session_id=thread_id, input=_input_preview(graph_input))
        except Exception as exc:
            log.warning("Initialisation du tracing Langfuse impossible : %r", exc)

    graph.invoke(graph_input, config)

    if client is not None:
        try:
            client.update_current_trace(output=_last_ai_text(graph, config))
        except Exception as exc:
            log.warning("Mise à jour de la trace impossible : %r", exc)
    return trace_id


def flush_traces() -> None:
    """Force l'envoi des traces en attente (à appeler avant de quitter)."""
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:  # pragma: no cover
            log.warning("flush Langfuse impossible : %r", exc)


def record_tool_scores(trace_id: str | None, tool_calls: list[ToolCallRecord]) -> None:
    """Attache à la trace Langfuse `trace_id` des scores décrivant la qualité de chaque
    appel d'outil effectué pendant le tour (`tool_calls`).

    Entrées
    -------
    trace_id   : identifiant de la trace du tour (None si le tracing est désactivé).
    tool_calls : appels d'outils du tour, avec nom, arguments, contenu, `artifact`
                 (sources pour la recherche documentaire, résultat structuré pour les
                 autres outils) et `status`.

    Comportement attendu
    --------------------
    - Chaque famille d'outil produit au moins un score nommé et interprétable dans
      Langfuse : un score de pertinence pour la recherche documentaire, un score de
      réussite/échec pour les actions simulées, un indicateur de validité pour le
      calcul de frais.
    - Un appel d'outil en erreur (`status == "error"`) est repérable dans les scores.
    - Chaque score porte un `comment` qui permet de comprendre sa valeur.

    Cas limites
    -----------
    - `trace_id` vide/None, ou Langfuse désactivé : no-op silencieux.
    - `tool_calls` vide : no-op.
    - Cette fonction ne doit JAMAIS faire échouer le tour de conversation : toute
      erreur Langfuse est journalisée, pas propagée.

    Ne doit PAS
    -----------
    - appeler un LLM (les scores sont calculés à partir des données déjà disponibles) ;
    - modifier le contenu des `tool_calls`.
    """
    raise NotImplementedError("BRIQUE 6 — voir README")
