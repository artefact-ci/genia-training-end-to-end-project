"""Exécution d'un tour de conversation — FOURNI.

`ask` et `resume` sont les deux seules fonctions utilisées par la CLI, l'UI et
l'évaluation pour faire tourner le graphe. Elles garantissent que tous les
consommateurs voient le même `AgentResult`.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from orange_bank_agent.observability import tracing
from orange_bank_agent import guardrails
from orange_bank_agent.schemas import AgentResult, ToolCallRecord
from orange_bank_agent.stubs import is_stub

log = logging.getLogger(__name__)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else p.get("text", "") for p in content if p]
        return "".join(parts)
    return str(content)


def _pending_interrupt(snapshot) -> dict | None:
    for task in snapshot.tasks:
        for interrupt in getattr(task, "interrupts", ()):
            value = interrupt.value
            return value if isinstance(value, dict) else {"message": str(value)}
    return None


def _build_result(graph, thread_id: str, trace_id: str | None) -> AgentResult:
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    messages = snapshot.values.get("messages", [])

    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    turn = messages[last_human + 1 :]

    requested = {
        call["id"]: call for m in turn if isinstance(m, AIMessage) for call in m.tool_calls
    }
    records: list[ToolCallRecord] = []
    sources: list[dict] = []
    seen: set[tuple] = set()
    for m in turn:
        if not isinstance(m, ToolMessage):
            continue
        call = requested.get(m.tool_call_id, {})
        records.append(
            ToolCallRecord(
                name=m.name or call.get("name", "?"),
                args=call.get("args", {}),
                content=_text(m.content),
                artifact=m.artifact,
                status=getattr(m, "status", "success") or "success",
            )
        )
        if isinstance(m.artifact, list):  # convention : liste de chunks sérialisés
            for item in m.artifact:
                if isinstance(item, dict) and "source" in item and "text" in item:
                    key = (item["source"], item.get("chunk_id"))
                    if key not in seen:
                        seen.add(key)
                        sources.append(item)

    answer = ""
    for m in reversed(turn):
        if isinstance(m, AIMessage) and not m.tool_calls:
            answer = _text(m.content)
            break

    return AgentResult(
        answer=answer,
        sources=sources,
        tool_calls=records,
        pending_confirmation=_pending_interrupt(snapshot),
        trace_id=trace_id,
    )


def _finish(graph, thread_id: str, trace_id: str | None) -> AgentResult:
    result = _build_result(graph, thread_id, trace_id)
    if result.tool_calls and not is_stub(tracing.record_tool_scores):  # Brique 6
        try:
            tracing.record_tool_scores(trace_id, result.tool_calls)
        except Exception as exc:
            log.warning("record_tool_scores a échoué : %r", exc)
    return result


def _apply_output_guardrail(result: AgentResult) -> None:
    """Brique 8 : filtre la réponse finale (no-op tant que la brique n'est pas faite)."""
    if not result.answer or is_stub(guardrails.check_output):
        return
    try:
        verdict = guardrails.check_output(result.answer, result.sources, result.tool_calls)
    except Exception as exc:
        log.warning("check_output a échoué : %r", exc)
        return
    if not verdict.allowed:
        result.guardrail_events.append(f"réponse bloquée ({verdict.category}) : {verdict.reason}")
        result.answer = verdict.replacement or "Je ne peux pas fournir cette réponse. Contactez le Service Client au 0 800 123 456."
    elif verdict.sanitized_text is not None and verdict.sanitized_text != result.answer:
        result.guardrail_events.append(f"réponse nettoyée ({verdict.category}) : {verdict.reason}")
        result.answer = verdict.sanitized_text


def ask(graph, question: str, thread_id: str) -> AgentResult:
    """Envoie `question` à l'agent dans la conversation `thread_id`."""
    events: list[str] = []
    if not is_stub(guardrails.check_input):  # Brique 8
        verdict = guardrails.check_input(question)
        if not verdict.allowed:
            return AgentResult(
                answer=verdict.replacement or "Je ne peux pas traiter cette demande.",
                guardrail_events=[f"message bloqué ({verdict.category}) : {verdict.reason}"],
            )
        if verdict.sanitized_text is not None and verdict.sanitized_text != question:
            events.append(f"message nettoyé ({verdict.category}) : {verdict.reason}")
            question = verdict.sanitized_text
    trace_id = tracing.invoke_traced(graph, {"messages": [HumanMessage(content=question)]}, thread_id)
    result = _finish(graph, thread_id, trace_id)
    result.guardrail_events = events + result.guardrail_events
    _apply_output_guardrail(result)
    return result


def resume(graph, approved: bool, thread_id: str) -> AgentResult:
    """Reprend un tour interrompu (`pending_confirmation`) avec la décision du client."""
    trace_id = tracing.invoke_traced(graph, Command(resume=approved), thread_id)
    result = _finish(graph, thread_id, trace_id)
    _apply_output_guardrail(result)
    return result
