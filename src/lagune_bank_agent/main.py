"""CLI de l'agent — FOURNI, FONCTIONNEL dès la Brique 0.

Lancement :  python -m lagune_bank_agent.main
Commandes :  /new (nouvelle conversation), /quit
"""
from __future__ import annotations

import uuid

from lagune_bank_agent.config import settings
from lagune_bank_agent.graph.agent_graph import build_graph
from lagune_bank_agent.observability.tracing import flush_traces
from lagune_bank_agent.runner import ask, resume
from lagune_bank_agent.schemas import AgentResult
from lagune_bank_agent.tools import tool_status

YES = {"o", "oui", "y", "yes"}


def _banner() -> None:
    status = tool_status()
    active = [name for name, ok in status.items() if ok] or ["aucun"]
    print("=" * 60)
    print(" Lagune Bank — assistant virtuel (CLI)")
    print(f" Modèle     : {settings.chat_model}")
    print(f" Outils     : {', '.join(active)}")
    print(f" Langfuse   : {'actif' if settings.langfuse_enabled else 'désactivé (clés absentes)'}")
    print(" Commandes  : /new  /quit")
    print("=" * 60)
    if not settings.llm_configured:
        print("⚠  LLM_API_KEY absente : copiez .env.example en .env.")


def _display(result: AgentResult) -> None:
    if result.answer:
        print(f"\nAssistant > {result.answer}")
    if result.tool_calls:
        print("\n  Outils appelés :")
        for call in result.tool_calls:
            print(f"   • {call.label}" + ("  [ERREUR]" if call.status == "error" else ""))
    for event in result.guardrail_events:
        print(f"\n  [garde-fou] {event}")
    if result.sources:
        print("\n  Sources citées :")
        for src in result.sources:
            print(f"   • {src['source']} (score {src.get('score', 0):.2f})")


def main() -> None:
    _banner()
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    try:
        while True:
            try:
                question = input("\nVous > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question in {"/quit", "/exit", "/q"}:
                break
            if question == "/new":
                thread_id = str(uuid.uuid4())
                print("Nouvelle conversation.")
                continue
            try:
                result = ask(graph, question, thread_id)
                while result.pending_confirmation:
                    pending = result.pending_confirmation
                    print(f"\n⚠  Confirmation requise : {pending.get('message', pending)}")
                    answer = input("Confirmer ? [o/N] > ").strip().lower()
                    result = resume(graph, answer in YES, thread_id)
                _display(result)
            except Exception as exc:  # l'agent ne doit jamais faire planter la boucle
                print(f"\n[Erreur] {type(exc).__name__}: {exc}")
    finally:
        flush_traces()
        print("\nAu revoir.")


if __name__ == "__main__":
    main()
