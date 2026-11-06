"""Registre des outils — FOURNI.

Chaque module d'outil expose `TOOL_BINDINGS = [(tool_langchain, fonction_coeur), ...]`.
Un outil n'est proposé à l'agent que si sa fonction cœur n'est PAS un stub
(`raise NotImplementedError`). Implémenter une brique suffit donc à brancher
son outil ; ne pas l'implémenter laisse l'agent fonctionnel, sans cet outil.
"""
from __future__ import annotations

import importlib
import sys

from langchain_core.tools import BaseTool

from lagune_bank_agent.stubs import is_stub

TOOL_MODULES = (
    "lagune_bank_agent.tools.rag_tool",
    "lagune_bank_agent.tools.fee_calculator",
    "lagune_bank_agent.tools.transaction_tool",
)


def _iter_bindings():
    for module_name in TOOL_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # un module cassé ne doit pas planter l'agent
            print(f"[tools] ATTENTION : import de {module_name} impossible : {exc!r}", file=sys.stderr)
            continue
        for tool_obj, core_fn in getattr(module, "TOOL_BINDINGS", []):
            yield tool_obj, not is_stub(core_fn)


def load_tools() -> list[BaseTool]:
    """Outils actuellement implémentés (liste vide en Brique 0)."""
    return [tool_obj for tool_obj, implemented in _iter_bindings() if implemented]


def tool_status() -> dict[str, bool]:
    """{nom_outil: implémenté ?} — utilisé par la CLI et l'UI pour s'afficher."""
    return {tool_obj.name: implemented for tool_obj, implemented in _iter_bindings()}
