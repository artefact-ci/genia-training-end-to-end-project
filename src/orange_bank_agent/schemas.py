"""Structures de données partagées — FOURNI.

Ces types sont le « langage commun » entre les briques, le runner, la CLI et l'UI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """Passage de documentation retourné par la recherche."""

    text: str
    source: str  # nom du fichier d'origine, ex. "grille_frais.txt"
    score: float = 0.0  # plus grand = plus pertinent
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeeResult:
    """Résultat d'un calcul de frais."""

    amount_eur: float  # montant total, arrondi au centime
    explanation: str  # explication lisible du calcul
    operation: str = ""
    account_type: str = ""


@dataclass
class ActionResult:
    """Résultat d'une action bancaire simulée."""

    success: bool
    message: str
    reference: str | None = None  # renseignée si et seulement si success est vrai

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardrailResult:
    """Verdict d'un garde-fou (Brique 8)."""

    allowed: bool = True
    category: str = ""  # ex. "prompt_injection", "sensitive_data", "ungrounded_figures"
    reason: str = ""  # explication courte, pour les logs et l'UI
    sanitized_text: str | None = None  # texte nettoyé à utiliser à la place de l'original
    replacement: str | None = None  # message de refus à montrer quand allowed est faux


@dataclass
class ToolCallRecord:
    """Trace d'un appel d'outil effectué pendant un tour de conversation."""

    name: str
    args: dict[str, Any]
    content: str  # texte renvoyé au LLM
    artifact: Any = None  # donnée structurée renvoyée par l'outil (sources, résultat d'action…)
    status: str = "success"  # "success" | "error"

    @property
    def label(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.name}({args})"


@dataclass
class AgentResult:
    """Ce que reçoivent la CLI et l'UI après un tour de conversation."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    pending_confirmation: dict[str, Any] | None = None  # voir README, Brique 3
    trace_id: str | None = None
    guardrail_events: list[str] = field(default_factory=list)  # Brique 8

    @property
    def tools_called(self) -> list[str]:
        return [call.name for call in self.tool_calls]
