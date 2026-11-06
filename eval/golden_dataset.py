"""BRIQUE 7 (bonus) — Jeu de référence et comparaison de deux variantes de l'agent.

Lancement prévu :  python -m eval.golden_dataset  (depuis la racine du projet)
"""
from dataclasses import dataclass, field
from typing import Callable

from orange_bank_agent.schemas import AgentResult

# Une variante = une façon de faire répondre l'agent : question -> résultat.
AgentVariant = Callable[[str], AgentResult]


@dataclass
class GoldenItem:
    """Une question de référence."""

    question: str
    expected_keywords: list[str] = field(default_factory=list)  # faits attendus dans la réponse
    expected_sources: list[str] = field(default_factory=list)  # documents attendus (au moins un)
    expected_tool: str | None = None  # outil attendu ; None = aucun outil attendu


# À vous de constituer ce jeu à partir de data/docs/ (8 à 12 questions couvrant
# documentation, frais, action, et au moins une question hors périmètre).
GOLDEN_ITEMS: list[GoldenItem] = []


def upload_dataset(dataset_name: str, items: list[GoldenItem]) -> None:
    """Crée (ou met à jour) un dataset Langfuse `dataset_name` contenant `items`.

    Contrat : après l'appel, le dataset existe côté Langfuse avec un item par élément
    de `items`, chaque item portant la question en entrée et les attentes
    (mots-clés, sources, outil) en sortie attendue. L'opération est idempotente :
    la relancer ne crée pas de doublons.

    Cas limites : Langfuse désactivé -> lève `RuntimeError` avec un message clair ;
    `items` vide -> lève `ValueError`.
    Ne doit PAS : appeler l'agent.
    """
    raise NotImplementedError("BRIQUE 7 — voir README")


def run_experiment(dataset_name: str, run_name: str, variant: AgentVariant) -> dict[str, float]:
    """Exécute `variant` sur chaque item du dataset Langfuse et enregistre un run nommé.

    Contrat : pour chaque item, la question est posée à `variant` ; la trace de
    l'agent est rattachée au run `run_name` du dataset ; des scores déterministes
    (sans LLM-juge) sont attachés par item, au minimum la couverture des mots-clés
    attendus dans la réponse et le respect de l'outil attendu (et des sources, quand
    elles sont attendues).

    Retourne les moyennes de chaque score sur le dataset, ex. {"keyword_coverage": 0.8}.

    Cas limites : une erreur sur un item n'interrompt pas le run ; elle est
    journalisée et l'item compte pour 0 sur les scores concernés. Deux runs
    distincts ne doivent pas partager d'historique de conversation.
    """
    raise NotImplementedError("BRIQUE 7 — voir README")


def compare_variants(dataset_name: str, variants: dict[str, AgentVariant]) -> dict[str, dict[str, float]]:
    """Lance `run_experiment` pour chaque variante (le nom de la variante devient le
    nom du run) et retourne {nom_variante: moyennes_des_scores}. Affiche un tableau
    comparatif lisible. Les variantes sont évaluées sur exactement les mêmes items.
    """
    raise NotImplementedError("BRIQUE 7 — voir README")
