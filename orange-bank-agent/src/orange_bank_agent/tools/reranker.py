"""BRIQUE 5 (OPTIONNELLE) — Reranking via Azure AI Foundry.

Cette brique est un choix de conception : l'intégrer (ou non) à `rag_tool.py` et
justifier la décision. Voir README.
"""
from orange_bank_agent.schemas import Chunk


def foundry_rerank(question: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
    """Réordonne `chunks` par pertinence vis-à-vis de `question` grâce au reranker
    hébergé sur Azure AI Foundry (variables `RERANK_*`), et en garde `top_n`.

    Compatible avec le type `Reranker` de `rag_tool.py`.

    Entrées
    -------
    question : la requête utilisateur.
    chunks   : candidats issus du retrieval vectoriel (peut être vide).
    top_n    : nombre maximal de chunks retournés (>= 1).

    Sortie
    ------
    Une liste de nouveaux `Chunk` (les originaux ne sont pas mutés), de longueur
    <= top_n, triée par pertinence décroissante, dont `score` contient le score de
    pertinence du reranker. Aucun chunk n'est inventé ni dupliqué.

    Cas limites
    -----------
    - `chunks` vide : retourne une liste vide, sans appel réseau.
    - Reranker non configuré (`settings.rerank_enabled` faux) ou service en erreur :
      le comportement face à la panne fait partie de votre choix de conception et
      doit être documenté ; il ne doit jamais rester silencieux.

    Ne doit PAS
    -----------
    - passer par un SDK de fournisseur : l'appel passe par LiteLLM ;
    - écrire une clé ou un endpoint en dur.
    """
    raise NotImplementedError("BRIQUE 5 — voir README")
