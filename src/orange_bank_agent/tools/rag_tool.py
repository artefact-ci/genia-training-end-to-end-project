"""BRIQUE 1 — Outil RAG (recherche documentaire).

À implémenter : `retrieve`. Le wrapper LangChain `search_documentation` est
FOURNI : il fixe le format que l'UI sait afficher (sources citées).
"""
from typing import Callable

from langchain_core.tools import tool

from orange_bank_agent.schemas import Chunk

DEFAULT_TOP_K = 4

# Signature d'un reranker (Brique 5, optionnelle) :
#   (question, candidats, top_n) -> au plus top_n chunks, du plus au moins pertinent.
Reranker = Callable[[str, list[Chunk], int], list[Chunk]]

# Point d'extension de la Brique 5 : `None` = pas de reranking.
# `search_documentation` transmet cette valeur à `retrieve`.
RERANKER: Reranker | None = None


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    reranker: Reranker | None = None,
) -> list[Chunk]:
    """Retourne les passages de la documentation les plus pertinents pour `question`.

    Entrées
    -------
    question : texte libre en français. Ne doit pas être vide ou composé d'espaces.
    top_k    : nombre maximal de passages retournés. Doit être >= 1.
    reranker : hook optionnel (Brique 5). S'il est fourni, la liste finale retournée
               doit refléter l'ordre qu'il établit ; s'il vaut None, le comportement
               est celui du retrieval seul.

    Sortie
    ------
    Une liste de `Chunk` :
      - de longueur <= top_k (et < top_k seulement si l'index contient moins de passages) ;
      - triée par `score` décroissant (score plus grand = plus pertinent) ;
      - chaque Chunk porte un `text` non vide, un `source` égal au nom du fichier
        d'origine (ex. "grille_frais.txt"), un `chunk_id` et un `score` ;
      - sans doublons.

    Cas limites
    -----------
    - `question` vide ou blanche, ou `top_k` < 1 : lève `ValueError` (message explicite).
    - Index vide ou absent : l'appelant doit obtenir un résultat exploitable ou une
      erreur explicite ; jamais une liste vide silencieuse due à un index non construit.
    - Aucun passage pertinent : une liste (éventuellement vide) est un résultat valide.

    Ne doit PAS
    -----------
    - appeler un LLM de chat pour générer ou reformuler une réponse ;
    - passer par un SDK de fournisseur : les embeddings viennent de
      `orange_bank_agent.config` (LiteLLM) ;
    - modifier l'index ni réindexer à chaque appel ;
    - dépendre de l'état d'un appel précédent.
    """
    raise NotImplementedError("BRIQUE 1 — voir README")


@tool("search_documentation", response_format="content_and_artifact")
def search_documentation(question: str) -> tuple[str, list[dict]]:
    """Recherche dans la documentation interne d'Orange Bank (CGU, FAQ, fiches produits,
    grille tarifaire, procédures) et retourne les passages pertinents avec leur source.
    À utiliser pour toute question sur les conditions, délais, plafonds, procédures
    ou caractéristiques des produits. Argument : la question du client, reformulée
    si nécessaire en une requête autonome."""
    chunks = retrieve(question, top_k=DEFAULT_TOP_K, reranker=RERANKER)
    if not chunks:
        return "Aucun passage pertinent trouvé dans la documentation.", []
    content = "\n\n".join(
        f"[{i}] (source : {c.source})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )
    return content, [c.to_dict() for c in chunks]


# (outil exposé à l'agent, fonction dont l'implémentation active l'outil)
TOOL_BINDINGS = [(search_documentation, retrieve)]
