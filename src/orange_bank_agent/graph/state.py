"""BRIQUE 4 — État du graphe.

`AgentState` ne contient que l'historique de messages : c'est suffisant pour le graphe
minimal de la Brique 0. À compléter pour la Brique 4.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """État partagé par les noeuds du graphe.

    Contrat du graphe complet (Brique 4) — l'état doit permettre de :
      - conserver l'historique de la conversation d'un tour à l'autre (thread) ;
      - connaître l'intention détectée pour le message courant, parmi
        "documentation", "frais", "action", "general" ;
      - accumuler les résultats des outils appelés (nom, arguments, sortie) sans
        écraser ceux des étapes précédentes.

    Contraintes :
      - le champ `messages` reste présent, avec le reducer `add_messages`
        (le runner, la CLI et l'UI lisent l'état via ce champ) ;
      - tout champ ajouté doit avoir une valeur par défaut raisonnable quand il
        n'a pas encore été écrit (le graphe minimal n'écrit que `messages`).
    """

    messages: Annotated[list[AnyMessage], add_messages]
