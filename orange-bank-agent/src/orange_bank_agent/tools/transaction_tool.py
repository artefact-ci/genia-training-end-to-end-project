"""BRIQUE 3 — Actions bancaires simulées (effets de bord fictifs, en mémoire).

À implémenter : `reset_state`, `block_card`, `open_dispute`.
Les wrappers LangChain `block_card_tool` et `open_dispute_tool` sont FOURNIS
en l'état (exécution directe). Les adapter fait partie du choix de conception
demandé par le README.
"""
from langchain_core.tools import tool

from orange_bank_agent.schemas import ActionResult

# Données de démonstration (lecture seule : copiez-les, ne les mutez pas).
SEED_CARDS = {
    "4242": {"holder": "Camille Martin", "type": "Gold", "status": "active"},
    "1881": {"holder": "Camille Martin", "type": "Classic", "status": "active"},
    "0007": {"holder": "Camille Martin", "type": "Prestige", "status": "blocked"},
}
SEED_TRANSACTIONS = {
    "TX-1001": {"date": "2026-10-12", "amount_eur": 89.90, "merchant": "ElectroMax"},
    "TX-1002": {"date": "2026-10-14", "amount_eur": 12.50, "merchant": "Café du Coin"},
    "TX-1003": {"date": "2026-10-20", "amount_eur": 249.00, "merchant": "VoyagesPlus"},
}


def reset_state() -> None:
    """Remet l'état en mémoire (cartes, blocages, dossiers) dans l'état initial
    décrit par `SEED_CARDS` / `SEED_TRANSACTIONS`.

    Après l'appel : "4242" et "1881" sont actives, "0007" est déjà bloquée, et aucun
    dossier de contestation n'existe. Ne doit PAS muter les constantes `SEED_*`
    (sans quoi un second reset restaurerait un état déjà modifié).
    """
    raise NotImplementedError("BRIQUE 3 — voir README")


def block_card(card_last4: str, reason: str) -> ActionResult:
    """Bloque (opposition définitive) la carte identifiée par ses 4 derniers chiffres.

    Entrées
    -------
    card_last4 : chaîne de 4 chiffres.
    reason     : motif en texte libre (ex. "vol"), obligatoire.

    Sortie
    ------
    Un `ActionResult` :
      - succès : `success=True`, `reference` non vide et unique par blocage, `message`
        confirmant le blocage et rappelant le délai de livraison de la carte de
        remplacement tel que documenté dans procedures.txt ;
      - échec métier : `success=False`, `reference=None`, `message` explicite pour
        le client (carte inconnue, carte déjà bloquée, format invalide, motif vide).

    Effet de bord (fictif) : la carte passe à l'état "blocked" dans l'état en
    mémoire, et une ligne de log/print trace l'action.

    Cas limites
    -----------
    - Les erreurs métier ne lèvent PAS d'exception : elles retournent `success=False`.
    - Bloquer deux fois la même carte : le second appel est un échec, sans effet.

    Ne doit PAS
    -----------
    - appeler un LLM ni le réseau ;
    - écrire sur disque (aucune vraie persistance) ;
    - modifier une autre carte que celle ciblée.
    """
    raise NotImplementedError("BRIQUE 3 — voir README")


def open_dispute(transaction_id: str, reason: str) -> ActionResult:
    """Ouvre un dossier de contestation sur une opération.

    Entrées
    -------
    transaction_id : identifiant d'une opération connue (ex. "TX-1001").
    reason         : motif en texte libre, obligatoire.

    Sortie
    ------
    Un `ActionResult` :
      - succès : `success=True`, `reference` non vide et unique, `message` annonçant
        le délai d'instruction du dossier tel que documenté dans procedures.txt ;
      - échec métier : `success=False`, `reference=None`, `message` explicite
        (opération inconnue, dossier déjà ouvert pour cette opération, motif vide).

    Effet de bord (fictif) : un dossier est enregistré en mémoire, et une ligne de
    log/print trace l'action.

    Cas limites et interdits : identiques à `block_card` (pas d'exception pour les
    erreurs métier, un seul dossier par opération, pas de LLM, pas de persistance).
    """
    raise NotImplementedError("BRIQUE 3 — voir README")


@tool("block_card", response_format="content_and_artifact")
def block_card_tool(card_last4: str, reason: str) -> tuple[str, dict]:
    """Bloque définitivement (opposition) une carte bancaire du client. ACTION IRRÉVERSIBLE :
    la carte ne pourra pas être réactivée et une nouvelle carte sera fabriquée.
    Arguments : card_last4 (les 4 derniers chiffres de la carte), reason (motif : perte,
    vol, fraude…). Ne l'appelle que si le client a clairement demandé le blocage."""
    result = block_card(card_last4, reason)
    return result.message, result.to_dict()


@tool("open_dispute", response_format="content_and_artifact")
def open_dispute_tool(transaction_id: str, reason: str) -> tuple[str, dict]:
    """Ouvre un dossier de contestation pour une opération bancaire que le client conteste.
    Arguments : transaction_id (identifiant de l'opération, ex. TX-1001), reason (motif :
    opération non reconnue, article non livré…). Ne l'appelle que si le client a
    clairement demandé la contestation."""
    result = open_dispute(transaction_id, reason)
    return result.message, result.to_dict()


TOOL_BINDINGS = [(block_card_tool, block_card), (open_dispute_tool, open_dispute)]
