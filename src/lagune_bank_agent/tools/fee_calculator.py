"""BRIQUE 2 — Calculateur de frais (logique pure Python, aucun appel LLM).

À implémenter : `calculate_fee`, à partir de `data/docs/grille_frais.txt`.
Le wrapper LangChain `estimate_fees` est FOURNI.
"""
from langchain_core.tools import tool

from lagune_bank_agent.schemas import FeeResult

ACCOUNT_TYPES = ("essentiel", "premium")

# Codes d'opération reconnus (les montants sont ceux de grille_frais.txt) :
OPERATIONS = (
    "account_maintenance",  # tenue de compte, `quantity` = nombre de mois
    "card_replacement",  # remplacement carte Classic/Gold, `quantity` = nombre de cartes
    "card_express_delivery",  # envoi express 48 h, `quantity` = nombre de cartes
    "prestige_card_annual",  # cotisation annuelle carte Prestige (Premium uniquement)
    "prestige_card_replacement",  # remplacement carte Prestige (Premium uniquement)
    "atm_withdrawal_euro_other_bank",  # `quantity` = nombre TOTAL de retraits du mois civil
    "atm_withdrawal_non_euro",  # `amount` = montant d'UN retrait, `quantity` = nombre de retraits
    "card_payment_non_euro",  # `amount` = montant total des paiements
    "transfer_instant",  # `quantity` = nombre de virements
    "transfer_international",  # `quantity` = nombre de virements
    "intervention_fee",  # `quantity` = nombre d'opérations irrégulières du mois
    "paper_statement",  # `quantity` = nombre de relevés papier
    # Opérations gratuites (doivent retourner 0.0 pour les deux types de compte) :
    "transfer_sepa",
    "atm_withdrawal_own_network",
    "card_payment_euro",
    "card_opposition",
    "account_closing",
)


def calculate_fee(
    operation: str,
    account_type: str,
    amount: float = 0.0,
    quantity: int = 1,
) -> FeeResult:
    """Calcule les frais d'une opération selon la grille tarifaire Lagune Bank.

    Entrées
    -------
    operation    : un des codes de `OPERATIONS` (insensible à la casse et aux espaces).
    account_type : "essentiel" ou "premium" (insensible à la casse et aux espaces).
    amount       : montant en euros concerné par l'opération (>= 0) ; ignoré quand
                   l'opération n'a pas de composante proportionnelle.
    quantity     : nombre d'occurrences (>= 1) ; sa signification par opération est
                   précisée dans les commentaires de `OPERATIONS`.

    Sortie
    ------
    Un `FeeResult` dont :
      - `amount_eur` est le total en euros, arrondi au centime (float >= 0) ;
      - `explanation` détaille le calcul en français (forfait, franchise, pourcentage,
        plafond appliqués) de façon exploitable par un conseiller ;
      - `operation` et `account_type` sont les valeurs normalisées.

    Règles à respecter (cf. grille_frais.txt) : franchises mensuelles, plafonds
    mensuels, composantes fixe + proportionnelle, tarifs différents selon le compte.
    Exemples de résultats attendus :
      - ("atm_withdrawal_euro_other_bank", "essentiel", quantity=5)  -> 2.00
      - ("atm_withdrawal_non_euro", "essentiel", amount=200)         -> 8.00
      - ("intervention_fee", "essentiel", quantity=12)               -> 80.00 (plafonné)

    Cas limites
    -----------
    - Opération ou type de compte inconnu : `ValueError` listant les valeurs valides.
    - `amount` < 0 ou `quantity` < 1 : `ValueError`.
    - Opération indisponible pour ce type de compte (ex. carte Prestige avec un compte
      "essentiel") : `ValueError` au message explicite.

    Ne doit PAS
    -----------
    - appeler un LLM ni le réseau ;
    - modifier un état global ;
    - arrondir avant le calcul final (pas d'erreurs d'arrondi cumulées).
    """
    raise NotImplementedError("BRIQUE 2 — voir README")


@tool("estimate_fees", response_format="content_and_artifact")
def estimate_fees(
    operation: str, account_type: str, amount: float = 0.0, quantity: int = 1
) -> tuple[str, dict]:
    """Calcule le montant exact des frais bancaires d'une opération (tarification Lagune Bank).
    Arguments : operation (code, parmi : account_maintenance, card_replacement,
    card_express_delivery, prestige_card_annual, prestige_card_replacement,
    atm_withdrawal_euro_other_bank, atm_withdrawal_non_euro, card_payment_non_euro,
    transfer_instant, transfer_international, intervention_fee, paper_statement,
    transfer_sepa, atm_withdrawal_own_network, card_payment_euro, card_opposition,
    account_closing), account_type ('essentiel' ou 'premium'), amount (montant en euros
    si pertinent), quantity (nombre d'occurrences). Si le type de compte du client
    n'est pas connu, demande-le avant d'appeler l'outil."""
    try:
        result = calculate_fee(operation, account_type, amount=amount, quantity=quantity)
    except ValueError as exc:
        return f"Calcul impossible : {exc}", {"ok": False}
    content = f"Frais : {result.amount_eur:.2f} €. {result.explanation}"
    return content, {"ok": True, "fee_eur": result.amount_eur}


TOOL_BINDINGS = [(estimate_fees, calculate_fee)]
