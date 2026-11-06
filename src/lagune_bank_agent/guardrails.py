"""BRIQUE 8 — Garde-fous (entrée et sortie).

À implémenter : `check_input` et `check_output`. Le runner les appelle
automatiquement autour de chaque tour dès qu'elles ne sont plus des stubs :
la CLI et l'UI affichent les événements de garde-fou déclenchés.
"""
from lagune_bank_agent.schemas import GuardrailResult, ToolCallRecord


def check_input(user_message: str) -> GuardrailResult:
    """Évalue un message client AVANT qu'il n'atteigne le graphe.

    Entrée : le texte brut saisi par le client.

    Sortie : un `GuardrailResult` :
      - `allowed=True`  : le message peut continuer. Si `sanitized_text` est renseigné,
        c'est CE texte (et non l'original) qui est transmis à l'agent et mémorisé
        dans l'historique ;
      - `allowed=False` : le message n'atteint jamais le graphe ni l'historique ;
        `replacement` est le message affiché au client, `category` et `reason`
        expliquent la décision.

    Menaces à couvrir (au minimum) :
      - tentatives de détournement des consignes de l'agent (prompt injection :
        « ignore tes instructions », demande de révéler le prompt système…) ;
      - données sensibles saisies par le client : numéro de carte complet, IBAN,
        code personnel ou codes à usage unique. Le choix entre masquer et refuser,
        selon la donnée, est une décision de conception à justifier.

    Cas limites :
      - une question bancaire légitime ne doit JAMAIS être bloquée (testez des
        formulations proches : « ma carte se termine par 4242 » est légitime) ;
      - message vide : pas d'exception ;
      - majuscules, accents manquants, espaces ou tirets dans un numéro de carte.

    Ne doit PAS : appeler le graphe, ni lever d'exception pour un message hostile
    (un refus est un résultat, pas une erreur).
    """
    raise NotImplementedError("BRIQUE 8 — voir README")


def check_output(
    answer: str,
    sources: list[dict],
    tool_calls: list[ToolCallRecord],
) -> GuardrailResult:
    """Évalue la réponse finale de l'agent AVANT son affichage au client.

    Entrées : la réponse, les passages de documentation cités ce tour-ci, et les
    appels d'outils de ce tour (avec leur contenu).

    Sortie : un `GuardrailResult` :
      - `allowed=True`, éventuellement avec `sanitized_text` (réponse nettoyée) ;
      - `allowed=False` avec `replacement` : message de repli montré à la place de
        la réponse (il doit orienter le client, par ex. vers le Service Client).

    Risques à couvrir (au minimum) :
      - fuite de données sensibles dans la réponse (numéro de carte ou IBAN complets) ;
      - chiffres bancaires (montants en euros, pourcentages) qui n'apparaissent ni
        dans les passages cités ni dans les sorties d'outils de ce tour : ils
        viennent probablement de la « mémoire » du modèle, pas de la documentation.
        Le niveau de sévérité (bloquer, avertir, ignorer les petits nombres…) est
        une décision de conception.

    Cas limites :
      - une réponse sans chiffre (salutation, refus poli) passe toujours ;
      - un calcul de l'agent à partir de chiffres sourcés (somme, produit) n'est pas
        forcément une hallucination : le comportement choisi doit être explicite ;
      - les formats numériques varient (« 22 950 € », « 22950 euros », « 2,00 » / « 2.00 »).

    Ne doit PAS : appeler un LLM, ni modifier `sources` ou `tool_calls`.
    """
    raise NotImplementedError("BRIQUE 8 — voir README")
