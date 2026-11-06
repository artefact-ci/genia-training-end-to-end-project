"""Détection des briques non implémentées — FOURNI.

Une fonction est un « stub » si son corps (hors docstring) se résume à
`raise NotImplementedError(...)`. Le registre d'outils et le runner s'en servent
pour que l'agent reste fonctionnel tant qu'une brique n'est pas faite.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable


def is_stub(fn: Callable) -> bool:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return False
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # retire la docstring
    return (
        len(body) == 1
        and isinstance(body[0], ast.Raise)
        and "NotImplementedError" in ast.dump(body[0])
    )
