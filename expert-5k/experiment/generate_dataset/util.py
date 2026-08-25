from __future__ import annotations

import ast
import re
from typing import Any

_ADJACENT_DICT_PATTERN = re.compile(r"}\s*\n\s*{")


def parse_numpy_repr_payload(text: str) -> Any:
    expression = ast.parse(_escape_newlines_inside_strings(text), mode="eval")
    simplified = _ArrayCallStripper().visit(expression)
    ast.fix_missing_locations(simplified)
    return ast.literal_eval(simplified)


def _escape_newlines_inside_strings(text: str) -> str:
    parts: list[str] = []
    quote: str | None = None
    escaped = False

    for char in text:
        if escaped:
            parts.append(char)
            escaped = False
            continue

        if char == "\\":
            parts.append(char)
            escaped = True
            continue

        if quote is None and char in {"'", '"'}:
            quote = char
            parts.append(char)
            continue

        if quote is not None and char == quote:
            quote = None
            parts.append(char)
            continue

        if quote is not None and char == "\n":
            parts.append("\\n")
            continue

        parts.append(char)

    normalized = "".join(parts)
    return _ADJACENT_DICT_PATTERN.sub("},\n {", normalized)


class _ArrayCallStripper(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "array":
            if node.args:
                return node.args[0]
            return ast.List(elts=[], ctx=ast.Load())
        return node
