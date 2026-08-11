from __future__ import annotations

import ast
import re
from pathlib import Path

from paicli.codeintel.models import Symbol

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".kt": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}

SYMBOL_NODE_KINDS = {
    "class_definition": "class",
    "class_declaration": "class",
    "interface_declaration": "interface",
    "function_definition": "function",
    "function_declaration": "function",
    "function_item": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "constructor_declaration": "constructor",
}


def extract_symbols(path: Path, *, relative_path: str, file_version: str) -> list[Symbol]:
    source = path.read_bytes()
    language = LANGUAGES.get(path.suffix.lower(), "")
    if language:
        parsed = _tree_sitter_symbols(
            source,
            language=language,
            relative_path=relative_path,
            file_version=file_version,
        )
        if parsed is not None:
            return parsed
    if path.suffix.lower() == ".py":
        return _python_ast_symbols(source, relative_path, file_version)
    return _regex_symbols(source, relative_path, file_version)


def _tree_sitter_symbols(
    source: bytes,
    *,
    language: str,
    relative_path: str,
    file_version: str,
) -> list[Symbol] | None:
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(language)
        tree = parser.parse(source)
    except Exception:  # noqa: BLE001 - optional parser must have a safe fallback
        return None

    symbols: list[Symbol] = []

    def visit(node, parent_name: str = "") -> None:
        kind = SYMBOL_NODE_KINDS.get(node.type)
        current_parent = parent_name
        if kind:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = source[name_node.start_byte : name_node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                first_line = (
                    source[node.start_byte : node.end_byte]
                    .decode("utf-8", errors="replace")
                    .splitlines()[0]
                )
                symbols.append(
                    Symbol(
                        path=relative_path,
                        name=name,
                        kind=kind,
                        parent_name=parent_name,
                        signature=first_line.strip()[:500],
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring="",
                        file_version=file_version,
                    )
                )
                if kind in {"class", "interface"}:
                    current_parent = name
        for child in node.children:
            visit(child, current_parent)

    visit(tree.root_node)
    return symbols


def _python_ast_symbols(source: bytes, relative_path: str, file_version: str) -> list[Symbol]:
    try:
        tree = ast.parse(source.decode("utf-8", errors="replace"))
    except SyntaxError:
        return []
    lines = source.decode("utf-8", errors="replace").splitlines()
    symbols: list[Symbol] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.parents: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._add(node, "class")
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._add(node, "method" if self.parents else "function")
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def _add(self, node, kind: str) -> None:
            line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else node.name
            symbols.append(
                Symbol(
                    path=relative_path,
                    name=node.name,
                    kind=kind,
                    parent_name=self.parents[-1] if self.parents else "",
                    signature=line[:500],
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    docstring=(ast.get_docstring(node) or "")[:1000],
                    file_version=file_version,
                )
            )

    Visitor().visit(tree)
    return symbols


def _regex_symbols(source: bytes, relative_path: str, file_version: str) -> list[Symbol]:
    text = source.decode("utf-8", errors="replace")
    pattern = re.compile(
        r"^\s*(?:(?:public|private|protected|static|async|export)\s+)*"
        r"(?:(class|interface|def|function|fn|func)\s+)([A-Za-z_$][\w$]*)",
        re.MULTILINE,
    )
    symbols = []
    for match in pattern.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        keyword, name = match.groups()
        kind = (
            "class" if keyword == "class" else "interface" if keyword == "interface" else "function"
        )
        signature = text[match.start() : text.find("\n", match.start())].strip()
        symbols.append(
            Symbol(
                path=relative_path,
                name=name,
                kind=kind,
                parent_name="",
                signature=signature[:500],
                start_line=line,
                end_line=line,
                docstring="",
                file_version=file_version,
            )
        )
    return symbols
