#!/usr/bin/env python3
"""Static checks over the code blocks in docs/.

This does NOT execute anything — SO3LR, JAX and a GPU are not available in the
docs CI, and running MD to lint a docs page would be absurd. It catches the
failure modes that actually bit us:

  1. Python blocks that are not valid Python (typos, unbalanced brackets).
  2. Python blocks referencing a name they never define — the `atoms`/`graph`
     class of bug, where a snippet cannot possibly run as pasted.
  3. `so3lr` CLI invocations using a flag that the documented flag tables do
     not mention, which usually means a flag was renamed in the code.

Run:  python scripts/check_snippets.py
Exit: 0 clean, 1 if anything is reported.
"""

from __future__ import annotations

import ast
import builtins
import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>[a-zA-Z0-9_+-]*)[^\n]*\n"
    r"(?P<body>.*?)"
    r"^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

# Names a snippet may use without defining: they are either introduced by an
# earlier block on the same page, or are deliberate placeholders.
ALLOWED_FREE_NAMES = {
    # carried between blocks on a page, or defined in a preceding example
    "atoms", "calc", "graph", "so3lr_potential", "potential", "energy_fn",
    "force_fn", "neighbor_fn", "neighbor_fn_lr", "nbrs", "nbrs_lr", "box",
    "positions", "atomic_numbers", "displacement", "shift", "hessian",
    "cell", "smearing", "k_grid", "aux", "energy", "forces", "n",
    "my_system", "system", "graph_template",
    # third-party / package names used illustratively
    "np", "jnp", "jax", "so3lr", "ase", "pysages", "plt", "yaml",
    # module-level names in a source file the reader is editing, shown in
    # excerpts on the contributing page
    "ModelSpec", "_MODELS", "_CANONICAL",
}

IGNORE_LANGS = {"", "text", "console", "output", "diff"}


def iter_blocks(path: pathlib.Path):
    text = path.read_text()
    for m in FENCE.finditer(text):
        indent = m.group("indent")
        body = m.group("body")
        if indent:  # de-indent blocks nested inside admonitions / tabs
            body = "\n".join(
                line[len(indent):] if line.startswith(indent) else line
                for line in body.split("\n")
            )
        line_no = text[: m.start()].count("\n") + 1
        yield m.group("lang").lower(), body, line_no


def defined_and_used(tree: ast.AST) -> tuple[set[str], set[str]]:
    defined: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (defined if isinstance(node.ctx, ast.Store) else used).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
                    defined.add(arg.arg)
                for arg in (a.vararg, a.kwarg):
                    if arg:
                        defined.add(arg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.comprehension):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    defined.add(sub.id)
    return defined, used


def check_python(body: str, where: str, problems: list[str], page_scope: set[str]) -> None:
    """Check one block. `page_scope` accumulates names defined earlier on the page.

    A docs page is read top to bottom, so a later block may legitimately use a
    name an earlier block imported. We therefore carry a per-page scope rather
    than treating each block as standalone — while still catching a name that
    is never introduced anywhere on the page.
    """
    # `...` placeholders inside a call are valid Python, so nothing to strip.
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        problems.append(f"{where}: python block does not parse — {exc.msg} (line {exc.lineno})")
        return

    defined, used = defined_and_used(tree)
    unknown = used - defined - page_scope - ALLOWED_FREE_NAMES - set(dir(builtins))
    if unknown:
        problems.append(
            f"{where}: python block uses undefined name(s) {sorted(unknown)}; "
            f"import them here, or in an earlier block on the same page"
        )
    page_scope |= defined


def documented_cli_flags() -> set[str]:
    """Every `--flag` mentioned anywhere in the docs."""
    flags: set[str] = set()
    for md in DOCS.rglob("*.md"):
        flags.update(re.findall(r"(--[a-z0-9][a-z0-9-]+)", md.read_text()))
    return flags


def check_shell(body: str, where: str, known_flags: set[str], problems: list[str]) -> None:
    if "so3lr " not in body:
        return
    for flag in re.findall(r"(--[a-z0-9][a-z0-9-]+)", body):
        if flag not in known_flags:
            problems.append(f"{where}: CLI flag {flag} is used but documented nowhere")


def main() -> int:
    problems: list[str] = []
    known_flags = documented_cli_flags()
    n_py = n_sh = 0

    for md in sorted(DOCS.rglob("*.md")):
        rel = md.relative_to(DOCS.parent)
        page_scope: set[str] = set()   # names introduced earlier on this page
        for lang, body, line_no in iter_blocks(md):
            where = f"{rel}:{line_no}"
            if lang in ("python", "py"):
                n_py += 1
                check_python(body, where, problems, page_scope)
            elif lang in ("shell", "bash", "sh", "console"):
                n_sh += 1
                check_shell(body, where, known_flags, problems)
            elif lang not in IGNORE_LANGS and lang not in {"yaml", "yml", "bibtex", "json"}:
                pass

    print(f"checked {n_py} python and {n_sh} shell blocks in {len(list(DOCS.rglob('*.md')))} pages")
    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for p in problems:
            print(f"  {p}")
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
