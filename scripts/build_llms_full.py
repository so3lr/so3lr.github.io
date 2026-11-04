#!/usr/bin/env python3
"""Concatenate every docs page into site/llms-full.txt.

The `llms.txt` convention pairs a curated index (hand-written, docs/llms.txt)
with an optional full dump that can be pasted into a context window in one go.
This generates the dump.

Pages are emitted in nav order, so the file reads in the same sequence as the
site rather than alphabetically. Run after `mkdocs build`:

    python scripts/build_llms_full.py
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SITE = ROOT / "site"
CONFIG = ROOT / "mkdocs.yml"

BASE_URL = "https://so3lr.github.io/"


def nav_order() -> list[pathlib.Path]:
    """Markdown files in the order mkdocs.yml lists them.

    Parsed with a regex rather than a YAML loader: mkdocs.yml uses
    `!!python/name:` tags that a safe loader refuses and an unsafe loader
    should not be pointed at casually.
    """
    text = CONFIG.read_text()
    nav = text[text.index("\nnav:"):]
    seen, ordered = set(), []
    for m in re.finditer(r":\s*([A-Za-z0-9_./-]+\.md)\s*$", nav, re.MULTILINE):
        rel = m.group(1)
        if rel not in seen and (DOCS / rel).exists():
            seen.add(rel)
            ordered.append(DOCS / rel)
    # anything not referenced in nav, appended so nothing is silently dropped
    for f in sorted(DOCS.rglob("*.md")):
        if f not in ordered:
            ordered.append(f)
    return ordered


def page_url(path: pathlib.Path) -> str:
    rel = path.relative_to(DOCS).as_posix()[:-3]
    if rel == "index":
        return BASE_URL
    return f"{BASE_URL}{rel}/"


def main() -> int:
    if not SITE.is_dir():
        print(f"error: {SITE} not found — run `mkdocs build` first", file=sys.stderr)
        return 2

    pages = nav_order()
    parts = [
        "# SO3LR documentation — full text",
        "",
        "Every page of https://so3lr.github.io/ concatenated, in nav order.",
        "The curated index is at https://so3lr.github.io/llms.txt",
        "",
    ]
    for f in pages:
        body = f.read_text()
        # strip YAML front matter; it is site plumbing, not content
        body = re.sub(r"\A---\n.*?\n---\n", "", body, flags=re.DOTALL)
        parts.append("\n" + "=" * 70)
        parts.append(f"SOURCE: {page_url(f)}")
        parts.append("=" * 70 + "\n")
        parts.append(body.rstrip())

    out = SITE / "llms-full.txt"
    out.write_text("\n".join(parts) + "\n")
    kb = out.stat().st_size / 1024
    print(f"wrote {out.relative_to(ROOT)} — {len(pages)} pages, {kb:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
