#!/usr/bin/env python3
"""Verify every cross-page `#anchor` in docs/ resolves in the built site.

`mkdocs build --strict` catches a link to a page that does not exist, but not a
link to a *heading* that does not exist: `[x](models.md#renamed-heading)` builds
happily and 404s in the browser. Headings get renamed constantly, so this is the
failure mode that actually happens.

Run after `mkdocs build`, from the repo root:
    python scripts/check_anchors.py
"""

from __future__ import annotations

import pathlib
import posixpath
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DOCS = ROOT / "docs"

LINK = re.compile(r"\]\((?!https?://|mailto:)([^)#\s]+\.md)(?:#([A-Za-z0-9_\-]+))?\)")
ID = re.compile(r'id="([^"]+)"')


def main() -> int:
    if not SITE.is_dir():
        print(f"error: {SITE} not found — run `mkdocs build` first", file=sys.stderr)
        return 2

    anchors: dict[str, set[str]] = {}
    for f in SITE.rglob("index.html"):
        rel = f.parent.relative_to(SITE).as_posix()
        anchors["" if rel == "." else rel] = set(ID.findall(f.read_text()))

    problems: list[str] = []
    n_links = 0

    for md in sorted(DOCS.rglob("*.md")):
        base = md.parent.relative_to(DOCS).as_posix()
        base = "" if base == "." else base
        for target, anchor in LINK.findall(md.read_text()):
            n_links += 1
            key = posixpath.normpath(posixpath.join(base, target))[:-3]
            if key == "index":
                key = ""
            elif key.endswith("/index"):
                key = key[: -len("/index")]
            src = md.relative_to(ROOT)
            if key not in anchors:
                problems.append(f"{src}: link to missing page '{target}'")
            elif anchor and anchor not in anchors[key]:
                problems.append(
                    f"{src}: '#{anchor}' does not exist in {key or 'index'} "
                    f"(heading renamed or removed?)"
                )

    print(f"checked {n_links} internal links across {len(anchors)} built pages")
    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for p in problems:
            print(f"  {p}")
        return 1
    print("all links and anchors resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
