---
name: testing-docs-site
description: How to run and browser-test the SO3LR MkDocs Material documentation site locally (serve, crawl all nav routes, check tables/admonitions/code blocks, search, palette, narrow viewport).
---

# Testing the SO3LR docs site (MkDocs Material)

Static site, no backend, no credentials, no secrets needed.

## Serve it

```bash
cd /path/to/so3lr-docs
source .venv/bin/activate          # existing venv: python 3.12 via uv, mkdocs 1.6.1 + mkdocs-material
mkdocs serve -a 127.0.0.1:8000
```

If the venv is missing/broken: `uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt`.
`python3.12` may not be on PATH — always use the venv interpreter or `uv venv --python 3.12`.
A Material warning about "MkDocs 2.0 incompatibility" on startup is normal and does not mean the build failed.

## Fast non-visual precheck before browsing

The nav list lives in `mkdocs.yml`. A small Python script that walks every nav route
and asserts (a) HTTP 200, (b) an `<h1>` inside `<article>`, (c) no raw markdown leaking
(`<p>|`, literal `!!! warning`, literal fences), (d) every `<img src>` returns 200 catches
almost all rendering regressions in seconds. Do this first, then use the browser to
verify what actually looks right.

## Things worth checking visually (these are the risky spots)

- Wide markdown tables: Material wraps them in a scrollable `<div>`; the **table** may scroll
  horizontally while the page body must not. Verify the last column is reachable, and repeat at
  a narrow width (resize with `wmctrl -r :ACTIVE: -b remove,maximized_vert,maximized_horz` then
  `xdotool getactivewindow windowsize 430 900`; restore with
  `wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz`).
- Long code blocks also get their own internal scrollbar — clipped trailing `# comments` are
  expected, not a bug. Verify by scrolling the block.
- `content.code.copy` is enabled: hover a block, click the copy button, and paste into a text
  field to prove the *whole* block was copied, not just that the icon exists.
- Admonitions (`!!! warning "..."`) must render as a colored callout with an icon; backticks in
  the title should become code chips.
- Toggle the palette (header icon) on any table/admonition-heavy page and check both modes.
- Instant navigation is on, so URL/anchor changes without a full reload — assert on the URL hash
  after clicking TOC entries.

## Known search limitation

The Material/lunr search tokenizer strips leading hyphens: querying a CLI flag verbatim
(e.g. `--md-cycles`) returns **"No matching documents"**, while `md-cycles` returns the right
pages. Do not report this as a page-content bug; if verbatim flag search is wanted it needs a
`plugins.search.separator` tweak in `mkdocs.yml` (e.g. a separator that does not split on `-`).

## Devin Secrets Needed

None.
