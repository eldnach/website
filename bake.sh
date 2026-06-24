#!/usr/bin/env bash
#
# bake.sh — splice a Markdown file into an HTML page between bake markers.
#
# Usage:
#   ./bake.sh <article.md> [target.html]
#
# The target HTML must contain a single pair of marker lines:
#   <!-- BAKE:START -->
#   <!-- BAKE:END -->
# Everything between them is replaced with the contents of <article.md>
# (wrapped by an <md-block> in the page, which renders it in the browser).
#
# Defaults to index.html when no target is given.

set -euo pipefail

md="${1:-}"
html="${2:-index.html}"

if [ -z "$md" ]; then
  echo "usage: $0 <article.md> [target.html]" >&2
  exit 2
fi
if [ ! -f "$md" ]; then
  echo "error: markdown file not found: $md" >&2
  exit 1
fi
if [ ! -f "$html" ]; then
  echo "error: target html not found: $html" >&2
  exit 1
fi
if ! grep -q '<!-- BAKE:START -->' "$html" || ! grep -q '<!-- BAKE:END -->' "$html"; then
  echo "error: $html is missing the <!-- BAKE:START --> / <!-- BAKE:END --> markers" >&2
  exit 1
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

# Walk the HTML: keep everything up to and including BAKE:START, drop the old
# baked body, insert the markdown verbatim, then resume at BAKE:END.
awk -v mdfile="$md" '
  /<!-- BAKE:START -->/ {
    print
    while ((getline line < mdfile) > 0) print line
    close(mdfile)
    skip = 1
    next
  }
  /<!-- BAKE:END -->/ { skip = 0 }
  !skip { print }
' "$html" > "$tmp"

mv "$tmp" "$html"
trap - EXIT
echo "baked $md -> $html"
