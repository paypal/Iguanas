#!/usr/bin/env bash
# Build a .docx from the manuscript.
# Pandoc's texmath (2.9) cannot render \bigvee, \bigcup, \setminus, \textstyle or \tag{},
# so these are rewritten to equivalents that do render. The markdown source stays canonical
# LaTeX for a journal submission; only the docx build is adapted.
set -euo pipefail
SRC="${1:-iguanas_eswa_submission.md}"
OUT="${2:-iguanas_eswa_submission.docx}"
TMP="$(mktemp /tmp/paper_docx_XXXX.md)"
sed -E \
  -e 's/\\tag\{([0-9]+)\}/\\qquad (\1)/g' \
  -e 's/\\bigvee/\\vee/g' \
  -e 's/\\bigcup/\\cup/g' \
  -e 's/\\setminus/\\smallsetminus/g' \
  -e 's/\\textstyle//g' \
  "$SRC" > "$TMP"
pandoc "$TMP" \
  -f markdown+pipe_tables+tex_math_dollars \
  -t docx --toc --toc-depth=2 \
  ${REFDOC:+--reference-doc="$REFDOC"} \
  -o "$OUT"
rm -f "$TMP"
echo "wrote $OUT"
