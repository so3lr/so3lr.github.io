#!/bin/sh
# Preview helper: activate one style variant from styles-preview/.
# Usage: ./swap-style.sh paper | graphite | editorial | current
set -e
V="$1"
[ -f "styles-preview/$V.css" ] || { echo "no such variant: $V"; ls styles-preview/*.css; exit 1; }
if [ "$V" = current ]; then
  cp styles-preview/current.css docs/stylesheets/extra.css
else
  cat styles-preview/_base.css "styles-preview/$V.css" > docs/stylesheets/extra.css
fi
echo "active style: $V"
