#!/usr/bin/env bash
# =============================================================================
# fetch_catalogs.sh
#
# Reproducible, dated fetch of the two source catalogs the IC pipeline needs.
# Provenance for the exact queries is your own repo:
#   supporting_data/TLEhistoric/discos_plus_TLE_historic.m documents the
#   Space-Track query used to build the yearly catalogs, and the two SATCAT
#   sources (CelesTrak + Space-Track) used for status/RCS.
#
#   2026.csv    Space-Track GP snapshot of the on-orbit LEO-region catalog.
#               The historic files used class/gp_history with a <=3-day EPOCH
#               window + MEAN_MOTION>3; 2026.csv is the SAME predicates against
#               class/gp (the current snapshot -- spread CREATION_DATE + live
#               TLE lines, one latest element set per object, no epoch window).
#               Documented query (historic form):
#   .../class/gp_history/EPOCH/2023-01-01--2023-01-03/DECAY_DATE/null-val/MEAN_MOTION/>3/orderby/NORAD_CAT_ID/format/csv
#               Feeds build_2026.m / size_debris_rcs.m.
#
#   satcat.csv  NORAD_CAT_ID -> OBJECT_ID (international designator) map used by
#               build_rb_injection_persat.py for exact launch pairing.
#               discos_plus_TLE_historic.m documents BOTH sources:
#                 CelesTrak  https://celestrak.org/satcat/search.php   (default)
#                 Space-Track  .../class/SATCAT/orderby/NORAD_CAT_ID/format/csv
#               Default is CelesTrak (matches the bare 'satcat.csv' filename you
#               have). Use --satcat-spacetrack to pull the Space-Track SATCAT
#               instead (same login as the GP fetch; keeps provenance uniform).
#
# Verifies each download's columns, saves a DATED archival copy + a manifest
# (sha256 + row count + exact query), so the fetch is a committed, dated step.
#
# ---------------------------------------------------------------------------
# CREDENTIALS (never hard-coded):
#   export ST_USER='you@example.com'; export ST_PASS='...'
#   or create ./space_track.env (sourced by this script) with ST_USER=/ST_PASS=
#   ADD space_track.env TO .gitignore -- do not commit credentials.
#
# USAGE:
#   ./fetch_catalogs.sh                     # both, CelesTrak SATCAT
#   ./fetch_catalogs.sh --satcat-spacetrack # both, Space-Track SATCAT
#   ./fetch_catalogs.sh --outdir .          # explicit output dir
#   ./fetch_catalogs.sh --satcat-only       # skip the GP pull
#   ./fetch_catalogs.sh --gp-only           # skip the SATCAT pull
#
# REPRODUCIBILITY CAVEAT: class/gp returns the CURRENT on-orbit catalog, so a
# later run is a LATER snapshot (hence the dated archive + sha256). A true
# historical freeze would need class/gp_history with an EPOCH window; the
# original 2026.csv was a class/gp snapshot, so that is what this reproduces.
# =============================================================================

set -euo pipefail

OUTDIR="."
DO_GP=1
DO_SATCAT=1
SATCAT_SRC="celestrak"     # or "spacetrack"

while [ $# -gt 0 ]; do
  case "$1" in
    --outdir)            OUTDIR="$2"; shift 2 ;;
    --satcat-only)       DO_GP=0; shift ;;
    --gp-only)           DO_SATCAT=0; shift ;;
    --satcat-spacetrack) SATCAT_SRC="spacetrack"; shift ;;
    -h|--help)           sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

command -v curl >/dev/null || { echo "curl not found" >&2; exit 1; }

DATE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DATE_TAG="$(date -u +%Y%m%d)"
SNAPDIR="${OUTDIR}/catalog_snapshots"
MANIFEST="${SNAPDIR}/MANIFEST.tsv"
mkdir -p "$SNAPDIR"

sha256() {
  if command -v shasum >/dev/null; then shasum -a 256 "$1" | awk '{print $1}';
  elif command -v sha256sum >/dev/null; then sha256sum "$1" | awk '{print $1}';
  else echo "NA"; fi
}

if [ ! -f "$MANIFEST" ]; then
  printf 'fetched_utc\tfile\tsource\trows\tsha256\tquery_or_url\n' > "$MANIFEST"
fi

# verify a CSV header contains all required columns (quote/CRLF tolerant)
verify_columns() {
  local file="$1"; shift
  local header; header="$(head -1 "$file" | tr -d '\r"')"
  local missing=0
  for col in "$@"; do
    case ",$header," in *",$col,"*) ;; *) echo "  MISSING column: $col" >&2; missing=1 ;; esac
  done
  return $missing
}

# count rows whose NORAD_CAT_ID (located by header name) satisfies op vs th
#   usage: norad_count FILE le|ge THRESHOLD
norad_count() {
  awk -F, -v op="$2" -v th="$3" '
    NR==1 { for(i=1;i<=NF;i++){h=$i;gsub(/"/,"",h);if(h=="NORAD_CAT_ID")col=i}; next }
    col   { v=$col; gsub(/"/,"",v); n=v+0;
            if(op=="le"){ if(n<=th)c++ } else { if(n>=th)c++ } }
    END   { print c+0 }' "$1"
}

archive_and_record() {
  local file="$1" source="$2" query="$3"
  local rows sha snap
  rows="$(($(wc -l < "$file") - 1))"
  sha="$(sha256 "$file")"
  snap="${SNAPDIR}/$(basename "$file").${DATE_TAG}"
  cp "$file" "$snap"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$DATE_UTC" "$(basename "$file")" "$source" "$rows" "$sha" "$query" >> "$MANIFEST"
  echo "  rows=$rows  sha256=${sha:0:12}...  archived -> $snap"
}

# --- shared Space-Track login (idempotent) -----------------------------------
ST_BASE="https://www.space-track.org"
ST_COOKIES=""
st_login() {
  [ -n "$ST_COOKIES" ] && return 0
  if [ -f "${OUTDIR}/space_track.env" ]; then
    # shellcheck disable=SC1090
    set -a; . "${OUTDIR}/space_track.env"; set +a
  fi
  : "${ST_USER:?Set ST_USER (env or space_track.env) for Space-Track login}"
  : "${ST_PASS:?Set ST_PASS (env or space_track.env) for Space-Track login}"
  ST_COOKIES="$(mktemp)"
  echo "  authenticating to Space-Track..."
  curl -sS --fail-with-body -c "$ST_COOKIES" \
    --data-urlencode "identity=${ST_USER}" \
    --data-urlencode "password=${ST_PASS}" \
    "${ST_BASE}/ajaxauth/login" -o /dev/null \
    || { echo "  Space-Track login failed" >&2; exit 1; }
}
cleanup() { [ -n "$ST_COOKIES" ] && rm -f "$ST_COOKIES"; }
trap cleanup EXIT

# =============================================================================
# 1) Space-Track GP -> 2026.csv
# =============================================================================
if [ "$DO_GP" -eq 1 ]; then
  echo "[1/2] Space-Track GP on-orbit LEO-region catalog -> ${OUTDIR}/2026.csv"
  st_login
  # documented predicates from discos_plus_TLE_historic.m, current-snapshot form:
  #   class/gp  DECAY_DATE=null  MEAN_MOTION>3  orderby NORAD_CAT_ID  CSV
  GP_QUERY="/basicspacedata/query/class/gp/DECAY_DATE/null-val/MEAN_MOTION/%3E3/orderby/NORAD_CAT_ID/format/csv"

  echo "  querying GP catalog..."
  curl -sS --fail-with-body -b "$ST_COOKIES" "${ST_BASE}${GP_QUERY}" -o "${OUTDIR}/2026.csv"

  if ! verify_columns "${OUTDIR}/2026.csv" \
        NORAD_CAT_ID OBJECT_TYPE RCS_SIZE SEMIMAJOR_AXIS MEAN_ANOMALY LAUNCH_DATE; then
    echo "  ERROR: 2026.csv missing expected GP columns (login or query issue)." >&2
    echo "  First line was:" >&2; head -1 "${OUTDIR}/2026.csv" >&2
    exit 1
  fi
  archive_and_record "${OUTDIR}/2026.csv" "space-track:gp" "${GP_QUERY}"

  # kept-count after the >80000 temp-ID drop the build scripts apply
  total="$(($(wc -l < "${OUTDIR}/2026.csv") - 1))"
  kept="$(norad_count "${OUTDIR}/2026.csv" le 80000)"
  big6="$(norad_count "${OUTDIR}/2026.csv" ge 100000)"
  echo "  total=${total}  kept(<=80000)=${kept}  (original June-2026 snapshot: 30086 -> 29039)"
  [ "${big6:-0}" -gt 0 ] && echo "  NOTE: ${big6} objects have 6-digit NORAD IDs; the >80000 build filter drops them."
else
  echo "[1/2] skipped GP (--satcat-only)"
fi

# =============================================================================
# 2) SATCAT -> satcat.csv   (CelesTrak default, or Space-Track)
# =============================================================================
if [ "$DO_SATCAT" -eq 1 ]; then
  if [ "$SATCAT_SRC" = "spacetrack" ]; then
    echo "[2/2] Space-Track SATCAT -> ${OUTDIR}/satcat.csv"
    st_login
    SATCAT_QUERY="/basicspacedata/query/class/satcat/orderby/NORAD_CAT_ID/format/csv"
    curl -sS --fail-with-body -b "$ST_COOKIES" "${ST_BASE}${SATCAT_QUERY}" -o "${OUTDIR}/satcat.csv"
    SATCAT_LABEL="space-track:satcat"; SATCAT_REF="${SATCAT_QUERY}"
  else
    echo "[2/2] CelesTrak SATCAT -> ${OUTDIR}/satcat.csv"
    CELESTRAK_URL="https://celestrak.org/pub/satcat.csv"
    UA="Mozilla/5.0 (fetch_catalogs.sh; research use)"
    curl -sS --fail-with-body -L -A "$UA" "$CELESTRAK_URL" -o "${OUTDIR}/satcat.csv" \
      || { echo "  CelesTrak fetch failed (anti-bot / network). Retry from a browser network, or use --satcat-spacetrack." >&2; exit 1; }
    SATCAT_LABEL="celestrak:satcat"; SATCAT_REF="${CELESTRAK_URL}"
  fi

  # need NORAD_CAT_ID and an intl-designator column (OBJECT_ID on CelesTrak,
  # INTLDES on Space-Track); build_rb_injection_persat.py accepts either.
  verify_columns "${OUTDIR}/satcat.csv" NORAD_CAT_ID || {
    echo "  ERROR: satcat.csv missing NORAD_CAT_ID." >&2; head -1 "${OUTDIR}/satcat.csv" >&2; exit 1; }
  hdr="$(head -1 "${OUTDIR}/satcat.csv" | tr -d '\r"')"
  case ",$hdr," in
    *,OBJECT_ID,*|*,INTLDES,*) ;;
    *) echo "  ERROR: satcat.csv missing OBJECT_ID/INTLDES." >&2; echo "  header: $hdr" >&2; exit 1 ;;
  esac
  archive_and_record "${OUTDIR}/satcat.csv" "${SATCAT_LABEL}" "${SATCAT_REF}"

  # rollover guard: build scripts drop NORAD>80000; real 6-digit (>=100000)
  # catalog numbers begin ~2026-07-12 and would be discarded by that filter.
  bignum="$(norad_count "${OUTDIR}/satcat.csv" ge 100000 2>/dev/null || echo 0)"
  if [ "${bignum:-0}" -gt 0 ]; then
    echo "  WARNING: satcat.csv has ${bignum} objects with 6-digit catalog numbers"
    echo "           (>=100000); the >80000 temp-ID filter in your build scripts"
    echo "           will discard these. Revisit that threshold before regenerating."
  fi
else
  echo "[2/2] skipped SATCAT (--gp-only)"
fi

echo
echo "Done. Manifest: ${MANIFEST}"
echo "Commit catalog_snapshots/ (dated copies + manifest); keep space_track.env untracked."