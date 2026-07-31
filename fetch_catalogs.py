#!/usr/bin/env python3
"""
fetch_catalogs.py

Dated fetch of the source catalogs.

  2026.csv    Space-Track GP snapshot of the on-orbit LEO-region catalog.
              Predicates: DECAY_DATE=null, MEAN_MOTION>3, ordered by NORAD_CAT_ID.

  satcat.csv  NORAD_CAT_ID -> OBJECT_ID map + operational status, used for exact
              launch pairing and live/dead payload classification.
              CelesTrak by default; --satcat-spacetrack for the Space-Track SATCAT.

Each download is written to a temp file, validated, and only then moved into place,
so a failed fetch never clobbers a good catalog. A dated archival copy + manifest
(sha256, row count, exact query) are recorded so the fetch is a committed step.

CREDENTIALS (never hard-coded):
    export ST_USER='you@example.com'; export ST_PASS='...'
  or put ST_USER=/ST_PASS= in space_track.env next to this script.
  ADD space_track.env TO .gitignore.

USAGE: python3 fetch_catalogs.py [--satcat-spacetrack] [--gp-only]
           [--satcat-only] [--outdir /data] [--gp-name 2026.csv]

REPRODUCIBILITY CAVEAT: class/gp returns the CURRENT on-orbit catalog, so a later
run is a LATER snapshot -- hence the dated archive + sha256. A true historical
freeze would need class/gp_history with an EPOCH window.
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ST_BASE = "https://www.space-track.org"
GP_QUERY = ("/basicspacedata/query/class/gp/DECAY_DATE/null-val"
            "/MEAN_MOTION/>3/orderby/NORAD_CAT_ID/format/csv")
SATCAT_QUERY = "/basicspacedata/query/class/satcat/orderby/NORAD_CAT_ID/format/csv"
CELESTRAK_URL = "https://celestrak.org/pub/satcat.csv"
UA = "fetch_catalogs.py (research use)"

# Columns the downstream scripts actually read.
GP_REQUIRED = ["NORAD_CAT_ID", "OBJECT_ID", "OBJECT_TYPE", "LAUNCH_DATE",
               "DECAY_DATE", "APOAPSIS", "PERIAPSIS", "RCS_SIZE"]
# CelesTrak uses OBJECT_ID, Space-Track SATCAT uses INTLDES; either is accepted.
SATCAT_REQUIRED = ["NORAD_CAT_ID", "OPS_STATUS_CODE"]
SATCAT_EITHER = ["OBJECT_ID", "INTLDES"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path):
    """Data rows (excludes the header). CSV-aware: quoted fields with embedded
    newlines (occasional in SATCAT names) count as one row, not two."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in csv.reader(f))
    return max(0, n - 1)


def verify_columns(path, required, either=()):
    """Raise if the CSV is missing any required column."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        header = next(csv.reader(f), [])
    header = [c.strip().lstrip("\ufeff") for c in header]
    missing = [c for c in required if c not in header]
    if either and not any(c in header for c in either):
        missing.append("/".join(either))
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {missing}\n  header was: {header[:12]}"
            f"{' ...' if len(header) > 12 else ''}")


def fetch(session, url, final_path, required, either=()):
    """Stream URL -> temp file -> verify -> atomically move into place. A failed
    or malformed download never overwrites the existing file."""
    with tempfile.NamedTemporaryFile(prefix=".fetch_", dir=str(final_path.parent),
                                     delete=False) as tf:
        tmp = tf.name
    try:
        with session.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
        verify_columns(tmp, required, either)
        shutil.move(tmp, final_path)          # only now do we clobber
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def archive_and_record(path, snapdir, manifest, source, query, fetched_utc):
    rows, sha = count_rows(path), sha256(path)
    snapdir.mkdir(parents=True, exist_ok=True)
    # Time-resolved tag: with a date-only tag, a second fetch the same day would
    # silently overwrite the first snapshot while the manifest kept both rows.
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    snap = snapdir / f"{path.name}.{tag}"
    shutil.copy2(path, snap)

    new = not manifest.exists()
    with open(manifest, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new:
            w.writerow(["fetched_utc", "file", "source", "rows", "sha256",
                        "query_or_url"])
        w.writerow([fetched_utc, path.name, source, rows, sha, query])

    print(f"  rows={rows}  sha256={sha[:12]}...  archived -> {snap}")


def new_session():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def space_track_session(outdir):
    """Authenticated Space-Track session. Credentials from env or space_track.env."""
    from_shell = bool(os.environ.get("ST_USER") and os.environ.get("ST_PASS"))
    for env in (Path(__file__).resolve().parent / "space_track.env",
                outdir / "space_track.env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            break

    user, pw = os.environ.get("ST_USER"), os.environ.get("ST_PASS")
    if not user or not pw:
        sys.exit("Set ST_USER and ST_PASS (env or space_track.env) for Space-Track.")
    print(f"  credentials from {'shell env (overrides space_track.env)' if from_shell else env}")

    s = new_session()
    print("  authenticating to Space-Track...")
    r = s.post(f"{ST_BASE}/ajaxauth/login",
               data={"identity": user, "password": pw}, timeout=60)
    r.raise_for_status()
    # 'chocolatechip' really is Space-Track's session cookie name; its absence
    # means the login didn't take even if the POST returned 200.
    if "Failed" in r.text or "chocolatechip" not in s.cookies:
        sys.exit("  Space-Track login failed (check ST_USER / ST_PASS).")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--gp-name", default="2026.csv",
                    help="GP output filename (downstream scripts default to 2026.csv)")
    ap.add_argument("--gp-only", action="store_true")
    ap.add_argument("--satcat-only", action="store_true")
    ap.add_argument("--satcat-spacetrack", action="store_true",
                    help="pull SATCAT from Space-Track instead of CelesTrak")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    snapdir = outdir / "catalog_snapshots"
    manifest = snapdir / "MANIFEST.tsv"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    st = None

    # ---- 1) Space-Track GP -> 2026.csv --------------------------------------
    if not a.satcat_only:
        gp_path = outdir / a.gp_name
        print(f"[1/2] Space-Track GP on-orbit LEO-region catalog -> {gp_path}")

        st = space_track_session(outdir)
        print("  querying GP catalog...")
        try:
            fetch(st, ST_BASE + GP_QUERY, gp_path, GP_REQUIRED)
        except (requests.RequestException, ValueError) as e:
            sys.exit(f"  GP fetch failed: {e}\n  (existing {gp_path} left untouched)")

        archive_and_record(gp_path, snapdir, manifest, "space-track:gp", GP_QUERY, now)
    else:
        print("[1/2] skipped GP (--satcat-only)")

    # ---- 2) SATCAT ----------------------------------------------------------
    if not a.gp_only:
        sc_path = outdir / "satcat.csv"
        if a.satcat_spacetrack:
            print(f"[2/2] Space-Track SATCAT -> {sc_path}")
            sess = st or space_track_session(outdir)
            url, label, ref = ST_BASE + SATCAT_QUERY, "space-track:satcat", SATCAT_QUERY
        else:
            print(f"[2/2] CelesTrak SATCAT -> {sc_path}")
            sess = new_session()
            url, label, ref = CELESTRAK_URL, "celestrak:satcat", CELESTRAK_URL

        try:
            fetch(sess, url, sc_path, SATCAT_REQUIRED, SATCAT_EITHER)
        except (requests.RequestException, ValueError) as e:
            sys.exit(f"  SATCAT fetch failed: {e}\n"
                     f"  (CelesTrak anti-bot? try --satcat-spacetrack)")
        archive_and_record(sc_path, snapdir, manifest, label, ref, now)
    else:
        print("[2/2] skipped SATCAT (--gp-only)")

    print(f"\nDone. Manifest: {manifest}")
    print("Commit catalog_snapshots/ (dated copies + manifest); "
          "keep space_track.env untracked.")


if __name__ == "__main__":
    main()