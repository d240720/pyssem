#!/usr/bin/env python3
"""discos_puller.py

Pull physical mass and average cross-section from ESA DISCOSweb for a list of
NORAD IDs (SATNOs), to build a physically-sized IC. Uses the bearer-token v2
API. Resumable, rate-limit aware, caches incrementally to CSV.

Usage (export DISCOS_TOKEN first; ALWAYS smoke-test with --limit 5 before the
full pull, which is safe to re-run -- it skips what's already cached):
  python3 discos_puller.py --satnos intact_satnos.csv --out discos_cache.csv
  ... --retry-misses     # re-fetch objects previously recorded as not-found

Input : CSV/txt, one NORAD ID per line (header tolerated).
Output: satno,cosparId,name,objectClass,mass_kg,xSectAvg_m2,radius_m,shape,found
        (found=1 with blank mass_kg = in DISCOS but no measured mass).

Notes:
  * radius_m = sqrt(xSectAvg/pi), equivalent-circle collision radius.
  * Append-only for durability: readers keep the LAST row per satno
    (load_cache does; --compact rewrites deduplicated).
  * --retry-misses keys on found=0; found=1 rows with blank mass are never
    re-fetched -- delete those rows (or the cache) if DISCOS may have gained data.
"""

import argparse
import csv
import math
import os
import sys
import time
import urllib.parse
from datetime import timezone
from email.utils import parsedate_to_datetime

import requests

BASE      = "https://discosweb.esoc.esa.int"
ENDPOINT  = BASE + "/api/objects"
API_VER   = "2"
PAGE_SIZE = 100          # JSON:API page size (DISCOS max is typically 100)
BATCH     = 100          # SATNOs per request via in(satno,(...))
BASE_SLEEP = 2.0         # polite delay between requests (DISCOS ~30 req/min)

MAX_429_RETRIES  = 6     # per page, before giving up on the batch
MAX_HTTP_RETRIES = 3     # transient 5xx / network errors, per page
BACKOFF = [2, 8, 30]     # seconds, indexed by attempt

FIELDS = ["satno", "cosparId", "name", "objectClass",
          "mass_kg", "xSectAvg_m2", "radius_m", "shape", "found"]


def load_satnos(path):
    out, seen = [], set()
    with open(path, newline="") as f:
        for line in f:
            s = line.strip().strip(",").strip('"')
            try:
                v = int(float(s))
            except ValueError:
                continue          # blank, header, or junk line
            if v > 0 and v not in seen:
                seen.add(v)
                out.append(v)
    return out


def load_cache(path):
    """Last row per satno wins, so re-runs supersede earlier misses."""
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    done[int(row["satno"])] = row
                except (KeyError, ValueError, TypeError):
                    pass
    return done


def cell(row, key):
    """DictReader yields None for missing trailing fields on truncated rows."""
    return (row.get(key) or "").strip()


def radius_from_xsect(x):
    try:
        x = float(x)
        if x > 0:
            return round(math.sqrt(x / math.pi), 6)
    except (TypeError, ValueError):
        pass
    return ""


def parse_retry_after(value, default=60):
    """Retry-After may be delta-seconds or an HTTP-date; tolerate both."""
    if not value:
        return default
    try:
        return max(1, min(int(float(value)), 300))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:                  # naive HTTP-date: RFC says GMT
            dt = dt.replace(tzinfo=timezone.utc)
        return max(1, min(int(dt.timestamp() - time.time()), 300))
    except (TypeError, ValueError, IndexError):
        return default


def throttle_from_headers(h):
    """Sleep if the rate-limit headers say we're near the cap."""
    rem   = h.get("X-RateLimit-Remaining") or h.get("RateLimit-Remaining")
    reset = h.get("X-RateLimit-Reset")     or h.get("RateLimit-Reset")
    try:
        if rem is not None and int(float(rem)) <= 1 and reset is not None:
            r = int(float(reset))
            wait = r - int(time.time()) if r > 100000 else r   # epoch vs seconds
            wait = max(1, min(wait, 120))
            print(f"    quota low; sleeping {wait}s to reset")
            time.sleep(wait + 1)
    except (TypeError, ValueError):
        pass


def get_with_retries(session, url):
    """GET one page. Transient network errors and 5xx share a backoff budget;
    429 honors Retry-After up to MAX_429_RETRIES; 401 exits with a hint.
    Raises requests.RequestException once the relevant budget is exhausted."""
    n429 = nerr = 0
    while True:
        try:
            resp = session.get(url, timeout=60)
        except requests.RequestException as e:
            if nerr >= MAX_HTTP_RETRIES:
                raise
            wait = BACKOFF[min(nerr, len(BACKOFF) - 1)]
            print(f"    network error ({e.__class__.__name__}); retry in {wait}s")
            nerr += 1
            time.sleep(wait)
            continue

        if resp.status_code == 401:
            sys.exit("401 Unauthorized -- check DISCOS_TOKEN (is it current?).")

        if resp.status_code == 429:
            if n429 >= MAX_429_RETRIES:
                raise requests.HTTPError(
                    f"429 persisted after {MAX_429_RETRIES} retries "
                    "(daily quota exhausted?)", response=resp)
            wait = parse_retry_after(resp.headers.get("Retry-After"))
            n429 += 1
            print(f"    429 rate-limited ({n429}/{MAX_429_RETRIES}); sleeping {wait}s")
            time.sleep(wait + 1)
            continue

        if 500 <= resp.status_code < 600:
            if nerr >= MAX_HTTP_RETRIES:
                resp.raise_for_status()
            wait = BACKOFF[min(nerr, len(BACKOFF) - 1)]
            print(f"    HTTP {resp.status_code}; retry in {wait}s")
            nerr += 1
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp


def fetch_batch(session, satnos, sleep=BASE_SLEEP):
    """Attribute dicts for the given SATNOs, following JSON:API pagination.
    Raises requests.RequestException if a page cannot be retrieved; the caller
    skips the batch and the SATNOs stay uncached so a re-run picks them up."""
    filt = "in(satno,({}))".format(",".join(str(s) for s in satnos))
    # Build the query manually: encode the filter value, leave page[...] literal.
    url = f"{ENDPOINT}?filter={urllib.parse.quote(filt)}&page[size]={PAGE_SIZE}&page[number]=1"
    results = []
    while url:
        resp = get_with_retries(session, url)
        body = resp.json()
        results += [item.get("attributes", {}) for item in body.get("data", [])]
        throttle_from_headers(resp.headers)
        nxt = (body.get("links") or {}).get("next")
        url = nxt and (nxt if nxt.startswith("http") else BASE + nxt)
        if url:
            time.sleep(sleep)      # pace pagination, not just batches
    return results


def compact(path):
    """Rewrite the cache deduplicated, last row per satno."""
    cache = load_cache(path)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        w.writeheader()
        for sn in sorted(cache):
            w.writerow({k: cache[sn].get(k, "") for k in FIELDS})
    os.replace(tmp, path)
    print(f"Compacted {path}: {len(cache)} unique SATNOs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--satnos", required=True, help="CSV/txt of NORAD IDs")
    ap.add_argument("--out", default="discos_cache.csv")
    ap.add_argument("--sleep", type=float, default=BASE_SLEEP)
    ap.add_argument("--limit", type=int, default=0, help="fetch only first N (smoke test)")
    ap.add_argument("--retry-misses", action="store_true",
                    help="re-fetch SATNOs previously recorded as found=0")
    ap.add_argument("--compact", action="store_true",
                    help="rewrite the cache deduplicated before summarizing")
    args = ap.parse_args()

    token = os.environ.get("DISCOS_TOKEN")
    if not token:
        sys.exit("Set DISCOS_TOKEN to your DISCOSweb API token first.")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}",
                            "DiscosWeb-Api-Version": API_VER})

    satnos = load_satnos(args.satnos)
    cache  = load_cache(args.out)

    if args.retry_misses:
        skip = {s for s, r in cache.items() if cell(r, "found") == "1"}
        n_retry = sum(1 for s in satnos if s in cache and s not in skip)
        print(f"--retry-misses: re-fetching {n_retry} previously-missing SATNOs")
    else:
        skip = set(cache)

    todo = [s for s in satnos if s not in skip]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"Requested: {len(satnos)} | already cached: {len(cache)} | to fetch: {len(todo)}")
    if not todo:
        print("Nothing to fetch. Cache is complete.")
        if args.compact:
            compact(args.out)
        summarize(args.out)
        return

    # An existing but empty file (crash before the header write) still needs one.
    need_header = (not os.path.exists(args.out)) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        if need_header:
            w.writeheader()
        nbatches = math.ceil(len(todo) / BATCH)
        for bi in range(0, len(todo), BATCH):
            batch = todo[bi:bi + BATCH]
            print(f"batch {bi//BATCH + 1}/{nbatches}: {len(batch)} satnos")
            try:
                attrs = fetch_batch(session, batch, sleep=args.sleep)
            except requests.RequestException as e:
                r = getattr(e, "response", None)
                if r is not None and r.status_code == 429:
                    # No point burning MAX_429_RETRIES on every remaining batch:
                    # a persistent 429 means the quota is gone for everyone.
                    sys.exit("  429 persisted through retries -- daily quota likely "
                             "exhausted. Progress so far is cached; re-run later "
                             "to resume.")
                print(f"  request failed: {e}; skipping this batch (rerun to retry)")
                time.sleep(args.sleep)         # stay polite even on the failure path
                continue

            got = set()
            for a in attrs:
                try:
                    sn = int(a.get("satno"))
                except (TypeError, ValueError):
                    continue
                got.add(sn)
                w.writerow({
                    "satno": sn,
                    "cosparId": a.get("cosparId") or "",
                    "name": a.get("name") or "",
                    "objectClass": a.get("objectClass") or "",
                    # is-not-None (not `or`) so a genuine 0 survives as 0
                    "mass_kg": a["mass"] if a.get("mass") is not None else "",
                    "xSectAvg_m2": a["xSectAvg"] if a.get("xSectAvg") is not None else "",
                    "radius_m": radius_from_xsect(a.get("xSectAvg")),
                    "shape": a.get("shape") or "",
                    "found": 1,
                })
            for sn in batch:                      # record misses so we don't refetch
                if sn not in got:
                    w.writerow({"satno": sn, "found": 0})
            fh.flush()
            os.fsync(fh.fileno())                 # incremental durability
            time.sleep(args.sleep)

    if args.compact:
        compact(args.out)
    summarize(args.out)


def summarize(path):
    cache = load_cache(path)          # deduplicated: last row per satno wins
    n = len(cache)
    found     = sum(cell(r, "found") == "1" for r in cache.values())
    withmass  = sum(cell(r, "found") == "1" and bool(cell(r, "mass_kg"))
                    for r in cache.values())
    withxsect = sum(cell(r, "found") == "1" and bool(cell(r, "xSectAvg_m2"))
                    for r in cache.values())
    pct = lambda a: f"{100*a/n:.1f}%" if n else "0%"
    print("\n=== DISCOS pull summary ===")
    print(f"  cached SATNOs        : {n}")
    print(f"  found in DISCOS      : {found} ({pct(found)})")
    print(f"  ...with a mass value : {withmass} ({pct(withmass)})")
    print(f"  ...with a xSect value: {withxsect} ({pct(withxsect)})")
    print(f"  missing / no mass    : {n - withmass} ({pct(n - withmass)})")


if __name__ == "__main__":
    main()