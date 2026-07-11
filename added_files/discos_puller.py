#!/usr/bin/env python3
"""discos_puller.py

Pull physical mass and average cross-section from ESA DISCOSweb for a list of
NORAD IDs (SATNOs), to build a physically-sized IC. Uses the bearer-token v2
API. Resumable, rate-limit aware, caches incrementally to CSV.

Usage:
  export DISCOS_TOKEN='your-token-here'
  # smoke test on the first 5 first, to confirm auth + filter syntax:
  python3 discos_puller.py --satnos intact_satnos.csv --out discos_cache.csv --limit 5
  # then the full pull (safe to re-run; it skips what's already cached):
  python3 discos_puller.py --satnos intact_satnos.csv --out discos_cache.csv

Input : CSV/txt with one NORAD ID (SATNO) per line (header line tolerated).
Output: discos_cache.csv with columns
        satno,cosparId,name,objectClass,mass_kg,xSectAvg_m2,radius_m,shape,found
        found=1 means DISCOS returned the object; mass_kg may still be blank if
        DISCOS has no measured mass for it.

Notes:
  * radius_m = sqrt(xSectAvg / pi): equivalent-circle radius from the average
    geometric cross-section -- a better collision radius than an RCS estimate.
  * The exact rate-limit header names / filter syntax can vary by API version;
    this handles the common cases and fails loudly otherwise. RUN --limit 5
    FIRST so you validate auth and syntax cheaply before the full pull.
"""

import os, sys, csv, time, math, argparse, urllib.parse
import requests

BASE      = "https://discosweb.esoc.esa.int"
ENDPOINT  = BASE + "/api/objects"
API_VER   = "2"
PAGE_SIZE = 100          # JSON:API page size (DISCOS max is typically 100)
BATCH     = 100          # SATNOs per request via in(satno,(...))
BASE_SLEEP = 2.0         # polite delay between requests (DISCOS ~30 req/min)

FIELDS = ["satno","cosparId","name","objectClass",
          "mass_kg","xSectAvg_m2","radius_m","shape","found"]


def load_satnos(path):
    out, seen = [], set()
    with open(path, newline="") as f:
        for line in f:
            s = line.strip().strip(",").strip('"')
            if not s:
                continue
            try:
                v = int(float(s))
            except ValueError:
                continue          # header or junk line
            if v > 0 and v not in seen:
                seen.add(v); out.append(v)
    return out


def load_cache(path):
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    done[int(row["satno"])] = row
                except (KeyError, ValueError, TypeError):
                    pass
    return done


def radius_from_xsect(x):
    try:
        x = float(x)
        if x > 0:
            return round(math.sqrt(x / math.pi), 6)
    except (TypeError, ValueError):
        pass
    return ""


def throttle_from_headers(h):
    """Sleep if the rate-limit headers say we're near the cap."""
    rem   = h.get("X-RateLimit-Remaining") or h.get("RateLimit-Remaining")
    reset = h.get("X-RateLimit-Reset")     or h.get("RateLimit-Reset")
    try:
        if rem is not None and int(rem) <= 1 and reset is not None:
            r = int(reset)
            wait = r - int(time.time()) if r > 100000 else r   # epoch vs seconds
            wait = max(1, min(wait, 120))
            print(f"    quota low; sleeping {wait}s to reset")
            time.sleep(wait + 1)
    except ValueError:
        pass


def fetch_batch(token, satnos):
    """Return a list of attribute dicts for the given SATNOs."""
    headers = {"Authorization": f"Bearer {token}", "DiscosWeb-Api-Version": API_VER}
    filt = "in(satno,({}))".format(",".join(str(s) for s in satnos))
    # Build the query manually: encode the filter value, leave page[...] literal.
    q = (f"?filter={urllib.parse.quote(filt)}"
         f"&page[size]={PAGE_SIZE}&page[number]=1")
    url = ENDPOINT + q
    results = []
    while url:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "60"))
            print(f"    429 rate-limited; sleeping {wait}s")
            time.sleep(wait + 1)
            continue
        if resp.status_code == 401:
            sys.exit("401 Unauthorized -- check DISCOS_TOKEN (is it current?).")
        resp.raise_for_status()
        body = resp.json()
        for item in body.get("data", []):
            results.append(item.get("attributes", {}))
        throttle_from_headers(resp.headers)
        nxt = body.get("links", {}).get("next")
        url = (BASE + nxt) if nxt else None
    return results


def miss_row(satno):
    row = {k: "" for k in FIELDS}
    row["satno"], row["found"] = satno, 0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--satnos", required=True, help="CSV/txt of NORAD IDs")
    ap.add_argument("--out", default="discos_cache.csv")
    ap.add_argument("--sleep", type=float, default=BASE_SLEEP)
    ap.add_argument("--limit", type=int, default=0, help="fetch only first N (smoke test)")
    args = ap.parse_args()

    token = os.environ.get("DISCOS_TOKEN")
    if not token:
        sys.exit("Set DISCOS_TOKEN to your DISCOSweb API token first.")

    satnos = load_satnos(args.satnos)
    cache  = load_cache(args.out)
    todo   = [s for s in satnos if s not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Requested: {len(satnos)} | already cached: {len(cache)} | to fetch: {len(todo)}")
    if not todo:
        print("Nothing to fetch. Cache is complete.")
        summarize(args.out)
        return

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        nbatches = -(-len(todo) // BATCH)
        for bi in range(0, len(todo), BATCH):
            batch = todo[bi:bi + BATCH]
            print(f"batch {bi//BATCH + 1}/{nbatches}: {len(batch)} satnos")
            try:
                attrs = fetch_batch(token, batch)
            except requests.HTTPError as e:
                print(f"  HTTP error: {e}; skipping this batch (rerun to retry)")
                continue
            got = {}
            for a in attrs:
                sn = a.get("satno")
                if sn is None:
                    continue
                sn = int(sn)
                got[sn] = True
                w.writerow({
                    "satno": sn,
                    "cosparId": a.get("cosparId", ""),
                    "name": a.get("name", ""),
                    "objectClass": a.get("objectClass", ""),
                    "mass_kg": a.get("mass", "") if a.get("mass") is not None else "",
                    "xSectAvg_m2": a.get("xSectAvg", "") if a.get("xSectAvg") is not None else "",
                    "radius_m": radius_from_xsect(a.get("xSectAvg")),
                    "shape": a.get("shape", ""),
                    "found": 1,
                })
            for sn in batch:                      # record misses so we don't refetch
                if sn not in got:
                    w.writerow(miss_row(sn))
            fh.flush()                            # incremental durability
            time.sleep(args.sleep)

    summarize(args.out)


def summarize(path):
    n = found = withmass = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n += 1
            if row.get("found") == "1":
                found += 1
                if row.get("mass_kg", "").strip():
                    withmass += 1
    print("\n=== DISCOS pull summary ===")
    print(f"  cached SATNOs        : {n}")
    print(f"  found in DISCOS      : {found} ({pct(found,n)})")
    print(f"  ...with a mass value : {withmass} ({pct(withmass,n)})  <- usable for sizing")
    print(f"  missing / no mass    : {n - withmass} ({pct(n-withmass,n)})  <- need fallback rule")


def pct(a, b):
    return f"{100*a/b:.1f}%" if b else "0%"


if __name__ == "__main__":
    main()