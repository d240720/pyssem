#!/usr/bin/env python3
"""
build_rb_injection.py

Per-satellite rocket-body injection model, in the spirit of build_rb_injection_persat.py
but driven by the same catalog CSVs this project already uses (2026.csv GP snapshot +
discos_cache.csv), and written to be consumed by targeted_launches.py.

Idea (unchanged from the original): for each satellite whose payload targets band b,
count the expected number of SURVIVING rocket-body stages deposited into each band b':

    R_persat[b][b'] = (surviving stages from launches targeting b, landing in b')
                      / (payloads targeting b)

"Surviving" = still on orbit in the snapshot, so recent low-altitude constellation
bands come out nearly stage-clean (stages have deorbited) and upper bands stage-dirty
(stages persist). Restricting to a recent launch window bakes in current practice.

Launch pairing is exact via the international designator (OBJECT_ID 'YYYY-NNN'), which
2026.csv carries directly -- no NORAD->intldes SATCAT step needed.

Output: rb_injection.json  { bands, R_persat, stage_mass_kg, stage_radius_m, ... }

USAGE: python3 build_rb_injection.py [--gp 2026.csv] [--discos discos_cache.csv]
           [--win 2020 2025] [--min-alt 200] [--max-alt 2000] [--out rb_injection.json]
"""

import argparse
import json

import numpy as np
import pandas as pd

# Physically-motivated altitude bands (km). Coarser than pyssem shells on purpose:
# surviving stages are sparse, so rates are estimated per band, then spread across
# the shells of each band when applied.
BANDS = [(200, 300), (300, 400), (400, 500), (500, 600),
         (600, 700), (700, 1000), (1000, 2000)]
NB = len(BANDS)


def band_of(alt):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= alt < hi:
            return i
    return -1


def launch_no(object_id):
    """International designator -> launch id 'YYYY-NNN' (drop the piece suffix)."""
    s = str(object_id).strip()
    if "-" in s:
        yr, rest = s.split("-", 1)
        return f"{yr}-{rest[:3]}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", default="2026.csv")
    ap.add_argument("--discos", default="discos_cache.csv")
    ap.add_argument("--win", type=float, nargs=2, default=[2020, 2025])
    ap.add_argument("--min-alt", type=float, default=200)
    ap.add_argument("--max-alt", type=float, default=2000)
    ap.add_argument("--out", default="rb_injection.json")
    a = ap.parse_args()

    gp = pd.read_csv(a.gp, low_memory=False)
    gp["NORAD_CAT_ID"] = pd.to_numeric(gp["NORAD_CAT_ID"], errors="coerce")
    gp["alt"] = (gp["APOAPSIS"] + gp["PERIAPSIS"]) / 2.0
    gp["ly"] = pd.to_datetime(gp["LAUNCH_DATE"], errors="coerce").dt.year

    # On-orbit survivors, launched in the window, within the altitude range.
    w = gp[(gp["DECAY_DATE"].isna())
           & (gp["ly"] >= a.win[0]) & (gp["ly"] <= a.win[1])
           & (gp["alt"] >= a.min_alt) & (gp["alt"] < a.max_alt)].copy()
    w["ln"] = w["OBJECT_ID"].map(launch_no)

    payloads_per_band = np.zeros(NB)      # payloads (satellites) targeting band b
    stages_band = np.zeros((NB, NB))      # surviving stages: payload-band b -> stage-band b'
    n_launch = 0
    stage_ids = []
    for _, grp in w.groupby("ln"):
        pay = grp[grp["OBJECT_TYPE"] == "PAYLOAD"]
        rb = grp[grp["OBJECT_TYPE"] == "ROCKET BODY"]
        pbs = [band_of(x) for x in pay["alt"] if band_of(x) >= 0]
        if not pbs:
            continue
        pb = max(set(pbs), key=pbs.count)          # this launch's payload band = mode
        payloads_per_band[pb] += len(pbs)          # count all payloads on the launch
        n_launch += 1
        for alt, nid in zip(rb["alt"], rb["NORAD_CAT_ID"]):
            sb = band_of(alt)
            if sb >= 0:
                stages_band[pb, sb] += 1
                if not np.isnan(nid):
                    stage_ids.append(int(nid))

    R = np.zeros((NB, NB))
    for b in range(NB):
        if payloads_per_band[b] > 0:
            R[b] = stages_band[b] / payloads_per_band[b]
    persat_total = R.sum(axis=1)

    # Stage mass/radius from DISCOS (median of the surviving stages).
    di = pd.read_csv(a.discos, low_memory=False)
    di["satno"] = pd.to_numeric(di["satno"], errors="coerce")
    st = di[di["satno"].isin(stage_ids)]
    stage_mass = float(st["mass_kg"].median()) if st["mass_kg"].notna().any() else 1421.0
    stage_radius = float(st["radius_m"].median()) if st["radius_m"].notna().any() else 1.82

    # ---- report ----
    print(f"exact launches in window {a.win[0]:.0f}-{a.win[1]:.0f}: {n_launch}")
    print(f"surviving stages: {len(stage_ids)} | stage mass {stage_mass:.0f} kg, "
          f"radius {stage_radius:.2f} m\n")
    print("per-satellite stage injection by payload band:")
    for b in range(NB):
        print(f"  B{b+1} {BANDS[b][0]:>4}-{BANDS[b][1]:<4}: "
              f"{int(payloads_per_band[b]):>6} sats | {persat_total[b]:.5f} stages/sat")
    print("\nR_persat[b -> b']  (rows = payload band, cols = stage band):")
    print("        " + "".join(f"  B{j+1:<5}" for j in range(NB)))
    for b in range(NB):
        print(f"  B{b+1:<5}" + "".join(f"{R[b, j]:7.4f}" for j in range(NB)))

    out = dict(
        bands=BANDS,
        R_persat=R.tolist(),
        stages_per_satellite=persat_total.tolist(),
        payloads_per_band=payloads_per_band.tolist(),
        stage_mass_kg=stage_mass,
        stage_radius_m=stage_radius,
        window=list(a.win),
        n_launch=n_launch,
        objectclass=5, controlled=0,
        note=("R_persat[b][bp] = expected surviving rocket-body stages deposited in "
              "band bp per satellite launched to band b. Apply by mapping each pyssem "
              "shell to its band and spreading each band->band rate uniformly across "
              "the destination band's shells."),
    )
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()