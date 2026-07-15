#!/usr/bin/env python3
"""
build_rb_injection.py

Per-satellite rocket-body injection model from the project catalogs
(2026.csv GP snapshot + discos_cache.csv).

    R_persat[b][b'] = (surviving stages from launches targeting b, landing in b')
                      / (payloads targeting b)

"Surviving" = still on orbit in the snapshot, so low bands come out nearly
stage-clean and upper bands stage-dirty. Launch pairing is exact via the
international designator (OBJECT_ID 'YYYY-NNN').

Stage mass/radius are emitted PER DESTINATION BAND: the 700-1000 km band holds
~4 t upper stages vs ~2.5 t at 400-600 km, and that band is where debris persists.

Caveats: the numerator counts survivors but pyssem re-applies drag, so low-band
rates are floors. The whole matrix rests on ~80-130 stages. Per-satellite
normalization is sensitive to launch batching -- run both windows, report the spread.

USAGE: python3 build_rb_injection.py [--gp 2026.csv] [--discos discos_cache.csv]
           [--win 2020 2025] [--min-alt 200] [--max-alt 2000]
           [--min-payloads 25] [--min-stages 3] [--no-exclude-kick-stages]
           [--out rb_injection.json]
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

BANDS = [(200, 300), (300, 400), (400, 500), (500, 600),
         (600, 700), (700, 1000), (1000, 2000)]
NB = len(BANDS)

# Light kick stages: cataloged ROCKET BODY but ~50 kg, not what the NASA SBM's
# rocket-body branch is calibrated on. Still counted in R_persat; only kept out
# of the mass/radius statistics.
KICK_STAGE_PATTERNS = ["Photon"]

MASS_BOUNDS = (100.0, 20000.0)     # kg
XSECT_BOUNDS = (0.1, 120.0)        # m^2


def band_of(alt):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= alt < hi:
            return i
    return -1


def launch_no(object_id):
    s = str(object_id).strip()
    if "-" in s:
        yr, rest = s.split("-", 1)
        return f"{yr}-{rest[:3]}"
    return s


def fit_area_law(di):
    """log(xSect) = c0 + c1*log(mass) over all rocket bodies in the cache.

    DISCOS xSect coverage is not missing-at-random: the L-15 class (n=22, mean
    5.8 t, dominant at 700-1000 km) has mass for every object and xSect for none.
    Dropping those rows biases B6 mass down ~40%, so impute instead.
    """
    rb = di[di["objectClass"].fillna("").str.contains("Rocket Body", na=False)]
    b = rb[rb["mass_kg"].between(*MASS_BOUNDS)
           & rb["xSectAvg_m2"].between(*XSECT_BOUNDS)]
    x, y = np.log(b["mass_kg"].to_numpy()), np.log(b["xSectAvg_m2"].to_numpy())
    c1, c0 = np.polyfit(x, y, 1)
    r2 = 1.0 - ((y - (c0 + c1 * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return c0, c1, r2, len(b)


def clean_stage_table(di, ids, exclude_kick, area_law):
    """DISCOS rows for `ids`: mass required, xSect imputed where absent."""
    c0, c1, _, _ = area_law
    s = di.reindex(sorted(set(int(i) for i in ids))).copy()
    s = s.dropna(subset=["mass_kg"])
    s = s[s["mass_kg"].between(*MASS_BOUNDS)]
    if exclude_kick and len(s):
        pat = "|".join(KICK_STAGE_PATTERNS)
        s = s[~s["name"].fillna("").str.contains(pat, case=False, regex=True)]
    s.loc[~s["xSectAvg_m2"].between(*XSECT_BOUNDS), "xSectAvg_m2"] = np.nan
    s["imputed"] = s["xSectAvg_m2"].isna()
    s.loc[s["imputed"], "xSectAvg_m2"] = np.exp(
        c0 + c1 * np.log(s.loc[s["imputed"], "mass_kg"]))
    return s


def mass_radius(s):
    """Mean mass (conserves the injected mass budget the SBM consumes) and
    area-equivalent radius sqrt(mean(xSect)/pi) (collisions scale with area).
    Median mass is unusable: the population spans 55 kg to 6 t and the median
    jumps 2x between nested samples."""
    return (float(s["mass_kg"].mean()),
            float(np.sqrt(s["xSectAvg_m2"].mean() / np.pi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", default="2026.csv")
    ap.add_argument("--discos", default="discos_cache.csv")
    ap.add_argument("--win", type=float, nargs=2, default=[2020, 2025])
    ap.add_argument("--min-alt", type=float, default=200)
    ap.add_argument("--max-alt", type=float, default=2000)
    ap.add_argument("--min-payloads", type=int, default=25,
                    help="suppress R_persat rows built from fewer payloads than this")
    ap.add_argument("--min-stages", type=int, default=3,
                    help="bands with fewer stages fall back to pooled mass/radius")
    ap.add_argument("--no-exclude-kick-stages", dest="exclude_kick",
                    action="store_false", default=True)
    ap.add_argument("--out", default="rb_injection.json")
    a = ap.parse_args()

    gp = pd.read_csv(a.gp, low_memory=False)
    gp["NORAD_CAT_ID"] = pd.to_numeric(gp["NORAD_CAT_ID"], errors="coerce")
    gp["alt"] = (gp["APOAPSIS"] + gp["PERIAPSIS"]) / 2.0
    gp["ly"] = pd.to_datetime(gp["LAUNCH_DATE"], errors="coerce").dt.year

    w = gp[(gp["DECAY_DATE"].isna())
           & (gp["ly"] >= a.win[0]) & (gp["ly"] <= a.win[1])
           & (gp["alt"] >= a.min_alt) & (gp["alt"] < a.max_alt)].copy()
    w["ln"] = w["OBJECT_ID"].map(launch_no)

    payloads_per_band = np.zeros(NB)
    stages_band = np.zeros((NB, NB))
    ids_by_band = {b: [] for b in range(NB)}      # stage ids by DESTINATION band
    n_launch = 0

    for _, grp in w.groupby("ln"):
        pay = grp[grp["OBJECT_TYPE"] == "PAYLOAD"]
        rb = grp[grp["OBJECT_TYPE"] == "ROCKET BODY"]
        pbs = [band_of(x) for x in pay["alt"] if band_of(x) >= 0]
        if not pbs:
            continue
        pb = max(set(pbs), key=pbs.count)          # launch's payload band = mode
        payloads_per_band[pb] += len(pbs)
        n_launch += 1
        for alt, nid in zip(rb["alt"], rb["NORAD_CAT_ID"]):
            sb = band_of(alt)
            if sb >= 0:
                stages_band[pb, sb] += 1
                if not np.isnan(nid):
                    ids_by_band[sb].append(int(nid))

    all_ids = sorted({i for v in ids_by_band.values() for i in v})

    R = np.zeros((NB, NB))
    suppressed = []
    for b in range(NB):
        if payloads_per_band[b] <= 0:
            continue
        if payloads_per_band[b] < a.min_payloads:
            suppressed.append(b)                   # leave row at zero
            continue
        R[b] = stages_band[b] / payloads_per_band[b]
    persat_total = R.sum(axis=1)

    # ---- stage properties, per destination band ----
    di = pd.read_csv(a.discos, low_memory=False)
    di["satno"] = pd.to_numeric(di["satno"], errors="coerce")
    di = di.dropna(subset=["satno"]).set_index("satno")

    area_law = fit_area_law(di)
    pooled = clean_stage_table(di, all_ids, a.exclude_kick, area_law)
    if len(pooled) == 0:
        sys.exit("FATAL: DISCOS join produced zero usable stages -- check satno dtype "
                 "and column names. (No hardcoded fallback: it would inject 1960s "
                 "Soviet stages into a 2020s sim.)")
    pooled_mass, pooled_radius = mass_radius(pooled)

    stage_mass, stage_radius, band_n, band_imp, used_pooled = [], [], [], [], []
    for b in range(NB):
        s = clean_stage_table(di, ids_by_band[b], a.exclude_kick, area_law)
        band_n.append(len(s))
        band_imp.append(int(s["imputed"].sum()) if len(s) else 0)
        if len(s) < a.min_stages:
            m, r, flag = pooled_mass, pooled_radius, True
        else:
            m, r = mass_radius(s)
            flag = False
        stage_mass.append(m)
        stage_radius.append(r)
        used_pooled.append(flag)

    # ---- report ----
    c0, c1, r2, nfit = area_law
    print(f"launches in window {a.win[0]:.0f}-{a.win[1]:.0f}: {n_launch}")
    print(f"surviving stages: {len(all_ids)} | {len(pooled)} usable after cleaning"
          f"{' (kick stages excluded)' if a.exclude_kick else ''}")
    print(f"area law: log A = {c0:.3f} + {c1:.3f} log m (R2={r2:.2f}, n={nfit}); "
          f"xSect imputed for {int(pooled['imputed'].sum())}/{len(pooled)}")
    print(f"pooled stage: {pooled_mass:.0f} kg, r_eff {pooled_radius:.2f} m\n")

    print("per-satellite stage injection by payload band:")
    for b in range(NB):
        flag = f"  [SUPPRESSED: <{a.min_payloads} payloads]" if b in suppressed else ""
        print(f"  B{b+1} {BANDS[b][0]:>4}-{BANDS[b][1]:<4}: {int(payloads_per_band[b]):>6}"
              f" sats | {persat_total[b]:.5f} stages/sat{flag}")

    print("\nR_persat[b -> b']  (rows = payload band, cols = stage band):")
    print("        " + "".join(f"  B{j+1:<5}" for j in range(NB)))
    for b in range(NB):
        print(f"  B{b+1:<5}" + "".join(f"{R[b, j]:7.4f}" for j in range(NB)))

    print("\nstage properties by DESTINATION band:")
    print(f"  {'band':<14}{'n':>4} {'imp':>4}  {'mass_kg':>8}  {'r_eff_m':>7}   source")
    for b in range(NB):
        print(f"  B{b+1} {BANDS[b][0]:>4}-{BANDS[b][1]:<5}{band_n[b]:>4} {band_imp[b]:>4}  "
              f"{stage_mass[b]:>8.0f}  {stage_radius[b]:>7.2f}   "
              f"{'pooled' if used_pooled[b] else 'band'}")

    out = dict(
        bands=BANDS,
        R_persat=R.tolist(),
        stages_per_satellite=persat_total.tolist(),
        payloads_per_band=payloads_per_band.tolist(),
        # per-DESTINATION-band vectors: index by stage band b' (the R_persat column)
        stage_mass_kg=stage_mass,
        stage_radius_m=stage_radius,
        stage_n_per_band=band_n,
        stage_n_imputed=band_imp,
        stage_band_used_pooled=used_pooled,
        stage_mass_kg_pooled=pooled_mass,
        stage_radius_m_pooled=pooled_radius,
        area_law=dict(c0=c0, c1=c1, r2=r2, n=nfit),
        window=list(a.win),
        n_launch=n_launch,
        suppressed_payload_bands=suppressed,
        kick_stages_excluded=bool(a.exclude_kick),
        objectclass=5, controlled=0,
        note=("R_persat[b][bp] = expected surviving rocket-body stages deposited in "
              "band bp per satellite launched to band b. Apply by mapping each pyssem "
              "shell to its band and spreading each band->band rate uniformly across "
              "the destination band's shells. stage_mass_kg / stage_radius_m are "
              "per-band VECTORS indexed by destination band bp."),
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()