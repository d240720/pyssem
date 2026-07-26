#!/usr/bin/env python3
"""
build_rb_injection.py -- per-satellite rocket-body injection from the catalogs
(GP snapshot + satcat.csv + discos_cache.csv).

    R_persat[b][b'] = surviving stages (launches targeting b, landing in b')
                      / payloads launched to b

Launch pairing is exact via the designator (OBJECT_ID 'YYYY-NNN'). Stage
mass/radius are per DESTINATION band: ~4 t at 700-1000 km vs ~2.5 t at
400-600 km, and the high band is where debris persists.

Caveats:
  * Stage bands are SNAPSHOT positions, not deposit orbits: survivors carry up
    to the window's decay, so low-band columns are biased low in altitude and
    count (pyssem re-applies drag; treat rates as residue floors).
  * The GP fetch is DECAY_DATE=null, so decayed payloads are counted from
    satcat.csv by designator. Its MEAN_MOTION>3 cut also means GTO-period
    stages never reach the snapshot (the eccentricity counter sees the rest).
  * Launches with no LEO payloads are excluded, their stages with them.
  * ~80-130 stages total; normalization is launch-batching-sensitive -- run
    both windows, report the spread.

USAGE: build_rb_injection.py [--gp 2026.csv] [--discos discos_cache.csv]
    [--satcat satcat.csv] [--win 2020 2025] [--min-alt 200] [--max-alt 2000]
    [--min-payloads 25] [--min-stages 3] [--no-exclude-kick-stages] [--out ...]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

BANDS = [(200, 300), (300, 400), (400, 500), (500, 600),
         (600, 700), (700, 1000), (1000, 2000)]
NB = len(BANDS)

# ~50 kg kick stages cataloged ROCKET BODY: counted in R_persat, excluded from
# the mass/radius stats (the SBM's RB branch isn't calibrated on them). The
# report prints the counted-but-excluded number so the bias stays visible.
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


def decayed_payloads_per_launch(path, win):
    """{launch designator: decayed payload count} from SATCAT (they're absent
    from the GP snapshot by construction). Tolerates both schemas (CelesTrak
    OBJECT_ID/DECAY_DATE, Space-Track INTLDES/DECAY); the window filter uses
    the designator's own year."""
    sc = pd.read_csv(path, low_memory=False)
    des = sc["OBJECT_ID"] if "OBJECT_ID" in sc else sc["INTLDES"]
    typ = sc["OBJECT_TYPE"].astype("string").str.strip().str.upper()
    dcol = next((c for c in ("DECAY_DATE", "DECAY") if c in sc), None)
    decayed = (sc[dcol].notna() & sc[dcol].astype(str).str.strip().ne("")) if dcol \
        else sc["OPS_STATUS_CODE"].astype(str).str.strip().eq("D")
    ln = des[typ.str.startswith("PAY").fillna(False) & decayed].map(launch_no)
    yr = pd.to_numeric(ln.str.slice(0, 4), errors="coerce")
    return ln[yr.between(win[0], win[1])].value_counts().to_dict()


def fit_area_law(di):
    """log(xSect) = c0 + c1*log(mass) over all cached rocket bodies. Gaps are
    structured (the L-15 class, dominant at 700-1000 km, has mass but no xSect;
    dropping it biases B6 mass ~40% low), so impute. exp() of the fit gives the
    MEDIAN; the smearing factor exp(s^2/2) corrects to the mean."""
    rb = di[di["objectClass"].fillna("").str.contains("Rocket Body", na=False)]
    b = rb[rb["mass_kg"].between(*MASS_BOUNDS)
           & rb["xSectAvg_m2"].between(*XSECT_BOUNDS)]
    x, y = np.log(b["mass_kg"].to_numpy()), np.log(b["xSectAvg_m2"].to_numpy())
    c1, c0 = np.polyfit(x, y, 1)
    resid = y - (c0 + c1 * x)
    r2 = 1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    smear = float(np.exp(resid.var(ddof=2) / 2.0))
    return c0, c1, smear, r2, len(b)


def clean_stage_table(di, ids, exclude_kick, area_law):
    """DISCOS rows for `ids`: mass required, xSect imputed where absent.
    Returns (table, n_excluded): stages counted in R_persat but dropped here
    (mass bounds / kick pattern) -- each gets injected at the band mean, so
    n_excluded bounds that overcount."""
    c0, c1, smear, _, _ = area_law
    s = di.reindex(sorted(set(int(i) for i in ids))).copy()
    s = s.dropna(subset=["mass_kg"])
    n0 = len(s)
    s = s[s["mass_kg"].between(*MASS_BOUNDS)]
    if exclude_kick and len(s):
        pat = "|".join(KICK_STAGE_PATTERNS)
        s = s[~s["name"].fillna("").str.contains(pat, case=False, regex=True)]
    n_excl = n0 - len(s)
    s.loc[~s["xSectAvg_m2"].between(*XSECT_BOUNDS), "xSectAvg_m2"] = np.nan
    s["imputed"] = s["xSectAvg_m2"].isna()
    s.loc[s["imputed"], "xSectAvg_m2"] = smear * np.exp(
        c0 + c1 * np.log(s.loc[s["imputed"], "mass_kg"]))
    return s, n_excl


def mass_radius(s):
    """Mean mass (conserves the SBM's injected mass budget) and area-equivalent
    radius sqrt(mean(xSect)/pi). Median mass is unusable on this multimodal
    55 kg - 6 t population."""
    return (float(s["mass_kg"].mean()),
            float(np.sqrt(s["xSectAvg_m2"].mean() / np.pi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", default="2026.csv")
    ap.add_argument("--discos", default="discos_cache.csv")
    ap.add_argument("--satcat", default="satcat.csv",
                    help="SATCAT for decayed-payload counts; pass '' to skip "
                         "(denominator then reverts to surviving payloads)")
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

    if a.satcat:
        if not os.path.exists(a.satcat):
            sys.exit(f"FATAL: {a.satcat} not found (decayed payloads only exist "
                     "there). Fetch it, or pass --satcat '' to explicitly accept "
                     "a surviving-payloads-only denominator.")
        dead = decayed_payloads_per_launch(a.satcat, a.win)
    else:
        dead = {}
        print("WARNING: --satcat '' -- denominator counts surviving payloads only")

    cat = gp[(gp["ly"] >= a.win[0]) & (gp["ly"] <= a.win[1])].copy()
    cat["ln"] = cat["OBJECT_ID"].map(launch_no)
    cat["alive"] = cat["DECAY_DATE"].isna()
    cat["in_rng"] = (cat["alt"] >= a.min_alt) & (cat["alt"] < a.max_alt)

    payloads_per_band = np.zeros(NB)
    stages_band = np.zeros((NB, NB))
    ids_by_band = {b: [] for b in range(NB)}      # stage ids by DESTINATION band
    n_launch = 0
    n_ecc = 0                                     # surviving stages with apo-peri > 100 km
    n_dead = 0                                    # decayed payloads counted via SATCAT

    for ln, grp in cat.groupby("ln"):
        pay = grp[grp["OBJECT_TYPE"] == "PAYLOAD"]
        rb = grp[(grp["OBJECT_TYPE"] == "ROCKET BODY")
                 & grp["alive"] & grp["in_rng"]]
        pbs = [band_of(x) for x in pay.loc[pay["alive"] & pay["in_rng"], "alt"]]
        pbs = [b for b in pbs if b >= 0]
        if not pbs:
            continue
        pb = max(set(pbs), key=lambda b: (pbs.count(b), -b))   # mode; ties -> lowest band
        # denominator = payloads LAUNCHED: live in-range ones banded above plus
        # this launch's SATCAT-decayed ones (dropping those inflates exactly the
        # low-band rates the docstring calls floors).
        n_d = dead.get(ln, 0)
        n_dead += n_d
        payloads_per_band[pb] += len(pbs) + n_d
        n_launch += 1
        n_ecc += int((rb["APOAPSIS"] - rb["PERIAPSIS"] > 100).sum())
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
    pooled, pooled_excl = clean_stage_table(di, all_ids, a.exclude_kick, area_law)
    if len(pooled) == 0:
        sys.exit("FATAL: DISCOS join produced zero usable stages -- check satno dtype "
                 "and column names. (No hardcoded fallback: it would inject 1960s "
                 "Soviet stages into a 2020s sim.)")
    pooled_mass, pooled_radius = mass_radius(pooled)

    stage_mass, stage_radius = [], []
    band_n, band_imp, band_excl, used_pooled = [], [], [], []
    for b in range(NB):
        s, n_excl = clean_stage_table(di, ids_by_band[b], a.exclude_kick, area_law)
        band_n.append(len(s))
        band_imp.append(int(s["imputed"].sum()) if len(s) else 0)
        band_excl.append(n_excl)
        if len(s) < a.min_stages:
            m, r, flag = pooled_mass, pooled_radius, True
        else:
            m, r = mass_radius(s)
            flag = False
        stage_mass.append(m)
        stage_radius.append(r)
        used_pooled.append(flag)

    # ---- report ----
    c0, c1, smear, r2, nfit = area_law
    print(f"launches in window {a.win[0]:.0f}-{a.win[1]:.0f}: {n_launch}")
    print(f"decayed payloads in denominator (via SATCAT): {n_dead}")
    print(f"surviving stages: {len(all_ids)} | {len(pooled)} usable after cleaning"
          f"{' (kick stages excluded)' if a.exclude_kick else ''}"
          f" | {pooled_excl} counted in R_persat but excluded from stats")
    print(f"eccentric stages (apo-peri > 100 km, banded by mean alt): {n_ecc}")
    print(f"area law: log A = {c0:.3f} + {c1:.3f} log m (R2={r2:.2f}, n={nfit}, "
          f"smear={smear:.2f}); xSect imputed for "
          f"{int(pooled['imputed'].sum())}/{len(pooled)}")
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
    print(f"  {'band':<14}{'n':>4} {'imp':>4} {'excl':>5}  {'mass_kg':>8}  {'r_eff_m':>7}   source")
    for b in range(NB):
        print(f"  B{b+1} {BANDS[b][0]:>4}-{BANDS[b][1]:<5}{band_n[b]:>4} {band_imp[b]:>4} "
              f"{band_excl[b]:>5}  {stage_mass[b]:>8.0f}  {stage_radius[b]:>7.2f}   "
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
        stage_n_excluded=band_excl,        # counted in R_persat, absent from mass stats
        stage_n_excluded_pooled=pooled_excl,
        stage_band_used_pooled=used_pooled,
        stage_mass_kg_pooled=pooled_mass,
        stage_radius_m_pooled=pooled_radius,
        n_eccentric_stages=n_ecc,
        n_decayed_payloads=n_dead,
        area_law=dict(c0=c0, c1=c1, smear=smear, r2=r2, n=nfit),
        window=list(a.win),
        n_launch=n_launch,
        suppressed_payload_bands=suppressed,
        kick_stages_excluded=bool(a.exclude_kick),
        objectclass=5, controlled=0,
        note=("R_persat[b][bp] = expected surviving rocket-body stages deposited in "
              "band bp per satellite launched to band b. Denominator counts launched "
              "payloads (decayed included, via SATCAT -- the GP snapshot cannot "
              "contain them); numerator counts snapshot survivors at "
              "their snapshot altitudes. Apply by mapping each pyssem shell to its "
              "band and spreading each band->band rate uniformly across the "
              "destination band's shells. stage_mass_kg / stage_radius_m are "
              "per-band VECTORS indexed by destination band bp."),
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()