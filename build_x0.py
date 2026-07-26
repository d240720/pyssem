"""
build_x0.py -- pyssem initial population (n_shells x species) from a catalog
snapshot: GP (which objects, where), DISCOS (mass), SATCAT (live vs derelict).

Returns a DataFrame shaped (n_shells x species) of counts, ready to assign to
scen_properties.x0 (exactly what pyssem's SEP_traffic_model produces).

Live-vs-dead comes from OPS_STATUS_CODE plus an age fallback: a single snapshot
cannot detect station-keeping. Constants live in CONFIG.

Known choices:
  * Debris is deliberately absent from the DISCOS pull, so ALL debris takes
    DEBRIS_DEFAULT_MASS_KG -- which is what selects its sink species. Payloads/
    rocket bodies missing mass take their category median; a category with NO
    masses at all means a broken cache and is a hard error.
  * Species matching is nearest mass in LINEAR space: with sinks at 0.5 and
    200 kg the crossover is ~100 kg, so mid-mass derelicts map light. Log-space
    (argmin |log m - log m_s|) would move it to ~10 kg; switch in `nearest`.
  * Altitude range is inclusive of max_alt here (top-edge objects clip into the
    last shell); build_rb_injection.py uses < max_alt.
"""

import numpy as np
import pandas as pd


# ----------------------------- CONFIG (edit me) -----------------------------
# CelesTrak OPS_STATUS_CODE: + operational, P partial, B backup, S spare,
# X extended, - nonoperational, D decayed, ? unknown.
ACTIVE_STATUS_CODES = {"+", "P", "B", "X", "S"}
KNOWN_STATUS_CODES = ACTIVE_STATUS_CODES | {"-", "D"}
ACTIVE_MAX_AGE_YEARS = 25   # unknown status: younger => assumed active
REFERENCE_YEAR = 2026       # "now" for the age calc (GP snapshot epoch year)

PAYLOAD_TYPES = {"PAYLOAD"}
ROCKETBODY_TYPES = {"ROCKET BODY"}   # everything else (DEBRIS/UNKNOWN/TBA) -> debris
# Debris is intentionally excluded from the DISCOS pull (see export_intact_satnos),
# so every debris object takes this mass -- it decides which sink species the
# whole debris population maps to. An explicit constant, not a silent fallback.
DEBRIS_DEFAULT_MASS_KG = 1.0
# ---------------------------------------------------------------------------


def _int_ids(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[col]).astype({col: int})


def build_x0(scen_properties, gp_path="2026.csv", satcat_path="satcat.csv",
             discos_path="discos_cache.csv", verbose=True):
    min_alt, max_alt = scen_properties.min_altitude, scen_properties.max_altitude
    n_shells = scen_properties.n_shells
    species_names = list(scen_properties.species_names)

    # Active vs. sink species (name, mass) from the configured scenario.
    active, sink = [], []
    for group in scen_properties.species.values():
        for s in group:
            (active if getattr(s, "active", False) else sink).append(
                (s.sym_name, float(s.mass)))
    if not active or not sink:
        raise ValueError("Scenario needs at least one active and one sink species.")

    # -------------------- load + join the three sources --------------------
    gp = _int_ids(pd.read_csv(gp_path, low_memory=False), "NORAD_CAT_ID")
    gp["alt"] = (gp["APOAPSIS"] + gp["PERIAPSIS"]) / 2.0   # already altitudes [km]
    gp = gp[gp["DECAY_DATE"].isna()
            & (gp["alt"] >= min_alt) & (gp["alt"] <= max_alt)].copy()

    # DISCOS cache is append-only; LAST row per satno wins (puller contract),
    # so a --retry-misses upgrade supersedes the earlier miss row.
    di = _int_ids(pd.read_csv(discos_path, low_memory=False), "satno")
    di = di[["satno", "mass_kg"]].drop_duplicates("satno", keep="last")
    gp = gp.merge(di, left_on="NORAD_CAT_ID", right_on="satno", how="left")

    sc = _int_ids(pd.read_csv(satcat_path, low_memory=False), "NORAD_CAT_ID")
    sc = sc[["NORAD_CAT_ID", "OPS_STATUS_CODE"]].drop_duplicates("NORAD_CAT_ID")
    gp = gp.merge(sc, on="NORAD_CAT_ID", how="left")

    # -------------------- categorize --------------------
    otype = gp["OBJECT_TYPE"].astype(str).str.upper().str.strip()
    gp["category"] = np.where(otype.isin(PAYLOAD_TYPES), "payload",
                     np.where(otype.isin(ROCKETBODY_TYPES), "rocket_body", "debris"))

    # Mass fill: category median for payloads/RBs, the configured constant for
    # debris. A payload/RB category with zero masses means a broken DISCOS join.
    gp["mass_kg"] = pd.to_numeric(gp["mass_kg"], errors="coerce")
    n_fill = {}
    for cat in ("payload", "rocket_body", "debris"):
        need = (gp["category"] == cat) & gp["mass_kg"].isna()
        have = gp.loc[gp["category"] == cat, "mass_kg"].dropna()
        if len(have):
            fill = float(have.median())
        elif cat == "debris":
            fill = DEBRIS_DEFAULT_MASS_KG
        else:
            raise ValueError(f"No DISCOS mass for any {cat} -- broken cache/join?")
        gp.loc[need, "mass_kg"] = fill
        n_fill[cat] = (int(need.sum()), fill)

    # Live payload vs. derelict: status says so, or status unknown AND young.
    age = REFERENCE_YEAR - pd.to_datetime(gp["LAUNCH_DATE"], errors="coerce").dt.year
    status = gp["OPS_STATUS_CODE"].astype(str).str.strip()
    is_payload = gp["category"] == "payload"
    payload_alive = is_payload & (status.isin(ACTIVE_STATUS_CODES)
                                  | (~status.isin(KNOWN_STATUS_CODES)
                                     & (age <= ACTIVE_MAX_AGE_YEARS)))
    gp["pool"] = np.where(payload_alive, "active", "sink")

    # -------------------- nearest-mass species, altitude shell, pivot --------------------
    def nearest(names, masses, m):
        # linear-space nearest; see docstring for the log-space alternative
        return np.array(names)[np.abs(np.array(masses)[None] - m[:, None]).argmin(1)]

    m = gp["mass_kg"].to_numpy()
    gp["species"] = np.where(gp["pool"] == "active",
                             nearest(*zip(*active), m), nearest(*zip(*sink), m))

    edges = np.linspace(min_alt, max_alt, n_shells + 1)
    gp["alt_bin"] = np.clip(np.digitize(gp["alt"], edges) - 1, 0, n_shells - 1)
    x0 = (gp.pivot_table(index="alt_bin", columns="species", aggfunc="size", fill_value=0)
            .reindex(index=range(n_shells), columns=species_names, fill_value=0)
            .astype(float))
    x0.index = range(n_shells)

    if verbose:
        print(f"Objects placed into x0: {int(x0.values.sum()):,}")
        print("By category:", gp["category"].value_counts().to_dict())
        print("Live payloads:", int(payload_alive.sum()),
              "| derelict payloads:", int((is_payload & ~payload_alive).sum()))
        for cat, (n, f) in n_fill.items():
            if n:
                print(f"  mass fallback: {n:,} {cat} objects filled at {f:.1f} kg")
        print("Totals by species:")
        for sp in species_names:
            print(f"  {sp:>10}: {int(x0[sp].sum()):,}")

    return x0


if __name__ == "__main__":
    # Standalone smoke test with a stub scenario (no pyssem import needed).
    class _S:
        def __init__(self, name, mass, active):
            self.sym_name, self.mass, self.active = name, mass, active

    class _Scen:
        min_altitude, max_altitude, n_shells = 200, 1400, 24
        species = {"active": [_S("S", 200, True)],
                   "debris": [_S("N", 0.5, False), _S("N_200kg", 200, False)]}
        species_names = ["S", "N", "N_200kg"]

    x0 = build_x0(_Scen())
    print("\nx0 shape:", x0.shape)
    print(x0.head(8))