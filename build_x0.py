"""
build_x0.py -- pyssem initial population (n_shells x species) from a catalog
snapshot: GP (which objects, where), DISCOS (mass), SATCAT (live vs derelict).

Returns a DataFrame shaped (n_shells x species) of counts, ready to assign to
scen_properties.x0 (exactly what pyssem's SEP_traffic_model produces).

Live-vs-dead comes from OPS_STATUS_CODE plus an age fallback: a single snapshot
cannot detect station-keeping. Constants live in CONFIG.

Known choices:
  * Debris is deliberately absent from the DISCOS pull, so debris WITHOUT a
    DISCOS mass -- normally all of it -- takes DEBRIS_DEFAULT_MASS_KG, which is
    what selects its sink species. The rare debris row that DOES carry a DISCOS
    mass (an object reclassified since the cache was built) keeps its real
    mass and is counted in the report, so classification drift between catalog
    snapshots stays visible instead of silently moving the fill value.
  * Payloads/rocket bodies missing mass take their category median; a category
    with NO masses at all means a broken cache and is a hard error.
  * Species matching is nearest mass in LINEAR space: with sinks at 0.5 and
    200 kg the crossover is ~100 kg, so mid-mass derelicts map light. Log-space
    (argmin |log m - log m_s|) would move it to ~10 kg; switch in `nearest`.
  * Altitude range is inclusive of max_alt here (top-edge objects clip into the
    last shell); build_rb_injection.py uses < max_alt.
  * ecc_policy handles eccentric orbits (45% of debris exceeds 100 km apo-peri
    against 50 km shells): "mean" pins each object at (apo+peri)/2 (legacy),
    "perigee" pins at perigee (decay-conservative), "spread" distributes each
    object across shells by Keplerian dwell time (fractional counts; the
    physically consistent choice for a shell model, and time spent outside the
    altitude range is dropped rather than renormalized).
  * No NORAD-ID range filter: analyst objects (traditionally 80000+) carry
    UNKNOWN/TBA types, fall into the debris category at the default mass, and
    are counted -- they are real tracked objects and belong in x0.
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
# Fill for debris rows with no DISCOS mass (normally all of them; see docstring).
DEBRIS_DEFAULT_MASS_KG = 1.0

GP_COLS = ["NORAD_CAT_ID", "OBJECT_TYPE", "LAUNCH_DATE", "DECAY_DATE",
           "APOAPSIS", "PERIAPSIS"]

RE_KM = 6378.137
# ---------------------------------------------------------------------------


def _dwell_weights(peri, apo, edges, n_shells):
    """Fraction of the Keplerian period each object spends in each altitude
    shell. For eccentric anomaly E, r = a(1 - e cos E) and time-from-perigee
    is proportional to M = E - e sin E; the fraction of the period with r in
    [r1, r2] is (M(E2) - M(E1)) / pi over the ascending half-orbit. Rows sum
    to the in-range fraction of the period (time spent outside [min_alt,
    max_alt] is correctly dropped, not renormalized)."""
    a = RE_KM + (apo + peri) / 2.0
    e = np.clip((apo - peri) / (2.0 * a), 0.0, None)
    r_edges = RE_KM + edges
    with np.errstate(divide="ignore", invalid="ignore"):
        cosE = (1.0 - r_edges[None, :] / a[:, None]) / e[:, None]
    E = np.arccos(np.clip(cosE, -1.0, 1.0))
    M = E - e[:, None] * np.sin(E)
    W = np.diff(M, axis=1) / np.pi
    circ = e < 1e-8                        # circular: delta at its one altitude
    if circ.any():
        W[circ] = 0.0
        idx = np.clip(np.digitize((apo + peri)[circ] / 2.0, edges) - 1,
                      0, n_shells - 1)
        W[np.where(circ)[0], idx] = 1.0
    return W


def _int_ids(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[col]).astype({col: int})


def build_x0(scen_properties, gp_path="2026.csv", satcat_path="satcat.csv",
             discos_path="discos_cache.csv", ecc_policy="mean", verbose=True):
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
    unlisted = {n for n, _ in active + sink} - set(species_names)
    if unlisted:
        raise ValueError(f"Species {sorted(unlisted)} exist in the scenario but "
                         f"not in species_names -- their objects would be "
                         f"silently dropped from x0.")

    # -------------------- load + join the three sources --------------------
    gp = _int_ids(pd.read_csv(gp_path, usecols=GP_COLS), "NORAD_CAT_ID")
    gp["alt"] = (gp["APOAPSIS"] + gp["PERIAPSIS"]) / 2.0   # already altitudes [km]
    gp = gp[gp["DECAY_DATE"].isna()
            & (gp["alt"] >= min_alt) & (gp["alt"] <= max_alt)].copy()

    # DISCOS cache is append-only; LAST row per satno wins (puller contract),
    # so a --retry-misses upgrade supersedes the earlier miss row.
    di = _int_ids(pd.read_csv(discos_path, usecols=["satno", "mass_kg"]), "satno")
    di = di.drop_duplicates("satno", keep="last")
    gp = gp.merge(di, left_on="NORAD_CAT_ID", right_on="satno", how="left")

    sc = pd.read_csv(satcat_path, low_memory=False)
    if "OPS_STATUS_CODE" not in sc.columns:
        raise ValueError(f"{satcat_path} has no OPS_STATUS_CODE column -- "
                         f"live/derelict classification needs the CelesTrak "
                         f"SATCAT schema (fetch_catalogs.py default).")
    sc = _int_ids(sc, "NORAD_CAT_ID")
    sc = sc[["NORAD_CAT_ID", "OPS_STATUS_CODE"]].drop_duplicates("NORAD_CAT_ID")
    gp = gp.merge(sc, on="NORAD_CAT_ID", how="left")

    # -------------------- categorize --------------------
    otype = gp["OBJECT_TYPE"].astype(str).str.upper().str.strip()
    gp["category"] = np.where(otype.isin(PAYLOAD_TYPES), "payload",
                     np.where(otype.isin(ROCKETBODY_TYPES), "rocket_body", "debris"))

    # Mass fill. Debris: ALWAYS the configured constant (a stray cached debris
    # mass must not become a median that silently retargets the whole debris
    # population's sink species); rows that carry a real mass keep it and are
    # counted. Payloads/RBs: category median; zero masses = broken DISCOS join.
    gp["mass_kg"] = pd.to_numeric(gp["mass_kg"], errors="coerce")
    n_fill = {}
    n_debris_with_mass = int(((gp["category"] == "debris")
                              & gp["mass_kg"].notna()).sum())
    for cat in ("payload", "rocket_body", "debris"):
        need = (gp["category"] == cat) & gp["mass_kg"].isna()
        if cat == "debris":
            fill = DEBRIS_DEFAULT_MASS_KG
        else:
            have = gp.loc[gp["category"] == cat, "mass_kg"].dropna()
            if not len(have):
                raise ValueError(f"No DISCOS mass for any {cat} -- broken cache/join?")
            fill = float(have.median())
        gp.loc[need, "mass_kg"] = fill
        n_fill[cat] = (int(need.sum()), fill)

    # Live payload vs. derelict: status says so, or status unknown AND young.
    age = REFERENCE_YEAR - pd.to_datetime(gp["LAUNCH_DATE"], errors="coerce").dt.year
    status = gp["OPS_STATUS_CODE"].astype(str).str.strip()
    is_payload = gp["category"] == "payload"
    alive_by_status = is_payload & status.isin(ACTIVE_STATUS_CODES)
    alive_by_age = (is_payload & ~status.isin(KNOWN_STATUS_CODES)
                    & (age <= ACTIVE_MAX_AGE_YEARS))
    payload_alive = alive_by_status | alive_by_age
    gp["pool"] = np.where(payload_alive, "active", "sink")

    # -------------------- nearest-mass species, altitude shell, pivot --------------------
    def nearest(names, masses, m):
        # linear-space nearest; see docstring for the log-space alternative
        return np.array(names)[np.abs(np.array(masses)[None] - m[:, None]).argmin(1)]

    m = gp["mass_kg"].to_numpy()
    gp["species"] = np.where(gp["pool"] == "active",
                             nearest(*zip(*active), m), nearest(*zip(*sink), m))

    edges = np.linspace(min_alt, max_alt, n_shells + 1)
    if ecc_policy in ("mean", "perigee"):
        alt = gp["alt"] if ecc_policy == "mean" else gp["PERIAPSIS"]
        gp["alt_bin"] = np.clip(np.digitize(alt, edges) - 1, 0, n_shells - 1)
        x0 = (gp.pivot_table(index="alt_bin", columns="species", aggfunc="size",
                             fill_value=0)
                .reindex(index=range(n_shells), columns=species_names, fill_value=0)
                .astype(float))
    elif ecc_policy == "spread":
        # Dwell-time fractional counts: each object contributes its per-shell
        # time fraction, so eccentric objects are distributed, not pinned.
        W = _dwell_weights(gp["PERIAPSIS"].to_numpy(float),
                           gp["APOAPSIS"].to_numpy(float), edges, n_shells)
        x0 = pd.DataFrame(0.0, index=range(n_shells), columns=species_names)
        sp_arr = gp["species"].to_numpy()
        for sp in species_names:
            x0[sp] = W[sp_arr == sp].sum(axis=0)
    else:
        raise ValueError(f"ecc_policy must be mean|perigee|spread, got {ecc_policy!r}")

    if verbose:
        print(f"Eccentricity policy: {ecc_policy}")
        placed = x0.values.sum()
        if ecc_policy == "spread":
            print(f"Objects placed into x0: {placed:,.1f} object-period "
                  f"fractions from {len(gp):,} objects "
                  f"({100 * placed / max(len(gp), 1):.1f}% of orbital time in range)")
        else:
            print(f"Objects placed into x0: {int(placed):,}")
        print("By category:", gp["category"].value_counts().to_dict())
        print("Live payloads:", int(payload_alive.sum()),
              f"({int(alive_by_status.sum())} by status + {int(alive_by_age.sum())} "
              f"by age fallback, unknown status & age <= {ACTIVE_MAX_AGE_YEARS} yr)",
              "| derelict payloads:", int((is_payload & ~payload_alive).sum()))
        n_ecc = int((gp["APOAPSIS"] - gp["PERIAPSIS"] > 100).sum())
        print(f"Eccentric objects (apo-peri > 100 km, binned at mean alt): {n_ecc} "
              f"({100 * n_ecc / max(len(gp), 1):.1f}% of x0)")
        for cat, (n, f) in n_fill.items():
            if n:
                print(f"  mass fallback: {n:,} {cat} objects filled at {f:.1f} kg")
        if n_debris_with_mass:
            print(f"  NOTE: {n_debris_with_mass} debris object(s) carry a DISCOS "
                  f"mass (reclassified since the cache was built?) and keep it")
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

    import sys
    policy = sys.argv[1] if len(sys.argv) > 1 else "mean"
    x0 = build_x0(_Scen(), ecc_policy=policy)
    print("\nx0 shape:", x0.shape)
    print(x0.head(8))