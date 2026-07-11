"""
build_x0.py

Build a pyssem initial population (x0) from a real catalog snapshot:
  - Space-Track GP / OMM  (current on-orbit elements)      -> which objects, and where
  - DISCOS                (mass, radius)                    -> physical size for mass-binning
  - CelesTrak/Space-Track SATCAT (OPS_STATUS_CODE)         -> live payload vs. derelict

Returns a DataFrame shaped (n_shells x species) of object counts, ready to assign
to scen_properties.x0 (that is exactly what pyssem's SEP_traffic_model produces).

Classification is deliberately explicit and editable (see the CONFIG block): with a
single snapshot you cannot detect station-keeping, so live-vs-dead is decided from
operational status flags plus age, not from a TLE time series.
"""

import numpy as np
import pandas as pd


# ----------------------------- CONFIG (edit me) -----------------------------
# Which OPS_STATUS_CODE values count as a LIVE (operational) payload.
# CelesTrak: + operational, P partially, B backup/standby, S spare, X extended,
#            - nonoperational, D decayed, ? unknown.
ACTIVE_STATUS_CODES = {"+", "P", "B", "X", "S"}

# For payloads with missing/unknown status, fall back to age: a payload younger
# than this many years is assumed still active, older is assumed derelict.
ACTIVE_MAX_AGE_YEARS = 25
REFERENCE_YEAR = 2026  # "now" for the age calc (the GP snapshot epoch year)

# GP OBJECT_TYPE -> broad physical category
PAYLOAD_TYPES = {"PAYLOAD"}
ROCKETBODY_TYPES = {"ROCKET BODY"}
DEBRIS_TYPES = {"DEBRIS", "UNKNOWN", "TBA"}
# ---------------------------------------------------------------------------


def build_x0(
    scen_properties,
    gp_path="2026.csv",
    satcat_path="satcat.csv",
    discos_path="discos_cache.csv",
    verbose=True,
):
    min_alt = scen_properties.min_altitude
    max_alt = scen_properties.max_altitude
    n_shells = scen_properties.n_shells
    species_names = list(scen_properties.species_names)

    # Active vs. sink species (name, mass) from the configured scenario.
    active_species, sink_species = [], []
    for group in scen_properties.species.values():
        for s in group:
            entry = (s.sym_name, float(s.mass))
            (active_species if getattr(s, "active", False) else sink_species).append(entry)
    if not active_species:
        raise ValueError("No active species found in the scenario.")
    if not sink_species:
        raise ValueError("No debris/derelict (sink) species found in the scenario.")

    # -------------------- load + join the three sources --------------------
    gp = pd.read_csv(gp_path, low_memory=False)
    gp["NORAD_CAT_ID"] = pd.to_numeric(gp["NORAD_CAT_ID"], errors="coerce")
    gp = gp.dropna(subset=["NORAD_CAT_ID"])
    gp["NORAD_CAT_ID"] = gp["NORAD_CAT_ID"].astype(int)

    # Mean altitude [km]. APOAPSIS/PERIAPSIS in this feed are already altitudes.
    gp["alt"] = (gp["APOAPSIS"] + gp["PERIAPSIS"]) / 2.0
    gp["ecc"] = pd.to_numeric(gp["ECCENTRICITY"], errors="coerce")

    # Drop anything already decayed, and restrict to the shell range.
    gp = gp[gp["DECAY_DATE"].isna()]
    gp = gp[(gp["alt"] >= min_alt) & (gp["alt"] <= max_alt)].copy()

    # DISCOS mass/radius by NORAD id.
    di = pd.read_csv(discos_path, low_memory=False)
    di["satno"] = pd.to_numeric(di["satno"], errors="coerce")
    di = di.dropna(subset=["satno"])
    di["satno"] = di["satno"].astype(int)
    di = di[["satno", "mass_kg", "radius_m"]].drop_duplicates("satno")
    gp = gp.merge(di, left_on="NORAD_CAT_ID", right_on="satno", how="left")

    # SATCAT operational status by NORAD id.
    sc = pd.read_csv(satcat_path, low_memory=False)
    sc["NORAD_CAT_ID"] = pd.to_numeric(sc["NORAD_CAT_ID"], errors="coerce")
    sc = sc.dropna(subset=["NORAD_CAT_ID"])
    sc["NORAD_CAT_ID"] = sc["NORAD_CAT_ID"].astype(int)
    sc = sc[["NORAD_CAT_ID", "OPS_STATUS_CODE"]].drop_duplicates("NORAD_CAT_ID")
    gp = gp.merge(sc, on="NORAD_CAT_ID", how="left")

    # -------------------- categorize each object --------------------
    otype = gp["OBJECT_TYPE"].astype(str).str.upper().str.strip()
    category = np.where(otype.isin(PAYLOAD_TYPES), "payload",
               np.where(otype.isin(ROCKETBODY_TYPES), "rocket_body", "debris"))
    gp["category"] = category

    # Fill missing mass with the median mass of that category (data-driven fallback).
    gp["mass_kg"] = pd.to_numeric(gp["mass_kg"], errors="coerce")
    for cat in ("payload", "rocket_body", "debris"):
        m = gp.loc[gp["category"] == cat, "mass_kg"]
        fill = m.median() if m.notna().any() else 1.0
        gp.loc[(gp["category"] == cat) & (gp["mass_kg"].isna()), "mass_kg"] = fill

    # Live payload vs. derelict.
    launch_year = pd.to_datetime(gp["LAUNCH_DATE"], errors="coerce").dt.year
    age = REFERENCE_YEAR - launch_year
    status = gp["OPS_STATUS_CODE"].astype(str).str.strip()

    is_payload = gp["category"] == "payload"
    status_alive = status.isin(ACTIVE_STATUS_CODES)
    status_known = status.isin(ACTIVE_STATUS_CODES | {"-", "D"})
    age_alive = age <= ACTIVE_MAX_AGE_YEARS
    # alive if status says so, or (status unknown AND young)
    payload_alive = is_payload & (status_alive | (~status_known & age_alive))

    # Target pool: active satellites vs. everything that is a sink.
    gp["target_pool"] = np.where(payload_alive, "active", "sink")

    # -------------------- map to a concrete species by nearest mass --------------------
    active_names = np.array([n for n, _ in active_species])
    active_masses = np.array([m for _, m in active_species])
    sink_names = np.array([n for n, _ in sink_species])
    sink_masses = np.array([m for _, m in sink_species])

    def nearest(mass, names, masses):
        return names[int(np.argmin(np.abs(masses - mass)))]

    gp["species"] = [
        nearest(m, active_names, active_masses) if pool == "active"
        else nearest(m, sink_names, sink_masses)
        for m, pool in zip(gp["mass_kg"].to_numpy(), gp["target_pool"].to_numpy())
    ]

    # -------------------- altitude -> shell, then pivot to counts --------------------
    edges = np.linspace(min_alt, max_alt, n_shells + 1)
    gp["alt_bin"] = np.clip(np.digitize(gp["alt"], edges) - 1, 0, n_shells - 1)

    counts = gp.pivot_table(index="alt_bin", columns="species", aggfunc="size", fill_value=0)
    x0 = pd.DataFrame(0.0, index=range(n_shells), columns=species_names)
    for sp in counts.columns:
        if sp in x0.columns:
            x0.loc[counts.index, sp] = counts[sp].values

    if verbose:
        print(f"Objects placed into x0: {int(x0.values.sum()):,} "
              f"(from {len(gp):,} in-range catalog objects)")
        print("By category (in range):", gp["category"].value_counts().to_dict())
        print("Live payloads:", int(payload_alive.sum()),
              "| derelict payloads:", int((is_payload & ~payload_alive).sum()))
        print("Totals by species:")
        for sp in species_names:
            print(f"  {sp:>10}: {int(x0[sp].sum()):,}")
        missing = gp.loc[gp["category"] == "payload", "mass_kg"].isna().sum()
        print(f"(payloads still missing mass after fallback: {missing})")

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

    x0 = build_x0(_Scen(), gp_path="2026.csv", satcat_path="satcat.csv",
                  discos_path="discos_cache.csv")
    print("\nx0 shape:", x0.shape)
    print(x0.head(8))
