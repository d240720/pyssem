"""
targeted_launches.py

Set time-varying launches to specific orbits (altitude shells) in pyssem.

pyssem uses the same launch_func_lambda_fun mechanism as MOCAT-SSEM (MATLAB),
but the per-shell entry is a NUMPY ARRAY of launch rates sampled on
scen_properties.scen_times (units: satellites/year), or None for no launches.
pyssem interpolates each array with scipy.interp1d (cubic) during integration.

Inject the profiles AFTER model.configure_species() and BEFORE model.run_model()
(run_model calls build_model internally).
"""

import json
import types
import numpy as np
import pandas as pd

# Match pyssem's own import fallback (installed package vs running inside pyssem/).
try:
    from pyssem.model import Model
    from pyssem.utils.launch.launch import launch_func_lambda_fun
except ImportError:
    from model import Model
    from utils.launch.launch import launch_func_lambda_fun

from build_x0 import build_x0   # x0 builder (build_x0.py in the same folder)

# Catalog files used to seed the initial population (see build_x0.py).
GP_PATH = "2026.csv"            # Space-Track GP / OMM snapshot
SATCAT_PATH = "satcat.csv"     # CelesTrak / Space-Track SATCAT (OPS_STATUS_CODE)
DISCOS_PATH = "discos_cache.csv"  # DISCOS mass / radius


# ----------------------------------------------------------------------
# 1) Point this at the same config your smoketest uses.
# ----------------------------------------------------------------------
CONFIG_PATH = "config_rb.json"

with open(CONFIG_PATH) as f:
    cfg = json.load(f)
sp = cfg["scenario_properties"]

model = Model(
    start_date=sp["start_date"].split("T")[0],
    simulation_duration=sp["simulation_duration"],   # YEARS
    steps=sp["steps"],
    min_altitude=sp["min_altitude"],
    max_altitude=sp["max_altitude"],
    n_shells=sp["n_shells"],
    launch_function=sp["launch_function"],
    integrator=sp["integrator"],
    density_model=sp["density_model"],
    LC=sp["LC"],
    v_imp=sp.get("v_imp"),
    baseline=sp.get("baseline", False),              # keep False for lambda launches
    fragment_spreading=sp.get("fragment_spreading", False),
    parallel_processing=sp.get("parallel_processing", False),
    indicator_variables=sp.get("indicator_variables"),
)


# ----------------------------------------------------------------------
# Seed the initial population from the real catalog (bypasses the broken
# launch-file download). build_x0 reads the GP/DISCOS/SATCAT files and returns
# an (n_shells x species) count table. It runs DURING configure_species, after
# species_names is populated, so self.species / self.species_names are ready.
# ----------------------------------------------------------------------
def _seed_initial_population(self, baseline=False, launch_file=None):
    # Build the real starting population from the catalog snapshot.
    self.x0 = build_x0(
        self, gp_path=GP_PATH, satcat_path=SATCAT_PATH, discos_path=DISCOS_PATH
    )

    # ---- OPTIONAL: hand-tweak the seeded population here ----
    # e.g. add extra derelicts to a band of shells:
    # if "N_200kg" in self.x0.columns:
    #     self.x0.loc[8:12, "N_200kg"] += 30.0
    # --------------------------------------------------------

    self.FLM_steps = None
    # Launches are set later via each species' lambda_funs, so we do NOT call
    # self.future_launch_model here.


model.scenario_properties.initial_pop_and_launch = types.MethodType(
    _seed_initial_population, model.scenario_properties
)

model.configure_species(cfg["species"])

scen     = model.scenario_properties
t        = np.asarray(scen.scen_times)   # years, shape (steps,)
HMid     = np.asarray(scen.HMid)         # shell midpoint altitudes [km]
n_shells = scen.n_shells


def shell_for(alt_km):
    """Nearest shell to an altitude, using pyssem's own rule."""
    return int(np.argmin(np.abs(HMid - alt_km)))


def to_rate_array(profile, t):
    """Sample a launch profile onto scen_times `t` (sats/year, clamped >= 0).

    profile may be:
      - a callable  f(t) -> rate      (analytic, vectorized over the t array), or
      - a sequence of ANNUAL rates [year0, year1, ...] one value per year,
        which is linearly interpolated onto scen_times.
    """
    if callable(profile):
        return np.maximum(0.0, np.asarray(profile(t), dtype=float))
    annual = np.asarray(profile, dtype=float)        # one value per year
    years = np.arange(len(annual))                   # 0, 1, 2, ...
    return np.maximum(0.0, np.interp(t, years, annual))


# ----------------------------------------------------------------------
# 2) SET YOUR LAUNCH PROFILES HERE.
#    Each campaign: (species sym_name, target_altitude_km, profile(t) -> sats/year).
#    t is a numpy array of years, so write profiles vectorized (np.*, &, |).
# ----------------------------------------------------------------------
#    NOTE: the first field must be an ACTIVE species name from your config.
#    three_species.json has only 'S' active (no 'Su'); other configs may add 'Su'.
#    The profile is EITHER a formula lambda t: ...  OR a list of ANNUAL rates
#    (one per year), which is interpolated onto scen_times.
campaigns = [
    ("S", 550,  lambda t: 100 + 12 * t),                          # ramp at ~550 km
    ("S", 340,  lambda t: 400 * ((t >= 10) & (t <= 30))),         # on/off window at ~340 km
    ("S", 1200, lambda t: 800 / (1 + np.exp(-0.18 * (t - 25)))),  # S-curve at ~1200 km

    # Per-year example: explicit sats/year for years 0,1,2,... (length need not
    # equal steps; it's interpolated onto scen_times). Uncomment to use:
    ("S", 700, [0, 0, 30, 45, 60, 60, 60, 80, 120, 150] + [150]*90),
]


# ----------------------------------------------------------------------
# 3) Build per-species lambda_funs: list (length n_shells) of arrays or None.
# ----------------------------------------------------------------------
launch_by_species = {}     # sym_name -> list of (array | None), length n_shells
for sym, alt, prof in campaigns:
    launch_by_species.setdefault(sym, [None] * n_shells)
    k = shell_for(alt)
    rate = to_rate_array(prof, t)                               # sats/year on scen_times
    if launch_by_species[sym][k] is None:
        launch_by_species[sym][k] = rate
    else:
        launch_by_species[sym][k] = launch_by_species[sym][k] + rate   # stack campaigns in same shell


# ----------------------------------------------------------------------
# 4) Attach to every ACTIVE species. Targeted species get their arrays;
#    other active species get no launches (all None). Debris/derelict untouched.
# ----------------------------------------------------------------------
active_names = [s.sym_name for g in scen.species.values() for s in g if getattr(s, "active", False)]
print("Active species in this config:", active_names)
for sym in launch_by_species:
    if sym not in active_names:
        print(f"  WARNING: campaign species '{sym}' is not an active species here; it will be ignored.")

for group in scen.species.values():
    for species in group:
        if getattr(species, "active", False):
            species.launch_func = launch_func_lambda_fun
            species.lambda_funs = launch_by_species.get(species.sym_name, [None] * n_shells)


# ----------------------------------------------------------------------
# 5) Sanity check, then run.
# ----------------------------------------------------------------------
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)  # renamed in NumPy 2.0
print("Shell mapping and total satellites launched per campaign:")
for sym, alt, prof in campaigns:
    k = shell_for(alt)
    total = _trapz(to_rate_array(prof, t), t)                    # sats/yr integrated over yrs
    print(f"  {sym:>3}  {alt:>5} km -> shell {k:>2} (HMid={HMid[k]:.0f} km)  ~{total:,.0f} sats total")

model.run_model()

# ----------------------------------------------------------------------
# 6) Results. run_model() stores the SciPy solve_ivp result on
#    scen.output (it returns None). Layout:
#      out.t : times in years, shape (n_times,)
#      out.y : state, shape (n_species * n_shells, n_times),
#              ordered species-major / shell-minor
#              -> row for (species si, shell k) is  si * n_shells + k
# ----------------------------------------------------------------------
out = scen.output
species_names = scen.species_names

print(f"\nFinal total population by species (t = {out.t[-1]:.0f} yr):")
for si, name in enumerate(species_names):
    block = out.y[si * n_shells:(si + 1) * n_shells, :]   # (n_shells, n_times)
    print(f"  {name:>8}: {block[:, -1].sum():,.0f}")

# --- Optional: export JSON exactly like your smoketest ---
# with open("targeted_launch_results.json", "w") as f:
#     json.dump(model.results_to_json(), f)


# ----------------------------------------------------------------------
# 7) Plots.
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")   # write PNGs without needing a display; remove to show windows
import matplotlib.pyplot as plt

# Per-species population over time (sum across shells) -> (n_species, n_times)
totals = {name: out.y[si * n_shells:(si + 1) * n_shells, :].sum(axis=0)
          for si, name in enumerate(species_names)}

# Calendar-year x-axis if we can, else years-from-start.
start_year = getattr(getattr(scen, "start_date", None), "year", None)
xt = (start_year + out.t) if start_year else out.t
xlabel = "year" if start_year else "years from start"

# --- Figure 1: total population over time, per species (+ grand total) ---
fig1, ax = plt.subplots(figsize=(9, 5))
for name, series in totals.items():
    ax.plot(xt, series, linewidth=2, label=name)
ax.plot(xt, np.sum(list(totals.values()), axis=0),
        linewidth=1.5, linestyle="--", color="black", label="total")
ax.set_xlabel(xlabel)
ax.set_ylabel("number of objects")
ax.set_title("LEO population over time by species")
ax.grid(True, alpha=0.3)
ax.legend()
fig1.tight_layout()
fig1.savefig("population_over_time.png", dpi=130)

# --- Figure 2: population vs altitude, initial (dashed) vs final (solid) ---
fig2, ax2 = plt.subplots(figsize=(9, 5))
for si, name in enumerate(species_names):
    block = out.y[si * n_shells:(si + 1) * n_shells, :]   # (n_shells, n_times)
    line, = ax2.plot(HMid, block[:, -1], linewidth=2, label=f"{name} (final)")
    ax2.plot(HMid, block[:, 0], linewidth=1.2, linestyle="--",
             color=line.get_color(), alpha=0.7, label=f"{name} (initial)")
ax2.set_xlabel("altitude [km]")
ax2.set_ylabel("number of objects in shell")
ax2.set_title(f"Population by altitude: start vs. year {out.t[-1]:.0f}")
ax2.grid(True, alpha=0.3)
ax2.legend(ncol=2, fontsize=8)
fig2.tight_layout()
fig2.savefig("population_by_altitude.png", dpi=130)

print("Saved plots: population_over_time.png, population_by_altitude.png")
print("Done.")