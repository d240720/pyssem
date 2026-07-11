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
    # ("S", 700, [0, 0, 30, 45, 60, 60, 60, 80, 120, 150] + [150]*90),
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
# 4b) Rocket-body injection: for every satellite launched, deposit the
#     expected number of surviving spent stages (per rb_injection.json).
#     B_rate[k'] = sum_k S_rate[k] * R_persat[band(k)][band(k')] spread over
#     the shells of band(k').  Set RB_INJECTION = None to disable.
# ----------------------------------------------------------------------
RB_INJECTION = "rb_injection.json"   # from build_rb_injection.py; None to skip

if RB_INJECTION:
    with open(RB_INJECTION) as f:
        inj = json.load(f)
    bands = inj["bands"]
    R_band = np.asarray(inj["R_persat"])           # (n_band, n_band)

    def band_of_alt(alt):
        for i, (lo, hi) in enumerate(bands):
            if lo <= alt < hi:
                return i
        return -1

    shell_band = np.array([band_of_alt(h) for h in HMid])       # band index per shell
    shells_in_band = {b: np.where(shell_band == b)[0] for b in range(len(bands))}

    # Total satellite launch rate per shell over time: (n_shells, n_times)
    n_times = len(t)
    S_rate = np.zeros((n_shells, n_times))
    for sym in active_names:
        for k, arr in enumerate(launch_by_species.get(sym, [None] * n_shells)):
            if arr is not None:
                S_rate[k] += arr

    # Map through the injection matrix into rocket-body launches per shell.
    B_rate = np.zeros((n_shells, n_times))
    for k in range(n_shells):
        b = shell_band[k]
        if b < 0 or not S_rate[k].any():
            continue
        for bp in range(len(bands)):
            dest = shells_in_band.get(bp, [])
            if R_band[b, bp] == 0 or len(dest) == 0:
                continue
            per_shell = R_band[b, bp] / len(dest)      # spread uniformly within dest band
            for kp in dest:
                B_rate[kp] += S_rate[k] * per_shell

    # Assign to the rocket-body species (RBflag == 1).
    rb_species = [s for g in scen.species.values() for s in g
                  if getattr(s, "RBflag", 0) == 1]
    if not rb_species:
        print("  WARNING: RB_INJECTION set but no species with RBflag==1 found; skipping.")
    else:
        b_funs = [B_rate[kp] if B_rate[kp].any() else None for kp in range(n_shells)]
        for s in rb_species:
            s.launch_func = launch_func_lambda_fun
            s.lambda_funs = b_funs
        _trapz0 = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        print(f"Rocket-body injection ON ({rb_species[0].sym_name}): "
              f"~{_trapz0(B_rate.sum(axis=0), t):,.0f} stages launched over the run "
              f"(mass {inj['stage_mass_kg']:.0f} kg).")


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

# --- Figure 1: population over time, ONE PANEL PER SPECIES (own y-scale) ---
n_sp = len(species_names)
ncols = 2 if n_sp > 1 else 1
nrows = int(np.ceil(n_sp / ncols))
fig1, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows),
                          sharex=True, squeeze=False)
flat = axes.ravel()
for idx, name in enumerate(species_names):
    axp = flat[idx]
    axp.plot(xt, totals[name], linewidth=2, color=f"C{idx}")
    axp.set_title(name)
    axp.set_ylabel("objects")
    axp.grid(True, alpha=0.3)
for idx in range(n_sp, len(flat)):        # hide any unused panels
    flat[idx].axis("off")
for c in range(ncols):                     # x-labels on the bottom row
    flat[min((nrows - 1) * ncols + c, len(flat) - 1)].set_xlabel(xlabel)
fig1.suptitle("Population over time by species")
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


# ----------------------------------------------------------------------
# 8) Collision probability per satellite, per shell (height band), over time.
#    Reproduces pyssem's own collision rate  gamma * phi * N_i * N_j, where
#      phi[k] = pi * v_imp_all[k] / (V[k] in km^3) * (r_i + r_j)^2 (km^2) * sec/yr
#      gamma  = collision-avoidance factor (alpha / alpha_active / slotting)
#    For an active species S:
#      p_S[k,t] = sum_j  gamma(S,j) * phi(S,j)[k] * N_j[k,t]
#    Units: expected collisions per satellite per year (~ annual collision prob).
# ----------------------------------------------------------------------
from matplotlib.colors import LogNorm

V_shell = np.asarray(scen.V, dtype=float)                 # shell volume [m^3]
v_imp_all = np.asarray(scen.v_imp_all, dtype=float)       # per-shell impact vel [km/s]
if v_imp_all.ndim == 0:
    v_imp_all = np.full(n_shells, float(v_imp_all))
M2KM, SEC_PER_YEAR = 1e-3, 86400.0 * 365.25

info, active_flag = {}, {}
for g in scen.species.values():
    for s in g:
        info[s.sym_name] = dict(
            r=float(s.radius), man=bool(s.maneuverable), trk=bool(s.trackable),
            alpha=float(s.alpha or 0.0), alpha_active=float(s.alpha_active or 0.0),
            slotted=bool(getattr(s, "slotted", False)),
            slot_eff=float(getattr(s, "slotting_effectiveness", 0.0) or 0.0))
        active_flag[s.sym_name] = getattr(s, "active", False)

def phi_pair(ri, rj):
    sigma = (ri * M2KM + rj * M2KM) ** 2                  # km^2
    return np.pi * v_imp_all / (V_shell * M2KM ** 3) * sigma * SEC_PER_YEAR   # (n_shells,)

def gamma_factor(a, b):
    if a["man"] and b["man"]:
        gf = a["alpha_active"] * b["alpha_active"]
        if a["slotted"] and b["slotted"]:
            gf *= min(a["slot_eff"], b["slot_eff"])
        return gf
    if a["man"] ^ b["man"]:
        man, non = (a, b) if a["man"] else (b, a)
        return man["alpha"] if non["trk"] else 1.0
    return 1.0

def species_block(nm):
    i = species_names.index(nm)
    return out.y[i * n_shells:(i + 1) * n_shells, :]      # (n_shells, n_times)

sats = [nm for nm in species_names if active_flag.get(nm, False)]
coll_prob = {}
for S in sats:
    p = np.zeros((n_shells, len(out.t)))
    for j in species_names:
        p += gamma_factor(info[S], info[j]) * phi_pair(info[S]["r"], info[j]["r"])[:, None] * species_block(j)
    coll_prob[S] = p

np.savez("collision_probability.npz", time=out.t, altitude_km=HMid,
         **{f"p_{S}": coll_prob[S] for S in sats})

for S in sats:
    p = coll_prob[S]
    fig, axh = plt.subplots(figsize=(9, 5))
    pos = p[p > 0]
    if pos.size:
        norm = LogNorm(vmin=pos.min(), vmax=p.max())
    else:
        norm = None
    mesh = axh.pcolormesh(xt, HMid, p, shading="auto", norm=norm, cmap="magma")
    axh.set_xlabel(xlabel)
    axh.set_ylabel("altitude [km]")
    axh.set_title(f"{S}: collision probability per satellite per year")
    fig.colorbar(mesh, ax=axh, label="expected collisions / sat / yr")
    fig.tight_layout()
    fig.savefig(f"collision_prob_{S}.png", dpi=130)
    # peak shell at final time
    kmax = int(np.argmax(p[:, -1]))
    print(f"{S}: peak collision prob at t={out.t[-1]:.0f} is "
          f"{p[kmax, -1]:.2e} /sat/yr at {HMid[kmax]:.0f} km")

print("Saved collision_probability.npz and collision_prob_<species>.png")
print("Done.")