"""
rbflag_sensitivity.py

A/B test for the RBflag -> objclass mapping in the NASA breakup model.

pyssem (faithfully copying MOCAT-SSEM MATLAB) uses:
    objclass = 5 if RBflag == 0 else 0
and func_Am treats ObjClass in (4.5, 8.5) as the ROCKET-BODY area/mass branch.
Traced through, that sends rocket-body collisions (RBflag==1 -> objclass 0) to the
SPACECRAFT distribution and vice-versa -- i.e. the mapping appears inverted.

This script runs the identical scenario twice:
  * "as-shipped"  : stock mapping
  * "corrected"   : the mapping with RB<->SC swapped
and overlays the debris-population trajectories so you can see the magnitude for
your scenario. The correction is applied by intercepting func_Am and swapping the
ObjClass it receives (0<->5), which is exactly the corrected assignment at the
point of use -- no edits to the installed pyssem source.

Fragmentation draws are stochastic, so both runs are seeded identically; the two
variants consume the RNG in lockstep (same fragment diameters -> same number of
draws), so the only difference is which A/m distribution parameters are used.
"""

import json
import types
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_x0 import build_x0

try:
    from pyssem.model import Model
    from pyssem.utils.launch.launch import launch_func_lambda_fun
    import pyssem.utils.collisions.NASA_SBM_Evolve as sbm
    try:
        import pyssem.utils.collisions.collisions as col
    except ImportError:
        col = None
except ImportError:
    from model import Model
    from utils.launch.launch import launch_func_lambda_fun
    import utils.collisions.NASA_SBM_Evolve as sbm
    try:
        import utils.collisions.collisions as col
    except ImportError:
        col = None

CONFIG_PATH = "config_rb.json"
GP_PATH, SATCAT_PATH, DISCOS_PATH = "2026.csv", "satcat.csv", "discos_cache.csv"
SEED = 12345

# Launch campaigns (identical for both runs; kept modest so debris signal is readable).
CAMPAIGNS = [
    ("S", 550,  lambda t: 100 + 12 * t),
    ("S", 1200, lambda t: 800 / (1 + np.exp(-0.18 * (t - 25)))),
]

# ---- keep an untouched reference to the real func_Am(s) ----
_ORIG_SBM_FUNC_AM = sbm.func_Am
_ORIG_COL_FUNC_AM = getattr(col, "func_Am", None) if col is not None else None


def _swap_objclass(oc):
    # 0 <-> 5; pass anything else through unchanged
    return 0 if oc == 5 else (5 if oc == 0 else oc)


def _apply_mapping(corrected: bool):
    """Patch func_Am to the corrected mapping, or restore the stock one."""
    if corrected:
        sbm.func_Am = lambda d, oc: _ORIG_SBM_FUNC_AM(d, _swap_objclass(oc))
        if col is not None and _ORIG_COL_FUNC_AM is not None:
            col.func_Am = lambda d, oc: _ORIG_COL_FUNC_AM(d, _swap_objclass(oc))
    else:
        sbm.func_Am = _ORIG_SBM_FUNC_AM
        if col is not None and _ORIG_COL_FUNC_AM is not None:
            col.func_Am = _ORIG_COL_FUNC_AM


def build_and_run(corrected: bool):
    # Patch BEFORE configure_species -- collision fragments are computed there.
    _apply_mapping(corrected)
    np.random.seed(SEED)
    random.seed(SEED)

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    sp = cfg["scenario_properties"]

    model = Model(
        start_date=sp["start_date"].split("T")[0],
        simulation_duration=sp["simulation_duration"], steps=sp["steps"],
        min_altitude=sp["min_altitude"], max_altitude=sp["max_altitude"],
        n_shells=sp["n_shells"], launch_function=sp["launch_function"],
        integrator=sp["integrator"], density_model=sp["density_model"],
        LC=sp["LC"], v_imp=sp.get("v_imp"), baseline=sp.get("baseline", False),
        fragment_spreading=sp.get("fragment_spreading", False),
        parallel_processing=sp.get("parallel_processing", False),
        indicator_variables=sp.get("indicator_variables"),
    )

    def _seed_pop(self, baseline=False, launch_file=None):
        self.x0 = build_x0(self, gp_path=GP_PATH, satcat_path=SATCAT_PATH,
                           discos_path=DISCOS_PATH, verbose=False)
        self.FLM_steps = None
    model.scenario_properties.initial_pop_and_launch = types.MethodType(
        _seed_pop, model.scenario_properties)

    model.configure_species(cfg["species"])

    scen = model.scenario_properties
    t = np.asarray(scen.scen_times)
    HMid = np.asarray(scen.HMid)
    n_shells = scen.n_shells

    by = {}
    for sym, alt, prof in CAMPAIGNS:
        by.setdefault(sym, [None] * n_shells)
        k = int(np.argmin(np.abs(HMid - alt)))
        rate = np.maximum(0.0, np.asarray(prof(t), dtype=float))
        by[sym][k] = rate if by[sym][k] is None else by[sym][k] + rate
    for g in scen.species.values():
        for s in g:
            if getattr(s, "active", False):
                s.launch_func = launch_func_lambda_fun
                s.lambda_funs = by.get(s.sym_name, [None] * n_shells)

    model.run_model()

    out = scen.output
    names = list(scen.species_names)
    per_species = {nm: out.y[i * n_shells:(i + 1) * n_shells, :].sum(axis=0)
                   for i, nm in enumerate(names)}
    active_flags = {s.sym_name: getattr(s, "active", False)
                    for g in scen.species.values() for s in g}
    return np.asarray(out.t), per_species, active_flags


# ----------------------------- run both -----------------------------
print("Run 1/2: as-shipped mapping ...")
t, ship, active_flags = build_and_run(corrected=False)
print("Run 2/2: corrected mapping ...")
_, corr, _ = build_and_run(corrected=True)

_apply_mapping(False)  # restore stock, just in case

debris_names = [nm for nm, a in active_flags.items() if not a]
ship_debris = np.sum([ship[nm] for nm in debris_names], axis=0)
corr_debris = np.sum([corr[nm] for nm in debris_names], axis=0)

# ----------------------------- report -----------------------------
def pct(a, b):
    return 100.0 * (a - b) / b if b else float("nan")

print(f"\nFinal debris totals (t = {t[-1]:.0f} yr):")
print(f"  as-shipped: {ship_debris[-1]:,.0f}")
print(f"  corrected : {corr_debris[-1]:,.0f}")
print(f"  difference: {pct(corr_debris[-1], ship_debris[-1]):+.1f}%  (corrected vs as-shipped)")
print("\nPer debris species (final):")
for nm in debris_names:
    print(f"  {nm:>10}: as-shipped {ship[nm][-1]:>12,.0f} | "
          f"corrected {corr[nm][-1]:>12,.0f} | {pct(corr[nm][-1], ship[nm][-1]):+.1f}%")

# ----------------------------- plots -----------------------------
start_year = 2025
xt = start_year + t

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(xt, ship_debris, lw=2, label="as-shipped")
ax1.plot(xt, corr_debris, lw=2, ls="--", label="corrected (RB/SC swapped)")
ax1.set_xlabel("year"); ax1.set_ylabel("total debris objects")
ax1.set_title("Total debris: RBflag mapping sensitivity")
ax1.grid(True, alpha=0.3); ax1.legend()

for nm in debris_names:
    line, = ax2.plot(xt, ship[nm], lw=2, label=f"{nm} (as-shipped)")
    ax2.plot(xt, corr[nm], lw=1.5, ls="--", color=line.get_color(),
             label=f"{nm} (corrected)")
ax2.set_xlabel("year"); ax2.set_ylabel("objects")
ax2.set_title("By debris species")
ax2.grid(True, alpha=0.3); ax2.legend(ncol=2, fontsize=8)

fig.tight_layout()
fig.savefig("rbflag_sensitivity.png", dpi=130)
print("\nSaved rbflag_sensitivity.png")
