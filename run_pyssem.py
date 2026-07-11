#!/usr/bin/env python3
"""
run_pyssem.py -- set conditions and run. All machinery is in pyssem_core.py.

run(CONFIG) RETURNS the time series in memory (writes no files):
    results = {
        "time":                  (n_times,)   years from start,
        "altitude_km":           (n_shells,),
        "populations":           {species:   (n_shells, n_times)},
        "collision_probability": {satellite: (n_shells, n_times)},
    }
e.g.  results["populations"]["S"]           -> satellite counts by shell over time
      results["collision_probability"]["S"] -> per-sat collision prob by shell over time

Grid is fixed at 200-2000 km, 50 km shells (36 shells).

Launch profiles accept either form:
    lambda t: <sats/year>     callable of time (years from start)
    [r0, r1, r2, ...]         one rate per year (index i -> year i), held flat after the last
"""

import numpy as np                 # available for the launch-profile lambdas below
from pyssem_core import run, summarize, plot_results

CONFIG_PATH = "config_rb.json"     # pyssem species/scenario JSON


# ============================ EDIT HERE ============================
CONFIG = {
    "config_path": CONFIG_PATH,

    # --- scenario grid: FIXED at 200-2000 km, 50 km shells ---
    "years": 100,
    "steps": 101,                  # 1/yr
    "min_altitude": 200,
    "max_altitude": 2000,
    "n_shells": 36,                # (2000-200)/36 = 50 km shells
    "density_model": None,         # None = keep JSON; or "static_exp_dens_func" / "JB2008_dens_func"

    # --- satellite properties (applied to every ACTIVE species; None = keep JSON) ---
    "sat_mass": None,              # kg
    "sat_radius": None,            # m  (characteristic size / length)
    "sat_lifetime": 5,             # years active before disposal (deltat)
    "pmd_fail": 0.1,               # PMD failure rate; Pm = 1 - pmd_fail

    # --- initial population ---
    "seed_from_catalog": True,     # base: build x0 from catalog, else empty
    "gp": "2026.csv",
    "satcat": "satcat.csv",
    "discos": "discos_cache.csv",

    # overrides on top of the base population (bands span the full 200-2000 range):
    "ic_bands": [(200, 300), (300, 400), (400, 500), (500, 600),
                 (600, 700), (700, 1000), (1000, 2000)],
    "ic_scale": {},                # species -> multiplier, e.g. {"N": 1.5}
    "ic_override": {},             # species -> per-band OR per-shell counts (replaces base)

    # --- satellite launch schedule: (active_species, altitude_km, profile) ---
    "launches": [
        ("S", 550,  lambda t: 100 + 12 * t),                          # ramp
        ("S", 340,  lambda t: 400 * ((t >= 10) & (t <= 30))),         # on/off window
        ("S", 1200, lambda t: 800 / (1 + np.exp(-0.18 * (t - 25)))),  # S-curve
    ],

    # --- rocket-body injection (spent stages per satellite) ---
    "rb_injection": "rb_injection.json",   # None to disable

    # --- collision probability ---
    "collision_avoidance": True,   # include alpha/alpha_active; False = raw conjunctions

    # when running this file directly, also draw figures (run() itself never saves)
    "plots": True,
    "outdir": ".",
}
# ==================================================================


if __name__ == "__main__":
    results = run(CONFIG)          # <- time series returned here
    time=results["time"] #time steps
    alt_km=results["altitude_km"] #center of each of the 36 bands
    pop=results["populations"] # populations, takes the keys: ['S', 'N', 'N_200kg', 'B'] where 'N_200kg' depends on set sat mass. S is sats, N is debris, B is rocket bodies, N_200kg is derelicts
    pop_S=pop['S']
    pop_N=pop['N']
    pop_D=pop['N_200kg'] #or whatever the mass of a sat is
    pop_B=pop['B']
    collision_prob_S=results["collision_probability"]['S'] #sat only
    
    if False:  #set to true if want to save data as json
        with open("results.json", "w") as file:
            json.dump(my_dict, file, indent=4) 
    
    #summarize(results)
    if CONFIG.get("plots"):
        plot_results(results, CONFIG.get("outdir", "."))