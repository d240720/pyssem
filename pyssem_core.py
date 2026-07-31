#!/usr/bin/env python3
"""
pyssem_core.py -- backend machinery for run_pyssem.py.

You should not need to edit this file. It builds a pyssem model from a plain
CONFIG dict, applies overrides (grid, density, satellite params, initial
conditions), sets launch schedules (+ rocket-body injection), runs the model,
and RETURNS the population and collision-probability time series in memory.

Public entry point:
    run(config) -> results dict   (writes no files)

results = {
    "time":                  (n_times,)   years from start,
    "altitude_km":           (n_shells,),
    "populations":           {species:   (n_shells, n_times)},
    "collision_probability": {satellite: (n_shells, n_times)},
}

Optional helpers: summarize(results), plot_results(results, outdir).
"""

import json
import types

import numpy as np
import matplotlib
matplotlib.use("Agg")            # only used by the optional plot_results()
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from build_x0 import build_x0
try:
    from pyssem.model import Model
    from pyssem.utils.launch.launch import launch_func_lambda_fun
except ImportError:
    from model import Model
    from utils.launch.launch import launch_func_lambda_fun


# --------------------------------------------------------------------------
# Rocket-body injection file (per-band format)
# --------------------------------------------------------------------------
def _load_rb_injection(path):
    """Load rb_injection.json, normalising old (scalar) and new (per-band) formats.

    New format: stage_mass_kg / stage_radius_m are length-n_bands VECTORS indexed by
    the DESTINATION band (the column of R_persat) -- stages landing at 700-1000 km are
    ~4 t upper stages, those at 400-600 km are ~2.5 t. Old files carry scalars; those
    are broadcast across all bands so this stays backward compatible.
    """
    with open(path) as f:
        inj = json.load(f)
    nb = len(inj["bands"])

    def as_vec(key, fallback_key, default):
        v = inj.get(key, inj.get(fallback_key, default))
        if np.isscalar(v):
            return np.full(nb, float(v))
        v = np.asarray(v, float)
        if len(v) != nb:
            raise ValueError(f"{path}: '{key}' has length {len(v)}, expected {nb} bands")
        return v

    inj["_mass"] = as_vec("stage_mass_kg", "stage_mass_kg_pooled", 1500.0)
    inj["_radius"] = as_vec("stage_radius_m", "stage_radius_m_pooled", 2.0)
    inj["_R"] = np.asarray(inj["R_persat"], float)
    return inj


def _profile_to_rate(profile, t):
    """Launch profile -> nonnegative rate on grid t. Callables are evaluated
    (constant lambdas broadcast from 0-d); array-likes are year-indexed
    (index i = year i), linearly interpolated, held flat after the last entry.
    Single definition shared by set_launches and the RB flux weighting so the
    same profile can never mean two different things to the two consumers."""
    if callable(profile):
        r = np.maximum(0.0, np.asarray(profile(t), float))
        return np.broadcast_to(r, t.shape)
    a = np.asarray(profile, float)
    return np.maximum(0.0, np.interp(t, np.arange(len(a)), a))


def _band_of_factory(bands):
    def band_of(alt):
        for i, (lo, hi) in enumerate(bands):
            if lo <= alt < hi:
                return i
        return -1
    return band_of


def _rb_stage_properties(cfg, years, steps, active_names=None):
    """Collapse the per-band stage mass/radius vectors to the single (mass, radius)
    that pyssem's one-RB-species model can represent, weighted by the stage flux this
    scenario's launch schedule will actually deposit in each band.

    Must run BEFORE configure_species(): species mass/radius are baked into the
    symbolic equations there, so mutating them later is a no-op.

    Mass is flux-weighted (conserves the deposited mass budget the breakup model
    consumes). Radius is flux-weighted in AREA, then converted back -- collision rate
    scales with cross-section, so averaging radius directly would under-count area.
    """
    inj = _load_rb_injection(cfg["rb_injection"])
    bands, R = inj["bands"], inj["_R"]
    band_of = _band_of_factory(bands)
    nb = len(bands)

    # Payload volume launched into each band over the run (arbitrary common units --
    # only the relative weights matter). Uses the sim's own horizon and only
    # launches that set_launches will actually schedule (active species).
    t = np.linspace(0.0, float(years), int(steps))
    vol = np.zeros(nb)
    for sym, alt, prof in cfg.get("launches", []):
        if active_names is not None and sym not in active_names:
            continue
        b = band_of(alt)
        if b < 0:
            continue
        rate = _profile_to_rate(prof, t)
        vol[b] += float(np.trapezoid(rate, t)) if hasattr(np, "trapezoid") \
            else float(np.trapz(rate, t))

    w = vol @ R                                  # stages deposited per destination band
    if w.sum() <= 0:                             # no launches -> fall back to a plain mean
        w = np.ones(nb)

    mass = float(np.average(inj["_mass"], weights=w))
    area = np.pi * inj["_radius"] ** 2
    radius = float(np.sqrt(np.average(area, weights=w) / np.pi))
    return mass, radius, w, inj


# --------------------------------------------------------------------------
# Config -> scenario / species overrides
# --------------------------------------------------------------------------
def _apply_scenario_params(sp, cfg):
    """Override scenario-grid and physics fields from CONFIG (None = keep JSON)."""
    sp["simulation_duration"] = cfg.get("years", sp["simulation_duration"])
    sp["steps"] = cfg.get("steps", sp["steps"])
    for key in ("min_altitude", "max_altitude", "n_shells", "density_model"):
        if cfg.get(key) is not None:
            sp[key] = cfg[key]


def _apply_species_params(conf, cfg):
    """Override lifetime, PMD failure, mass, and size for every ACTIVE species,
    before the equations are built."""
    life, pmdf = cfg.get("sat_lifetime"), cfg.get("pmd_fail")
    mass, radius = cfg.get("sat_mass"), cfg.get("sat_radius")
    for s in conf["species"]:
        if s.get("active"):
            if life is not None:
                s["deltat"] = life
            if pmdf is not None:
                s["Pm"] = 1.0 - pmdf
            if mass is not None:
                s["mass"] = mass
            if radius is not None:
                s["radius"] = radius
                s["A"] = "Calculated based on radius"   # recompute area from new radius


def _apply_rb_species_params(conf, cfg):
    """Set the RBflag species' mass/radius from the rb_injection file, flux-weighted
    across destination bands. Runs before configure_species() so the values reach the
    symbolic equations. Explicit CONFIG values (rb_mass / rb_radius) win."""
    if not cfg.get("rb_injection"):
        return
    sp = conf["scenario_properties"]            # post-override values
    active_names = {s["sym_name"] for s in conf["species"] if s.get("active")}
    mass, radius, w, inj = _rb_stage_properties(
        cfg, sp.get("simulation_duration", 100), sp.get("steps", 200), active_names)
    mass = cfg.get("rb_mass") if cfg.get("rb_mass") is not None else mass
    radius = cfg.get("rb_radius") if cfg.get("rb_radius") is not None else radius

    rb = [s for s in conf["species"] if s.get("RBflag", 0) == 1]
    if not rb:
        print("  warning: rb_injection set but no RBflag==1 species in config; "
              "stage mass/radius not applied")
        return
    for s in rb:
        s["mass"] = float(mass)
        s["radius"] = float(radius)
        s["A"] = "Calculated based on radius"

    shares = w / w.sum() if w.sum() > 0 else w
    print(f"  rb stage props: {mass:.0f} kg, r_eff {radius:.2f} m "
          f"(flux-weighted over bands; per-band mass "
          f"{np.array2string(inj['_mass'], precision=0, separator=',')})")
    print(f"  rb stage flux share by destination band: "
          f"{np.array2string(shares, precision=3, separator=',')}")


def _apply_ic(x0, cfg, HMid):
    """Apply ic_scale (multipliers) and ic_override (per-band or per-shell counts)
    to the base initial-condition DataFrame in place."""
    n = len(x0)
    bands = cfg.get("ic_bands")
    shell_band = None
    if bands:
        band_of = _band_of_factory(bands)
        shell_band = np.array([band_of(h) for h in HMid])

    for sp, f in cfg.get("ic_scale", {}).items():
        if sp in x0.columns:
            x0[sp] = x0[sp] * float(f)
        else:
            print(f"  warning: ic_scale species '{sp}' not in model; skipped")

    for sp, vec in cfg.get("ic_override", {}).items():
        if sp not in x0.columns:
            print(f"  warning: ic_override species '{sp}' not in model; skipped")
            continue
        vec = np.asarray(vec, float)
        if len(vec) == n:                                   # per-shell counts
            x0[sp] = vec
        elif shell_band is not None and len(vec) == len(bands):   # per-band totals
            col = np.zeros(n)
            for b in range(len(bands)):
                shells = np.where(shell_band == b)[0]
                if len(shells):
                    col[shells] = vec[b] / len(shells)      # spread total over band's shells
            x0[sp] = col
        else:
            print(f"  warning: ic_override['{sp}'] length {len(vec)} is neither "
                  f"n_shells({n}) nor n_bands({len(bands) if bands else 0}); skipped")
    return x0


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------
def build_model(cfg):
    with open(cfg["config_path"]) as f:
        conf = json.load(f)
    sp = conf["scenario_properties"]
    _apply_scenario_params(sp, cfg)      # grid / density overrides
    _apply_species_params(conf, cfg)     # lifetime / pmd / mass / size overrides
    _apply_rb_species_params(conf, cfg)  # RB stage mass/radius from rb_injection

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

    # Seed the initial population (bypasses the broken launch-file download).
    def _seed(self, baseline=False, launch_file=None):
        if cfg["seed_from_catalog"]:
            self.x0 = build_x0(self, gp_path=cfg["gp"], satcat_path=cfg["satcat"],
                               discos_path=cfg["discos"],
                               ecc_policy=cfg.get("ecc_policy", "mean"), verbose=False)
        else:
            import pandas as pd
            self.x0 = pd.DataFrame(np.zeros((self.n_shells, len(self.species_names))),
                                   index=range(self.n_shells), columns=self.species_names)
        _apply_ic(self.x0, cfg, np.asarray(self.HMid))   # scale / override
        self.FLM_steps = None
    model.scenario_properties.initial_pop_and_launch = types.MethodType(
        _seed, model.scenario_properties)

    model.configure_species(conf["species"])
    return model


# --------------------------------------------------------------------------
# Launch schedule (satellites + rocket-body injection)
# --------------------------------------------------------------------------
def set_launches(model, cfg):
    scen = model.scenario_properties
    t = np.asarray(scen.scen_times)
    HMid = np.asarray(scen.HMid)
    n = scen.n_shells

    def to_rate(profile):
        return _profile_to_rate(profile, t)

    def shell_for(alt):
        return int(np.argmin(np.abs(HMid - alt)))

    by = {}
    for sym, alt, prof in cfg["launches"]:
        by.setdefault(sym, [None] * n)
        k = shell_for(alt)
        r = to_rate(prof)
        by[sym][k] = r if by[sym][k] is None else by[sym][k] + r

    active = [s.sym_name for g in scen.species.values() for s in g
              if getattr(s, "active", False)]
    for sym in by:
        if sym not in active:
            print(f"  warning: launch species '{sym}' is not active; ignored")

    for g in scen.species.values():
        for s in g:
            if getattr(s, "active", False):
                s.launch_func = launch_func_lambda_fun
                s.lambda_funs = by.get(s.sym_name, [None] * n)

    if cfg.get("rb_injection"):
        _apply_rb_injection(scen, cfg["rb_injection"], by, active, t, HMid, n)


def _apply_rb_injection(scen, path, by, active, t, HMid, n):
    inj = _load_rb_injection(path)
    bands = inj["bands"]
    R_band = inj["_R"]
    band_of = _band_of_factory(bands)

    shell_band = np.array([band_of(h) for h in HMid])
    in_band = {b: np.where(shell_band == b)[0] for b in range(len(bands))}
    empty = [bp for bp in range(len(bands))
             if len(in_band[bp]) == 0 and R_band[:, bp].any()]
    if empty:
        print(f"  warning: rb_injection destination band(s) "
              f"{[tuple(bands[bp]) for bp in empty]} contain no shells in this "
              f"grid; their stage rates are dropped")

    S_rate = np.zeros((n, len(t)))
    for sym in active:
        for k, arr in enumerate(by.get(sym, [None] * n)):
            if arr is not None:
                S_rate[k] += arr

    n_unbanded = sum(1 for k in range(n)
                     if shell_band[k] < 0 and S_rate[k].any())
    if n_unbanded:
        print(f"  warning: {n_unbanded} shell(s) with launches lie outside all "
              f"rb_injection bands; those launches generate no stages")
    suppressed = set(inj.get("suppressed_payload_bands", []))
    hit = sorted({int(shell_band[k]) for k in range(n)
                  if shell_band[k] in suppressed and S_rate[k].any()})
    if hit:
        print(f"  warning: launches target payload band(s) "
              f"{[tuple(bands[b]) for b in hit]} whose R_persat row was "
              f"SUPPRESSED at build time (too few catalog payloads); those "
              f"launches generate no stages -- rebuild rb_injection with a "
              f"lower --min-payloads or accept the zero")

    B_rate = np.zeros((n, len(t)))
    for k in range(n):
        b = shell_band[k]
        if b < 0 or not S_rate[k].any():
            continue
        for bp in range(len(bands)):
            dest = in_band.get(bp, [])
            if R_band[b, bp] == 0 or len(dest) == 0:
                continue
            per_shell = R_band[b, bp] / len(dest)
            for kp in dest:
                B_rate[kp] += S_rate[k] * per_shell

    rb = [s for g in scen.species.values() for s in g if getattr(s, "RBflag", 0) == 1]
    if not rb:
        print("  warning: rb_injection set but no RBflag==1 species; skipping")
        return
    b_funs = [B_rate[kp] if B_rate[kp].any() else None for kp in range(n)]
    for s in rb:
        s.launch_func = launch_func_lambda_fun
        s.lambda_funs = b_funs


# --------------------------------------------------------------------------
# Outputs: population + collision-probability time series
# --------------------------------------------------------------------------
def population_series(scen):
    """Return t (years), altitude (km), and {species: (n_shells, n_times)}."""
    out = scen.output
    names = list(scen.species_names)
    n = scen.n_shells
    per = {nm: out.y[i * n:(i + 1) * n, :] for i, nm in enumerate(names)}
    return np.asarray(out.t), np.asarray(scen.HMid), per


def collision_probability(scen, avoidance=True):
    """Per-satellite collision probability {sat: (n_shells, n_times)} using
    pyssem's own kinetic factor phi and collision-avoidance gamma."""
    out = scen.output
    names = list(scen.species_names)
    n = scen.n_shells
    V = np.asarray(scen.V, float)
    v_imp = np.asarray(scen.v_imp_all, float)
    if v_imp.ndim == 0:
        v_imp = np.full(n, float(v_imp))
    M2KM, SEC = 1e-3, 86400.0 * 365.25

    unset = [s.sym_name for g in scen.species.values() for s in g
             if getattr(s, "maneuverable", False)
             and (s.alpha is None or s.alpha_active is None)]
    if avoidance and unset:
        print(f"  warning: maneuverable species {unset} have alpha/alpha_active="
              f"None -> treated as 0 (perfect avoidance) in reported probabilities")

    info, active = {}, []
    for g in scen.species.values():
        for s in g:
            info[s.sym_name] = dict(
                r=float(s.radius), man=bool(s.maneuverable), trk=bool(s.trackable),
                alpha=float(s.alpha or 0.0), alpha_active=float(s.alpha_active or 0.0),
                slotted=bool(getattr(s, "slotted", False)),
                slot_eff=float(getattr(s, "slotting_effectiveness", 0.0) or 0.0))
            if getattr(s, "active", False):
                active.append(s.sym_name)

    def phi(ri, rj):
        sigma = (ri * M2KM + rj * M2KM) ** 2
        return np.pi * v_imp / (V * M2KM ** 3) * sigma * SEC       # (n_shells,)

    def gamma(a, b):
        if not avoidance:
            return 1.0
        if a["man"] and b["man"]:
            gf = a["alpha_active"] * b["alpha_active"]
            if a["slotted"] and b["slotted"]:
                gf *= min(a["slot_eff"], b["slot_eff"])
            return gf
        if a["man"] ^ b["man"]:
            man, non = (a, b) if a["man"] else (b, a)
            return man["alpha"] if non["trk"] else 1.0
        return 1.0

    def block(nm):
        i = names.index(nm)
        return out.y[i * n:(i + 1) * n, :]

    prob = {}
    for S in active:
        p = np.zeros((n, len(out.t)))
        for j in names:
            p += gamma(info[S], info[j]) * phi(info[S]["r"], info[j]["r"])[:, None] * block(j)
        prob[S] = p
    return prob


def collect_results(scen, avoidance=True):
    """Assemble the in-memory time series returned by run() (no files written)."""
    t, HMid, per = population_series(scen)
    prob = collision_probability(scen, avoidance=avoidance)
    return {
        "time": t,                        # (n_times,) years from start
        "altitude_km": HMid,              # (n_shells,)
        "populations": per,               # {species: (n_shells, n_times)}
        "collision_probability": prob,    # {satellite: (n_shells, n_times)}
    }


def summarize(results):
    """Print final totals and peak collision probability from a results dict."""
    t, HMid = results["time"], results["altitude_km"]
    print(f"\nFinal totals (t = {t[-1]:.0f} yr):")
    for nm, arr in results["populations"].items():
        print(f"  {nm:>10}: {arr[:, -1].sum():,.0f}")
    for S, p in results["collision_probability"].items():
        k = int(np.argmax(p[:, -1]))
        print(f"  peak collision prob [{S}]: {p[k, -1]:.2e} /sat/yr at {HMid[k]:.0f} km")


def plot_results(results, outdir="."):
    """Optional: write population panels + collision-probability heatmaps from a
    results dict. run() does NOT call this; invoke it yourself if you want figures."""
    t, HMid = results["time"], results["altitude_km"]
    per, prob = results["populations"], results["collision_probability"]
    names = list(per.keys())
    od = outdir.rstrip("/")
    totals = {nm: per[nm].sum(axis=0) for nm in names}

    ncols = 2 if len(names) > 1 else 1
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows),
                             sharex=True, squeeze=False)
    flat = axes.ravel()
    for i, nm in enumerate(names):
        flat[i].plot(t, totals[nm], lw=2, color=f"C{i}")
        flat[i].set_title(nm)
        flat[i].set_ylabel("objects")
        flat[i].grid(True, alpha=0.3)
    for i in range(len(names), len(flat)):
        flat[i].axis("off")
    for c in range(ncols):
        flat[min((nrows - 1) * ncols + c, len(flat) - 1)].set_xlabel("years from start")
    fig.suptitle("Population over time by species")
    fig.tight_layout()
    fig.savefig(f"{od}/population_over_time.png", dpi=130)
    plt.close(fig)

    for S, p in prob.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        pos = p[p > 0]
        norm = LogNorm(vmin=pos.min(), vmax=p.max()) if pos.size else None
        mesh = ax.pcolormesh(t, HMid, p, shading="auto", norm=norm, cmap="magma")
        ax.set_xlabel("years from start")
        ax.set_ylabel("altitude [km]")
        ax.set_title(f"{S}: collision probability per satellite per year")
        fig.colorbar(mesh, ax=ax, label="collisions / sat / yr")
        fig.tight_layout()
        fig.savefig(f"{od}/collision_prob_{S}.png", dpi=130)
        plt.close(fig)
    print(f"Wrote figures to {od}/")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run(cfg):
    """Build, launch, run; RETURN the in-memory time series (writes no files)."""
    model = build_model(cfg)
    set_launches(model, cfg)
    model.run_model()
    return collect_results(model.scenario_properties, cfg.get("collision_avoidance", True))