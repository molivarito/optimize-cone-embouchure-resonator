# blueopt_search.py
from __future__ import annotations
import json
import time
from typing import Dict, List, Tuple
import numpy as np
from scipy import optimize
from scipy.stats import qmc

from blueopt_objective import BlueOptSpace, evaluar_candidato

def sobol_samples(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Muestras Sobol en [0,1]^d."""
    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    # SciPy Sobol requiere potencia de 2 para balance; usamos next_pow2>=n
    m = int(np.ceil(np.log2(max(2, n))))
    u = eng.random_base2(m=m)
    if u.shape[0] > n:
        u = u[:n]
    return u  # (n,d)

def run_global_then_local(cfg: Dict,
                          n_samples: int = 200,
                          top_k: int = 5,
                          seed: int = 42,
                          scan_steps: int = 10,
                          L_min_frac: float = 0.50) -> Dict:
    """
    Estrategia NUEVA:
      1) Global: Sobol n_samples
      2) Local: Powell desde los top_k
    """
    # Espacio
    space = BlueOptSpace.from_json(cfg)
    weights = cfg.get("weights", {"w_mag":1.0, "w_var":0.5, "w_max":0.5, "w_miss":50.0})
    temperatura = float(cfg.get("fixed", {}).get("temperatura", 20.0))

    # Muestras Sobol y escala a bounds
    u = sobol_samples(n_samples, d=len(space.var_names), seed=seed)
    X0 = np.vstack([space.scale01_to_bounds(ui) for ui in u])

    # Evaluación global
    results = []
    t0 = time.time()
    for i, x in enumerate(X0):
        r = evaluar_candidato(space, x, weights, scan_steps=scan_steps,
                              L_min_frac=L_min_frac, temperatura=temperatura)
        results.append({"x": x, "eval": r})
    t1 = time.time()

    # Orden por costo
    feasible = [r for r in results if r["eval"]["ok"]]
    if not feasible:
        return {"ok": False, "reason": "no_feasible_global"}

    feasible.sort(key=lambda rr: rr["eval"]["cost"])
    shortlist = feasible[:min(top_k, len(feasible))]

    # Refinamiento local (Powell) con bounds
    bounds = [(lo, hi) for (lo, hi) in space.bounds]

    def f_obj(x: np.ndarray) -> float:
        rr = evaluar_candidato(space, x, weights, scan_steps=scan_steps,
                               L_min_frac=L_min_frac, temperatura=temperatura)
        return float(rr["cost"])

    best_local = None
    for j, cand in enumerate(shortlist):
        x0 = cand["x"]
        res = optimize.minimize(f_obj, x0, method="Powell", bounds=bounds,
                                options={"maxiter": 250, "xtol": 1e-3, "ftol": 1e-3, "disp": False})
        val = f_obj(res.x)
        pack = {"x": res.x, "cost": val, "res": res}
        if (best_local is None) or (val < best_local["cost"]):
            best_local = pack

    # Armar salida con el mejor global y el mejor local
    best_global = feasible[0]
    out = {
        "ok": True,
        "space": {
            "var_names": space.var_names,
            "bounds": space.bounds,
            "fixed": space.fixed,
        },
        "global": {
            "best_x": best_global["x"].tolist(),
            "best_cost": float(best_global["eval"]["cost"]),
            "elapsed_s": float(t1 - t0),
        },
        "local": {
            "best_x": best_local["x"].tolist(),
            "best_cost": float(best_local["cost"]),
        },
    }
    # Evaluación completa del mejor local para reportar curvas
    best_eval = evaluar_candidato(space, np.array(out["local"]["best_x"]), weights,
                                  scan_steps=scan_steps, L_min_frac=L_min_frac,
                                  temperatura=temperatura)
    out["best_eval"] = best_eval
    return out