# blueopt_objective.py
from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from escaneo_longitud import escanear_por_longitud  # usamos tu módulo

# ---------------- Config & mapping ----------------

@dataclass
class BlueOptSpace:
    """Espacio de búsqueda: nombres y bounds."""
    var_names: List[str]
    bounds: List[Tuple[float, float]]
    fixed: Dict[str, float]

    @classmethod
    def from_json(cls, cfg: Dict) -> "BlueOptSpace":
        vars_cfg = cfg["variables"]
        var_names = list(vars_cfg.keys())
        bounds = [tuple(vars_cfg[n]) for n in var_names]
        fixed = cfg.get("fixed", {})
        return cls(var_names=var_names, bounds=bounds, fixed=fixed)

    def scale01_to_bounds(self, u: np.ndarray) -> np.ndarray:
        """Escala un vector u∈[0,1]^d a los bounds reales."""
        lo = np.array([b[0] for b in self.bounds], dtype=float)
        hi = np.array([b[1] for b in self.bounds], dtype=float)
        return lo + u*(hi - lo)

    def vec_to_params(self, x: np.ndarray) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.var_names, x)}

# ---------------- Costo “curva azul” (Δ2) ----------------

def costo_curva_azul(deltas2: np.ndarray,
                     w_mag: float = 1.0,
                     w_var: float = 0.5,
                     w_max: float = 0.5,
                     miss_lambda: float = 50.0) -> float:
    """
    deltas2: array de Δ2 (cents), puede contener NaN si faltan picos.
    Penaliza magnitud, variación y outliers; y puntos faltantes.
    """
    deltas2 = np.asarray(deltas2, dtype=float)
    valid = np.isfinite(deltas2)
    n_total = deltas2.size
    n_valid = int(valid.sum())
    if n_total == 0:
        return 1e9

    # Penalización por faltantes
    J_miss = miss_lambda * (n_total - n_valid) / max(1, n_total)

    if n_valid < max(3, 0.5*n_total):
        # Muy inestable: devuelve gran costo
        return 1e6 + J_miss

    d = deltas2[valid]
    J_mag = np.median(np.abs(d))
    J_var = float(np.std(d))
    J_max = float(np.percentile(np.abs(d), 95))
    return w_mag*J_mag + w_var*J_var + w_max*J_max + J_miss

# ---------------- Evaluación de un candidato ----------------

def evaluar_candidato(space: BlueOptSpace,
                      x: np.ndarray,
                      weights: Dict[str, float],
                      scan_steps: int = 10,
                      L_min_frac: float = 0.50,
                      temperatura: float = 20.0) -> Dict:
    """
    Construye geometría, escanea 10 pasos y calcula costo sobre Δ2 (curva azul).
    Devuelve dict con costo, params, deltas y metadatos.
    """
    params = space.vec_to_params(x)
    params_all = dict(space.fixed)
    params_all.update(params)

    # Obtener r1_len y radios
    r1_len = params_all.get("r1_longitud")
    r1_rad = params_all.get("r1_radio")
    r3_len_param = params_all.get("r3_longitud")
    r3_rad_out = params_all.get("radio_cono_final")
    L_total = params_all.get("L_total")

    # Compatibilidad: si no hay L_total pero sí r3_len, calcular L_total
    if L_total is None and r3_len_param is not None and r1_len is not None:
        L_total = r1_len + r3_len_param
    # Si L_total está dado y r1_len está dado, deducir r3_len
    if L_total is not None and r1_len is not None:
        r3_len = L_total - r1_len
    elif r3_len_param is not None:
        r3_len = r3_len_param
    else:
        r3_len = None

    # Validaciones de geometría
    if any(v is None for v in [r1_len, r1_rad, r3_len, r3_rad_out]):
        return {"ok": False, "cost": 1e9, "reason": "faltan parámetros geométricos"}

    if r1_len <= 0 or r3_len <= 0 or r1_rad <= 0 or r3_rad_out <= 0:
        return {"ok": False, "cost": 1e9, "reason": "geometría inválida"}

    # Geometría y agujero (embocadura)
    geometria = [
        [0.0, r1_len,         r1_rad,       r1_rad,    'cone'],
        [r1_len, r1_len+r3_len, r1_rad,     r3_rad_out,'cone'],
    ]
    h_pos = params_all.get("agujero_posicion", 0.02)
    h_in  = params_all.get("agujero_radio_in", 0.005)
    h_out = params_all.get("agujero_radio_out", 0.005)
    h_len = params_all.get("agujero_largo", 0.002)
    agujeros = [
        ['label','position','radius','length','radius_out'],
        ['embocadura', h_pos, h_in, h_len, h_out]
    ]

    # Escaneo por longitud (usa tu módulo)
    try:
        resultados = escanear_por_longitud(
            geometria_base=geometria,
            agujeros=agujeros,
            L_full=L_total,
            L_min_frac=L_min_frac,
            n_steps=scan_steps,
            temperatura=temperatura
        )
    except Exception as e:
        return {"ok": False, "cost": 1e9, "reason": f"sim_error: {e}"}

    # Extraer Δ2 (delta_oct)
    deltas2 = np.array([r.get("delta_oct", np.nan) for r in resultados], dtype=float)

    # Costo
    cost = costo_curva_azul(
        deltas2,
        w_mag=weights.get("w_mag", 1.0),
        w_var=weights.get("w_var", 0.5),
        w_max=weights.get("w_max", 0.5),
        miss_lambda=weights.get("w_miss", 50.0),
    )

    return {
        "ok": True,
        "cost": float(cost),
        "params": params,
        "fixed": space.fixed,
        "L_total": float(L_total),
        "scan_steps": int(scan_steps),
        "L_min_frac": float(L_min_frac),
        "temperatura": float(temperatura),
        "deltas2": deltas2.tolist(),
    }