# workers.py
# -*- coding: utf-8 -*-
"""
Workers en segundo plano para:
- Escaneo por longitud (ScanWorker)
- Optimización de la curva azul (BlueOptWorker)

Compatible con PyQt6. No depende de matplotlib ni de la GUI.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Tuple, Any

from PyQt6.QtCore import QObject, pyqtSignal
from scipy import optimize
from scipy.stats import qmc

# Motor de escaneo (tu módulo existente)
from escaneo_longitud import escanear_por_longitud


# ------------------------------------------------------------
# ScanWorker: dispara el escaneo por longitud
# ------------------------------------------------------------
class ScanWorker(QObject):
    """
    Ejecuta escanear_por_longitud(**kwargs) en hilo.
    Señales:
      - finished(list): lista de diccionarios con resultados del escaneo
      - error(str): mensaje de error
      - progress(int): reservado (0–100) si en el futuro queremos granularidad
    """
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, kwargs_escanear: Dict[str, Any]):
        super().__init__()
        self.kwargs = dict(kwargs_escanear)

    def run(self) -> None:
        try:
            res = escanear_por_longitud(**self.kwargs)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


# ------------------------------------------------------------
# BlueOptWorker: búsqueda global + refinamiento local
# ------------------------------------------------------------
class BlueOptWorker(QObject):
    """
    Optimiza la inarmonicidad del segundo armónico (Δ2) a través de:
      1) Muestreo global Sobol en el espacio de variables.
      2) Refinamiento local (Powell) desde el mejor candidato.

    Señales:
      - finished(dict): resumen con mejor solución y evaluación final
      - error(str): mensaje de error si algo falla
      - progress_global(int, int): (i, total) avance del muestreo Sobol
      - progress_local(int, int): (eval_count, eval_budget_estimado)
      - message(str): mensajes informativos (reservado)
      - progress_snapshot(dict): {'d2': list, 'geom': {...}} para vista en vivo
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress_global = pyqtSignal(int, int)
    progress_local = pyqtSignal(int, int)
    message = pyqtSignal(str)
    progress_snapshot = pyqtSignal(dict)

    def __init__(
        self,
        space_cfg: dict,
        weights: dict,
        scan_steps: int,
        L_min_frac: float,
        temperatura: float,
        n_samples: int,
        top_k: int,
        seed: int,
        maxiter_local: int
    ):
        super().__init__()
        self.space_cfg = dict(space_cfg)
        self.weights = dict(weights)
        self.scan_steps = int(scan_steps)
        self.L_min_frac = float(L_min_frac)
        self.temperatura = float(temperatura)
        self.n_samples = int(n_samples)
        self.top_k = int(top_k)
        self.seed = int(seed)
        self.maxiter_local = int(maxiter_local)

        self._last_emit = 0.0
        self._eval_counter = 0

    # ---------------- Utilidades internas ----------------
    @staticmethod
    def _sobol(n: int, d: int, seed: int = 42) -> np.ndarray:
        """Matriz (n, d) en [0,1]^d con secuencia de Sobol (scrambled)."""
        eng = qmc.Sobol(d=d, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(2, n))))
        u = eng.random_base2(m=m)
        return u[:n]

    def _space_from_cfg(self) -> Tuple[List[str], List[Tuple[float, float]], dict]:
        """
        Devuelve:
          - var_names: nombres de variables
          - bounds: lista de (lo, hi)
          - fixed: dict de parámetros fijos (p.ej. temperatura, L_total_fixed)
        """
        vars_cfg = self.space_cfg.get("variables", {})
        var_names = list(vars_cfg.keys())
        bounds = [tuple(vars_cfg[n]) for n in var_names]
        fixed = dict(self.space_cfg.get("fixed", {}))
        return var_names, bounds, fixed

    @staticmethod
    def _scale01_to_bounds(u: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        return lo + u * (hi - lo)

    # ---------------- Núcleo de evaluación ----------------
    def _evaluar_candidato(self, var_names: List[str], fixed: dict, x: np.ndarray) -> dict:
        """
        Construye la geometría a partir de (x), ejecuta escaneo por longitud
        y devuelve métrica de costo + detalles (incluyendo Δ2).
        """
        # 1) Mezcla de variables + fijos
        params = {n: float(v) for n, v in zip(var_names, x)}
        P = dict(fixed)
        P.update(params)

        # Si se fija L_total desde la GUI:
        if "L_total_fixed" in fixed:
            P["L_total"] = float(fixed["L_total_fixed"])

        # 2) Geometría preferente: L_total conocido; r3_len derivado
        r1_len = P.get("r1_longitud")
        r1_rad = P.get("r1_radio")
        r3_rad_out = P.get("radio_cono_final")
        L_total = P.get("L_total", None)
        r3_len = P.get("r3_longitud", None)

        if L_total is None and r3_len is not None and r1_len is not None:
            L_total = float(r1_len) + float(r3_len)
        if r3_len is None and L_total is not None and r1_len is not None:
            r3_len = float(L_total) - float(r1_len)

        if any(v is None for v in [r1_len, r1_rad, r3_len, r3_rad_out]):
            return {"ok": False, "cost": 1e9, "reason": "faltan parametros"}

        # Normaliza tipos
        r1_len = float(r1_len)
        r1_rad = float(r1_rad)
        r3_len = float(r3_len)
        r3_rad_out = float(r3_rad_out)
        if L_total is None:
            L_total = r1_len + r3_len
        else:
            L_total = float(L_total)

        # Validación geométrica
        if (L_total <= r1_len) or (r1_len <= 0) or (r1_rad <= 0) or (r3_len <= 0) or (r3_rad_out <= 0):
            return {"ok": False, "cost": 1e9, "reason": "geom invalida"}

        # Segmentos (cilindro + cono)
        geometria = [
            [0.0, r1_len, r1_rad, r1_rad, 'cone'],
            [r1_len, L_total, r1_rad, r3_rad_out, 'cone'],
        ]

        # Embocadura
        h_pos = float(P.get("agujero_posicion", 0.02))
        h_in = float(P.get("agujero_radio_in", 0.005))
        h_out = float(P.get("agujero_radio_out", 0.005))
        h_len = float(P.get("agujero_largo", 0.002))
        agujeros = [
            ['label', 'position', 'radius', 'length', 'radius_out'],
            ['embocadura', h_pos, h_in, h_len, h_out]
        ]

        # 3) Simulación (escaneo de longitud)
        try:
            resultados = escanear_por_longitud(
                geometria_base=geometria,
                agujeros=agujeros,
                L_full=L_total,
                L_min_frac=self.L_min_frac,
                n_steps=self.scan_steps,
                temperatura=self.temperatura
            )
        except Exception as e:
            return {"ok": False, "cost": 1e9, "reason": f"sim_error: {e}"}

        # 4) Costo en base a Δ2
        deltas2 = np.array([r.get("delta_oct", np.nan) for r in resultados], dtype=float)
        valid = np.isfinite(deltas2)
        n_total = deltas2.size
        n_valid = int(valid.sum())

        miss_lambda = float(self.weights.get("w_miss", 50.0))
        J_miss = miss_lambda * (n_total - n_valid) / max(1, n_total)

        if n_valid < max(3, 0.5 * n_total):
            return {"ok": False, "cost": 1e6 + J_miss, "reason": "pocos puntos"}

        d = deltas2[valid]
        J_mag = float(np.median(np.abs(d)))
        J_var = float(np.std(d))
        J_max = float(np.percentile(np.abs(d), 95))

        cost = float(
            self.weights.get("w_mag", 1.0) * J_mag +
            self.weights.get("w_var", 0.5) * J_var +
            self.weights.get("w_max", 0.5) * J_max +
            J_miss
        )

        return {
            "ok": True,
            "cost": cost,
            "params": params,
            "fixed": fixed,
            "deltas2": deltas2.tolist(),
            "L_total": float(L_total),
            "r3_longitud": float(r3_len)
        }

    # ---------------- Bucle principal ----------------
    def run(self) -> None:
        try:
            var_names, bounds, fixed = self._space_from_cfg()
            d = len(var_names)

            # Presupuesto estimado de evaluaciones para la barra local
            local_max_evals = int(self.maxiter_local * (2 * d + 5))
            if local_max_evals <= 0:
                local_max_evals = 100
            self._eval_counter = 0

            # 1) Muestreo global Sobol
            U = self._sobol(self.n_samples, d, seed=self.seed)
            X0 = np.vstack([self._scale01_to_bounds(u, bounds) for u in U])

            evals = []
            best_cost_so_far = np.inf

            for i, x in enumerate(X0, start=1):
                r = self._evaluar_candidato(var_names, fixed, x)
                evals.append({"x": x, "eval": r})
                self.progress_global.emit(i, self.n_samples)

                # Snapshot live del mejor (throttling 0.5 s)
                now = time.time()
                if r.get("ok") and r["cost"] <= best_cost_so_far - 1e-9:
                    best_cost_so_far = r["cost"]
                    if now - self._last_emit > 0.5:
                        try:
                            self.progress_snapshot.emit({
                                "d2": r.get("deltas2", []),
                                "geom": {
                                    "L_total": r.get("L_total"),
                                    "r1_longitud": r.get("params", {}).get("r1_longitud"),
                                    "r1_radio": r.get("params", {}).get("r1_radio"),
                                    "r3_longitud": r.get("r3_longitud"),
                                    "radio_cono_final": r.get("params", {}).get("radio_cono_final"),
                                    "agujero_posicion": r.get("params", {}).get("agujero_posicion"),
                                    "agujero_radio_in": r.get("params", {}).get("agujero_radio_in"),
                                    "agujero_radio_out": r.get("params", {}).get("agujero_radio_out"),
                                    "agujero_largo": r.get("params", {}).get("agujero_largo"),
                                }
                            })
                        except Exception:
                            pass
                        self._last_emit = now

            feasible = [e for e in evals if e["eval"].get("ok")]
            if not feasible:
                self.error.emit("No hubo candidatos factibles en la fase global.")
                return

            feasible.sort(key=lambda e: e["eval"]["cost"])
            shortlist = feasible[:min(self.top_k, len(feasible))]

            # 2) Refinamiento local (Powell) — desde el mejor global
            bounds_tuple = [(b[0], b[1]) for b in bounds]
            best = shortlist[0]
            x0 = best["x"]
            it_counter = {"k": 0}

            def _cb(_xk):
                it_counter["k"] += 1
                self.progress_local.emit(it_counter["k"], self.maxiter_local)

            def _f_obj(x: np.ndarray) -> float:
                rr = self._evaluar_candidato(var_names, fixed, x)
                self._eval_counter += 1
                self.progress_local.emit(min(self._eval_counter, local_max_evals), local_max_evals)
                return float(rr["cost"]) if rr.get("ok") else float(rr.get("cost", 1e9)) + 1e3

            res = optimize.minimize(
                _f_obj, x0, method="Powell", bounds=bounds_tuple,
                options={"maxiter": self.maxiter_local, "xtol": 1e-3, "ftol": 1e-3, "disp": False},
                callback=_cb
            )

            # 3) Evaluación final y salida
            final_eval = self._evaluar_candidato(var_names, fixed, res.x)
            out = {
                "space": {"var_names": var_names, "bounds": bounds, "fixed": fixed},
                "global_best": {"x": best["x"].tolist(), "cost": best["eval"]["cost"]},
                "local_best": {"x": res.x.tolist(), "cost": float(final_eval.get("cost", np.nan))},
                "best_eval": final_eval,
            }
            self.finished.emit(out)

        except Exception as e:
            self.error.emit(str(e))