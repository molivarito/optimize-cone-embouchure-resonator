#!/usr/bin/env python3
import os, sys, json, time, uuid, hashlib
from pathlib import Path
from datetime import datetime, timezone
from skopt.space import Real, Categorical
import numpy as np

# opcionales
try:
    from tqdm import trange
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

# scikit-optimize
HAVE_SKOPT = True
try:
    from skopt import Optimizer
    from skopt.space import Real
except Exception:
    HAVE_SKOPT = False

# fallback local
from scipy import optimize
from scipy.stats import qmc

from escaneo_longitud import escanear_por_longitud

def sha1_of_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

# ---------------- evaluación (misma métrica del batch) ----------------
def evaluate_candidate(params: dict, model: dict, weights: dict):
    """
    Usa L_total y r1_longitud, deduce r3_longitud = L_total - r1_longitud.
    Robusto: nunca levanta excepción; si algo falla → costo alto.
    Devuelve (eval_out: dict, resultados: list).
    """
    try:
        L_total = float(params["L_total"])
        r1_len  = float(params["r1_longitud"])
        r1_rad  = float(params["r1_radio"])
        r3_out  = float(params["radio_cono_final"])

        r3_len = L_total - r1_len

        # geometría inválida → costo alto
        if not (L_total > r1_len > 0.0 and r1_rad > 0.0 and r3_out > 0.0 and r3_len > 0.0):
            return {
                "cost": 1e9, "n_total": 0, "n_valid": 0,
                "J_mag": float('nan'), "J_var": float('nan'), "J_p95": float('nan'),
                "deltas2": np.array([], dtype=float)
            }, []

        geometria_base = [
            [0.0, r1_len, r1_rad, r1_rad, 'cone'],
            [r1_len, L_total, r1_rad, r3_out, 'cone'],
        ]

        h_pos = float(params["agujero_posicion"])
        h_in  = float(params["agujero_radio_in"])
        h_out = float(params["agujero_radio_out"])
        h_len = float(params["agujero_largo"])

        agujeros = [
            ['label','position','radius','length','radius_out'],
            ['embocadura', h_pos, h_in, h_len, h_out]
        ]

        try:
            resultados = escanear_por_longitud(
                geometria_base=geometria_base,
                agujeros=agujeros,
                L_full=L_total,
                L_min_frac=float(model["L_min_frac"]),
                n_steps=int(model["scan_steps"]),
                temperatura=float(model["temperature_C"])
            )
        except Exception:
            return {
                "cost": 1e9, "n_total": 0, "n_valid": 0,
                "J_mag": float('nan'), "J_var": float('nan'), "J_p95": float('nan'),
                "deltas2": np.array([], dtype=float)
            }, []

        deltas2 = np.array([r.get("delta_oct", np.nan) for r in resultados], dtype=float)
        valid = np.isfinite(deltas2)
        n_total = deltas2.size
        n_valid = int(valid.sum())
        miss_lambda = float(weights.get("w_miss", 50.0))
        J_miss = miss_lambda * (n_total - n_valid) / max(1, n_total)

        if n_valid < max(3, 0.5*n_total):
            cost = 1e6 + J_miss
        else:
            d = deltas2[valid]
            J_mag = float(np.median(np.abs(d)))
            J_var = float(np.std(d))
            J_max = float(np.percentile(np.abs(d), 95))
            cost = float(weights.get("w_mag",1.0)*J_mag +
                         weights.get("w_var",0.5)*J_var +
                         weights.get("w_max",0.5)*J_max + J_miss)

        out = {
            "cost": float(cost),
            "n_total": int(n_total),
            "n_valid": int(n_valid),
            "J_mag": float(np.median(np.abs(deltas2[valid])) if n_valid>0 else np.nan),
            "J_var": float(np.std(deltas2[valid]) if n_valid>0 else np.nan),
            "J_p95": float(np.percentile(np.abs(deltas2[valid]), 95) if n_valid>0 else np.nan),
            "deltas2": deltas2
        }
        return out, resultados
    except Exception:
        return {
            "cost": 1e9, "n_total": 0, "n_valid": 0,
            "J_mag": float('nan'), "J_var": float('nan'), "J_p95": float('nan'),
            "deltas2": np.array([], dtype=float)
        }, []

# ---------------- util persistencia ----------------
def write_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_jsonl(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------- optimizador bayesiano ----------------
def build_space(space_cfg: dict):
    """
    Construye dimensiones para skopt.
    - Si low == high -> parámetro fijo usando Categorical([valor]).
    - Si low < high  -> parámetro continuo usando Real(low, high).
    """
    dims = []
    keys = []
    for k, (lo, hi) in space_cfg.items():
        lo = float(lo); hi = float(hi)
        keys.append(k)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Espacio inválido para '{k}': [{lo}, {hi}]")
        if hi <= lo:  # incluye el caso hi == lo (fijo)
            # Parámetro fijo
            dims.append(Categorical([lo], name=k))
        else:
            dims.append(Real(lo, hi, name=k, prior="uniform"))
    return keys, dims

def suggest_random(space_cfg: dict, n=1, seed=42):
    rng = np.random.default_rng(seed)
    keys = list(space_cfg.keys())
    X = []
    for _ in range(n):
        x = {}
        for k,(lo,hi) in space_cfg.items():
            lo = float(lo); hi=float(hi)
            val = lo if abs(hi-lo)<1e-15 else float(rng.uniform(lo,hi))
            x[k]=val
        X.append(x)
    return X

def run_bayes(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    meta = cfg.get("meta", {})
    model = cfg.get("model", {})
    space_cfg = cfg.get("space", {})
    weights = cfg.get("objective", {}).get("weights", {})
    runtime = cfg.get("runtime", {})

    outdir = Path(runtime.get("output_dir", "bayes_results"))
    outdir.mkdir(parents=True, exist_ok=True)

    seed = int(runtime.get("seed", 42))
    rnd_starts = int(runtime.get("random_starts", 20))
    n_calls = int(runtime.get("n_calls", 200))
    use_tqdm = bool(runtime.get("use_tqdm", True))
    ckpt_every = int(runtime.get("checkpoint_every", 5))
    resume_from = runtime.get("resume_from", "").strip()
    save_best_curve = bool(runtime.get("save_best_curve", True))

    log_jsonl = outdir / "trace.jsonl"
    best_json = outdir / "best.json"
    meta_json = outdir / "META.json"

    write_json(meta_json, {
        "run_id": f"{meta.get('name','bayes')}-{uuid.uuid4().hex[:8]}",
        "config": cfg,
        "created_at": datetime.now(timezone.utc).isoformat()
    })

    # Espacio
    keys, dims = build_space(space_cfg)

    # Estado para resume
    X_prev = []
    y_prev = []

    if resume_from:
        prev = Path(resume_from)
        if prev.exists():
            # cargar puntos previos
            with open(prev, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        x = [float(rec["params"][k]) for k in keys]
                        y = float(rec["eval"]["cost"])
                        X_prev.append(x); y_prev.append(y)
                    except Exception:
                        pass
            print(f"[bayes] Resume: cargados {len(X_prev)} puntos de {prev.name}")

    # Inicializar optimizador
    if HAVE_SKOPT:
        opt = Optimizer(
            dimensions=dims,
            base_estimator="GP",
            acq_func="EI",
            acq_optimizer="auto",
            random_state=seed
        )
        if X_prev:
            opt.tell(X_prev, y_prev)
    else:
        print("[bayes] skopt no disponible — fallback a random+Powell local.")

    # Bucle de optimización
    it_range = trange(n_calls, desc="[bayes] BO", unit="it") if (use_tqdm and HAVE_TQDM) else range(n_calls)
    best = {"cost": float("inf"), "params": None, "eval": None}

    # Semillas aleatorias reproducibles
    rng = np.random.default_rng(seed)

    for it in it_range:
        # Sugerencia
        if HAVE_SKOPT:
            x_suggest = opt.ask()  # lista de valores en orden de keys
            params = {k: float(v) for k,v in zip(keys, x_suggest)}
        else:
            params = suggest_random(space_cfg, 1, seed + it)[0]

        # Evaluar
        eval_out, resultados = evaluate_candidate(params, model, weights)
        y = float(eval_out["cost"])

        # Actualizar BO
        if HAVE_SKOPT:
            opt.tell([params[k] for k in keys], y)

        # Guardar traza
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "iter": it,
            "params": params,
            "eval": {
                "cost": y,
                "n_total": int(eval_out["n_total"]),
                "n_valid": int(eval_out["n_valid"]),
                "J_mag": float(eval_out["J_mag"]),
                "J_var": float(eval_out["J_var"]),
                "J_p95": float(eval_out["J_p95"])
            }
        }
        append_jsonl(log_jsonl, rec)

        # Mejor hasta ahora
        if y < best["cost"]:
            best = {"cost": y, "params": params, "eval": eval_out}
            write_json(best_json, {
                "best_cost": float(y),
                "best_params": params,
                "best_eval": {
                    k: (float(v) if isinstance(v, (int,float,np.floating)) else
                        (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v))
                    for k,v in eval_out.items()
                }
            })
            # curva Δ2
            if save_best_curve and eval_out.get("deltas2") is not None:
                np.save(outdir / "best_d2.npy", np.asarray(eval_out["deltas2"], dtype=np.float32))

        # Checkpoint periódico (ya se escribe trace.jsonl continuo)
        if ckpt_every > 0 and (it+1) % ckpt_every == 0:
            # nada extra que hacer; trace.jsonl/best.json ya están al día
            pass

    print("[bayes] Listo. Best guardado en:", best_json)

def run_fallback_random_powell(cfg_path: Path):
    # Usado solo si no tenemos skopt; hace random search + Powell local en el mejor.
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    meta = cfg.get("meta", {})
    model = cfg.get("model", {})
    space_cfg = cfg.get("space", {})
    weights = cfg.get("objective", {}).get("weights", {})
    runtime = cfg.get("runtime", {})

    outdir = Path(runtime.get("output_dir", "bayes_results_fallback"))
    outdir.mkdir(parents=True, exist_ok=True)

    seed = int(runtime.get("seed", 42))
    rnd_starts = int(runtime.get("random_starts", 40))
    n_calls = int(runtime.get("n_calls", 200))
    use_tqdm = bool(runtime.get("use_tqdm", True))

    log_jsonl = outdir / "trace.jsonl"
    best_json = outdir / "best.json"

    keys = list(space_cfg.keys())
    lows = np.array([float(space_cfg[k][0]) for k in keys], dtype=float)
    highs= np.array([float(space_cfg[k][1]) for k in keys], dtype=float)

    def sample(rng):
        x = lows + rng.random(len(keys))*(highs - lows)
        return {k: float(v) for k,v in zip(keys, x)}

    rng = np.random.default_rng(seed)
    it_range = trange(n_calls, desc="[fallback] random", unit="it") if (use_tqdm and HAVE_TQDM) else range(n_calls)
    best = {"cost": float('inf'), "params": None, "eval": None}

    for it in it_range:
        params = sample(rng)
        eval_out, _ = evaluate_candidate(params, model, weights)
        y = float(eval_out["cost"])
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "iter": it,
            "params": params,
            "eval": {"cost": y}
        }
        append_jsonl(log_jsonl, rec)
        if y < best["cost"]:
            best = {"cost": y, "params": params, "eval": eval_out}
            write_json(best_json, {"best_cost": y, "best_params": params})

    # Powell local (si hay algo razonable)
    if best["params"] is not None:
        x0 = np.array([best["params"][k] for k in keys], dtype=float)
        bounds = [(float(space_cfg[k][0]), float(space_cfg[k][1])) for k in keys]
        def f_obj(x):
            p = {k: float(v) for k,v in zip(keys, x)}
            r, _ = evaluate_candidate(p, model, weights)
            return float(r["cost"])
        res = optimize.minimize(f_obj, x0, method="Powell", bounds=bounds,
                                options={"maxiter": 200, "xtol": 1e-3, "ftol": 1e-3, "disp": False})
        rfin, _ = evaluate_candidate({k: float(v) for k,v in zip(keys, res.x)}, model, weights)
        write_json(outdir / "best_local.json", {
            "best_cost_local": float(rfin["cost"]),
            "best_params_local": {k: float(v) for k,v in zip(keys, res.x)}
        })
    print("[fallback] Listo.")

def main():
    if len(sys.argv) < 2:
        print("Uso: python bayes_explorer.py bayes_config.json")
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    if HAVE_SKOPT:
        run_bayes(cfg_path)
    else:
        run_fallback_random_powell(cfg_path)

if __name__ == "__main__":
    main()