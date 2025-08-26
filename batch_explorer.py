#!/usr/bin/env python3
import os, sys, json, math, time, hashlib, signal, uuid
import numpy as np
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

# opcionales (para parquet/csv cómodo)
try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

from joblib import Parallel, delayed
from escaneo_longitud import escanear_por_longitud

# ------------------------ utilidades ------------------------
def sha1_of_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def linspace_inclusive(start, stop, steps):
    """Soporta start<=stop y start>stop. Devuelve 'steps' puntos incluyendo extremos."""
    start = float(start); stop = float(stop); steps = int(steps)
    if steps <= 1:
        return np.array([start], dtype=float)
    arr = np.linspace(start, stop, steps, dtype=float)
    return arr

def build_grid(variables: dict):
    """
    Genera dicts de parámetros para todas las combinaciones.
    Acepta rangos descendentes (p.ej. 0.017 → 0.0019).
    """
    keys = list(variables.keys())
    axes = []
    for k in keys:
        spec = variables[k]
        arr = linspace_inclusive(spec["start"], spec["stop"], spec["steps"])
        axes.append(arr)
    for vals in product(*axes):
        yield dict(zip(keys, map(float, vals)))

# ------------------------ evaluación ------------------------
def evaluate_candidate(params: dict, model: dict, weights: dict):
    """
    Usa L_total y r1_longitud del 'params' y deduce r3_longitud = L_total - r1_longitud.
    Si r3_longitud<=0, marca costo alto y sigue.
    Robustez total: nunca levanta excepción.
    """
    try:
        # Geometría base
        L_total = float(params["L_total"])
        r1_len  = float(params["r1_longitud"])
        r1_rad  = float(params["r1_radio"])
        r3_out  = float(params["radio_cono_final"])

        r3_len = L_total - r1_len
        if not (L_total > r1_len > 0.0 and r1_rad > 0.0 and r3_out > 0.0 and r3_len > 0.0):
            # Geometría inválida → costo alto
            return {
                "cost": 1e9, "n_total": 0, "n_valid": 0,
                "J_mag": float('nan'), "J_var": float('nan'), "J_p95": float('nan'),
                "deltas2": np.array([], dtype=float)
            }, []

        geometria_base = [
            [0.0, r1_len, r1_rad, r1_rad, 'cone'],
            [r1_len, L_total, r1_rad, r3_out, 'cone'],
        ]

        # Embocadura
        h_pos = float(params["agujero_posicion"])
        h_in  = float(params["agujero_radio_in"])
        h_out = float(params["agujero_radio_out"])
        h_len = float(params["agujero_largo"])
        agujeros = [
            ['label','position','radius','length','radius_out'],
            ['embocadura', h_pos, h_in, h_len, h_out]
        ]

        # Escaneo robusto
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
            # candidato entero falló → costo alto
            return {
                "cost": 1e9, "n_total": 0, "n_valid": 0,
                "J_mag": float('nan'), "J_var": float('nan'), "J_p95": float('nan'),
                "deltas2": np.array([], dtype=float)
            }, []

        # Objetivo sobre Δ2
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
            "J_mag": float(np.median(np.abs(deltas2[valid]))) if n_valid>0 else float('nan'),
            "J_var": float(np.std(deltas2[valid])) if n_valid>0 else float('nan'),
            "J_p95": float(np.percentile(np.abs(deltas2[valid]), 95)) if n_valid>0 else float('nan'),
            "deltas2": deltas2
        }
        return out, resultados

    except Exception:
        # cualquier otro fallo inesperado → costo alto y seguimos
        return {
            "cost": 1e9, "n_total": 0, "n_valid": 0,
            "J_mag": float('nan'), "J_var": float('nan'), "J_p95": float('nan'),
            "deltas2": np.array([], dtype=float)
        }, []

# ------------------------ persistencia ------------------------
class Sink:
    def __init__(self, outdir: Path, fmt: str, save_curves: bool, flush_every: int, resume: bool):
        self.outdir = outdir; self.fmt = fmt.lower()
        self.save_curves = save_curves
        self.flush_every = int(flush_every)
        self.resume = resume
        self.rows = []
        self.curves_dir = self.outdir / "curves"
        self.index_path = self.outdir / "index.jsonl"
        self.meta_path = self.outdir / "META.json"
        self._stop = False
        self._seen = set()
        self._load_index_if_resume()

        os.makedirs(self.outdir, exist_ok=True)
        if self.save_curves:
            os.makedirs(self.curves_dir, exist_ok=True)

        # Ctrl+C
        signal.signal(signal.SIGINT, self._sigint_handler)

    def _sigint_handler(self, sig, frame):
        print("\n[batch] SIGINT recibido — guardando…")
        self._stop = True
        self.flush()

    def _load_index_if_resume(self):
        if not self.resume or not self.index_path.exists():
            return
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    self._seen.add(rec["geom_hash"])
                except Exception:
                    pass

    def already_done(self, geom_hash: str) -> bool:
        return geom_hash in self._seen

    def add(self, meta: dict, params: dict, eval_out: dict, resultados: list):
        geom_hash = meta["geom_hash"]
        self._seen.add(geom_hash)

        curve_path = None
        if self.save_curves and eval_out.get("deltas2") is not None:
            curve_path = self.curves_dir / f"{geom_hash}.npy"
            np.save(curve_path, np.array(eval_out["deltas2"], dtype=np.float32))

        row = {
            "run_id": meta["run_id"],
            "ts": meta["ts"],
            "geom_hash": geom_hash,
            **params,
            "cost": float(eval_out["cost"]),
            "n_total": int(eval_out["n_total"]),
            "n_valid": int(eval_out["n_valid"]),
            "J_mag": float(eval_out["J_mag"]),
            "J_var": float(eval_out["J_var"]),
            "J_p95": float(eval_out["J_p95"]),
            "curve_path": str(curve_path) if curve_path else ""
        }
        self.rows.append(row)

        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"geom_hash": geom_hash}) + "\n")

        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.rows:
            return

        if not HAVE_PANDAS:
            # CSV simple
            csv_path = self.outdir / "results.csv"
            write_header = not csv_path.exists()
            import csv
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                if write_header: w.writeheader()
                for r in self.rows: w.writerow(r)
            self.rows = []
            return

        df = pd.DataFrame(self.rows)
        self.rows = []
        if self.fmt == "parquet" and HAVE_PARQUET:
            pq_path = self.outdir / "results.parquet"
            if pq_path.exists():
                existing = pq.read_table(pq_path).to_pandas()
                df = pd.concat([existing, df], ignore_index=True)
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, pq_path)
        else:
            csv_path = self.outdir / "results.csv"
            if csv_path.exists():
                old = pd.read_csv(csv_path)
                df = pd.concat([old, df], ignore_index=True)
            df.to_csv(csv_path, index=False)

    def write_meta(self, meta_obj: dict):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, indent=2, ensure_ascii=False)

# ------------------------ main ------------------------
def main():
    if len(sys.argv) < 2:
        print("Uso: python batch_explorer.py sweep_config.json")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as e:
        print(f"[batch] No pude leer el JSON '{cfg_path}': {e}")
        sys.exit(1)

    meta = cfg.get("meta", {})
    model = cfg.get("model", {})
    variables = cfg.get("variables", {})
    weights = cfg.get("objective", {}).get("weights", {})
    runtime = cfg.get("runtime", {})

    outdir = Path(runtime.get("output_dir", "batch_results"))
    fmt = runtime.get("format", "parquet")
    save_curves = bool(runtime.get("save_curves", True))
    flush_every = int(runtime.get("flush_every", 50))
    resume = bool(runtime.get("resume", True))
    n_jobs = int(runtime.get("n_jobs", 1))
    use_tqdm = bool(runtime.get("use_tqdm", True))
    progress_every = int(runtime.get("progress_every", 50))

    outdir.mkdir(parents=True, exist_ok=True)

    # expand grid
    grid_iter = list(build_grid(variables))
    total = len(grid_iter)
    print(f"[batch] Candidatos totales: {total}")

    # sink
    run_id = f"{meta.get('name','run')}-{uuid.uuid4().hex[:8]}"
    sink = Sink(outdir, fmt, save_curves, flush_every, resume)
    sink.write_meta({
        "run_id": run_id,
        "config": cfg,
        "created_at": datetime.now(timezone.utc).isoformat()
    })

    def _task(params, i=None):
        geom_hash = sha1_of_dict(params)
        if sink.already_done(geom_hash):
            return None  # skip
        try:
            eval_out, resultados = evaluate_candidate(params, model, weights)
        except Exception:
            eval_out, resultados = {
                "cost": 1e9, "n_total": 0, "n_valid": 0,
                "J_mag": float('nan'), "J_var": float('nan'), "J_p95": float('nan'),
                "deltas2": np.array([], dtype=float)
            }, []
        sink.add({
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "geom_hash": geom_hash
        }, params, eval_out, resultados)
        return True

    if n_jobs == 1:
        iterator = enumerate(grid_iter, 1)
        if use_tqdm and HAVE_TQDM:
            iterator = tqdm(iterator, total=total, desc="[batch] exploración", unit="cand")
        for i, p in iterator:
            if sink._stop: break
            _task(p, i=i)
            if not (use_tqdm and HAVE_TQDM) and (i % progress_every == 0):
                print(f"[batch] {i}/{total}…")
        sink.flush()
    else:
        # multiproceso
        Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
            delayed(_task)(p) for p in grid_iter
        )
        sink.flush()

    print("[batch] Listo.")

if __name__ == "__main__":
    main()