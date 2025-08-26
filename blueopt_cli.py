# blueopt_cli.py
from __future__ import annotations
import argparse
import json
import csv
from pathlib import Path
import numpy as np

from blueopt_search import run_global_then_local

def main():
    ap = argparse.ArgumentParser(description="BlueOpt (curva azul) optimizer")
    ap.add_argument("--config", required=True, help="Ruta a blueopt_config.json")
    ap.add_argument("--samples", type=int, default=200, help="N de muestras Sobol")
    ap.add_argument("--topk", type=int, default=5, help="Top-k para refinamiento local")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=10, help="Pasos de escaneo (curva azul)")
    ap.add_argument("--lmin", type=float, default=0.50, help="L_min_frac (0.50 por defecto)")
    ap.add_argument("--outdir", default="blueopt_out", help="Carpeta de salida")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = run_global_then_local(cfg, n_samples=args.samples, top_k=args.topk,
                                seed=args.seed, scan_steps=args.steps, L_min_frac=args.lmin)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Guardar resumen JSON
    (outdir/"summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Exportar curva Δ2 del mejor
    best = out.get("best_eval", {})
    deltas = best.get("deltas2", [])
    csv_path = outdir/"best_curve_d2.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "delta2_cents"])
        for i, d in enumerate(deltas):
            w.writerow([i, d])

    print(f"[BlueOpt] listo.\n- Resumen: {outdir/'summary.json'}\n- Curva Δ2: {csv_path}")

if __name__ == "__main__":
    main()