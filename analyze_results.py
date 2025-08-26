#!/usr/bin/env python3
import os, sys, json
from pathlib import Path
import numpy as np

try:
    import pandas as pd
except Exception as e:
    print("Este script requiere pandas.")
    sys.exit(1)

import matplotlib.pyplot as plt

def load_table(outdir: Path):
    pq = outdir / "results.parquet"
    csv = outdir / "results.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    elif csv.exists():
        return pd.read_csv(csv)
    else:
        raise FileNotFoundError("No se encontró results.parquet ni results.csv")

def main():
    if len(sys.argv) < 2:
        print("Uso: python analyze_results.py batch_results")
        sys.exit(1)
    outdir = Path(sys.argv[1])
    df = load_table(outdir)
    print(df.describe(numeric_only=True))

    # Top-10 por costo
    df2 = df.sort_values("cost", ascending=True).head(10)
    print("\nTop-10 candidatos por costo:\n", df2[["cost","r1_longitud","r1_radio","r3_longitud","radio_cono_final",
                                               "agujero_posicion","agujero_radio_in","agujero_radio_out","agujero_largo"]])

    # Histograma de costos
    plt.figure()
    df["cost"].plot.hist(bins=50)
    plt.title("Histograma de costos"); plt.xlabel("Costo"); plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

    # Curvas Δ2 de los mejores (si existen)
    curves_dir = outdir / "curves"
    if curves_dir.exists():
        for _, row in df2.iterrows():
            p = row.get("curve_path", "")
            if isinstance(p, str) and p and Path(p).exists():
                d2 = np.load(p)
                plt.plot(d2, "-o", label=f"cost={row['cost']:.3f}")
        if plt.gca().has_data():
            plt.axhline(0, color='k', lw=1)
            plt.title("Curvas Δ2 de los mejores")
            plt.xlabel("índice de paso (L_eff)")
            plt.ylabel("Δ2 (cents)")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.show()

    # Heatmap simple (ejemplo: r1_radio vs radio_cono_final)
    try:
        pivot = df.pivot_table(index="r1_radio", columns="radio_cono_final", values="cost", aggfunc="min")
        plt.figure()
        im = plt.imshow(pivot.values, origin="lower", aspect="auto")
        plt.colorbar(im, label="cost")
        plt.xticks(np.arange(pivot.shape[1]), [f"{c:.4f}" for c in pivot.columns], rotation=90)
        plt.yticks(np.arange(pivot.shape[0]), [f"{r:.4f}" for r in pivot.index])
        plt.title("Mapa de costos (r1_radio vs radio_cono_final)")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()