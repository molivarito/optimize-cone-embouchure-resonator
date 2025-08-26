# escaneo_longitud.py
import numpy as np
import json, hashlib, csv, datetime
from scipy.signal import find_peaks
from openwind import ImpedanceComputation, Player

# ------------ Utilidades ------------
def sha_config(obj)->str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]

def truncar_geometria(geom, L_eff):
    """
    geom: lista [[x0,x1,r0,r1,'cone'|'cyl'], ...] con x crecientes.
    Devuelve una nueva lista recortada en L_eff.
    Para tramos 'cone' se interpola linealmente el radio → **mantiene el ángulo pendiente**
    del cono al truncar; para 'cyl' se preserva r0 (radio constante).
    """
    out = []
    for (x0,x1,r0,r1,kind) in geom:
        if L_eff <= x0:
            break
        if L_eff >= x1:
            out.append([x0,x1,r0,r1,kind])
        else:
            # recorte dentro de este tramo
            # 'cone': interpolación lineal (ángulo constante)
            # 'cyl' : radio constante (r0)
            if kind == 'cyl':
                r_eff = r0
            else:  # default: 'cone' y cualquier otro "lineal"
                t = (L_eff - x0) / max(1e-12, (x1 - x0))
                r_eff = r0 + t*(r1 - r0)
            out.append([x0, L_eff, r0, r_eff, kind])
            break
    return out

def truncar_agujeros(holes, L_eff):
    """Devuelve una copia de la tabla de agujeros dejando solo los que están antes de L_eff.
    Se asume formato [[header...], [label, position, radius, length, radius_out], ...].
    """
    if not holes or len(holes) < 2:
        return holes
    header = holes[0]
    body = [row for row in holes[1:] if len(row) >= 3 and float(row[1]) < L_eff - 1e-9]
    return [header] + body

def prominencia_adaptativa(db_values):
    p05, p95 = np.percentile(db_values, [5, 95])
    return float(max(1.0, 0.1*abs(p95 - p05)))

def inharmonicidades(fm):
    """ fm: array de picos [f1,f2,f3,f4,...]; devuelve dict con Δoct, Δduo, Δ4 """
    out = {"delta_oct": np.nan, "delta_duo": np.nan, "delta_4": np.nan}
    if len(fm) >= 2 and fm[0] > 0 and fm[1] > 0:
        out["delta_oct"] = 1200*np.log2(fm[1]/(2*fm[0]))
    if len(fm) >= 3 and fm[0] > 0 and fm[2] > 0:
        out["delta_duo"] = 1200*np.log2(fm[2]/(3*fm[0]))
    if len(fm) >= 4 and fm[0] > 0 and fm[3] > 0:
        out["delta_4"]  = 1200*np.log2(fm[3]/(4*fm[0]))
    return out

# ------------ Núcleo de escaneo ------------
def escanear_por_longitud(geometria_base, agujeros, L_full, L_min_frac=0.5, n_steps=60,
                          frecuencias=None, temperatura=20.0,
                          radiation={'bell':'unflanged','entrance':'closed','holes':'unflanged'},
                          player_name="FLUTE"):
    if frecuencias is None:
        frecuencias = np.linspace(50, 4000, 2000)

    L_min = max(L_full*L_min_frac, 1e-3)
    L_grid = np.linspace(L_full, L_min, n_steps)

    player = Player(player_name)
    resultados = []  # lista de dicts (una fila por L_eff)

    for L_eff in L_grid:
        geom_L = truncar_geometria(geometria_base, L_eff)
        holes_L = truncar_agujeros(agujeros, L_eff)
        if not geom_L or geom_L[-1][1] <= geom_L[-1][0]:
            # geometría inválida tras recorte
            resultados.append({"L_eff": L_eff, "f1": np.nan, "f2": np.nan, "f3": np.nan, "f4": np.nan,
                               "delta_oct": np.nan, "delta_duo": np.nan, "delta_4": np.nan, "n_peaks": 0})
            continue

        sim = ImpedanceComputation(frecuencias, geom_L, holes_L,
                                   player=player, source_location='embocadura',
                                   radiation_category=radiation, temperature=temperatura,
                                   interp=True, interp_grid=1e-3)

        imp_db = 20*np.log10(np.abs(sim.impedance))
        prom = prominencia_adaptativa(imp_db)
        idx, _ = find_peaks(-imp_db, prominence=prom, distance=10)
        fm = frecuencias[idx]
        f1 = fm[0] if len(fm)>=1 else np.nan
        f2 = fm[1] if len(fm)>=2 else np.nan
        f3 = fm[2] if len(fm)>=3 else np.nan
        f4 = fm[3] if len(fm)>=4 else np.nan
        deltas = inharmonicidades(fm)

        resultados.append({"L_eff": L_eff, "f1": f1, "f2": f2, "f3": f3, "f4": f4,
                           **deltas, "n_peaks": len(fm)})
    return resultados

def guardar_csv_y_plot(resultados, ruta_csv, ruta_png):
    """
    Guarda un CSV con columnas [L_eff,f1,f2,f3,f4,delta_oct,delta_duo,delta_4,n_peaks]
    y un gráfico PNG de inarmonicidad (Δ2, Δ3, Δ4) vs f1.
    """
    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    cols = ["L_eff", "f1", "f2", "f3", "f4", "delta_oct", "delta_duo", "delta_4", "n_peaks"]
    with open(ruta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in resultados:
            w.writerow([r.get(c, np.nan) for c in cols])

    f1 = np.array([r.get("f1", np.nan) for r in resultados], dtype=float)
    d2 = np.array([r.get("delta_oct", np.nan) for r in resultados], dtype=float)
    d3 = np.array([r.get("delta_duo", np.nan) for r in resultados], dtype=float)
    d4 = np.array([r.get("delta_4",  np.nan) for r in resultados], dtype=float)

    plt.figure(figsize=(7, 4))
    if np.isfinite(d2).any():
        plt.plot(f1, d2, '-o', label="2º vs 2f1")
    if np.isfinite(d3).any():
        plt.plot(f1, d3, '-o', label="3º vs 3f1")
    if np.isfinite(d4).any():
        plt.plot(f1, d4, '-o', label="4º vs 4f1")
    plt.axhline(0, linewidth=1)
    plt.xlabel("f1 (Hz)")
    plt.ylabel("Desviación (cents)")
    plt.title("Inarmonicidad vs f1 (escaneo por longitud)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150)
    plt.close()