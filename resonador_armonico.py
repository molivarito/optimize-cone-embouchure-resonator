import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QLabel, QDoubleSpinBox, QPushButton, QTabWidget, QTextEdit,
    QComboBox, QFileDialog, QSpinBox, QProgressBar, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle
from openwind import ImpedanceComputation, Player
from escaneo_longitud import guardar_csv_y_plot

# Importar workers y builders externos
from workers import ScanWorker, BlueOptWorker
from ui_builders import (
    build_scan_tab, build_opt_tab, build_parameter_panel, build_results_panel, MplCanvas
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador de Instrumentos Acústicos (v7.1 con L_total)")
        self.setGeometry(100, 100, 1400, 950)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Panel de parámetros (izquierda)
        build_parameter_panel(self)
        # Panel de resultados/tabs (derecha)
        build_results_panel(self)

        # Enlace para mantener coherencia del radio inicial del cono
        self.r1_radio.valueChanged.connect(self.update_r3_radius_label)
        self.update_r3_radius_label(self.r1_radio.value())
        self.r1_longitud.valueChanged.connect(self.update_derived_lengths)
        self.L_total.valueChanged.connect(self.update_derived_lengths)
        self.update_derived_lengths()

        # Estado de último escaneo
        self._ultimo_resultado = None
        self._opt_summary = None


    def update_r3_radius_label(self, value):
        # value viene del spinbox de r1_radio (mm)
        self.r3_radio_inicio_label.setText(f"{value:.2f} mm (igual a R1)")

    def update_derived_lengths(self):
        """Actualiza r3_longitud (derivado) en cm."""
        try:
            r1_cm = self.r1_longitud.value()
            Ltot_cm = self.L_total.value()
            r3_cm = max(Ltot_cm - r1_cm, 0.0)
            self.lbl_r3_len.setText(f"{r3_cm:.2f} cm")
        except Exception:
            pass


    def _start_opt(self):
        try:
            def around(val, frac=0.2, lo=1e-6):
                a = max(lo, val*(1.0-frac)); b = val*(1.0+frac)
                if a > b: a, b = b, a
                return [float(a), float(b)]

            # Convertir desde UI (cm/mm) a SI (m) para la optimización
            r1_len = self.r1_longitud.value() / 100.0
            r1_rad = self.r1_radio.value() / 1000.0
            L_total = self.L_total.value() / 100.0
            r3_len = max(L_total - r1_len, 0.0)
            r3_rad_out = self.r3_radio_final.value() / 1000.0
            h_pos = self.agujero_posicion.value() / 100.0
            h_in  = self.agujero_radio_entrada.value() / 1000.0
            h_out = self.agujero_radio_salida.value() / 1000.0
            h_len = self.agujero_largo.value() / 1000.0

            space_cfg = {
                "variables": {
                    "L_total": around(L_total),
                    "r1_longitud": around(r1_len),
                    "r1_radio": around(r1_rad),
                    "radio_cono_final": around(r3_rad_out),
                    "agujero_posicion": around(h_pos),
                    "agujero_radio_in": around(h_in),
                    "agujero_radio_out": around(h_out),
                    "agujero_largo": around(h_len),
                },
                "fixed": {"temperatura": 20.0}
            }
            # Si el usuario decide mantener L_total fijo, pasamos la restricción
            if hasattr(self, 'chk_fix_Ltot') and self.chk_fix_Ltot.isChecked():
                space_cfg.setdefault("fixed", {})["L_total_fixed"] = float(L_total)

            weights = {"w_mag":1.0, "w_var":0.5, "w_max":0.5, "w_miss":50.0}
            n_samples = int(self.opt_samples.value())
            top_k = int(self.opt_topk.value())
            seed = int(self.opt_seed.value())
            steps = int(self.opt_steps.value())
            lmin = float(self.opt_lmin.value())
            maxiter = int(self.opt_maxiter.value())

            self.opt_prog_global.setRange(0, n_samples)
            self.opt_prog_global.setValue(0)
            # Estimar presupuesto de evaluaciones para la fase local (Powell)
            d = len(space_cfg["variables"])  # número de variables
            est_local_max = int(maxiter * (2*d + 5))
            if est_local_max <= 0:
                est_local_max = 100
            self.opt_prog_local.setRange(0, est_local_max)
            self.opt_prog_local.setValue(0)

            self.opt_log.clear()
            # Clear axes and persistent lines
            self.line_d2.set_data([], [])
            self.canvas_opt.draw()
            self.line_geo.set_data([], [])
            # Embocadura (círculos: overlay)
            if hasattr(self, 'circ_hole_out'):
                self.circ_hole_out.set_radius(0.0)
                self.circ_hole_out.set_visible(False)
            if hasattr(self, 'circ_hole_in'):
                self.circ_hole_in.set_radius(0.0)
                self.circ_hole_in.set_visible(False)
            if hasattr(self, 'ax_hole'):
                # sincronizar límites x con el eje principal
                self.ax_hole.set_xlim(self.canvas_opt_geo.axes.get_xlim())
                self.ax_hole.set_ylim(-1, 1)
            self.canvas_opt_geo.draw()
            # Limpiar texto de parámetros
            if hasattr(self, 'lbl_geom_params'):
                self.lbl_geom_params.setText("")
            self.btn_run_opt.setEnabled(False)

            self._opt_thread = QThread()
            self._opt_worker = BlueOptWorker(space_cfg, weights, steps, lmin, 20.0,
                                             n_samples, top_k, seed, maxiter)
            self._opt_worker.moveToThread(self._opt_thread)
            self._opt_thread.started.connect(self._opt_worker.run)
            self._opt_worker.progress_global.connect(self._on_opt_prog_global)
            self._opt_worker.progress_local.connect(self._on_opt_prog_local)
            self._opt_worker.finished.connect(self._on_opt_finished)
            self._opt_worker.error.connect(self._on_opt_error)
            self._opt_worker.progress_snapshot.connect(self._on_opt_snapshot)
            self._opt_worker.finished.connect(self._opt_thread.quit)
            self._opt_worker.finished.connect(self._opt_worker.deleteLater)
            self._opt_thread.finished.connect(self._opt_thread.deleteLater)
            self._opt_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error al iniciar optimización", str(e))

    def _on_opt_snapshot(self, snap: dict):
        try:
            if not self.chk_live.isChecked():
                return
            d2 = np.array(snap.get('d2', []), dtype=float)
            if d2.size > 0:
                x_idx = np.arange(len(d2))
                self.line_d2.set_data(x_idx, d2)
                ax = self.canvas_opt.axes
                ax.relim(); ax.autoscale_view()
                self.canvas_opt.draw_idle()

            geom = snap.get('geom', {})
            r1_len = geom.get('r1_longitud'); r1_rad = geom.get('r1_radio')
            r3_len = geom.get('r3_longitud'); r3_out = geom.get('radio_cono_final')
            Ltot = geom.get('L_total')

            if (r3_len is None) and (Ltot is not None) and (r1_len is not None):
                try:
                    r3_len = float(Ltot) - float(r1_len)
                except Exception:
                    r3_len = None

            # Geometría principal
            end_x = None
            if Ltot is not None:
                end_x = Ltot
            elif (r1_len is not None and r3_len is not None):
                end_x = float(r1_len) + float(r3_len)
            if None not in (r1_len, r1_rad, r3_out) and (end_x is not None):
                x = np.array([0.0, r1_len, end_x])
                y = np.array([r1_rad, r1_rad, r3_out])
                self.line_geo.set_data(x, y)

            # Embocadura (círculos en eje overlay con aspecto igual)
            x_h = geom.get('agujero_posicion')
            r_in_h = geom.get('agujero_radio_in')
            r_out_h = geom.get('agujero_radio_out')
            t_h = geom.get('agujero_largo')
            if None not in (x_h, r_in_h, r_out_h):
                # Centros: separados sobre eje vertical por el largo de pared
                t = float(t_h) if (t_h is not None) else 0.0
                y_in  = -0.5 * t
                y_out =  0.5 * t
                x_c = float(x_h)
                # Actualizar círculos (radio en metros)
                self.circ_hole_in.center = (x_c, y_in)
                self.circ_hole_in.set_radius(float(r_in_h))
                self.circ_hole_in.set_visible(True)

                self.circ_hole_out.center = (x_c, y_out)
                self.circ_hole_out.set_radius(float(r_out_h))
                self.circ_hole_out.set_visible(True)

                # Sincronizar límites x con el eje principal y ajustar y según t y radios
                axg = self.canvas_opt_geo.axes
                self.ax_hole.set_xlim(axg.get_xlim())
                rmax = max(float(r_in_h), float(r_out_h)) if (r_in_h is not None and r_out_h is not None) else 0.0
                ypad = 0.2 * (t + 2*rmax + 1e-6)
                self.ax_hole.set_ylim(-0.5*t - rmax - ypad, 0.5*t + rmax + ypad)
            else:
                if hasattr(self, 'circ_hole_in'):
                    self.circ_hole_in.set_radius(0.0); self.circ_hole_in.set_visible(False)
                if hasattr(self, 'circ_hole_out'):
                    self.circ_hole_out.set_radius(0.0); self.circ_hole_out.set_visible(False)

            # Texto con parámetros (live) en cm/mm
            if hasattr(self, 'lbl_geom_params'):
                try:
                    txt = []
                    if None not in (r1_len, r1_rad, r3_out):
                        r1_len_cm = float(r1_len) * 100.0
                        r1_rad_mm = float(r1_rad) * 1000.0
                        if Ltot is not None:
                            Ltot_cm = float(Ltot) * 100.0
                            r3_len_cm = max(Ltot_cm - r1_len_cm, 0.0)
                        else:
                            r3_len_cm = float(r3_len) * 100.0 if r3_len is not None else 0.0
                            Ltot_cm = r1_len_cm + r3_len_cm
                        r3_out_mm = float(r3_out) * 1000.0
                        txt.append(f"R1: L={r1_len_cm:.2f} cm, r={r1_rad_mm:.2f} mm")
                        txt.append(f"R3: L(der.)={r3_len_cm:.2f} cm, r_out={r3_out_mm:.2f} mm")
                        txt.append(f"L_total={Ltot_cm:.2f} cm")
                    if None not in (x_h, r_in_h, r_out_h):
                        x_cm     = float(x_h) * 100.0
                        r_in_mm  = float(r_in_h) * 1000.0
                        r_out_mm = float(r_out_h) * 1000.0
                        t_mm     = float(t_h) * 1000.0 if (t_h is not None) else 0.0
                        txt.append(f"Embocadura: x={x_cm:.2f} cm, r_in={r_in_mm:.2f} mm, r_out={r_out_mm:.2f} mm, t={t_mm:.2f} mm")
                    self.lbl_geom_params.setText("\n".join(txt))
                except Exception:
                    pass

            ag = self.canvas_opt_geo.axes
            ag.relim(); ag.autoscale_view()
            self.canvas_opt_geo.draw_idle()
            if hasattr(self, 'ax_hole'):
                self.canvas_opt_geo.draw_idle()
        except Exception:
            pass

    def _on_opt_prog_global(self, i, total):
        self.opt_prog_global.setValue(i)

    def _on_opt_prog_local(self, it, maxit):
        self.opt_prog_local.setValue(min(it, maxit))


    def _on_opt_finished(self, out: dict):
        self._opt_summary = out
        ax = self.canvas_opt.axes
        d2 = np.array(out.get("best_eval", {}).get("deltas2", []), dtype=float)
        if d2.size > 0:
            x_idx = np.arange(len(d2))
            self.line_d2.set_data(x_idx, d2)
            ax.relim(); ax.autoscale_view()
            self.canvas_opt.draw_idle()
        self.btn_run_opt.setEnabled(True)
        try:
            cost_val = out.get('local_best',{}).get('cost', np.nan)
            msg_cost = f"{float(cost_val):.3f}" if np.isfinite(cost_val) else "?"
        except Exception:
            msg_cost = "?"
        QMessageBox.information(self, "Optimización finalizada",
                                f"Costo (local): {msg_cost}")

    def _on_opt_error(self, msg):
        self.btn_run_opt.setEnabled(True)
        QMessageBox.critical(self, "Error durante optimización", msg)

    def _export_opt_curve(self):
        if not self._opt_summary or 'best_eval' not in self._opt_summary:
            QMessageBox.information(self, "Sin resultados", "Aún no hay una optimización para exportar.")
            return
        try:
            csv_new, _ = QFileDialog.getSaveFileName(self, "Guardar curva Δ2", "best_curve_d2.csv", "CSV (*.csv)")
            if not csv_new:
                return
            png_new, _ = QFileDialog.getSaveFileName(self, "Guardar gráfico Δ2", "best_curve_d2.png", "PNG (*.png)")
            if not png_new:
                return
            # Recalcular f1..f4 y deltas antes de guardar, usando los parámetros óptimos
            best_eval = self._opt_summary['best_eval']
            # Extraer geometría y parámetros
            L_total = best_eval.get("L_total")
            r1_len = best_eval.get("params", {}).get("r1_longitud")
            r1_rad = best_eval.get("params", {}).get("r1_radio")
            r3_len = best_eval.get("r3_longitud")
            r3_rad_out = best_eval.get("params", {}).get("radio_cono_final")
            h_pos = best_eval.get("params", {}).get("agujero_posicion")
            h_in = best_eval.get("params", {}).get("agujero_radio_in")
            h_out = best_eval.get("params", {}).get("agujero_radio_out")
            h_len = best_eval.get("params", {}).get("agujero_largo")
            geometria_base = [
                [0, r1_len, r1_rad, r1_rad, 'cone'],
                [r1_len, L_total, r1_rad, r3_rad_out, 'cone']
            ]
            agujeros = [
                ['label', 'position', 'radius', 'length', 'radius_out'],
                ['embocadura', h_pos, h_in, h_len, h_out]
            ]
            L_min_frac = self.opt_lmin.value() if hasattr(self, 'opt_lmin') else 0.5
            n_steps = self.opt_steps.value() if hasattr(self, 'opt_steps') else 10
            temperatura = 20.0
            from escaneo_longitud import escanear_por_longitud
            resultados = escanear_por_longitud(
                geometria_base=geometria_base,
                agujeros=agujeros,
                L_full=L_total,
                L_min_frac=L_min_frac,
                n_steps=n_steps,
                temperatura=temperatura
            )
            guardar_csv_y_plot(resultados, csv_new, png_new)
            QMessageBox.information(self, "Exportación completada", "Se guardaron los archivos seleccionados.")
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))

    # ---------- Utilidad para refrescar contenedores ----------
    def _update_geo_canvas(self, container, new_canvas):
        if container.layout() is None:
            container.setLayout(QVBoxLayout())
        # limpiar
        while container.layout().count():
            item = container.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        container.layout().addWidget(new_canvas)

    # ---------- Simulación instantánea (tab 1–4) ----------
    def run_simulation(self):
        try:
            print("Iniciando simulación...")
            frecuencias = np.linspace(50, 4000, 2000)
            temperatura_global = 20

            # Lectura de UI (cm/mm) y conversión a SI (m)
            r1_len = self.r1_longitud.value() / 100.0       # cm → m
            r1_rad = self.r1_radio.value() / 1000.0         # mm → m
            h_pos = self.agujero_posicion.value() / 100.0   # cm → m
            h_rad_in = self.agujero_radio_entrada.value() / 1000.0   # mm → m
            h_rad_out = self.agujero_radio_salida.value() / 1000.0   # mm → m
            h_len = self.agujero_largo.value() / 1000.0     # mm → m
            L_total = self.L_total.value() / 100.0          # cm → m
            r3_len = max(L_total - r1_len, 0.0)
            r3_rad_out = self.r3_radio_final.value() / 1000.0  # mm → m
            r3_rad_in = r1_rad

            if L_total <= r1_len:
                QMessageBox.warning(self, "Parámetros inválidos",
                                    "L_total debe ser mayor que la longitud del cilindro (R1).")
                return

            modelo_seleccionado = self.modelo_combo.currentText()

            agujeros = [
                ['label', 'position', 'radius', 'length', 'radius_out'],
                ['embocadura', h_pos, h_rad_in, h_len, h_rad_out]
            ]
            digitacion_abierta = [['label', 'abierto'], ['embocadura', 'o']]
            geometria_concatenada = [
                [0, r1_len, r1_rad, r1_rad, 'cone'],
                [r1_len, L_total, r3_rad_in, r3_rad_out, 'cone']
            ]

            # Cálculo
            if "Flauta" in modelo_seleccionado:
                print("Calculando como 'Modelo de Flauta'...")
                titulo_analisis = "Modelo de Flauta (Impedancia en Embocadura)"
                player = Player("FLUTE")
                condiciones = {'bell': 'unflanged', 'entrance': 'closed', 'holes': 'unflanged'}
                resultado_sim = ImpedanceComputation(
                    frecuencias, geometria_concatenada, agujeros,
                    player=player, source_location='embocadura',
                    radiation_category=condiciones, temperature=temperatura_global,
                    interp=True, interp_grid=1e-3
                )
                impedancia_db = 20 * np.log10(np.abs(resultado_sim.impedance))
                indices_resonancia, _ = find_peaks(-impedancia_db, prominence=1)
                tipo_resonancia = "Mínimos"
            else:
                print("Calculando como 'Modelo Matriz de Prop.'...")
                titulo_analisis = "Modelo Matriz de Prop. (Impedancia en Extremo)"
                resultado_sim = ImpedanceComputation(
                    frecuencias, geometria_concatenada, agujeros, digitacion_abierta,
                    note='abierto', temperature=temperatura_global, interp=True, interp_grid=1e-3
                )
                impedancia_db = 20 * np.log10(np.abs(resultado_sim.impedance))
                indices_resonancia, _ = find_peaks(-impedancia_db, prominence=1)
                tipo_resonancia = "Mínimos"

            f_resonancias = frecuencias[indices_resonancia]

            # --- Geometría (tab 1)
            plt.figure()
            resultado_sim.plot_instrument_geometry()
            self._update_geo_canvas(self.geo_container, FigureCanvas(plt.gcf()))
            plt.close('all')

            # --- Impedancia (tab 2)
            ax_imp = self.canvas_analisis_impedancia.axes
            ax_imp.cla()
            ax_imp.plot(frecuencias, impedancia_db, label='Impedancia Calculada', color='k')
            ax_imp.plot(f_resonancias, impedancia_db[indices_resonancia], 'x', color='red',
                      markersize=8, label=f'Resonancias ({tipo_resonancia})')
            ax_imp.set_title(titulo_analisis)
            ax_imp.legend()
            ax_imp.grid(True)
            self.canvas_analisis_impedancia.draw()

            # --- Campos internos (tab 3)
            ax_campos = self.canvas_campos_internos.axes
            ax_campos.cla()
            if len(f_resonancias) >= 3:
                x, pressure_matrix, _ = resultado_sim.get_pressure_flow()
                for i in range(3):
                    idx = np.argmin(np.abs(frecuencias - f_resonancias[i]))
                    pressure_dist = np.abs(pressure_matrix[idx, :])
                    pressure_dist_norm = pressure_dist / (np.max(pressure_dist) + 1e-12)
                    ax_campos.plot(x * 1000, pressure_dist_norm,
                                   label=f'Modo #{i+1} ({f_resonancias[i]:.1f} Hz)')
                ax_campos.axvline(x=r1_len * 1000, color='k', linestyle='--', label='Unión R1-R3')
                plt.figure()
                resultado_sim.plot_instrument_geometry()
                self._update_geo_canvas(self.geo_concat_container, FigureCanvas(plt.gcf()))
                plt.close('all')
            ax_campos.set_title(f"Distribución de Presión en Resonancias ({tipo_resonancia})")
            ax_campos.set_xlabel("Posición (mm)")
            ax_campos.set_ylabel("Presión Normalizada")
            ax_campos.legend()
            ax_campos.grid(True)
            self.canvas_campos_internos.draw()

            print("...simulación completada.")
        except Exception as e:
            QMessageBox.critical(self, "Error en simulación", str(e))

    # ---------- Escaneo por longitud (tab 5) ----------
    def _start_scan(self):
        try:
            # Tomamos geometría actual de la UI (cm/mm) y convertimos a m
            r1_len = self.r1_longitud.value() / 100.0
            r1_rad = self.r1_radio.value() / 1000.0
            L_total = self.L_total.value() / 100.0
            r3_len = max(L_total - r1_len, 0.0)
            r3_rad_in = r1_rad
            r3_rad_out = self.r3_radio_final.value() / 1000.0

            h_pos = self.agujero_posicion.value() / 100.0
            h_rad_in = self.agujero_radio_entrada.value() / 1000.0
            h_rad_out = self.agujero_radio_salida.value() / 1000.0
            h_len = self.agujero_largo.value() / 1000.0

            if L_total <= r1_len:
                QMessageBox.warning(self, "Parámetros inválidos",
                                    "L_total debe ser mayor que la longitud del cilindro (R1).")
                return

            geometria_base = [
                [0, r1_len, r1_rad, r1_rad, 'cone'],
                [r1_len, L_total, r3_rad_in, r3_rad_out, 'cone']
            ]
            agujeros = [
                ['label', 'position', 'radius', 'length', 'radius_out'],
                ['embocadura', h_pos, h_rad_in, h_len, h_rad_out]
            ]

            # Parámetros de escaneo
            L_min_frac = float(self.scan_frac.value())
            n_steps = int(self.scan_steps.value())
            temperatura = float(self.scan_temp.value())

            kwargs = dict(
                geometria_base=geometria_base,
                agujeros=agujeros,
                L_full=L_total,
                L_min_frac=L_min_frac,
                n_steps=n_steps,
                temperatura=temperatura
            )

            # Iniciar worker en thread
            self.btn_run_scan.setEnabled(False)
            self.scan_progress.setRange(0, 0)  # indeterminado
            self.canvas_scan.axes.cla()
            self.canvas_scan.draw()

            self._thread = QThread()
            self._worker = ScanWorker(kwargs)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.finished.connect(self._scan_finished)
            self._worker.error.connect(self._scan_error)
            self._worker.finished.connect(self._thread.quit)
            self._worker.finished.connect(self._worker.deleteLater)
            self._thread.finished.connect(self._thread.deleteLater)
            self._thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error al iniciar escaneo", str(e))

    def _scan_finished(self, resultados):
        # Guardar estado
        self._ultimo_resultado = resultados

        # Actualizar gráfico: inarmonicidad vs f1
        ax = self.canvas_scan.axes
        ax.cla()
        import numpy as np

        f1 = np.array([r.get("f1", np.nan) for r in resultados], dtype=float)
        d2 = np.array([r.get("delta_oct", np.nan) for r in resultados], dtype=float)
        d3 = np.array([r.get("delta_duo", np.nan) for r in resultados], dtype=float)
        d4 = np.array([r.get("delta_4", np.nan) for r in resultados], dtype=float)

        if np.isfinite(d2).any():
            ax.plot(f1, d2, '-o', label="2º vs 2f1")
        if np.isfinite(d3).any():
            ax.plot(f1, d3, '-o', label="3º vs 3f1")
        if np.isfinite(d4).any():
            ax.plot(f1, d4, '-o', label="4º vs 4f1")
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("f1 (Hz)")
        ax.set_ylabel("Desviación (cents)")
        ax.set_title("Inarmonicidad vs f1 (escaneo por longitud)")
        ax.grid(True)
        ax.legend()
        self.canvas_scan.figure.tight_layout()
        self.canvas_scan.draw()

        self.scan_progress.setRange(0, 1)  # done
        self.scan_progress.setValue(1)
        self.btn_run_scan.setEnabled(True)

    def _scan_error(self, msg):
        self.scan_progress.setRange(0, 1)
        self.scan_progress.setValue(1)
        self.btn_run_scan.setEnabled(True)
        QMessageBox.critical(self, "Error durante escaneo", msg)

    def _export_last_outputs(self):
        if not self._ultimo_resultado or not isinstance(self._ultimo_resultado, list) or len(self._ultimo_resultado) == 0:
            QMessageBox.information(self, "Sin resultados",
                                    "Aún no hay un escaneo para exportar.")
            return
        try:
            csv_new, _ = QFileDialog.getSaveFileName(
                self, "Guardar CSV de escaneo", "scan_longitud.csv", "CSV (*.csv)"
            )
            if not csv_new:
                return
            png_new, _ = QFileDialog.getSaveFileName(
                self, "Guardar gráfico PNG", "scan_longitud.png", "PNG (*.png)"
            )
            if not png_new:
                return
            guardar_csv_y_plot(self._ultimo_resultado, csv_new, png_new)
            QMessageBox.information(self, "Exportación completada",
                                    "Se guardaron los archivos seleccionados.")
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))


# -------------------- main --------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())