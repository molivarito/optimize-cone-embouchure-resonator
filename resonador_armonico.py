import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Importaciones de la librería PyQt6 para la GUI
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QFormLayout, QLabel,
                             QDoubleSpinBox, QPushButton, QTabWidget, QTextEdit,
                             QComboBox)
from PyQt6.QtCore import Qt

# Importaciones para integrar Matplotlib en PyQt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importaciones de OpenWind
from openwind import ImpedanceComputation, Player

# Clase auxiliar para el lienzo de Matplotlib
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador de Instrumentos Acústicos (v6.3 - Final)")
        self.setGeometry(100, 100, 1300, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._create_parameter_panel()
        self._create_results_panel()
        
        self.r1_radio.valueChanged.connect(self.update_r3_radius_label)
        self.update_r3_radius_label(self.r1_radio.value())

    def _create_parameter_panel(self):
        panel_izquierdo = QWidget()
        layout_parametros = QVBoxLayout(panel_izquierdo)
        
        group_model_selector = QGroupBox("1. Modelo de Análisis Principal")
        form_model = QFormLayout(); self.modelo_combo = QComboBox(); self.modelo_combo.addItems(["Modelo Matriz de Prop.", "Modelo de Flauta"]); form_model.addRow("Seleccionar Modelo:", self.modelo_combo); group_model_selector.setLayout(form_model)

        grupo_r1 = QGroupBox("Componente 1 (Cilindro)"); form_r1 = QFormLayout(); self.r1_longitud = QDoubleSpinBox(decimals=3, value=0.5, minimum=0.1, maximum=5.0, singleStep=0.01, suffix=" m"); self.r1_radio = QDoubleSpinBox(decimals=4, value=0.015, minimum=0.001, maximum=0.1, singleStep=0.001, suffix=" m"); form_r1.addRow("Longitud:", self.r1_longitud); form_r1.addRow("Radio:", self.r1_radio); grupo_r1.setLayout(form_r1)
        
        grupo_agujero = QGroupBox("Agujero/Embocadura"); form_agujero = QFormLayout(); self.agujero_posicion = QDoubleSpinBox(decimals=4, value=0.02, minimum=0.0, maximum=5.0, singleStep=0.001, suffix=" m"); self.agujero_radio_entrada = QDoubleSpinBox(decimals=4, value=0.005, minimum=0.001, maximum=0.05, singleStep=0.001, suffix=" m"); self.agujero_radio_salida = QDoubleSpinBox(decimals=4, value=0.005, minimum=0.001, maximum=0.05, singleStep=0.001, suffix=" m"); self.agujero_largo = QDoubleSpinBox(decimals=4, value=0.002, minimum=0.001, maximum=0.05, singleStep=0.001, suffix=" m"); form_agujero.addRow("Posición desde el Inicio:", self.agujero_posicion); form_agujero.addRow("Radio Interno:", self.agujero_radio_entrada); form_agujero.addRow("Radio Externo:", self.agujero_radio_salida); form_agujero.addRow("Largo (Pared):", self.agujero_largo); grupo_agujero.setLayout(form_agujero)

        grupo_r3 = QGroupBox("Componente 2 (Cono Decreciente)"); form_r3 = QFormLayout(); self.r3_longitud = QDoubleSpinBox(decimals=3, value=0.5, minimum=0.1, maximum=5.0, singleStep=0.01, suffix=" m"); self.r3_radio_inicio_label = QLabel(); self.r3_radio_final = QDoubleSpinBox(decimals=4, value=0.005, minimum=0.001, maximum=0.1, singleStep=0.001, suffix=" m"); form_r3.addRow("Longitud:", self.r3_longitud); form_r3.addRow("Radio Inicial:", self.r3_radio_inicio_label); form_r3.addRow("Radio Final:", self.r3_radio_final); grupo_r3.setLayout(form_r3)
        
        self.boton_calcular = QPushButton("Calcular y Actualizar Gráficos")
        self.boton_calcular.clicked.connect(self.run_simulation)
        
        layout_parametros.addWidget(group_model_selector); layout_parametros.addWidget(grupo_r1); layout_parametros.addWidget(grupo_agujero); layout_parametros.addWidget(grupo_r3); layout_parametros.addWidget(self.boton_calcular)
        
        group_bar_chart = QGroupBox("Inarmonicidad (Cents)"); bar_chart_layout = QVBoxLayout(); self.canvas_inarmonicidad_bar = MplCanvas(self, height=3); bar_chart_layout.addWidget(self.canvas_inarmonicidad_bar); group_bar_chart.setLayout(bar_chart_layout)
        layout_parametros.addWidget(group_bar_chart); layout_parametros.addStretch()
        self.main_layout.addWidget(panel_izquierdo, 1)
        
    def update_r3_radius_label(self, value):
        self.r3_radio_inicio_label.setText(f"{value:.4f} m (igual a R1)")

    def _create_results_panel(self):
        self.tabs = QTabWidget(); self.tab_geometria = QWidget(); self.tab_analisis_impedancia = QWidget(); self.tab_campos_internos = QWidget(); self.tab_inarmonicidad_texto = QWidget()
        self.layout_geo = QVBoxLayout(self.tab_geometria); self.geo_container = QWidget(); self.layout_geo.addWidget(self.geo_container)
        self.canvas_analisis_impedancia = MplCanvas(self); self.tab_analisis_impedancia.setLayout(QVBoxLayout()); self.tab_analisis_impedancia.layout().addWidget(self.canvas_analisis_impedancia)
        self.texto_inarmonicidad = QTextEdit("Presiona 'Calcular' para ver los resultados."); self.texto_inarmonicidad.setReadOnly(True); self.texto_inarmonicidad.setStyleSheet("font-size: 14px;"); self.tab_inarmonicidad_texto.setLayout(QVBoxLayout()); self.tab_inarmonicidad_texto.layout().addWidget(self.texto_inarmonicidad)
        layout_campos_internos = QVBoxLayout(self.tab_campos_internos); self.geo_concat_container = QWidget(); self.canvas_campos_internos = MplCanvas(self); layout_campos_internos.addWidget(self.geo_concat_container, 1); layout_campos_internos.addWidget(self.canvas_campos_internos, 2)
        self.tabs.addTab(self.tab_geometria, "1. Geometría"); self.tabs.addTab(self.tab_analisis_impedancia, "2. Análisis de Impedancia"); self.tabs.addTab(self.tab_campos_internos, "3. Campos Internos"); self.tabs.addTab(self.tab_inarmonicidad_texto, "4. Inarmonicidad (Texto)")
        self.main_layout.addWidget(self.tabs, 2)
        
    def _update_geo_canvas(self, container, new_canvas):
        if container.layout() is None: container.setLayout(QVBoxLayout())
        while container.layout().count():
            item = container.layout().takeAt(0)
            if item.widget(): item.widget().deleteLater()
        container.layout().addWidget(new_canvas)

    def run_simulation(self):
        print("Iniciando simulación...")
        frecuencias = np.linspace(50, 4000, 2000); temperatura_global = 20
        r1_len, r1_rad = self.r1_longitud.value(), self.r1_radio.value(); h_pos = self.agujero_posicion.value(); h_rad_in, h_rad_out = self.agujero_radio_entrada.value(), self.agujero_radio_salida.value(); h_len = self.agujero_largo.value(); r3_len, r3_rad_out = self.r3_longitud.value(), self.r3_radio_final.value(); r3_rad_in = self.r1_radio.value()
        
        modelo_seleccionado = self.modelo_combo.currentText()
        agujeros = [['label', 'position', 'radius', 'length', 'radius_out'], ['embocadura', h_pos, h_rad_in, h_len, h_rad_out]]
        digitacion_abierta = [['label', 'abierto'], ['embocadura', 'o']]
        geometria_concatenada = [[0, r1_len, r1_rad, r1_rad, 'cone'], [r1_len, r1_len + r3_len, r3_rad_in, r3_rad_out, 'cone']]
        
        # --- Selección del Modelo y Cálculo ---
        if "Flauta" in modelo_seleccionado:
            print("Calculando como 'Modelo de Flauta'...")
            titulo_analisis = "Modelo de Flauta (Impedancia en Embocadura)"
            player = Player("FLUTE")
            condiciones = {'bell': 'unflanged', 'entrance': 'closed', 'holes': 'unflanged'}
            resultado_sim = ImpedanceComputation(frecuencias, geometria_concatenada, agujeros, player=player, source_location='embocadura', radiation_category=condiciones, temperature=temperatura_global, interp=True, interp_grid=1e-3)
            impedancia_db = 20 * np.log10(np.abs(resultado_sim.impedance))
            indices_resonancia, _ = find_peaks(-impedancia_db, prominence=1)
            tipo_resonancia = "Mínimos"
        else: # Modelo Matriz de Prop.
            print("Calculando como 'Modelo Matriz de Prop.'...")
            titulo_analisis = "Modelo Matriz de Prop. (Impedancia en Extremo)"
            resultado_sim = ImpedanceComputation(frecuencias, geometria_concatenada, agujeros, digitacion_abierta, note='abierto', temperature=temperatura_global, interp=True, interp_grid=1e-3)
            
            # --- LÓGICA CORREGIDA Y DEFINITIVA ---
            # Se calcula la impedancia y se buscan sus MÍNIMOS.
            impedancia_db = 20 * np.log10(np.abs(resultado_sim.impedance))
            indices_resonancia, _ = find_peaks(-impedancia_db, prominence=1)
            tipo_resonancia = "Mínimos"
        
        f_resonancias = frecuencias[indices_resonancia]

        # --- Actualización de Pestañas ---
        plt.figure(); resultado_sim.plot_instrument_geometry(); self._update_geo_canvas(self.geo_container, FigureCanvas(plt.gcf())); plt.close('all')
        
        ax_imp = self.canvas_analisis_impedancia.axes; ax_imp.cla()
        ax_imp.plot(frecuencias, impedancia_db, label='Impedancia Calculada', color='k')
        ax_imp.plot(f_resonancias, impedancia_db[indices_resonancia], 'x', color='red', markersize=8, label=f'Resonancias ({tipo_resonancia})')
        ax_imp.set_title(titulo_analisis); ax_imp.legend(); ax_imp.grid(True); self.canvas_analisis_impedancia.draw()

        ax_campos = self.canvas_campos_internos.axes; ax_campos.cla();
        if len(f_resonancias) >= 3:
            x, pressure_matrix, _ = resultado_sim.get_pressure_flow()
            for i in range(3):
                idx = np.argmin(np.abs(frecuencias - f_resonancias[i])); pressure_dist = np.abs(pressure_matrix[idx, :]); pressure_dist_norm = pressure_dist / (np.max(pressure_dist) + 1e-12); ax_campos.plot(x * 1000, pressure_dist_norm, label=f'Modo #{i+1} ({f_resonancias[i]:.1f} Hz)')
            ax_campos.axvline(x=r1_len * 1000, color='k', linestyle='--', label='Unión R1-R3')
            plt.figure(); resultado_sim.plot_instrument_geometry(); self._update_geo_canvas(self.geo_concat_container, FigureCanvas(plt.gcf())); plt.close('all')
        ax_campos.set_title(f"Distribución de Presión en Resonancias ({tipo_resonancia})"); ax_campos.set_xlabel("Posición (mm)"); ax_campos.set_ylabel("Presión Normalizada"); ax_campos.legend(); ax_campos.grid(True); self.canvas_campos_internos.draw()
        
        texto_analisis = f"--- Análisis de Inarmonicidad ({tipo_resonancia}) ---\n\n"; deltas = []
        if len(f_resonancias) >= 4:
            f1, f2, f3, f4 = f_resonancias[0], f_resonancias[1], f_resonancias[2], f_resonancias[3]
            deltas = [1200 * np.log2(f2/(2*f1)), 1200 * np.log2(f3/(3*f1)), 1200 * np.log2(f4/(4*f1))]
            texto_analisis += f"Fundamental (f1): {f1:.2f} Hz\n\n"; texto_analisis += f"Segundo modo (f2 vs 2f1): {deltas[0]:.2f} cents\n"; texto_analisis += f"Tercer modo (f3 vs 3f1): {deltas[1]:.2f} cents\n"; texto_analisis += f"Cuarto modo (f4 vs 4f1): {deltas[2]:.2f} cents"
        else: texto_analisis += "No se encontraron suficientes resonancias para el análisis."
        self.texto_inarmonicidad.setText(texto_analisis)

        ax_bar = self.canvas_inarmonicidad_bar.axes; ax_bar.cla()
        if deltas:
            labels = ['2º Modo', '3er Modo', '4º Modo']; colors = ['skyblue' if d > 0 else 'salmon' for d in deltas]; bars = ax_bar.bar(labels, deltas, color=colors); ax_bar.axhline(0, color='k', linewidth=0.8, linestyle='--'); ax_bar.set_ylabel('Desviación (cents)'); ax_bar.set_title('Inarmonicidad'); ax_bar.bar_label(bars, fmt='%.1f', padding=3)
        else: ax_bar.text(0.5, 0.5, 'Datos insuficientes', ha='center', va='center')
        self.canvas_inarmonicidad_bar.fig.tight_layout(); self.canvas_inarmonicidad_bar.draw()
        
        print("...simulación completada.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())