# ui_builders.py
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel,
    QDoubleSpinBox, QPushButton, QTabWidget, QTextEdit, QComboBox,
    QFileDialog, QSpinBox, QProgressBar, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle


# -------------------- Canvas auxiliar --------------------
class MplCanvas(FigureCanvas):
    """
    Canvas auxiliar para incrustar Matplotlib en PyQt6.

    Importante: expone tanto `.figure` (estándar) como `.fig` (alias de
    compatibilidad) para que el resto del código pueda usar cualquiera.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # Atributo estándar
        self.figure = fig
        # Alias de compatibilidad para código que usa `.fig`
        self.fig = fig
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        fig.tight_layout()


    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        fig.tight_layout()


# -------------------- Builders de la GUI --------------------
def build_parameter_panel(self):
    """Panel izquierdo con parámetros de geometría y botón de cálculo."""
    panel_izquierdo = QWidget()
    layout_parametros = QVBoxLayout(panel_izquierdo)

    # Selector de modelo
    group_model_selector = QGroupBox("1. Modelo de Análisis Principal")
    form_model = QFormLayout()
    self.modelo_combo = QComboBox()
    self.modelo_combo.addItems(["Modelo Matriz de Prop.", "Modelo de Flauta"])
    form_model.addRow("Seleccionar Modelo:", self.modelo_combo)
    group_model_selector.setLayout(form_model)

    # R1: cilindro (UI en cm / mm)
    grupo_r1 = QGroupBox("Componente 1 (Cilindro)")
    form_r1 = QFormLayout()
    self.r1_longitud = QDoubleSpinBox(decimals=2, value=50.0, minimum=10.0, maximum=500.0,
                                      singleStep=0.5, suffix=" cm")
    self.r1_radio = QDoubleSpinBox(decimals=2, value=15.0, minimum=1.0, maximum=100.0,
                                   singleStep=0.1, suffix=" mm")
    form_r1.addRow("Longitud:", self.r1_longitud)
    form_r1.addRow("Radio:", self.r1_radio)
    grupo_r1.setLayout(form_r1)

    # Agujero / Embocadura (UI en cm / mm)
    grupo_agujero = QGroupBox("Agujero/Embocadura")
    form_agujero = QFormLayout()
    self.agujero_posicion = QDoubleSpinBox(decimals=2, value=2.0, minimum=0.0, maximum=500.0,
                                           singleStep=0.1, suffix=" cm")
    self.agujero_radio_entrada = QDoubleSpinBox(decimals=2, value=5.0, minimum=1.0, maximum=50.0,
                                                singleStep=0.1, suffix=" mm")
    self.agujero_radio_salida = QDoubleSpinBox(decimals=2, value=5.0, minimum=1.0, maximum=50.0,
                                               singleStep=0.1, suffix=" mm")
    self.agujero_largo = QDoubleSpinBox(decimals=2, value=2.0, minimum=0.1, maximum=50.0,
                                        singleStep=0.1, suffix=" mm")
    form_agujero.addRow("Posición desde el Inicio:", self.agujero_posicion)
    form_agujero.addRow("Radio Interno:", self.agujero_radio_entrada)
    form_agujero.addRow("Radio Externo:", self.agujero_radio_salida)
    form_agujero.addRow("Largo (Pared):", self.agujero_largo)
    grupo_agujero.setLayout(form_agujero)

    # L_total (cm)
    group_total = QGroupBox("Largo Total del Instrumento")
    form_total = QFormLayout()
    self.L_total = QDoubleSpinBox(decimals=2, value=100.0, minimum=10.0, maximum=500.0,
                                  singleStep=0.5, suffix=" cm")
    form_total.addRow("L_total:", self.L_total)
    group_total.setLayout(form_total)

    # R3: cono (parámetros dependientes)
    grupo_r3 = QGroupBox("Componente 2 (Cono Decreciente)")
    form_r3 = QFormLayout()
    self.lbl_r3_len = QLabel("0.00 cm")  # se actualiza por update_derived_lengths()
    self.r3_radio_inicio_label = QLabel()
    self.r3_radio_final = QDoubleSpinBox(decimals=2, value=5.0, minimum=1.0, maximum=100.0,
                                         singleStep=0.1, suffix=" mm")
    form_r3.addRow("Longitud (derivada):", self.lbl_r3_len)
    form_r3.addRow("Radio Inicial:", self.r3_radio_inicio_label)
    form_r3.addRow("Radio Final:", self.r3_radio_final)
    grupo_r3.setLayout(form_r3)

    # Botón cálculo instantáneo (modelo seleccionado)
    self.boton_calcular = QPushButton("Calcular y Actualizar Gráficos")
    self.boton_calcular.clicked.connect(self.run_simulation)

    # Montaje del panel izquierdo
    layout_parametros.addWidget(group_model_selector)
    layout_parametros.addWidget(grupo_r1)
    layout_parametros.addWidget(grupo_agujero)
    layout_parametros.addWidget(group_total)
    layout_parametros.addWidget(grupo_r3)
    layout_parametros.addWidget(self.boton_calcular)
    layout_parametros.addStretch()

    self.main_layout.addWidget(panel_izquierdo, 1)


def build_results_panel(self):
    """Panel derecho con tabs (Geometría, Impedancia, Campos, Inarmonicidad, Escaneo, Optimización)."""
    self.tabs = QTabWidget()

    # 1. Geometría
    self.tab_geometria = QWidget()
    self.layout_geo = QVBoxLayout(self.tab_geometria)
    self.geo_container = QWidget()
    self.layout_geo.addWidget(self.geo_container)

    # 2. Análisis de impedancia
    self.tab_analisis_impedancia = QWidget()
    self.tab_analisis_impedancia.setLayout(QVBoxLayout())
    self.canvas_analisis_impedancia = MplCanvas(self)
    self.tab_analisis_impedancia.layout().addWidget(self.canvas_analisis_impedancia)

    # 3. Campos internos
    self.tab_campos_internos = QWidget()
    layout_campos_internos = QVBoxLayout(self.tab_campos_internos)
    self.geo_concat_container = QWidget()
    self.canvas_campos_internos = MplCanvas(self)
    layout_campos_internos.addWidget(self.geo_concat_container, 1)
    layout_campos_internos.addWidget(self.canvas_campos_internos, 2)

    # 4. Inarmonicidad (texto)
    self.tab_inarmonicidad_texto = QWidget()
    self.tab_inarmonicidad_texto.setLayout(QVBoxLayout())
    self.texto_inarmonicidad = QTextEdit("Presiona 'Calcular' para ver los resultados.")
    self.texto_inarmonicidad.setReadOnly(True)
    self.texto_inarmonicidad.setStyleSheet("font-size: 14px;")
    self.tab_inarmonicidad_texto.layout().addWidget(self.texto_inarmonicidad)

    # 5. Escaneo por longitud
    self.tab_scan = QWidget()
    build_scan_tab(self, self.tab_scan)

    # 6. Optimización (Curva Azul)
    self.tab_opt = QWidget()
    build_opt_tab(self, self.tab_opt)

    # Agregar tabs
    self.tabs.addTab(self.tab_geometria, "1. Geometría")
    self.tabs.addTab(self.tab_analisis_impedancia, "2. Análisis de Impedancia")
    self.tabs.addTab(self.tab_campos_internos, "3. Campos Internos")
    self.tabs.addTab(self.tab_scan, "4. Escaneo por Longitud")
    self.tabs.addTab(self.tab_opt, "5. Optimización (Curva Azul)")

    self.main_layout.addWidget(self.tabs, 2)


def build_scan_tab(self, tab_widget):
    """Construye la pestaña de escaneo por longitud (no bloqueante)."""
    lay = QVBoxLayout(tab_widget)

    # Controles
    box_ctrl = QGroupBox("Parámetros de Escaneo")
    form = QFormLayout()

    self.scan_frac = QDoubleSpinBox(decimals=2, value=0.50, minimum=0.30, maximum=0.90,
                                    singleStep=0.05)
    self.scan_frac.setSuffix(" × L_total")
    self.scan_steps = QSpinBox()
    self.scan_steps.setRange(10, 400)
    self.scan_steps.setValue(10)

    self.scan_temp = QDoubleSpinBox(decimals=1, value=20.0, minimum=0.0, maximum=45.0,
                                    singleStep=0.5, suffix=" °C")

    form.addRow("L_min (fracción de L_total):", self.scan_frac)
    form.addRow("N° de pasos:", self.scan_steps)
    form.addRow("Temperatura:", self.scan_temp)
    box_ctrl.setLayout(form)

    # Botones
    row_btns = QHBoxLayout()
    self.btn_run_scan = QPushButton("Escanear (no bloqueante)")
    self.btn_run_scan.clicked.connect(self._start_scan)
    self.btn_save_last = QPushButton("Exportar Último CSV/PNG…")
    self.btn_save_last.clicked.connect(self._export_last_outputs)
    row_btns.addWidget(self.btn_run_scan)
    row_btns.addWidget(self.btn_save_last)

    # Progreso
    self.scan_progress = QProgressBar()
    self.scan_progress.setRange(0, 0)  # indeterminado mientras corre

    # Gráfico resultado
    self.canvas_scan = MplCanvas(self, height=4)

    # Armado
    lay.addWidget(box_ctrl)
    lay.addLayout(row_btns)
    lay.addWidget(self.scan_progress)
    lay.addWidget(self.canvas_scan)


def build_opt_tab(self, tab_widget):
    """Construye la pestaña de optimización (curva azul) y su vista en vivo de geometría."""
    lay = QVBoxLayout(tab_widget)

    # Controles
    box_ctrl = QGroupBox("Parámetros de Optimización (curva azul)")
    form = QFormLayout()
    self.opt_samples = QSpinBox(); self.opt_samples.setRange(20, 2000); self.opt_samples.setValue(200)
    self.opt_topk = QSpinBox(); self.opt_topk.setRange(1, 20); self.opt_topk.setValue(5)
    self.opt_seed = QSpinBox(); self.opt_seed.setRange(0, 999999); self.opt_seed.setValue(42)
    self.opt_steps = QSpinBox(); self.opt_steps.setRange(5, 50); self.opt_steps.setValue(10)
    self.opt_lmin = QDoubleSpinBox(decimals=2, value=0.50, minimum=0.30, maximum=0.90, singleStep=0.05)
    self.opt_lmin.setSuffix(" × L_total")
    self.opt_maxiter = QSpinBox(); self.opt_maxiter.setRange(10, 2000); self.opt_maxiter.setValue(250)
    self.chk_fix_Ltot = QCheckBox("Mantener L_total fijo")
    self.chk_fix_Ltot.setChecked(True)

    form.addRow("Muestras Sobol (global):", self.opt_samples)
    form.addRow("Top-K para refinamiento:", self.opt_topk)
    form.addRow("Seed:", self.opt_seed)
    form.addRow("Pasos de escaneo:", self.opt_steps)
    form.addRow("L_min (fracción):", self.opt_lmin)
    form.addRow("Iteraciones Powell (máx):", self.opt_maxiter)
    form.addRow("Fijar L_total:", self.chk_fix_Ltot)
    box_ctrl.setLayout(form)

    # Botones
    row_btns = QHBoxLayout()
    self.btn_run_opt = QPushButton("Optimizar (no bloqueante)")
    self.btn_run_opt.clicked.connect(self._start_opt)
    self.btn_save_opt = QPushButton("Exportar curva Δ2 del mejor…")
    self.btn_save_opt.clicked.connect(self._export_opt_curve)
    row_btns.addWidget(self.btn_run_opt)
    row_btns.addWidget(self.btn_save_opt)

    # Barras de progreso
    self.opt_prog_global = QProgressBar(); self.opt_prog_global.setFormat("Global: %p%")
    self.opt_prog_local = QProgressBar(); self.opt_prog_local.setFormat("Local: %p%")

    # (Opcional) Log — lo conservamos porque tu código aún lo referencia
    self.opt_log = QTextEdit(); self.opt_log.setReadOnly(True)

    # Gráfico de Δ2
    self.canvas_opt = MplCanvas(self, height=4)
    self.line_d2, = self.canvas_opt.axes.plot([], [], '-o', label='Δ2 (cents)')
    self.canvas_opt.axes.axhline(0, linewidth=1)
    self.canvas_opt.axes.set_xlabel('índice de paso (L_eff)')
    self.canvas_opt.axes.set_ylabel('Δ2 (cents)')
    self.canvas_opt.axes.grid(True); self.canvas_opt.axes.legend()

    # Live preview (curva y geometría)
    self.chk_live = QCheckBox('Live preview (curva y geometría)')
    self.chk_live.setChecked(True)

    # Canvas para geometría simplificada (perfil 1D)
    self.canvas_opt_geo = MplCanvas(self, height=2.5)
    axg = self.canvas_opt_geo.axes
    self.line_geo, = axg.plot([], [], '-', linewidth=2, label='perfil')
    axg.set_xlabel('x (m)')
    axg.set_ylabel('radio (m)')
    axg.grid(True)
    axg.legend()

    # Eje overlay para embocadura (círculos)
    self.ax_hole = self.canvas_opt_geo.figure.add_axes(axg.get_position(), frameon=False)
    self.ax_hole.set_aspect('equal', adjustable='box')
    self.ax_hole.set_xticks([]); self.ax_hole.set_yticks([])
    for spine in self.ax_hole.spines.values():
        spine.set_visible(False)

    # Círculos embocadura (interno/externo)
    self.circ_hole_out = Circle((0.0, 0.0), radius=0.0, fill=False, linestyle='-', linewidth=2)
    self.circ_hole_in  = Circle((0.0, 0.0), radius=0.0, fill=False, linestyle='--', linewidth=1.5)
    self.ax_hole.add_patch(self.circ_hole_out)
    self.ax_hole.add_patch(self.circ_hole_in)

    # Label con parámetros de la geometría (live)
    self.lbl_geom_params = QLabel()
    self.lbl_geom_params.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    self.lbl_geom_params.setStyleSheet("font-family: monospace; font-size: 12px;")
    self.lbl_geom_params.setWordWrap(True)

    # Montaje
    lay.addWidget(box_ctrl)
    lay.addLayout(row_btns)
    lay.addWidget(self.opt_prog_global)
    lay.addWidget(self.opt_prog_local)
    lay.addWidget(self.opt_log)                 # si luego lo quieres ocultar, comentarlo aquí y en el .py principal
    lay.addWidget(self.canvas_opt)
    lay.addWidget(self.chk_live)
    lay.addWidget(self.canvas_opt_geo)
    lay.addWidget(self.lbl_geom_params)