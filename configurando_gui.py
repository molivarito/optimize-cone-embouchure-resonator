import sys
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QFormLayout, QLabel,
                             QDoubleSpinBox, QPushButton, QComboBox,
                             QLineEdit, QCheckBox, QFileDialog)
from PyQt6.QtGui import QFont

class ConfiguratorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configurador de Experimentos de Optimización")
        self.setGeometry(150, 150, 800, 700)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        self.params = {}
        self.create_config_panel()
        self.save_button = QPushButton("Guardar Configuración para el Lanzador")
        self.save_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white;")
        self.save_button.clicked.connect(self.save_config)
        layout.addWidget(self.config_panel)
        layout.addWidget(self.save_button)

    def create_config_panel(self):
        self.config_panel = QWidget()
        layout = QVBoxLayout(self.config_panel)
        group_optimizer = QGroupBox("1. Configuración del Optimizador")
        form_optimizer = QFormLayout()
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Differential Evolution", "Nelder-Mead"])
        form_optimizer.addRow("Algoritmo:", self.optimizer_combo)
        self.w_oct = QDoubleSpinBox(value=1.0, minimum=0.0, maximum=100.0, singleStep=0.1)
        self.w_duo = QDoubleSpinBox(value=1.0, minimum=0.0, maximum=100.0, singleStep=0.1)
        form_optimizer.addRow("Peso Octava (w_oct):", self.w_oct)
        form_optimizer.addRow("Peso Duodécima (w_duo):", self.w_duo)
        group_optimizer.setLayout(form_optimizer)
        
        group_params = QGroupBox("2. Parámetros del Instrumento")
        params_layout = QVBoxLayout(group_params)
        help_text = QLabel("Para cada parámetro: marque 'Optimizar' y defina un rango,\nO escriba valores a explorar (separados por comas).")
        help_text.setStyleSheet("font-style: italic; color: grey;")
        params_layout.addWidget(help_text)
        form_layout = QFormLayout()
        param_list = {
            "largo_total": {"label": "Largo Total Instrumento", "default": "1.0"},
            "r1_radio": {"label": "Radio Cilindro R1", "default": "0.015"},
            "agujero_radio_in": {"label": "Radio Interno Embocadura", "default": "0.005"},
            "agujero_radio_out": {"label": "Radio Externo Embocadura", "default": "0.005, 0.01"},
            "agujero_largo": {"label": "Largo Pared Embocadura", "default": "0.002, 0.006, 0.01"},
            "longitud_r1": {"label": "Longitud Cilindro R1", "optim": True, "min": 0.1, "max": 0.9},
            "radio_cono_final": {"label": "Radio Final Cono R3", "optim": True, "min": 0.001, "max": 0.02}
        }
        for name, config in param_list.items():
            row_layout = QHBoxLayout()
            checkbox = QCheckBox()
            min_box = QDoubleSpinBox(decimals=4, minimum=0.0001, maximum=10.0, value=config.get("min", 0.1))
            max_box = QDoubleSpinBox(decimals=4, minimum=0.0001, maximum=10.0, value=config.get("max", 1.0))
            values_edit = QLineEdit(config.get("default", ""))
            
            checkbox.setChecked(config.get("optim", False))
            min_box.setEnabled(checkbox.isChecked())
            max_box.setEnabled(checkbox.isChecked())
            values_edit.setEnabled(not checkbox.isChecked())

            checkbox.toggled.connect(lambda state, mb=min_box, mxb=max_box, ve=values_edit: self.toggle_param_mode(state, mb, mxb, ve))

            row_layout.addWidget(checkbox)
            row_layout.addWidget(QLabel("Optimizar"))
            row_layout.addWidget(min_box)
            row_layout.addWidget(max_box)
            row_layout.addWidget(QLabel("Explorar/Fijo Vals:"))
            row_layout.addWidget(values_edit)
            form_layout.addRow(config["label"], row_layout)
            self.params[name] = {"checkbox": checkbox, "min": min_box, "max": max_box, "values": values_edit}

        params_layout.addLayout(form_layout)
        layout.addWidget(group_optimizer)
        layout.addWidget(group_params)
        layout.addStretch()
        self.config_panel.setLayout(layout)

    def toggle_param_mode(self, state, min_box, max_box, values_edit):
        min_box.setEnabled(state)
        max_box.setEnabled(state)
        values_edit.setEnabled(not state)
        
    def save_config(self):
        config_data = {
            "optimizer_settings": {
                "name": self.optimizer_combo.currentText(),
                "cost_weights": {
                    "oct": self.w_oct.value(),
                    "duo": self.w_duo.value()
                }
            },
            "parameters": {}
        }
        for name, widgets in self.params.items():
            param_config = {}
            if widgets["checkbox"].isChecked():
                param_config["mode"] = "optimize"
                param_config["bounds"] = [widgets["min"].value(), widgets["max"].value()]
            else:
                values_str = widgets["values"].text().strip()
                if not values_str:
                    print(f"Advertencia: El campo para '{name}' está vacío y será omitido.")
                    continue
                try:
                    values = [float(v.strip()) for v in values_str.split(',')]
                    if len(values) > 1:
                        param_config["mode"] = "explore"
                        param_config["values"] = values
                    else:
                        param_config["mode"] = "fixed"
                        param_config["value"] = values[0]
                except (ValueError, IndexError):
                    print(f"Error: Valor inválido para '{name}'. Se omitirá."); continue
            config_data["parameters"][name] = param_config
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Guardar Configuración", "config.json", "JSON Files (*.json)")
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=4)
            print(f"Configuración guardada exitosamente en: {filepath}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfiguratorWindow()
    window.show()
    sys.exit(app.exec())