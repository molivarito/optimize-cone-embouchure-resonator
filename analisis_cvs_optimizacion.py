import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QLabel)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MplCanvas(FigureCanvas):
    """Clase base para un lienzo de Matplotlib."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)

class InteractiveAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador Interactivo de Resultados de Optimización")
        self.setGeometry(100, 100, 1600, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Crear los dos paneles principales
        self.create_scatter_panel()
        self.create_barchart_panel()
        
        main_layout.addWidget(self.scatter_panel, 2) # El scatter plot es más grande
        main_layout.addWidget(self.barchart_panel, 1)

        self.df = None
        self.load_data()

    def create_scatter_panel(self):
        self.scatter_panel = QWidget()
        layout = QVBoxLayout(self.scatter_panel)
        self.scatter_canvas = MplCanvas(self, width=8, height=7, dpi=100)
        
        # Conectar el evento de clic del ratón
        self.scatter_canvas.mpl_connect('pick_event', self.on_pick)
        
        layout.addWidget(QLabel("Haz clic en un punto para analizar su inarmonicidad:"))
        layout.addWidget(self.scatter_canvas)

    def create_barchart_panel(self):
        self.barchart_panel = QWidget()
        layout = QVBoxLayout(self.barchart_panel)
        self.barchart_canvas = MplCanvas(self, width=5, height=5, dpi=100)
        layout.addWidget(QLabel("Gráfico de Inarmonicidad (Cents):"))
        layout.addWidget(self.barchart_canvas)
        # Inicializar con un mensaje
        self.update_barchart(None)

    def load_data(self):
        """Abre un diálogo para cargar el fichero CSV."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Abrir Fichero de Log de Optimización", "", "CSV Files (*.csv)")
        
        if not filepath:
            sys.exit()
            
        try:
            # Lectura robusta del CSV
            self.df = pd.read_csv(filepath, index_col=False)
            # Asegurarse de que las columnas son numéricas
            for col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df.dropna(subset=['costo'], inplace=True)
            
            print(f"Fichero '{filepath}' cargado con {len(self.df)} puntos válidos.")
            self.plot_scatter()
        except Exception as e:
            print(f"Error al cargar o procesar el fichero: {e}")
            sys.exit()

    def plot_scatter(self):
        """Dibuja el mapa del espacio de búsqueda."""
        ax = self.scatter_canvas.ax
        ax.cla() # Limpiar el gráfico anterior

        # Identificar las columnas de los parámetros optimizados
        param_cols = [col for col in self.df.columns if col not in ['costo', 'f0', 'f1', 'f2', 'f3', 'f4']]
        if len(param_cols) < 2:
            print("Error: El fichero CSV no tiene suficientes columnas de parámetros para un gráfico 2D.")
            return
            
        self.x_col, self.y_col = param_cols[0], param_cols[1]

        # Encontrar la mejor solución para marcarla
        df_sorted = self.df.sort_values(by='costo').reset_index()
        if not df_sorted.empty:
            mejor_solucion = df_sorted.iloc[0]
        else:
            mejor_solucion = None

        # Filtramos valores de costo extremos para una mejor visualización del color
        vmin = self.df['costo'].quantile(0.0)
        vmax = self.df['costo'].quantile(0.95)

        # El argumento 'picker=True' es la clave para que los puntos se puedan "clickear"
        sc = ax.scatter(self.df[self.x_col], self.df[self.y_col], c=self.df['costo'], 
                        cmap='viridis_r', alpha=0.8, s=50, picker=True, pickradius=5, vmin=vmin, vmax=vmax)
        
        if mejor_solucion is not None:
            ax.scatter(mejor_solucion[self.x_col], mejor_solucion[self.y_col], 
                       marker='*', color='red', s=400, edgecolors='white', zorder=5,
                       label=f"Mejor Solución (Costo={mejor_solucion['costo']:.0f})")

        cbar = self.scatter_canvas.fig.colorbar(sc, ax=ax)
        cbar.set_label('Valor de la Función de Costos (más bajo = mejor)')
        ax.set_xlabel(self.x_col)
        ax.set_ylabel(self.y_col)
        ax.set_title('Mapa del Espacio de Búsqueda de la Optimización')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        self.scatter_canvas.draw()

    def on_pick(self, event):
        """Función que se ejecuta cuando se hace clic en un punto."""
        if not hasattr(event, 'ind') or not event.ind:
            return

        # Obtener el índice del punto clickeado
        index = event.ind[0]
        
        # Obtener la fila de datos correspondiente
        punto_seleccionado = self.df.iloc[index]
        
        # Actualizar el gráfico de barras
        self.update_barchart(punto_seleccionado)
        
    def update_barchart(self, data_point):
        """Dibuja o actualiza el gráfico de barras de inarmonicidad."""
        ax = self.barchart_canvas.ax
        ax.cla()

        if data_point is None or data_point[['f0', 'f1', 'f2']].isnull().any():
            ax.text(0.5, 0.5, "Haz clic en un punto del mapa\npara ver su inarmonicidad.", 
                    ha='center', va='center', fontsize=12, color='grey')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            f0, f1, f2 = data_point['f0'], data_point['f1'], data_point['f2']
            deltas = []
            if f0 > 0 and f1 > 0: deltas.append(1200 * np.log2(f1 / (2 * f0)))
            if f0 > 0 and f2 > 0: deltas.append(1200 * np.log2(f2 / (3 * f0)))
            
            if 'f3' in data_point and pd.notna(data_point['f3']):
                deltas.append(1200 * np.log2(data_point['f3'] / (4 * f0)))
            
            labels = ['2º Arm.', '3er Arm.', '4º Arm.'][:len(deltas)]
            colors = ['skyblue' if d > 0 else 'salmon' for d in deltas]

            bars = ax.bar(labels, deltas, color=colors)
            ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
            ax.set_ylabel('Desviación (cents)')
            ax.set_title(f"Inarmonicidad del Punto #{data_point.name}")
            ax.bar_label(bars, fmt='%.1f', padding=3)

        self.barchart_canvas.fig.tight_layout()
        self.barchart_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InteractiveAnalyzer()
    window.show()
    sys.exit(app.exec())