import numpy as np
import csv
import datetime
import itertools
import json
import sys
from scipy.optimize import differential_evolution, minimize
from openwind import ImpedanceComputation, Player
from scipy.signal import find_peaks # The missing import statement

class TareaOptimizacion:
    def __init__(self, parametros_fijos, optim_vars_names, bounds, optimizer_options, cost_weights, output_filename):
        self.parametros_fijos = parametros_fijos
        self.optim_vars_names = optim_vars_names
        self.bounds = bounds
        self.x0 = [np.mean(b) for b in bounds]
        self.optimizer_options = optimizer_options
        self.cost_weights = cost_weights
        self.output_filename = output_filename
        self.log_file = None
        self.csv_writer = None

    def __call__(self, parametros_variables):
        # Assign fixed and optimization variables
        params = self.parametros_fijos.copy()
        params.update(dict(zip(self.optim_vars_names, parametros_variables)))

        # Boundary check for Nelder-Mead
        for i, param in enumerate(parametros_variables):
            if not (self.bounds[i][0] <= param <= self.bounds[i][1]):
                return 2e12 

        print(f"  Testing: {' '.join([f'{k}={v:.4f}' for k, v in zip(self.optim_vars_names, parametros_variables)])} ... ", end="")
        
        try:
            # --- Flute Model Simulation Logic ---
            
            longitud_r3 = params['largo_total'] - params['longitud_r1']
            geometria_concatenada = [
                [0, params['longitud_r1'], params['r1_radio'], params['r1_radio'], 'cone'],
                [params['longitud_r1'], params['largo_total'], params['r1_radio'], params['radio_cono_final'], 'cone']
            ]
            agujeros = [['label', 'position', 'radius', 'length', 'radius_out'], 
                        ['embocadura', params.get('agujero_posicion', 0.02), params['agujero_radio_in'], params['agujero_largo'], params['agujero_radio_out']]]

            player_flauta = Player("FLUTE")
            condiciones_flauta = {'bell': 'unflanged', 'entrance': 'closed', 'holes': 'unflanged'}
            
            resultado_sim = ImpedanceComputation(params['frecuencias'], geometria_concatenada, agujeros, 
                                                 player=player_flauta, 
                                                 source_location='embocadura', 
                                                 radiation_category=condiciones_flauta, 
                                                 temperature=params['temperatura'])
            
            # Find resonances (MINIMA) for the flute model
            impedancia_db = 20 * np.log10(np.abs(resultado_sim.impedance))
            indices_minimos, _ = find_peaks(-impedancia_db, prominence=1)
            f_modos = params['frecuencias'][indices_minimos]

            # Calculate cost based on inharmonicity
            costo = 1e12
            if len(f_modos) >= 3:
                f0, f1, f2 = f_modos[0], f_modos[1], f_modos[2]
                if f0 > 0 and f1 > 0 and f2 > 0:
                    delta_oct = 1200 * np.log2(f1 / (2 * f0))
                    delta_duo = 1200 * np.log2(f2 / (3 * f0))
                    costo = self.cost_weights['oct'] * (delta_oct**2) + self.cost_weights['duo'] * (delta_duo**2)
            
            print(f"Cost = {costo:.2f}")

        except Exception as e:
            error_msg = f"Simulation Error: {e}"
            print(error_msg)
            with open("errores_optimizacion_flauta.txt", "a") as f_error:
                f_error.write(f"[{datetime.datetime.now()}] Parameters: {dict(zip(self.optim_vars_names, parametros_variables))}\\n  -> {error_msg}\\n\\n")
            costo = 1e12

        # Log results to CSV
        fila = [costo] + list(parametros_variables)
        frecuencias_a_guardar = list(f_modos if 'f_modos' in locals() and len(f_modos)>0 else [])
        fila.extend(frecuencias_a_guardar); fila.extend([np.nan] * (1 + len(self.optim_vars_names) + 5 - len(fila)))
        self.csv_writer.writerow(fila)
        self.log_file.flush()
        return costo

    def ejecutar(self):
        with open(self.output_filename, 'w', newline='') as f:
            self.log_file = f; self.csv_writer = csv.writer(f)
            self.csv_writer.writerow(['costo'] + self.optim_vars_names + ['f0', 'f1', 'f2', 'f3', 'f4'])
            if self.optimizer_options.pop('name', '') == 'Differential Evolution':
                resultado = differential_evolution(self, self.bounds, **self.optimizer_options)
            else:
                resultado = minimize(self, self.x0, method='Nelder-Mead', options=self.optimizer_options)
        return resultado

if __name__ == '__main__':
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: 'config.json' not found. Please generate it with 'configurador_gui.py'."); sys.exit(1)
        
    optim_vars_names = [k for k,v in config['parameters'].items() if v['mode'] == 'optimize']
    bounds = [v['bounds'] for v in config['parameters'].values() if v['mode'] == 'optimize']
    explore_configs = {k: v['values'] for k,v in config['parameters'].items() if v['mode'] == 'explore'}
    fixed_vars = {k: v['value'] for k,v in config['parameters'].items() if v['mode'] == 'fixed'}

    explore_param_names = list(explore_configs.keys()); explore_param_values = list(explore_configs.values())
    combinaciones = list(itertools.product(*explore_param_values)) if explore_param_values else [()]
    
    opt_name_str = config['optimizer_settings']['name']
    print(f"Executing {len(combinaciones)} optimizations with the '{opt_name_str}' method.")

    for i, combo_values in enumerate(combinaciones):
        current_explore_params = dict(zip(explore_param_names, combo_values))
        print(f"\\n--- STARTING OPTIMIZATION #{(i+1)}/{len(combinaciones)} ---")
        if current_explore_params: print(f"Exploration config: {current_explore_params}")
        
        parametros_fijos_run = {"frecuencias": np.linspace(50, 4000, 2000), "temperatura": 20}
        parametros_fijos_run.update(fixed_vars); parametros_fijos_run.update(current_explore_params)
        
        optimizer_options = {'maxiter': 50, 'disp': True} if "Evolution" in opt_name_str else {'maxiter': 200, 'disp': True}
        optimizer_options['name'] = opt_name_str
        
        config_str = "_".join([f"{k.replace('_','')[0:4]}{v}" for k, v in current_explore_params.items()])
        output_filename = f'log_FLUTE_{opt_name_str.replace(" ","")}_{config_str if config_str else "base"}.csv'
        
        tarea = TareaOptimizacion(parametros_fijos_run, optim_vars_names, bounds, optimizer_options, config['optimizer_settings']['cost_weights'], output_filename)
        resultado = tarea.ejecutar()
        
        print(f"\\nOptimization #{(i+1)} finished.")
        print(f"Best result for this configuration:"); print(f"  - Optimized parameters: {dict(zip(optim_vars_names, resultado.x))}"); print(f"  - Final cost: {resultado.fun:.2f}"); print(f"Results saved to '{output_filename}'")
        
    print("\\n--- ALL OPTIMIZATIONS HAVE FINISHED ---")