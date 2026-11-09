import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

app = Flask(__name__)

# Simulación o carga de la integración metodológica de datos (SIVIGILA + IDEAM)
def cargar_datos_modelo():
    # En un entorno de producción real, aquí se leen los CSV procesados según el protocolo metodológico.
    np.random.seed(42)
    n_registros = 108  # 9 años (2015-2023) * 12 meses
    
    fechas = pd.date_range(start='2015-01-01', periods=n_registros, freq='ME')
    data = {
        'ano': fechas.year,
        'mes': fechas.month,
        'T_max_mensual': np.random.uniform(16.0, 24.0, n_registros),
        'T_min_mensual': np.random.uniform(4.0, 12.0, n_registros),
        'T_media_mensual': np.random.uniform(10.0, 18.0, n_registros),
        'HR_media_mensual': np.random.uniform(65.0, 92.0, n_registros),
        'num_bro': np.random.poisson(lam=2, size=n_registros)
    }
    df = pd.DataFrame(data)
    
    # Construcción de la variable de salida binaria (Mediana histórica)
    mediana_brotes = df['num_bro'].median()
    df['riesgo'] = (df['num_bro'] > mediana_brotes).astype(int)
    return df

df_global = cargar_datos_modelo()

@app.route('/')
def index():
    return render_template('index.html', total_registros=len(df_global))

@app.route('/datos')
def ver_datos():
    # Muestra una tabla con los registros integrados del dataset
    datos_tabla = df_global.head(20).to_html(classes='table table-striped table-bordered', index=False)
    return render_template('datos.html', tabla=datos_tabla)

@app.route('/modelo', methods=['GET', 'POST'])
def modelo():
    resultados = None
    if request.method == 'POST':
        # Extracción de features y división temporal metodológica (2015-2021 train, 2022-2023 test)
        X = df_global[['T_max_mensual', 'T_min_mensual', 'T_media_mensual', 'HR_media_mensual', 'mes']]
        y = df_global['riesgo']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42, shuffle=False)
        
        # 1. Incorporación de class_weight='balanced' para mitigar el desbalance de clases
        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # 2. Obtención de probabilidades y ajuste del umbral de decisión (35%) para elevar la sensibilidad
        y_probs = clf.predict_proba(X_test)[:, 1]
        umbral_sensible = 0.35  
        y_pred = (y_probs >= umbral_sensible).astype(int)
        
        resultados = {
            'sensibilidad': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            'exactitud': round(accuracy_score(y_test, y_pred) * 100, 2)
        }
    return render_template('modelo.html', resultados=resultados)

@app.route('/alerta', methods=['GET', 'POST'])
def alerta_temprana():
    alerta_resultado = None
    if request.method == 'POST':
        t_media = float(request.form['t_media'])
        hr_media = float(request.form['hr_media'])
        mes = int(request.form['mes'])
        
        # Simulación de inferencia basada en umbrales críticos de la cadena de frío
        riesgo_alto = (t_media > 18.0) and (hr_media > 80.0)
        alerta_resultado = {
            'estado': 'ALERTA ROJA: Alto Riesgo de Contaminación Microbiológica' if riesgo_alto else 'ESTABLE: Condiciones Seguras en la Cadena',
            'recomendacion': 'Intensificar control de cadena de frío y verificar tiempos de transporte en la Sabana de Bogotá.' if riesgo_alto else 'Monitoreo continuo estándar bajo normas del Codex Alimentarius.'
        }
    return render_template('alerta.html', alerta=alerta_resultado)

if __name__ == '__main__':
    app.run(debug=True)