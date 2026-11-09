# AgroRed 4.0 - Chía: Sistema de Alerta Temprana e Inocuidad Agroindustrial

Prototipo de aplicación web desarrollado en **Flask** que integra un repositorio de datos multivariado (SIVIGILA e IDEAM) y un modelo predictivo de aprendizaje automático (*Random Forest Classifier*) para anticipar y prevenir riesgos de contaminación microbiológica en la cadena de suministro agroindustrial de Chía y la Sabana Centro (Cundinamarca).

---

## 🚀 Características Principales
- **Repositorio Multivariado:** Integración histórica de 108 periodos mensuales (2015–2023) uniendo variables fisicoquímicas/hidroclimáticas (temperaturas y humedad relativa) con el conteo de brotes de enfermedades transmitidas por alimentos (ETA).
- **Modelo Predictivo Optimizado:** Algoritmo *Random Forest* configurado con ponderación de clases (`class_weight='balanced'`) y un umbral de decisión probabilístico del 35% para priorizar la sensibilidad epidemiológica (62.50%).
- **Módulo de Alerta Temprana:** Interfaz interactiva de simulación de riesgos en la cadena de frío bajo parámetros del *Codex Alimentarius*.

---

## 📊 Métricas Oficiales de Rendimiento (Conjunto de Prueba 2022–2023)
Evaluado sobre una partición temporal estricta de 24 meses:
- **Sensibilidad (Recall):** 62.50%
- **Especificidad:** 50.00%
- **Precisión:** 38.46%
- **Exactitud Global:** 54.17%
- **Área Bajo la Curva (AUC-ROC):** 0.5898

---

## 🛠️ Tecnologías y Librerías Utilizadas
- **Backend:** Python, Flask
- **Machine Learning & Análisis de Datos:** Scikit-Learn, Pandas, NumPy
- **Visualización:** Matplotlib, Seaborn, Tailwind CSS (Frontend)

---

## ⚙️ Instrucciones de Instalación y Ejecución Local

Sigue estos pasos para clonar y levantar el proyecto en tu entorno local:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/app_prediccion_riesgo_inocuidad.git](https://github.com/SantiagoSanchezVargas/app_prediccion_riesgo_inocuidad.git)
   cd app_prediccion_riesgo_inocuidad
2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
4. **Ejecutar la aplicación web:**
    ```bash
     python app.py
5. **Abrir en el navegador:**
   Entra a http://127.0.0.1:5000 para interactuar con la plataforma.

## 👥 Autores
Universidad de Cundinamarca - Sede Chía (2026)
