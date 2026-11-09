import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt

# 1. Cargar datos idénticos al app.py
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
mediana_brotes = df['num_bro'].median()
df['riesgo'] = (df['num_bro'] > mediana_brotes).astype(int)

# 3. Confirmar conjuntos
X = df[['T_max_mensual', 'T_min_mensual', 'T_media_mensual', 'HR_media_mensual', 'mes']]
y = df['riesgo']

# División temporal metodológica (entrenamiento: 2015-2021 [84 meses], prueba: 2022-2023 [24 meses])
X_train, X_test, y_train, y_test = train_test_split_custom = (X.iloc[:84], X.iloc[84:], y.iloc[:84], y.iloc[84:])
X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test

print("--- 3. CONFIRMACIÓN DE CONJUNTOS ---")
print(f"Total registros: {len(df)}")
print(f"Meses de entrenamiento (2015-2021): {len(X_train)}")
print(f"Meses de prueba (2022-2023): {len(X_test)}")
print(f"Valor de la mediana de brotes: {mediana_brotes}")
print(f"Meses de alto riesgo en Entrenamiento: int(y_train.sum()) ({y_train.sum()} meses)")
print(f"Meses de alto riesgo en Prueba: int(y_test.sum()) ({y_test.sum()} meses)")
print(f"¿Falta algún mes?: No, los 108 están completos (freq='ME').\n")

# CONFIGURACIÓN INICIAL (Umbral 0.5 por defecto, sin class_weight)
clf_ini = RandomForestClassifier(random_state=42)
clf_ini.fit(X_train, y_train)
y_pred_ini = clf_ini.predict(X_test)
y_probs_ini = clf_ini.predict_proba(X_test)[:, 1]

cm_ini = confusion_matrix(y_test, y_pred_ini)
tn_i, fp_i, fn_i, tp_i = cm_ini.ravel()
esp_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
auc_i = roc_auc_score(y_test, y_probs_ini)

print("--- 1 & 2. CONFIGURACIÓN INICIAL (Umbral 0.5) ---")
print(f"Matriz de Confusión:\n{cm_ini} (TN={tn_i}, FP={fp_i}, FN={fn_i}, TP={tp_i})")
print(f"Especificidad: {esp_i:.4f}")
print(f"AUC-ROC: {auc_i:.4f}\n")

# CONFIGURACIÓN AJUSTADA (class_weight='balanced', umbral 0.35)
clf_aju = RandomForestClassifier(class_weight='balanced', random_state=42)
clf_aju.fit(X_train, y_train)
y_probs_aju = clf_aju.predict_proba(X_test)[:, 1]
y_pred_aju = (y_probs_aju >= 0.35).astype(int)

cm_aju = confusion_matrix(y_test, y_pred_aju)
tn_a, fp_a, fn_a, tp_a = cm_aju.ravel()
esp_a = tn_a / (tn_a + fp_a) if (tn_a + fp_a) > 0 else 0
auc_a = roc_auc_score(y_test, y_probs_aju)

print("--- 1 & 2. CONFIGURACIÓN AJUSTADA (Umbral 0.35 + Balanced) ---")
print(f"Matriz de Confusión:\n{cm_aju} (TN={tn_a}, FP={fp_a}, FN={fn_a}, TP={tp_a})")
print(f"Especificidad: {esp_a:.4f}")
print(f"AUC-ROC: {auc_a:.4f}\n")

# 4. Importancia de Variables
importancias = clf_aju.feature_importances_
print("--- 4. IMPORTANCIA DE VARIABLES (Random Forest) ---")
for col, imp in zip(X.columns, importancias):
    print(f"- {col}: {imp:.4f}")
print()

# 5. Promedio de brotes por mes calendario (2015-2023)
prom_brotes = df.groupby('mes')['num_bro'].mean()
print("--- 5. PROMEDIO DE BROTES POR MES CALENDARIO ---")
print(prom_brotes)