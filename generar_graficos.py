import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo académico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# 1. Datos de estacionalidad (Promedios reales arrojados por tu script)
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
promedios = [2.00, 2.33, 1.89, 1.67, 1.67, 2.44, 1.89, 2.22, 2.44, 1.44, 1.22, 1.56]

plt.figure(figsize=(9, 5), dpi=300)
bars = plt.bar(meses, promedios, color='#2e7d32', edgecolor='#1b5e20', alpha=0.85)
# Destacar los picos
for i in [1, 5, 8]:
    bars[i].set_color('#c62828')

plt.title('Figura 5. Promedio Mensual de Brotes ETA en Cundinamarca (2015–2023)', fontsize=12, pad=15, fontweight='bold')
plt.xlabel('Mes Calendario', fontsize=11, labelpad=10)
plt.ylabel('Promedio Histórico de Brotes (SIVIGILA)', fontsize=11, labelpad=10)
plt.ylim(0, 3.0)
plt.axhline(y=2.0, color='gray', linestyle='--', linewidth=1, label='Mediana Histórica (Umbral = 2.0)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figura_estacionalidad_brotes.png', dpi=300)
plt.close()
print("-> Generada: figura_estacionalidad_brotes.png")

# 2. Datos de Importancia de Variables (Random Forest)
features = ['T. Media Mensual', 'Humedad Relativa', 'T. Máxima Mensual', 'T. Mínima Mensual', 'Mes del Año']
importancias = [0.2482, 0.2151, 0.2078, 0.1781, 0.1509]

plt.figure(figsize=(9, 4.5), dpi=300)
y_pos = np.arange(len(features))
plt.barh(y_pos, [imp * 100 for imp in importancias], color='#1565c0', edgecolor='#0d47a1', alpha=0.85)
plt.yticks(y_pos, features)
plt.gca().invert_yaxis()  # Para mostrar la más importante arriba
plt.title('Figura 6. Importancia Relativa de Variables en el Random Forest Classifier', fontsize=12, pad=15, fontweight='bold')
plt.xlabel('Importancia Relativa (%)', fontsize=11, labelpad=10)
for index, value in enumerate(importancias):
    plt.text(value * 100 + 0.5, index, f"{value*100:.2f}%", va='center', fontsize=10, fontweight='bold')
plt.xlim(0, 30)
plt.tight_layout()
plt.savefig('figura_importancia_variables.png', dpi=300)
plt.close()
print("-> Generada: figura_importancia_variables.png")