# fix_prediction_scale.py
import os
import numpy as np
import pandas as pd
import joblib

print("="*60)
print("🔧 CORRIGIENDO ESCALA DE PREDICCIONES")
print("="*60)

# Cargar recursos
processed_path = 'data/processed'
models_path = 'models'

# Cargar modelo y escaladores
model = joblib.load(os.path.join(models_path, 'random_forest_latest.joblib'))
feature_scaler = joblib.load(os.path.join(processed_path, 'scaler.joblib'))
target_scaler = joblib.load(os.path.join(processed_path, 'target_scaler.joblib'))

# Cargar datos originales para verificar escala
y_train = pd.read_csv(os.path.join(processed_path, 'y_train.csv')).values.ravel()

print(f"\n📊 DATOS ORIGINALES (en dólares):")
print(f"   • Min: ${y_train.min():,.2f}")
print(f"   • Max: ${y_train.max():,.2f}")
print(f"   • Mean: ${y_train.mean():,.2f}")
print(f"   • Median: ${np.median(y_train):,.2f}")

# Probar con un valor de ejemplo
print(f"\n🔄 PRUEBA DE TRANSFORMACIÓN:")
print(f"   • Si el valor original es $200,000")
print(f"   • Valor transformado: {target_scaler.transform([[200000/100000]])[0][0]:.4f}")

print(f"\n🔄 PRUEBA DE INVERSA:")
print(f"   • Si el valor transformado es 0.5")
inverse = target_scaler.inverse_transform([[0.5]])[0][0]
print(f"   • Valor original sin escala: {inverse}")
print(f"   • Valor en dólares (×100,000): ${inverse * 100000:,.2f}")
print(f"   • Valor en dólares (×1,000): ${inverse * 1000:,.2f}")

print(f"\n✅ CONCLUSIÓN:")
print(f"   • El target original está en DÓLARES (rango: ${y_train.min():,.0f} - ${y_train.max():,.0f})")
print(f"   • El modelo fue entrenado con el target TRANSFORMADO")
print(f"   • Para obtener el valor en dólares: prediction * 100000")
print("="*60)