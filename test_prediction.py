# test_prediction.py
import os
import numpy as np
import pandas as pd
import joblib

print("="*60)
print("🏠 PRUEBA DE PREDICCIÓN CORREGIDA")
print("="*60)

# Cargar recursos
processed_path = 'data/processed'
models_path = 'models'

model = joblib.load(os.path.join(models_path, 'random_forest_latest.joblib'))
feature_scaler = joblib.load(os.path.join(processed_path, 'scaler.joblib'))
target_scaler = joblib.load(os.path.join(processed_path, 'target_scaler.joblib'))

# Cargar datos reales para comparar
X_test = pd.read_csv(os.path.join(processed_path, 'X_test_scaled.csv')).values
y_test = pd.read_csv(os.path.join(processed_path, 'y_test.csv')).values.ravel()

# Tomar 5 muestras aleatorias del test set
np.random.seed(42)
indices = np.random.choice(len(X_test), 5, replace=False)

print("\n📋 COMPARACIÓN DE PREDICCIONES:")
print("-" * 80)
print(f"{'No.':<5} {'Valor Real':<15} {'Predicción':<15} {'Diferencia':<15} {'Error %':<10}")
print("-" * 80)

for i, idx in enumerate(indices, 1):
    # Predecir
    X_sample = X_test[idx].reshape(1, -1)
    pred_scaled = model.predict(X_sample)[0]
    pred_original = target_scaler.inverse_transform([[pred_scaled]])[0][0]
    
    real_value = y_test[idx]
    error = abs(pred_original - real_value)
    error_pct = (error / real_value) * 100
    
    print(f"{i:<5} ${real_value:,.0f}       ${pred_original:,.0f}       ${error:,.0f}        {error_pct:.1f}%")

print("-" * 80)
print(f"\n✅ Rango esperado de predicciones: ${pred_original:,.0f} - ${real_value:,.0f}")
print(f"   (Debería estar entre ${y_test.min():,.0f} y ${y_test.max():,.0f})")

# Probar con el ejemplo
print(f"\n🏠 PREDICCIÓN DE EJEMPLO:")
ejemplo = np.array([[5.0, 30.0, 6.0, 1.0, 1000.0, 3.0, 34.0, -118.0]])
ejemplo_scaled = feature_scaler.transform(ejemplo)
pred_scaled = model.predict(ejemplo_scaled)[0]
pred_original = target_scaler.inverse_transform([[pred_scaled]])[0][0]

print(f"   • Predicción escalada: {pred_scaled:.4f}")
print(f"   • Predicción en dólares: ${pred_original:,.2f}")
print(f"   • Predicción en miles: ${pred_original/1000:,.1f}K")
print(f"   • Predicción en millones: ${pred_original/1e6:.2f}M")
print("="*60)