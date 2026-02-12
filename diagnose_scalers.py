import os
import numpy as np
import pandas as pd
import joblib

print("="*60)
print("🔍 DIAGNÓSTICO DE ESCALADORES Y PREDICCIONES")
print("="*60)

# Cargar datos
processed_path = 'data/processed'
models_path = 'models'

# 1. Verificar escaladores
print("\n📊 ESCALADORES:")
feature_scaler = joblib.load(os.path.join(processed_path, 'scaler.joblib'))
target_scaler = joblib.load(os.path.join(processed_path, 'target_scaler.joblib'))

print(f"   • Feature scaler type: {type(feature_scaler).__name__}")
print(f"   • Target scaler type: {type(target_scaler).__name__}")
print(f"   • Target scaler mean: {target_scaler.mean_[0] if hasattr(target_scaler, 'mean_') else 'N/A'}")
print(f"   • Target scaler scale: {target_scaler.scale_[0] if hasattr(target_scaler, 'scale_') else 'N/A'}")

# 2. Verificar datos originales
print("\n📈 DATOS ORIGINALES:")
y_train = pd.read_csv(os.path.join(processed_path, 'y_train.csv')).values.ravel()
print(f"   • y_train min: {y_train.min():.2f}")
print(f"   • y_train max: {y_train.max():.2f}")
print(f"   • y_train mean: {y_train.mean():.2f}")

y_train_transformed = pd.read_csv(os.path.join(processed_path, 'y_train_transformed.csv')).values.ravel()
print(f"\n📉 DATOS TRANSFORMADOS:")
print(f"   • y_train_transformed min: {y_train_transformed.min():.4f}")
print(f"   • y_train_transformed max: {y_train_transformed.max():.4f}")
print(f"   • y_train_transformed mean: {y_train_transformed.mean():.4f}")

# 3. Probar transformación inversa
print("\n🔄 PRUEBA DE TRANSFORMACIÓN INVERSA:")
sample_scaled = np.array([[0.5]])
sample_inverse = target_scaler.inverse_transform(sample_scaled)
print(f"   • Valor escalado: 0.5")
print(f"   • Valor original: {sample_inverse[0][0]:.2f}")

sample_scaled = np.array([[1.0]])
sample_inverse = target_scaler.inverse_transform(sample_scaled)
print(f"   • Valor escalado: 1.0")
print(f"   • Valor original: {sample_inverse[0][0]:.2f}")

sample_scaled = np.array([[2.0]])
sample_inverse = target_scaler.inverse_transform(sample_scaled)
print(f"   • Valor escalado: 2.0")
print(f"   • Valor original: {sample_inverse[0][0]:.2f}")

# 4. Cargar modelo y probar predicción real
print("\n🤖 PRUEBA DE MODELO REAL:")
model = joblib.load(os.path.join(models_path, 'random_forest_latest.joblib'))

# Usar el mismo ejemplo
ejemplo = np.array([[5.0, 30.0, 6.0, 1.0, 1000.0, 3.0, 34.0, -118.0]])
ejemplo_scaled = feature_scaler.transform(ejemplo)
pred_scaled = model.predict(ejemplo_scaled)[0]
pred_original = target_scaler.inverse_transform([[pred_scaled]])[0][0]

print(f"   • Predicción escalada: {pred_scaled:.4f}")
print(f"   • Predicción original: ${pred_original * 100000:.2f} USD")
print(f"   • Predicción (millones): ${pred_original * 100000 / 1e6:.2f}M USD")

print("\n" + "="*60)