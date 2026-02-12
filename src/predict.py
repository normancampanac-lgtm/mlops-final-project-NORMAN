"""
Script de predicción para California Housing
Autor: Norman Campana
Fecha: Febrero 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime

# Agregar directorio raíz al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HousingPredictor:
    """Clase para realizar predicciones con el modelo entrenado"""
    
    def __init__(self, model_path=None):
        """
        Inicializar el predictor
        
        Args:
            model_path (str, optional): Ruta del modelo a cargar
        """
        self.model = None
        self.feature_scaler = None
        self.feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                             'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        # Configurar rutas
        self.models_path = os.path.join('models')
        self.processed_path = os.path.join('data', 'processed')
        
        # Cargar recursos
        self.load_resources(model_path)
    
    def load_resources(self, model_path=None):
        """
        Cargar modelo y escaladores
        
        Args:
            model_path (str, optional): Ruta específica del modelo
        """
        print("📂 Cargando modelo y escaladores...")
        
        try:
            # 1. Cargar modelo
            if model_path is None:
                model_path = os.path.join(self.models_path, 'random_forest_latest.joblib')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
            
            self.model = joblib.load(model_path)
            print(f"   ✓ Modelo cargado: {os.path.basename(model_path)}")
            
            # 2. Cargar escalador de características
            scaler_paths = [
                os.path.join(self.processed_path, 'feature_scaler.joblib'),
                os.path.join(self.processed_path, 'scaler.joblib')
            ]
            
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    self.feature_scaler = joblib.load(scaler_path)
                    print(f"   ✓ Escalador de características cargado: {os.path.basename(scaler_path)}")
                    break
            
            if self.feature_scaler is None:
                print("   ⚠️  No se encontró escalador de características. Usando datos sin escalar.")
            
            print("✅ Recursos cargados exitosamente")
            
        except Exception as e:
            print(f"❌ Error cargando recursos: {e}")
            raise
    
    def predict_single(self, features_dict):
        """
        Realizar predicción para una sola casa
        
        Args:
            features_dict (dict): Diccionario con las características
            
        Returns:
            dict: Resultado de la predicción
        """
        try:
            # Convertir diccionario a array
            features = np.array([[features_dict[feature] for feature in self.feature_names]])
            
            # Escalar características si hay escalador
            if self.feature_scaler is not None:
                features_scaled = self.feature_scaler.transform(features)
            else:
                features_scaled = features
            
            # Realizar predicción (resultado en CIENTOS DE MILES de dólares)
            prediction_hundreds = self.model.predict(features_scaled)[0]
            
            # CONVERTIR A DÓLARES REALES (multiplicar por 100,000)
            prediction_usd = prediction_hundreds * 100000
            
            # Crear resultado
            result = {
                'prediction_hundreds': float(prediction_hundreds),  # Valor en cientos de miles
                'prediction_usd': float(prediction_usd),            # Valor en dólares reales
                'prediction_formatted': f"${prediction_usd:,.2f}",  # Valor formateado
                'features': features_dict,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            raise
    
    def predict_batch(self, X):
        """
        Realizar predicciones para múltiples muestras
        
        Args:
            X (array): Array de características
            
        Returns:
            np.array: Predicciones en dólares reales
        """
        try:
            # Escalar si es necesario
            if self.feature_scaler is not None:
                X_scaled = self.feature_scaler.transform(X)
            else:
                X_scaled = X
            
            # Predecir (resultados en CIENTOS DE MILES)
            predictions_hundreds = self.model.predict(X_scaled)
            
            # CONVERTIR A DÓLARES REALES
            predictions_usd = predictions_hundreds * 100000
            
            return predictions_usd
            
        except Exception as e:
            print(f"❌ Error en predicción por lotes: {e}")
            raise
    
    def get_model_info(self):
        """
        Obtener información del modelo
        
        Returns:
            dict: Información del modelo
        """
        info = {
            'model_type': type(self.model).__name__,
            'has_feature_scaler': self.feature_scaler is not None,
            'feature_names': self.feature_names,
            'target_unit': 'USD',
            'target_multiplier': 100000  # Factor para convertir a dólares
        }
        
        # Si es RandomForest, obtener parámetros
        if hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        # Si hay feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            info['feature_importances'] = {
                name: float(importance)
                for name, importance in zip(self.feature_names, self.model.feature_importances_)
            }
        
        return info

def main():
    """Función principal para probar el predictor"""
    print("\n" + "="*60)
    print("🔮 CALIFORNIA HOUSING - PREDICCIÓN DE PRECIOS")
    print(f"👤 Autor: Norman Campana")
    print("="*60)
    
    try:
        # Inicializar predictor
        predictor = HousingPredictor()
        
        # Mostrar información del modelo
        print("\n📋 Información del modelo:")
        info = predictor.get_model_info()
        print(f"   • Tipo: {info['model_type']}")
        print(f"   • Unidad target: {info['target_unit']}")
        if 'feature_importances' in info:
            print(f"   • Características más importantes:")
            sorted_features = sorted(info['feature_importances'].items(), key=lambda x: x[1], reverse=True)
            for name, importance in sorted_features[:3]:
                print(f"     - {name}: {importance:.4f}")
        
        # Ejemplo de predicción
        print("\n🏠 Probando predicción con ejemplo del dataset...")
        
        # Características de ejemplo (una casa típica)
        ejemplo = {
            'MedInc': 5.0,        # Ingreso medio
            'HouseAge': 30.0,      # Antigüedad de la casa
            'AveRooms': 6.0,       # Número promedio de habitaciones
            'AveBedrms': 1.0,      # Número promedio de dormitorios
            'Population': 1000.0,   # Población del bloque
            'AveOccup': 3.0,       # Ocupantes promedio
            'Latitude': 34.0,      # Latitud
            'Longitude': -118.0    # Longitud
        }
        
        result = predictor.predict_single(ejemplo)
        
        print(f"\n   📊 Características de entrada:")
        for key, value in ejemplo.items():
            print(f"      • {key}: {value}")
        
        print(f"\n   💰 RESULTADO DE LA PREDICCIÓN:")
        print(f"      • Valor en cientos de miles: {result['prediction_hundreds']:.4f}")
        print(f"      • VALOR EN DÓLARES REALES: {result['prediction_formatted']}")
        
        # Mostrar equivalencias
        print(f"\n   📈 Equivalencias:")
        print(f"      • En miles: ${result['prediction_usd']/1000:,.0f}K")
        if result['prediction_usd'] >= 1e6:
            print(f"      • En millones: ${result['prediction_usd']/1e6:.2f}M")
        
        # Guardar predicción de ejemplo
        output_path = os.path.join('reports', 'sample_prediction.json')
        os.makedirs('reports', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            # Crear una versión serializable del resultado
            serializable_result = {
                'prediction_hundreds': result['prediction_hundreds'],
                'prediction_usd': result['prediction_usd'],
                'prediction_formatted': result['prediction_formatted'],
                'features': result['features'],
                'timestamp': result['timestamp'],
                'author': 'Norman Campana'
            }
            json.dump(serializable_result, f, indent=4, ensure_ascii=False)
        print(f"\n   💾 Predicción guardada en: {output_path}")
        
        print("\n" + "="*60)
        print("✅ PRUEBA DE PREDICCIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Primero debes entrenar el modelo:")
        print("   python src/train.py")
    except Exception as e:
        print(f"\n❌ Error durante la predicción: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()