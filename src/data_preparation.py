"""
Preparación de datos para California Housing
Autor: Norman Campana
Fecha: Febrero 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataPreparer:
    """Clase para preparar los datos de California Housing"""
    
    def __init__(self):
        """Inicializar el preparador de datos"""
        self.raw_data_path = os.path.join('data', 'raw', 'california_housing.csv')
        self.processed_path = os.path.join('data', 'processed')
        self.features_path = os.path.join('data', 'features')
        
        # Crear directorios si no existen
        for path in [self.processed_path, self.features_path]:
            os.makedirs(path, exist_ok=True)
        
        print("🎯 Preparador de datos inicializado")
        print(f"   • Datos crudos: {self.raw_data_path}")
        print(f"   • Datos procesados: {self.processed_path}")
    
    def load_data(self):
        """Cargar datos desde el archivo CSV"""
        print("\n📂 Cargando datos...")
        
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.raw_data_path}")
        
        df = pd.read_csv(self.raw_data_path)
        print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        return df
    
    def prepare_features(self, df):
        """
        Preparar características y target
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            tuple: (X, y, feature_names)
        """
        print("\n🔧 Preparando características...")
        
        # California Housing tiene estas columnas
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        X = df[feature_names].copy()
        
        # IMPORTANTE: MedHouseVal está en CIENTOS DE MILES de dólares
        # 1.0 = $100,000 USD
        # 2.5 = $250,000 USD
        # 5.0 = $500,000 USD
        y = df['MedHouseVal'].copy()
        
        print(f"   • Características: {len(feature_names)}")
        print(f"   • Target: MedHouseVal (cientos de miles USD)")
        print(f"   • Rango: {y.min():.2f} - {y.max():.2f}")
        print(f"   • En USD: ${y.min()*100000:,.0f} - ${y.max()*100000:,.0f}")
        print(f"   • Precio promedio: {y.mean():.2f} (${y.mean()*100000:,.0f})")
        
        return X, y, feature_names
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Dividir datos en entrenamiento y prueba
        
        Args:
            X: Características
            y: Target
            test_size: Proporción de prueba
            random_state: Semilla aleatoria
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n✂️  Dividiendo datos...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"   • Train: {X_train.shape[0]} muestras ({int((1-test_size)*100)}%)")
        print(f"   • Test: {X_test.shape[0]} muestras ({int(test_size*100)}%)")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Escalar características (NO escalar el target)
        
        Args:
            X_train: Características de entrenamiento
            X_test: Características de prueba
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, scaler)
        """
        print("\n📏 Escalando características...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("   • Feature scaling: StandardScaler")
        print(f"   • Media (primer feature): {X_train.iloc[:, 0].mean():.2f} → {X_train_scaled[:, 0].mean():.4f}")
        print(f"   • Std (primer feature): {X_train.iloc[:, 0].std():.2f} → {X_train_scaled[:, 0].std():.4f}")
        
        return X_train_scaled, X_test_scaled, scaler
    
    def save_data(self, X_train, X_test, y_train, y_test, 
                X_train_scaled, X_test_scaled, scaler, feature_names):
        """
        Guardar todos los datos procesados
        
        Args:
            X_train: Características originales de entrenamiento
            X_test: Características originales de prueba
            y_train: Target de entrenamiento (en cientos de miles USD)
            y_test: Target de prueba (en cientos de miles USD)
            X_train_scaled: Características escaladas de entrenamiento
            X_test_scaled: Características escaladas de prueba
            scaler: Escalador ajustado
            feature_names: Nombres de las características
        """
        print("\n💾 Guardando datos procesados...")
        
        # Guardar datos originales (sin escalar)
        pd.DataFrame(X_train, columns=feature_names).to_csv(
            os.path.join(self.processed_path, 'X_train.csv'), index=False
        )
        pd.DataFrame(X_test, columns=feature_names).to_csv(
            os.path.join(self.processed_path, 'X_test.csv'), index=False
        )
        pd.DataFrame(y_train, columns=['MedHouseVal']).to_csv(
            os.path.join(self.processed_path, 'y_train.csv'), index=False
        )
        pd.DataFrame(y_test, columns=['MedHouseVal']).to_csv(
            os.path.join(self.processed_path, 'y_test.csv'), index=False
        )
        
        # Guardar datos escalados (solo características)
        pd.DataFrame(X_train_scaled, columns=feature_names).to_csv(
            os.path.join(self.processed_path, 'X_train_scaled.csv'), index=False
        )
        pd.DataFrame(X_test_scaled, columns=feature_names).to_csv(
            os.path.join(self.processed_path, 'X_test_scaled.csv'), index=False
        )
        
        # Guardar en formato numpy para compatibilidad
        np.save(os.path.join(self.processed_path, 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(self.processed_path, 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(self.processed_path, 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_path, 'y_test.npy'), y_test)
        
        # Guardar escalador de características
        joblib.dump(scaler, os.path.join(self.processed_path, 'feature_scaler.joblib'))
        joblib.dump(scaler, os.path.join(self.processed_path, 'scaler.joblib'))
        
        # Guardar metadatos
        metadata = {
            'n_samples': len(y_train) + len(y_test),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'target_name': 'MedHouseVal',
            'target_unit': 'cientos_de_miles_USD',
            'target_multiplier': 100000,  # Factor para convertir a USD
            'target_min': float(y_train.min()),
            'target_max': float(y_train.max()),
            'target_mean': float(y_train.mean()),
            'target_median': float(np.median(y_train)),
            'scaler_type': 'StandardScaler',
            'target_scaled': False,
            'timestamp': pd.Timestamp.now().isoformat(),
            'author': 'Norman Campana'
        }
        
        with open(os.path.join(self.processed_path, 'metadata.json'), 'w') as f:
            import json
            json.dump(metadata, f, indent=4)
        
        print(f"✅ Datos guardados en: {self.processed_path}")
        print(f"   • Archivos generados: {len(os.listdir(self.processed_path))}")
        
        return metadata
    
    def run(self):
        """Ejecutar el pipeline completo de preparación de datos"""
        print("\n" + "="*60)
        print("🚀 INICIANDO PREPARACIÓN DE DATOS")
        print("="*60)
        
        try:
            # 1. Cargar datos
            df = self.load_data()
            
            # 2. Preparar características
            X, y, feature_names = self.prepare_features(df)
            
            # 3. Dividir datos
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # 4. Escalar características (solo X, NO y)
            X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)
            
            # 5. Guardar datos
            metadata = self.save_data(
                X_train, X_test, y_train, y_test,
                X_train_scaled, X_test_scaled, scaler,
                feature_names
            )
            
            print("\n" + "="*60)
            print("🎉 PREPARACIÓN DE DATOS COMPLETADA")
            print("="*60)
            print(f"\n📋 RESUMEN:")
            print(f"   • Muestras totales: {metadata['n_samples']}")
            print(f"   • Features: {metadata['n_features']}")
            print(f"   • Target: {metadata['target_name']} ({metadata['target_unit']})")
            print(f"   • Rango de precios: ${metadata['target_min']:,.0f} - ${metadata['target_max']:,.0f}")
            print(f"   • Precio promedio: ${metadata['target_mean']:,.0f}")
            print(f"\n   • Train: {metadata['n_train']} muestras")
            print(f"   • Test: {metadata['n_test']} muestras")
            print(f"\n📂 Datos guardados en: data/processed/")
            
            return metadata
            
        except Exception as e:
            print(f"\n❌ Error durante la preparación: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Función principal"""
    preparer = DataPreparer()
    preparer.run()
    
    print("\n✅ Proceso completado. Revisa la carpeta data/processed/")
    print("\n📋 Para entrenar el modelo:")
    print("   python src/train.py")

if __name__ == "__main__":
    main()