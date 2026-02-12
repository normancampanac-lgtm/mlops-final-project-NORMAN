import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, use_mlflow=False):
        """Inicializar el entrenador de modelos"""
        self.use_mlflow = use_mlflow
        
        # Configurar MLflow solo si se solicita
        if self.use_mlflow:
            try:
                import mlflow
                # Crear directorio mlruns
                mlruns_dir = "mlruns"
                os.makedirs(mlruns_dir, exist_ok=True)
                
                # Configurar URI para Windows
                mlflow.set_tracking_uri(f"file:///{os.path.abspath(mlruns_dir).replace('\\\\', '/').replace('\\', '/')}")
                mlflow.set_experiment("California_Housing_Prediction")
                print("✅ MLflow configurado")
            except Exception as e:
                print(f"⚠️  Error MLflow: {e}. Continuando sin tracking...")
                self.use_mlflow = False
        
        self.processed_path = os.path.join('data', 'processed')
        self.models_path = os.path.join('models')
        self.reports_path = os.path.join('reports')
        
        # Crear directorios si no existen
        for path in [self.models_path, self.reports_path]:
            os.makedirs(path, exist_ok=True)
        
        print("🎯 Entrenador de modelo inicializado")
    
    def load_data(self):
        """Cargar datos procesados (target en cientos de miles USD)"""
        
        print("📂 Cargando datos procesados...")
        
        try:
            # Cargar características escaladas
            X_train = pd.read_csv(os.path.join(self.processed_path, 'X_train_scaled.csv')).values
            X_test = pd.read_csv(os.path.join(self.processed_path, 'X_test_scaled.csv')).values
            
            # Cargar target en CIENTOS DE MILES de dólares
            y_train = pd.read_csv(os.path.join(self.processed_path, 'y_train.csv')).values.ravel()
            y_test = pd.read_csv(os.path.join(self.processed_path, 'y_test.csv')).values.ravel()
            
            print(f"✅ Datos cargados:")
            print(f"   • X_train: {X_train.shape}")
            print(f"   • X_test: {X_test.shape}")
            print(f"   • y_train: {y_train.shape}")
            print(f"   • y_test: {y_test.shape}")
            print(f"   • Rango target: {y_train.min():.2f} - {y_train.max():.2f}")
            print(f"   • En USD: ${y_train.min()*100000:,.0f} - ${y_train.max()*100000:,.0f}")
            print(f"   • Precio promedio: ${y_train.mean()*100000:,.0f}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            raise
        
        
    
    def train_random_forest(self, X_train, y_train):
        """Entrenar modelo Random Forest"""
        print("🌲 Entrenando Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print("✅ Random Forest entrenado")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluar el modelo con métricas de regresión"""
        print("\n📊 Evaluando modelo...")
        
        # Realizar predicciones
        y_pred = model.predict(X_test)
        
        # Calcular métricas (en cientos de miles USD)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'samples_test': int(len(y_test))
        }
        
        print(f"📈 Métricas del modelo (en cientos de miles USD):")
        print(f"   • MSE: {mse:.4f}")
        print(f"   • RMSE: {rmse:.4f}")
        print(f"   • MAE: {mae:.4f}")
        print(f"   • R²: {r2:.4f}")
        print(f"\n   Equivalente en USD:")
        print(f"   • RMSE: ${rmse*100000:,.0f}")
        print(f"   • MAE: ${mae*100000:,.0f}")
        
        return metrics, y_pred
    
    def save_model(self, model, metrics, feature_names):
        """Guardar modelo y métricas"""
        print("💾 Guardando modelo...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"random_forest_{timestamp}.joblib"
        latest_filename = "random_forest_latest.joblib"
        
        model_path = os.path.join(self.models_path, model_filename)
        latest_path = os.path.join(self.models_path, latest_filename)
        
        # Guardar modelo
        joblib.dump(model, model_path)
        joblib.dump(model, latest_path)
        
        # Guardar métricas
        metrics_path = os.path.join(self.reports_path, f"metrics_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Guardar información del modelo
        model_info = {
            'model_name': 'RandomForestRegressor',
            'timestamp': timestamp,
            'metrics': metrics,
            'feature_names': feature_names,
            'model_path': model_path
        }
        
        info_path = os.path.join(self.models_path, f"model_info_{timestamp}.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"✅ Modelo guardado: {model_path}")
        print(f"✅ Último modelo: {latest_path}")
        print(f"✅ Métricas guardadas: {metrics_path}")
        
        return model_path
    
    def create_visualizations(self, y_test, y_pred, metrics):
        """Crear visualizaciones de resultados"""
        print("📈 Creando visualizaciones...")
        
        try:
            # Crear gráfico de predicciones vs valores reales
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.title('Predicciones vs Valores Reales - California Housing')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(self.reports_path, f"predictions_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            print(f"✅ Visualización guardada: {plot_path}")
            
        except Exception as e:
            print(f"⚠️  No se pudieron crear visualizaciones: {e}")
    
    def run(self):
        """Ejecutar el pipeline de entrenamiento completo"""
        print("🚀 Iniciando entrenamiento...")
        
        # Cargar datos
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Entrenar modelo
        model = self.train_random_forest(X_train, y_train)
        
        # Evaluar modelo
        metrics, y_pred = self.evaluate_model(model, X_test, y_test)
        
        # Obtener nombres de características
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        # Guardar modelo
        model_path = self.save_model(model, metrics, feature_names)
        
        # Crear visualizaciones
        self.create_visualizations(y_test, y_pred, metrics)
        
        # Log con MLflow si está activado
        if self.use_mlflow:
            try:
                import mlflow
                import mlflow.sklearn
                
                with mlflow.start_run():
                    # Log de parámetros
                    mlflow.log_param("model_type", "RandomForestRegressor")
                    mlflow.log_param("n_estimators", 100)
                    mlflow.log_param("max_depth", 10)
                    
                    # Log de métricas
                    mlflow.log_metrics(metrics)
                    
                    # Log del modelo
                    mlflow.sklearn.log_model(model, "random_forest_model")
                    
                    # Log de artefactos
                    mlflow.log_artifact(model_path)
                    
                print("✅ Métricas registradas en MLflow")
            except Exception as e:
                print(f"⚠️  Error registrando en MLflow: {e}")
        
        print("🎉 Entrenamiento completado exitosamente!")
        return model_path

def main():
    """Función principal"""
    print("=" * 50)
    print("🤖 ENTRENAMIENTO DE MODELO - CALIFORNIA HOUSING")
    print("=" * 50)
    
    try:
        # Preguntar si usar MLflow
        use_mlflow_input = input("¿Usar MLflow para tracking? (s/n): ").lower().strip()
        use_mlflow = use_mlflow_input == 's'
        
        # Inicializar y ejecutar entrenador
        trainer = ModelTrainer(use_mlflow=use_mlflow)
        trainer.run()
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()