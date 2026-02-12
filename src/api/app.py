"""
API para servir el modelo de predicción de precios de casas en California
Autor: Norman Campana
Fecha: Febrero 2024
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# Añadir el directorio raíz al path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR))

# Inicializar FastAPI
app = FastAPI(
    title="California Housing Price Prediction API",
    description="API para predecir el valor de casas en California - Desarrollada por Norman Campana",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# CONFIGURACIÓN DE RUTAS CORREGIDA
# ============================================
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = MODELS_DIR / "random_forest_latest.joblib"
SCALER_PATH = PROCESSED_DIR / "scaler.joblib"

# Variables globales
model = None
feature_scaler = None
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                 'Population', 'AveOccup', 'Latitude', 'Longitude']

# ============================================
# MODELOS DE DATOS ACTUALIZADOS (Pydantic V2)
# ============================================
class HouseFeatures(BaseModel):
    """Esquema para características de la vivienda"""
    MedInc: float = Field(..., description="Ingreso medio en el bloque (en decenas de miles)", ge=0, le=20)
    HouseAge: float = Field(..., description="Antigüedad promedio de las casas", ge=0, le=100)
    AveRooms: float = Field(..., description="Promedio de habitaciones", ge=1, le=20)
    AveBedrms: float = Field(..., description="Promedio de dormitorios", ge=0.5, le=10)
    Population: float = Field(..., description="Población del bloque", ge=1, le=50000)
    AveOccup: float = Field(..., description="Promedio de ocupantes", ge=1, le=1000)
    Latitude: float = Field(..., description="Latitud", ge=32, le=42)
    Longitude: float = Field(..., description="Longitud", ge=-125, le=-114)
    
    # Configuración actualizada para Pydantic V2
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
    )

class PredictionResponse(BaseModel):
    """Esquema para respuesta de predicción"""
    prediction_hundreds: float = Field(..., description="Valor en cientos de miles de dólares")
    prediction_usd: float = Field(..., description="Valor en dólares reales")
    prediction_usd_formatted: str = Field(..., description="Valor formateado con símbolo $")
    prediction_usd_k: str = Field(..., description="Valor en miles de dólares")
    prediction_usd_m: str = Field(..., description="Valor en millones de dólares")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    features: Dict[str, float] = Field(..., description="Características utilizadas")
    model_version: str = Field(..., description="Versión del modelo")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_hundreds": 2.0742,
                "prediction_usd": 207420.00,
                "prediction_usd_formatted": "$207,420.00",
                "prediction_usd_k": "$207K",
                "prediction_usd_m": "$0.21M",
                "timestamp": "2024-02-12T11:06:38",
                "features": {"MedInc": 8.3252},
                "model_version": "1.0.0"
            }
        }
    )

class BatchPredictionRequest(BaseModel):
    """Esquema para solicitud de predicción por lotes"""
    features: List[HouseFeatures]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [
                    {
                        "MedInc": 8.3252,
                        "HouseAge": 41.0,
                        "AveRooms": 6.984127,
                        "AveBedrms": 1.023810,
                        "Population": 322.0,
                        "AveOccup": 2.555556,
                        "Latitude": 37.88,
                        "Longitude": -122.23
                    }
                ]
            }
        }
    )

class BatchPredictionResponse(BaseModel):
    """Esquema para respuesta de predicción por lotes"""
    predictions: List[Dict[str, Any]]
    count: int
    total_value_usd: float
    total_value_formatted: str
    average_value_usd: float
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Esquema para información del modelo"""
    model_type: str
    feature_names: List[str]
    feature_importances: Dict[str, float]
    n_features: int
    target_unit: str
    target_multiplier: int
    parameters: Dict[str, Any]
    author: str
    version: str

# ============================================
# FUNCIONES DE CARGA CORREGIDAS
# ============================================
def load_resources():
    """Cargar modelo y escalador desde las rutas correctas"""
    global model, feature_scaler
    
    print("🚀 Cargando recursos del modelo...")
    
    try:
        # Verificar rutas absolutas
        print(f"   • Buscando modelo en: {MODEL_PATH}")
        
        if not MODEL_PATH.exists():
            # Intentar ruta alternativa (desarrollo local)
            alt_path = BASE_DIR / "src" / "models" / "random_forest_latest.joblib"
            if alt_path.exists():
                print(f"   • Usando ruta alternativa: {alt_path}")
                model_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Modelo no encontrado en {MODEL_PATH}\n"
                    f"   • Asegúrate de haber ejecutado: python src/train.py\n"
                    f"   • El modelo debería estar en: {MODELS_DIR}"
                )
        else:
            model_path = MODEL_PATH
        
        # Cargar modelo
        model = joblib.load(model_path)
        print(f"   ✓ Modelo cargado: {model_path.name}")
        print(f"   • Tipo: {type(model).__name__}")
        
        # Verificar escalador
        print(f"   • Buscando escalador en: {SCALER_PATH}")
        if SCALER_PATH.exists():
            feature_scaler = joblib.load(SCALER_PATH)
            print(f"   ✓ Escalador cargado: {SCALER_PATH.name}")
        else:
            print("   ⚠️  Escalador no encontrado. Usando datos sin escalar.")
            feature_scaler = None
        
        print("✅ Recursos cargados exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error cargando recursos: {e}")
        return False

# ============================================
# EVENTOS DE INICIO
# ============================================
@app.on_event("startup")
async def startup_event():
    """Cargar recursos al iniciar la aplicación"""
    print("\n" + "="*60)
    print("🚀 INICIANDO API DE PREDICCIÓN DE PRECIOS DE CASAS")
    print(f"👤 Autor: Norman Campana")
    print("="*60 + "\n")
    
    success = load_resources()
    if success:
        print("\n✨ API lista para recibir peticiones")
        print("📚 Documentación disponible en: http://localhost:8000/docs")
    else:
        print("\n⚠️  API iniciada con errores. Verifica los recursos.")

# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
async def root():
    """Endpoint raíz - Información de la API"""
    return {
        "message": "Bienvenido a la API de Predicción de Precios de Casas de California",
        "author": "Norman Campana",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "scaler_loaded": feature_scaler is not None,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "model_info": "/model/info",
            "features_info": "/features/info",
            "predict": "/predict (POST)",
            "predict_batch": "/predict/batch (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """Verificar estado de la API"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "scaler_loaded": feature_scaler is not None,
        "service": "California Housing Price Prediction API",
        "author": "Norman Campana",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Obtener información detallada del modelo"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    info = {
        "model_type": type(model).__name__,
        "feature_names": feature_names,
        "feature_importances": {},
        "n_features": len(feature_names),
        "target_unit": "USD",
        "target_multiplier": 100000,
        "parameters": {},
        "author": "Norman Campana",
        "version": "1.0.0"
    }
    
    # Obtener importancia de características
    if hasattr(model, 'feature_importances_'):
        info["feature_importances"] = {
            name: float(importance)
            for name, importance in zip(feature_names, model.feature_importances_)
        }
    
    # Obtener parámetros del modelo
    if hasattr(model, 'get_params'):
        params = model.get_params()
        info["parameters"] = {
            k: str(v) for k, v in params.items() 
            if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']
        }
    
    return info

@app.get("/features/info")
async def features_info():
    """Información sobre las características del modelo"""
    feature_descriptions = {
        "MedInc": "Ingreso mediano del área (en decenas de miles de dólares)",
        "HouseAge": "Edad mediana de las casas (años)",
        "AveRooms": "Promedio de habitaciones por vivienda",
        "AveBedrms": "Promedio de dormitorios por vivienda",
        "Population": "Población del bloque",
        "AveOccup": "Promedio de ocupantes por vivienda",
        "Latitude": "Latitud del área",
        "Longitude": "Longitud del área"
    }
    
    feature_ranges = {
        "MedInc": "0 - 20",
        "HouseAge": "0 - 100",
        "AveRooms": "1 - 20",
        "AveBedrms": "0.5 - 10",
        "Population": "1 - 50000",
        "AveOccup": "1 - 1000",
        "Latitude": "32 - 42",
        "Longitude": "-124 - -114"
    }
    
    return {
        "features": feature_names,
        "descriptions": feature_descriptions,
        "ranges": feature_ranges,
        "target": {
            "name": "MedHouseVal",
            "description": "Precio mediano de la vivienda",
            "unit": "cientos de miles USD",
            "multiplier": 100000,
            "unit_real": "USD"
        },
        "author": "Norman Campana"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures):
    """
    Predecir precio de vivienda basado en características
    
    Args:
        features (HouseFeatures): Características de la vivienda
        
    Returns:
        PredictionResponse: Predicción del precio en dólares reales
    """
    try:
        # Verificar que el modelo está cargado
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecuta primero: python src/train.py")
        
        # Convertir características a array numpy
        feature_dict = features.model_dump()  # Cambiado de .dict() a .model_dump() para Pydantic V2
        features_array = np.array([[feature_dict[name] for name in feature_names]])
        
        # Escalar características si el escalador está disponible
        if feature_scaler is not None:
            features_scaled = feature_scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Realizar predicción (resultado en CIENTOS DE MILES)
        prediction_hundreds = model.predict(features_scaled)[0]
        
        # CONVERTIR A DÓLARES REALES (multiplicar por 100,000)
        prediction_usd = prediction_hundreds * 100000
        
        # Formatear valores
        if prediction_usd >= 1e6:
            formatted = f"${prediction_usd/1e6:.2f}M"
            k_formatted = f"${prediction_usd/1000:,.0f}K"
            m_formatted = f"${prediction_usd/1e6:.2f}M"
        else:
            formatted = f"${prediction_usd:,.2f}"
            k_formatted = f"${prediction_usd/1000:,.0f}K"
            m_formatted = f"${prediction_usd/1e6:.2f}M"
        
        return PredictionResponse(
            prediction_hundreds=float(prediction_hundreds),
            prediction_usd=float(prediction_usd),
            prediction_usd_formatted=formatted,
            prediction_usd_k=k_formatted,
            prediction_usd_m=m_formatted,
            timestamp=datetime.now().isoformat(),
            features=feature_dict,
            model_version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error en la predicción: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predicción por lotes para múltiples viviendas
    
    Args:
        request (BatchPredictionRequest): Lista de características de viviendas
        
    Returns:
        BatchPredictionResponse: Predicciones para todas las viviendas
    """
    try:
        # Verificar que el modelo está cargado
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no disponible")
        
        predictions = []
        total_value = 0.0
        
        for features in request.features:
            # Convertir características a array
            feature_dict = features.model_dump()  # Cambiado de .dict() a .model_dump()
            features_array = np.array([[feature_dict[name] for name in feature_names]])
            
            # Escalar si es necesario
            if feature_scaler is not None:
                features_scaled = feature_scaler.transform(features_array)
            else:
                features_scaled = features_array
            
            # Predecir (en cientos de miles)
            prediction_hundreds = model.predict(features_scaled)[0]
            
            # Convertir a dólares reales
            prediction_usd = prediction_hundreds * 100000
            total_value += prediction_usd
            
            # Formatear
            if prediction_usd >= 1e6:
                formatted = f"${prediction_usd/1e6:.2f}M"
            else:
                formatted = f"${prediction_usd:,.2f}"
            
            predictions.append({
                "features": feature_dict,
                "prediction_hundreds": float(prediction_hundreds),
                "prediction_usd": float(prediction_usd),
                "prediction_formatted": formatted
            })
        
        # Calcular estadísticas
        average_value = total_value / len(predictions) if predictions else 0
        
        # Formatear total
        if total_value >= 1e6:
            total_formatted = f"${total_value/1e6:.2f}M"
        else:
            total_formatted = f"${total_value:,.2f}"
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            total_value_usd=float(total_value),
            total_value_formatted=total_formatted,
            average_value_usd=float(average_value),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción por lotes: {str(e)}"
        )

@app.get("/example")
async def get_example():
    """Obtener un ejemplo de predicción"""
    example_features = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    return {
        "message": "Ejemplo de características para predicción",
        "features": example_features,
        "curl_command": f'curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d \'{str(example_features).replace("'", '"')}\'',
        "python_example": """
import requests
import json

response = requests.post(
    "http://localhost:8000/predict",
    json=json.loads('{}')
)
print(response.json())
        """.format(str(example_features).replace("'", '"'))
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🌐 INICIANDO SERVIDOR API - CALIFORNIA HOUSING")
    print(f"👤 Autor: Norman Campana")
    print("="*60 + "\n")
    print(f"📚 Documentación: http://localhost:8000/docs")
    print(f"📋 Redoc: http://localhost:8000/redoc")
    print(f"🔍 Health check: http://localhost:8000/health")
    print(f"\n📌 Presiona Ctrl+C para detener el servidor\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )