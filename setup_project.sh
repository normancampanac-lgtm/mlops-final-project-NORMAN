#!/bin/bash

# ============================================
# SCRIPT PARA PROYECTO MLOPS - REUTILIZA .venv
# ============================================

echo "🚀 INICIANDO PROYECTO MLOPS: mlops-final-project_1"
echo "=================================================="
echo ""

# ----------------------------------------------------
# 1. VERIFICAR DIRECTORIO ACTUAL
# ----------------------------------------------------
echo "🔍 Verificando ubicación..."
CURRENT_DIR=$(pwd)
if [[ $CURRENT_DIR != *"mlops-final-project_1"* ]]; then
    echo "⚠️  Parece que no estás en el directorio correcto"
    echo "   Directorio actual: $CURRENT_DIR"
    echo "   Deberías estar en: .../mlops-final-project_1/"
    read -p "¿Continuar de todos modos? (s/n): " continue_anyway
    if [[ $continue_anyway != "s" && $continue_anyway != "S" ]]; then
        echo "❌ Ejecución cancelada"
        exit 1
    fi
fi
echo "✅ Directorio verificado"
echo ""

# ----------------------------------------------------
# 2. ACTIVAR ENTORNO VIRTUAL EXISTENTE (.venv)
# ----------------------------------------------------
echo "🔧 Activando entorno virtual..."
if [ -d ".venv" ]; then
    echo "✅ Encontrado: .venv/"
    
    # Intentar activar .venv (diferentes sistemas)
    if [ -f ".venv/bin/activate" ]; then
        # Linux/Mac
        source .venv/bin/activate
        echo "✅ Entorno virtual activado (Linux/Mac)"
    elif [ -f ".venv/Scripts/activate" ]; then
        # Windows Git Bash
        source .venv/Scripts/activate
        echo "✅ Entorno virtual activado (Windows)"
    else
        echo "⚠️  No se encontró script de activación en .venv/"
        echo "   Creando uno nuevo..."
        python -m venv .venv
        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        elif [ -f ".venv/Scripts/activate" ]; then
            source .venv/Scripts/activate
        fi
    fi
else
    echo "❌ No se encontró .venv/"
    echo "   Creando nuevo entorno virtual..."
    python -m venv .venv
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    fi
    echo "✅ Entorno virtual creado y activado"
fi

# Verificar que Python del entorno virtual está activo
PYTHON_PATH=$(which python)
echo "📌 Python activo: $PYTHON_PATH"
echo ""

# ----------------------------------------------------
# 3. INSTALAR/ACTUALIZAR DEPENDENCIAS
# ----------------------------------------------------
echo "📦 Gestionando dependencias..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "📋 Instalando desde requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Dependencias instaladas/actualizadas"
else
    echo "⚠️  requirements.txt no encontrado"
    echo "📦 Instalando paquetes básicos..."
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook
    pip install mlflow joblib fastapi uvicorn pydantic
    echo "✅ Paquetes básicos instalados"
fi
echo ""

# ----------------------------------------------------
# 4. VERIFICAR ESTRUCTURA DE DIRECTORIOS
# ----------------------------------------------------
echo "📁 Verificando estructura de directorios..."

# Directorios esenciales que DEBEN existir
essential_dirs=(
    "data/raw"
    "data/processed"
    "notebooks"
    "models"
    "reports"
    "src"
)

for dir in "${essential_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "   📂 Creando: $dir/"
        mkdir -p "$dir"
    else
        echo "   ✅ Existe: $dir/"
    fi
done

# Directorios opcionales
optional_dirs=(
    "data/features"
    "experiments"
    "tests"
    "docs"
    "resources/images"
    "src/api"
    "src/utils"
)

for dir in "${optional_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "   📁 Opcional: $dir/ (no creado)"
    else
        echo "   ✅ Opcional: $dir/"
    fi
done
echo ""

# ----------------------------------------------------
# 5. DESCARGAR DATASET (solo si no existe)
# ----------------------------------------------------
DATASET_PATH="data/raw/california_housing.csv"

if [ -f "$DATASET_PATH" ]; then
    echo "📊 Dataset ya existe: $DATASET_PATH"
    echo "   ℹ️  Para re-descargar, elimina el archivo primero"
else
    echo "📊 Descargando dataset California Housing..."
    python -c "
import sys
try:
    from sklearn.datasets import fetch_california_housing
    import pandas as pd
    import os
    
    print('🔍 Conectando a scikit-learn...')
    california = fetch_california_housing()
    
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['MedHouseVal'] = california.target * 100000
    
    print(f'📦 Dataset descargado: {df.shape[0]:,} filas, {df.shape[1]} columnas')
    print(f'💵 Rango de precios: \${df[\"MedHouseVal\"].min():,.0f} - \${df[\"MedHouseVal\"].max():,.0f}')
    
    df.to_csv('$DATASET_PATH', index=False)
    print(f'✅ Guardado en: $DATASET_PATH')
    
except Exception as e:
    print(f'❌ Error: {e}')
    print('⚠️  Posibles soluciones:')
    print('   1. Verifica tu conexión a internet')
    print('   2. Instala scikit-learn: pip install scikit-learn')
    sys.exit(1)
"
fi
echo ""

# ----------------------------------------------------
# 6. MENÚ INTERACTIVO
# ----------------------------------------------------
while true; do
    echo "🎮 MENÚ PRINCIPAL - ¿QUÉ QUIERES HACER?"
    echo "=================================================="
    echo "1. 📊 Ejecutar Análisis Exploratorio (Jupyter)"
    echo "2. 🔧 Ejecutar TODO el pipeline (preparar + entrenar + API)"
    echo "3. 🛠️  Solo preparar datos"
    echo "4. 🤖 Solo entrenar modelo"
    echo "5. 🌐 Solo iniciar API"
    echo "6. 🔮 Probar predicciones"
    echo "7. 📋 Verificar estado del proyecto"
    echo "8. ❌ Salir"
    echo ""

    read -p "Selecciona una opción (1-8): " option
    echo ""

    case $option in
        1)
            echo "📊 EJECUTANDO ANÁLISIS EXPLORATORIO..."
            echo "   Abriendo Jupyter Notebook..."
            
            # Buscar notebook de EDA
            if [ -f "notebooks/01_eda.ipynb" ]; then
                echo "   ✅ Encontrado: notebooks/01_eda.ipynb"
                echo "   ⚠️  Cierra el notebook cuando termines para volver al menú"
                echo ""
                jupyter notebook notebooks/01_eda.ipynb
            elif [ -f "01_eda.ipynb" ]; then
                echo "   ✅ Encontrado: 01_eda.ipynb"
                jupyter notebook 01_eda.ipynb
            else
                echo "❌ No se encontró notebook de EDA"
                echo "   Creando uno básico..."
                python -c "
import json

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': ['# Análisis Exploratorio\\n## mlops-final-project_1']
        }
    ],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

import os
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/01_eda.ipynb', 'w') as f:
    json.dump(notebook, f)

print('✅ Notebook creado: notebooks/01_eda.ipynb')
print('   Ejecuta esta opción nuevamente para abrirlo')
"
            fi
            ;;
            
        2)
            echo "🚀 EJECUTANDO PIPELINE COMPLETO..."
            echo ""
            
            # 6.2.1 Preparar datos
            echo "🔄 PASO 1: Preparando datos..."
            if [ -f "src/data_preparation.py" ]; then
                python src/data_preparation.py
                echo "✅ Datos preparados"
            else
                echo "⚠️  Archivo no encontrado: src/data_preparation.py"
                echo "   Creando versión básica..."
                python -c "
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Cargar datos
df = pd.read_csv('data/raw/california_housing.csv')
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar
os.makedirs('data/processed', exist_ok=True)
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/processed/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/processed/X_test_scaled.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
joblib.dump(scaler, 'data/processed/scaler.joblib')

print('✅ Datos preparados y guardados en data/processed/')
"
            fi
            echo ""
            
            # 6.2.2 Entrenar modelo
            echo "🤖 PASO 2: Entrenando modelo..."
            if [ -f "src/train.py" ]; then
                python src/train.py
            else
                echo "⚠️  Archivo no encontrado: src/train.py"
                echo "   Creando versión básica..."
                python -c "
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import os

print('📊 Cargando datos...')
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print('🌲 Entrenando Random Forest...')
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print('📈 Evaluando modelo...')
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'✅ Modelo entrenado:')
print(f'   • RMSE: \${rmse:,.2f}')
print(f'   • R²: {r2:.4f}')

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/random_forest_model.joblib')
print(f'💾 Modelo guardado: models/random_forest_model.joblib')
"
            fi
            echo ""
            
            # 6.2.3 Probar predicciones
            echo "🔮 PASO 3: Probando predicciones..."
            if [ -f "src/predict.py" ]; then
                python src/predict.py
            else
                echo "⚠️  Archivo no encontrado: src/predict.py"
                echo "   Probando con datos de ejemplo..."
                python -c "
import joblib
import numpy as np

try:
    model = joblib.load('models/random_forest_model.joblib')
    print('✅ Modelo cargado correctamente')
    
    # Datos de ejemplo (una casa)
    example_house = np.array([[8.3252, 41.0, 6.984127, 1.023810, 
                               322.0, 2.555556, 37.88, -122.23]])
    
    prediction = model.predict(example_house)[0]
    print(f'🏠 Predicción de ejemplo: \${prediction:,.2f}')
    
except Exception as e:
    print(f'❌ Error: {e}')
"
            fi
            echo ""
            
            # 6.2.4 Iniciar API
            echo "🌐 PASO 4: Iniciando API..."
            if [ -f "src/api/app.py" ]; then
                echo "   API disponible en: http://localhost:8000"
                echo "   Documentación: http://localhost:8000/docs"
                echo ""
                echo "📋 Para probar (en otra terminal):"
                echo "   curl -X POST \\"
                echo "     -H \"Content-Type: application/json\" \\"
                echo "     -d '{\"MedInc\":8.3,\"HouseAge\":41,\"AveRooms\":7,\"AveBedrms\":1,\"Population\":322,\"AveOccup\":2.6,\"Latitude\":37.88,\"Longitude\":-122.23}' \\"
                echo "     http://localhost:8000/predict"
                echo ""
                echo "   Presiona Ctrl+C para detener la API"
                echo ""
                python src/api/app.py
            else
                echo "⚠️  Archivo no encontrado: src/api/app.py"
                echo "   La API no se iniciará automáticamente"
            fi
            ;;
            
        3)
            echo "🛠️  PREPARANDO DATOS..."
            if [ -f "src/data_preparation.py" ]; then
                python src/data_preparation.py
            else
                echo "❌ Archivo no encontrado: src/data_preparation.py"
            fi
            ;;
            
        4)
            echo "🤖 ENTRENANDO MODELO..."
            if [ -f "src/train.py" ]; then
                python src/train.py
            else
                echo "❌ Archivo no encontrado: src/train.py"
            fi
            ;;
            
        5)
            echo "🌐 INICIANDO API..."
            if [ -f "src/api/app.py" ]; then
                echo "📌 Endpoints disponibles:"
                echo "   • http://localhost:8000/          - Página principal"
                echo "   • http://localhost:8000/docs      - Documentación interactiva"
                echo "   • http://localhost:8000/predict   - Endpoint de predicción"
                echo ""
                echo "📋 Ejemplo de uso:"
                echo "   curl -X POST http://localhost:8000/predict \\"
                echo "     -H \"Content-Type: application/json\" \\"
                echo "     -d '{\"MedInc\":8.3,\"HouseAge\":41,\"AveRooms\":7,\"AveBedrms\":1,\"Population\":322,\"AveOccup\":2.6,\"Latitude\":37.88,\"Longitude\":-122.23}'"
                echo ""
                echo "🔄 Iniciando servidor..."
                python src/api/app.py
            else
                echo "❌ Archivo no encontrado: src/api/app.py"
            fi
            ;;
            
        6)
            echo "🔮 PROBANDO PREDICCIONES..."
            if [ -f "src/predict.py" ]; then
                python src/predict.py
            else
                echo "❌ Archivo no encontrado: src/predict.py"
            fi
            ;;
            
        7)
            echo "📋 ESTADO DEL PROYECTO"
            echo "======================"
            
            # Verificar archivos importantes
            important_files=(
                "requirements.txt"
                "README.md"
                "setup.py"
                "data/raw/california_housing.csv"
                "notebooks/01_eda.ipynb"
                "src/data_preparation.py"
                "src/train.py"
                "src/predict.py"
                "src/api/app.py"
            )
            
            echo "📁 ARCHIVOS ESENCIALES:"
            for file in "${important_files[@]}"; do
                if [ -f "$file" ]; then
                    echo "   ✅ $file"
                else
                    echo "   ❌ $file (no encontrado)"
                fi
            done
            echo ""
            
            # Verificar directorios
            echo "📂 DIRECTORIOS:"
            if [ -d ".venv" ]; then
                echo "   ✅ .venv/ (entorno virtual)"
            else
                echo "   ❌ .venv/ (no encontrado)"
            fi
            
            if [ -d "models" ] && [ "$(ls -A models 2>/dev/null)" ]; then
                echo "   ✅ models/ (contenido: $(ls models | wc -l) archivos)"
            else
                echo "   ❌ models/ (vacío o no existe)"
            fi
            
            # Verificar Python y pip
            echo ""
            echo "🐍 CONFIGURACIÓN PYTHON:"
            echo "   Versión: $(python --version 2>&1)"
            echo "   Pip: $(pip --version 2>&1 | cut -d' ' -f1-2)"
            echo ""
            ;;
            
        8)
            echo "👋 ¡Hasta luego!"
            exit 0
            ;;
            
        *)
            echo "❌ Opción no válida. Intenta nuevamente."
            ;;
    esac
    
    echo ""
    echo "=================================================="
    read -p "¿Volver al menú principal? (s/n): " return_menu
    if [[ $return_menu != "s" && $return_menu != "S" ]]; then
        echo "👋 ¡Hasta luego!"
        break
    fi
    echo ""
done

# ----------------------------------------------------
# 7. MENSAJE FINAL
# ----------------------------------------------------
echo ""
echo "=================================================="
echo "🎉 PROYECTO CONFIGURADO CORRECTAMENTE"
echo "=================================================="
echo ""
echo "📌 RESUMEN:"
echo "   • Entorno virtual: .venv/"
echo "   • Dataset: data/raw/california_housing.csv"
echo "   • Dependencias instaladas desde requirements.txt"
echo ""
echo "🚀 PRÓXIMOS PASOS MANUALES:"
echo "   1. Ejecutar notebook: jupyter notebook notebooks/01_eda.ipynb"
echo "   2. Preparar datos: python src/data_preparation.py"
echo "   3. Entrenar modelo: python src/train.py"
echo "   4. Iniciar API: python src/api/app.py"
echo ""
echo "💡 TIP: Usa 'source .venv/bin/activate' para activar el entorno"
echo "      en nuevas terminales (o .venv\\Scripts\\activate en Windows)"
echo ""