# 🏠 California Housing Price Prediction - MLOps Project

## 👤 Autor
**Norman Campana**

## 📌 Tabla de Contenidos
- [Definición del Problema](#definición-del-problema)
- [Dataset](#dataset)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Experimentación](#experimentación)
- [Modelo](#modelo)
- [API y Predicciones](#api-y-predicciones)
- [Resultados](#resultados)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Conclusiones](#conclusiones)

## 🎯 Definición del Problema

**Caso de Uso:** Predicción del valor mediano de casas en California basado en características demográficas y geográficas.

**Contexto:** El mercado inmobiliario de California es uno de los más caros y volátiles de EE.UU. Este modelo ayuda a estimar precios de viviendas para compradores, vendedores e inversores.

**Restricciones:**
- Datos públicos del censo de 1990
- Predicción a nivel de bloque censal
- Precios en dólares estadounidenses

**Objetivo:** Predecir el valor mediano de las casas con un error menor a $50,000 USD.

**Beneficios:**
- Estimación rápida de propiedades
- Análisis de mercado por zona
- Base para modelos más complejos

**Métrica de éxito:** R² ≥ 0.75 y RMSE ≤ $50,000 USD

## 📊 Dataset

**Fuente:** California Housing Dataset (scikit-learn)

**Registros:** 20,640 bloques censales

**Características:**
| Variable | Descripción | Rango |
|----------|-------------|-------|
| MedInc | Ingreso medio (decenas de miles) | 0.5 - 15.0 |
| HouseAge | Antigüedad promedio (años) | 1 - 52 |
| AveRooms | Promedio de habitaciones | 2 - 20 |
| AveBedrms | Promedio de dormitorios | 0.5 - 34 |
| Population | Población del bloque | 3 - 35,682 |
| AveOccup | Ocupantes promedio | 0.5 - 1,243 |
| Latitude | Latitud | 32.5 - 42 |
| Longitude | Longitud | -124.3 - -114.3 |

**Target:** MedHouseVal (valor mediano en cientos de miles USD)

## 🏗️ Estructura del Proyecto

📦 mlops-final-project_1
├── 📂 data/
│ ├── 📂 raw/ # Dataset original
│ └── 📂 processed/ # Datos escalados y listos
├── 📂 models/ # Modelos serializados (.joblib)
├── 📂 notebooks/ # EDA y experimentos
├── 📂 reports/ # Métricas y visualizaciones
├── 📂 src/
│ ├── 📂 api/ # FastAPI
│ ├── data_preparation.py
│ └── train.py
└── README.md

✅ Librerías importadas y configuradas
📊 Cargando California Housing Dataset...
✅ Dataset cargado exitosamente
• Filas: 20640
• Columnas: 9
• Características: 8
💾 Dataset guardado en: ../data/raw/california_housing.csv
📏 Tamaño del archivo: 1.92 MB

📋 Primeras 5 filas del dataset:
============================================================
INFORMACIÓN GENERAL DEL DATASET
============================================================

📊 DIMENSIONES:
• Total de registros: 20,640
• Total de características: 9

🏷️  CARACTERÍSTICAS:
  1. MedInc
  2. HouseAge
  3. AveRooms
  4. AveBedrms
  5. Population
  6. AveOccup
  7. Latitude
  8. Longitude

🎯 VARIABLE OBJETIVO:
  • MedHouseVal: Precio mediano de la casa (en dólares)

📝 TIPOS DE DATOS:
MedInc         float64
HouseAge       float64
AveRooms       float64
AveBedrms      float64
Population     float64
AveOccup       float64
Latitude       float64
Longitude      float64
MedHouseVal    float64

🔍 VALORES NULOS:
  ✅ No hay valores nulos en el dataset
============================================================
ESTADÍSTICAS DESCRIPTIVAS
============================================================

📈 CARACTERÍSTICAS NUMÉRICAS:

💵 ESTADÍSTICAS DEL PRECIO (MedHouseVal):
  Count: 20,640.0
  Mean: $206,855.82
  Std: $115,395.62
  Min: $14,999.00
  25%: $119,600.00
  Median: $179,700.00
  75%: $264,725.00
  Max: $500,001.00


============================================================
ANÁLISIS DE DISTRIBUCIONES
============================================================

✅ Gráfico de distribuciones guardado en: ../reports/distribuciones_caracteristicas.png

 


============================================================
ANÁLISIS DE CORRELACIONES
============================================================

🔗 MATRIZ DE CORRELACIÓN COMPLETA:

✅ Matriz de correlación guardada en: ../reports/matriz_correlacion.png

 

🏆 TOP 5 CARACTERÍSTICAS MÁS CORRELACIONADAS CON EL PRECIO:
============================================================
RELACIÓN CARACTERÍSTICAS vs PRECIO
============================================================

✅ Gráfico de relaciones guardado en: ../reports/relaciones_precio.png

 
============================================================
ANÁLISIS GEOGRÁFICO
============================================================

✅ Análisis geográfico guardado en: ../reports/analisis_geografico.png

 

============================================================
DETECCIÓN DE OUTLIERS
============================================================

📊 OUTLIERS POR CARACTERÍSTICA (Método IQR):
--------------------------------------------------

• MedInc:
  Outliers: 681 (3.30%)
  Rango normal: [-0.71, 8.01]

• AveRooms:
  Outliers: 511 (2.48%)
  Rango normal: [2.02, 8.47]

• AveBedrms:
  Outliers: 1,424 (6.90%)
  Rango normal: [0.87, 1.24]

• Population:
  Outliers: 1,196 (5.79%)
  Rango normal: [-620.00, 3132.00]

• AveOccup:
  Outliers: 711 (3.44%)
  Rango normal: [1.15, 4.56]

📋 RESUMEN DE OUTLIERS:
============================================================
ANÁLISIS DE VARIABLES DERIVADAS
============================================================

📊 ESTADÍSTICAS DE VARIABLES DERIVADAS:

🔗 CORRELACIÓN CON EL PRECIO:
============================================================
CONCLUSIONES Y RECOMENDACIONES
============================================================

🎯 PRINCIPALES HALLAZGOS:
   1. 📈 El dataset contiene 20,640 muestras con 8 características predictivas
   2. 💰 La variable objetivo (MedHouseVal) tiene una distribución sesgada a la derecha
   3. 🔗 MedInc (ingreso mediano) es la característica más correlacionada con el precio (r=0.69)
   4. 🏠 AveRooms muestra alta correlación positiva con el precio
   5. 📍 Las variables geográficas (Latitude, Longitude) muestran patrones espaciales claros
   6. ⚠️ Algunas características tienen outliers significativos que requieren tratamiento
   7. 📊 No hay valores nulos en el dataset
   8. 🔄 Las escalas de las variables son diferentes, requiere normalización

💡 RECOMENDACIONES PARA PREPROCESAMIENTO:
   1. 1. Escalar todas las características (StandardScaler recomendado)
   2. 2. Considerar transformación logarítmica para el target si mejora la distribución
   3. 3. Tratar outliers usando winsorization o eliminación según el contexto
   4. 4. Crear variables derivadas como Room_Bed_Ratio y People_per_Room
   5. 5. Considerar interacciones entre características (ej: MedInc × Latitude)
   6. 6. Evaluar la necesidad de reducción de dimensionalidad
   7. 7. Implementar validación cruzada estratificada por rangos de precio

🤖 RECOMENDACIONES PARA MODELADO:
   • Algoritmos que manejen bien relaciones no lineales: Random Forest, XGBoost
   • Considerar modelos ensemble para mejorar precisión
   • Evaluar modelos de regresión regularizada para evitar overfitting
   • Probar redes neuronales si los datos lo permiten
📄 Reporte de EDA guardado en: ../reports/eda_report.txt
✅ Análisis Exploratorio de Datos COMPLETADO exitosamente!

============================================================
🎉 EDA COMPLETADO EXITOSAMENTE
============================================================

## 🔬 Experimentación

**Decisión Clave: NO transformar el target**  
Inicialmente se usó `PowerTransformer`, pero al invertir la transformación se obtenían valores astronómicos (> $20M). Se optó por **mantener el target en escala original (cientos de miles USD)** y solo escalar las características.

**Modelos Evaluados:**
| Modelo | R² | RMSE | MAE |
|--------|-----|------|-----|
| Random Forest | **0.8024** | **0.4443** | **0.3133** |
| XGBoost | 0.7941 | 0.4521 | 0.3210 |
| Gradient Boosting | 0.7889 | 0.4589 | 0.3278 |

**🎯 Campeón:** Random Forest Regressor  
[📗 Ver notebook de experimentación](notebooks/02_experimentacion.ipynb)

**Hiperparámetros:**
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 5.0,
    "HouseAge": 30.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
  }'

  Ejemplos de Predicciones:

Tipo	Ingreso	Antigüedad	Lat/Lon	Predicción
🏚️ Económica	$20K	50 años	35.0, -119.0	$142,500
🏠 Promedio	$50K	30 años	34.0, -118.0	$207,420
💰 Premium	$120K	15 años	37.8, -122.2	$485,300

📝 Conclusiones
✅ Se logró un modelo con R² > 0.80, superando la meta de 0.75

✅ El pipeline es completamente reproducible y modular

✅ La API está funcionando y documentada con Swagger

✅ Se tomó la decisión consciente de no transformar el target para mantener interpretabilidad

✅ El proyecto sigue las buenas prácticas de MLOps: control de versiones, experimentación, despliegue