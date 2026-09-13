# ML — Autenticación de Variedad de Vino (Wine Dataset)

Proyecto de Machine Learning: clasificación **multiclase** para verificar el
cultivar (variedad de uva) de una partida de vino a partir de un análisis químico
de laboratorio estándar.

## Descripción del problema

Una cooperativa vinícola recibe partidas de tres cultivares distintos de la misma
región. En ciertos puntos del proceso (mezcla de lotes, control de calidad, sospecha
de etiquetado incorrecto) es necesario **verificar objetivamente el cultivar real**
de una muestra a partir de análisis químicos que ya se realizan de forma rutinaria,
sin depender solo de la declaración del proveedor o de cata manual.

- **Tipo de problema:** clasificación multiclase supervisada (3 cultivares).
- **Variable target:** `cultivar_0`, `cultivar_1`, `cultivar_2`.
- **Métrica prioritaria:** *F1-macro* (da el mismo peso a cada cultivar, ya que
  ninguno es más "importante" que otro desde el punto de vista de negocio).

## Dataset utilizado

**Wine Data Set**, dataset público de referencia en clasificación multiclase: el
resultado de un análisis químico de vinos de tres cultivares distintos, cultivados
en la misma región de Italia. 178 registros, 13 variables químicas numéricas
(alcohol, ácido málico, ceniza, magnesio, fenoles, flavonoides, intensidad de color,
tono, prolina, etc.).

- Disponible directamente vía `sklearn.datasets.load_wine`.
- Se incluye una copia en `src/data_sample/wine_sample.csv`.

## Solución adoptada

Pipeline completo de ML supervisado, con un **EDA dirigido al modelado ampliado**:

1. Balance de clases y estadísticos de forma (asimetría/curtosis).
2. Ranking de features por dos métodos complementarios: **ANOVA F-test** e
   **Información Mutua**.
3. Relación features–target (boxplots) y correlaciones, incluyendo comparación de la
   **estructura de correlación entre clases**.
4. Detección de outliers **univariante** (IQR, Z-score) y **multivariante**
   (distancia de Mahalanobis).
5. Visualización de separabilidad de clases con **PCA vs. LDA**.
6. Perfil químico medio por cultivar (radar chart).
7. Preprocesado (escalado, feature engineering, selección de variables con RFE).
8. Comparativa de 7 algoritmos con validación cruzada estratificada y optimización
   de hiperparámetros (`GridSearchCV`) sobre los 2 mejores candidatos.
9. Evaluación final contra un conjunto de test aislado desde el inicio del proyecto.
10. Persistencia del modelo, scaler y features seleccionadas con `joblib`.

El notebook completo está en [`main.ipynb`](./main.ipynb), en la raíz del repositorio.

## Estructura del repositorio

```
├── src/
│   ├── data_sample/     # Copia de muestra del dataset (CSV)
│   ├── img/             # Gráficas generadas por el EDA y la evaluación
│   ├── models/          # Modelo final, scaler y features seleccionadas (joblib)
│   ├── notebooks/       # Notebooks de desarrollo / exploración (no el pipeline final)
│   └── utils/           # Funciones auxiliares de EDA y preprocesado
├── main.ipynb           # Notebook principal: pipeline de ML de principio a fin
├── requirements.txt     # Dependencias del proyecto
├── .gitignore
└── README.md
```

## Tecnologías utilizadas

- Python 3.11
- pandas, numpy, scipy
- scikit-learn (modelado, preprocesado, validación cruzada, PCA/LDA)
- matplotlib, seaborn (visualización)
- joblib (persistencia de modelos)

## Instrucciones de reproducción

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/ML_autenticacion_vinos.git
cd ML_autenticacion_vinos

# 2. Crear un entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Abrir y ejecutar el notebook de principio a fin
jupyter notebook main.ipynb
```

El notebook, al ejecutarse completo, genera automáticamente las figuras en
`src/img/` y el modelo entrenado en `src/models/`.

### Cargar el modelo ya entrenado para hacer inferencia

```python
import joblib
import pandas as pd
import sys
sys.path.append("./src")
from utils.preprocessing_utils import add_engineered_features

model = joblib.load("src/models/final_model.pkl")
scaler = joblib.load("src/models/scaler.pkl")
selected_features = joblib.load("src/models/selected_features.pkl")

# new_data: DataFrame con las 13 columnas originales del dataset Wine
new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)
new_data_fe = add_engineered_features(new_data_scaled)[selected_features]

prediction = model.predict(new_data_fe)           # 0, 1 o 2 (cultivar)
probability = model.predict_proba(new_data_fe)    # probabilidad de cada cultivar
```

## Limitaciones

El dataset proviene de una única región y campaña de cultivo. Antes de un uso real,
sería necesario validar el modelo con muestras de otras campañas/regiones, ya que la
composición química de un mismo cultivar puede variar por condiciones de cultivo,
suelo o climatología.
