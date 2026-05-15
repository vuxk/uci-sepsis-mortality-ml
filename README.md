# Diseño e implementación de un modelo interpretable para la predicción de mortalidad en pacientes con sepsis utilizando la base de datos MIMIC-IV

**Juan Sebastian Aristizabal Calderón · Guillermo Córdoba Fernández**
Pontificia Universidad Javeriana · 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vuxk/uci-sepsis-mortality-ml/blob/main/sepsis-mortality-interpretable-model.ipynb)

 **Datos:** MIMIC-IV requiere credenciales en PhysioNet. Ver `data/README.md`.

**SEED = 42 | Umbral óptimo siempre buscado en validación, evaluado en test**

## Descripción

Este repositorio contiene el código del **sepsis-mortality-interpretable-model.ipynb** de la tesis de grado, que desarrolla y evalúa modelos de aprendizaje supervisado para la predicción de mortalidad hospitalaria en pacientes de UCI, utilizando la base de datos clínica **MIMIC-IV**.

El pipeline compara seis algoritmos de machine learning contra los scores clínicos estándar SOFA y APACHE II, aplica cuatro estrategias de balanceo de clases, e incorpora análisis de interpretabilidad local y global mediante SHAP y LIME.

---

## Contenido del repositorio

```
├── Objetivo2_ML_SHAP_LIME.ipynb   # Notebook principal — pipeline completo
├── requirements.txt               # Dependencias Python
├── data/
│   └── README.md                  # Instrucciones para obtener los datos (MIMIC-IV)
└── README.md                      # Este archivo
```

> ⚠️ **Los datos no se incluyen.** MIMIC-IV es de acceso restringido vía PhysioNet.
> Consulta [`data/README.md`](data/README.md) para instrucciones de acceso.

---

## Datos

| Característica | Detalle |
|---|---|
| Fuente | MIMIC-IV v2.2 — PhysioNet |
| Cohorte | Pacientes adultos UCI, primera admisión, estancia ≥ 24h |
| N total | 20,884 pacientes |
| Variable objetivo | Mortalidad hospitalaria (binaria) |
| Prevalencia | 15.5% (3,241 fallecidos / 17,643 sobrevivientes) |
| Desbalance | 5.4 : 1 |
| División | Train 60% · Validación 20% · Test 20% (estratificada, SEED=42) |
| Variables | 23 numéricas + 1 categórica (gender) = 24 features |

---

## Pipeline

### Parte 1 — Modelos de ML

| Paso | Descripción |
|---|---|
| **1. Carga** | CSV con cohorte APACHE II — MIMIC-IV |
| **2. División** | Estratificada 60/20/20 · SEED=42 · sin data leakage |
| **3. Preprocesamiento** | Mediana + StandardScaler (numéricas) · Moda + OneHotEncoder (categóricas) · fit SOLO en train |
| **4. Balanceo** | 4 estrategias: sin balanceo · class_weight · SMOTE · undersampling manual |
| **5. Entrenamiento** | 5 modelos sklearn × 4 estrategias = 20 configuraciones |
| **6. XGBoost nativo** | xgb.train + DMatrix + early stopping · eval_metric=aucpr |
| **7. Evaluación** | Umbral óptimo F1 en validación → métricas en test · IC95% bootstrap N=2000 |
| **8. Comparación** | vs SOFA y APACHE II con mismo esquema de umbral |

### Modelos evaluados

| Modelo | Hiperparámetros clave |
|---|---|
| Regresión Logística (LR) | `C=0.01`, `penalty=l2`, `solver=lbfgs` |
| Random Forest (RF) | `n_estimators=800`, `max_depth=16`, `min_samples_leaf=5` |
| SVM — RBF | `C=0.5`, `gamma=scale`, `kernel=rbf` |
| Gradient Boosting (GB) | `lr=0.05`, `max_depth=4`, `l2_regularization=1.0` (HistGB) |
| Red Neuronal (MLP) | `(256,128,64)`, `alpha=0.01`, `early_stopping=True` |
| **XGBoost (XGB)** | `eta=0.05`, `max_depth=5`, `eval_metric=aucpr`, `early_stopping=50` |

### Métricas reportadas

`AUROC` · `AUPRC` · `Sensibilidad` · `Especificidad` · `VPP` · `F1-score` · `Brier Score` · `IC95% bootstrap (N=2000)`

---

### Parte 2 — Interpretabilidad

| Análisis | Descripción |
|---|---|
| **SHAP global** | TreeExplainer · beeswarm plot · barplot importancia media |
| **SHAP local 2×2** | Waterfall plots para 4 casos: TP · TN · FN · FP |
| **LIME local 2×2** | Contribuciones locales para los mismos 4 casos |
| **SHAP vs scores** | Ranking SHAP frente a variables de SOFA y APACHE II |

---

## Outputs generados

Al correr el notebook completo se generan los siguientes archivos:

```
# Tablas de resultados
resultados_completos.csv
tabla_final_IC95_completa.csv

# Figuras ML
ML_resultados_completos.png        # ROC · PR · AUROC · AUPRC · F1 · Calibración

# SHAP global
shap_importancia_global.csv
interp_shap_beeswarm.png
interp_shap_barplot.png

# SHAP local — 4 casos
interp_shap_waterfall_2x2.png      # Panel 2×2 para presentación
interp_waterfall_TP.png
interp_waterfall_TN.png
interp_waterfall_FN.png
interp_waterfall_FP.png
shap_caso_TP.csv · shap_caso_TN.csv · shap_caso_FN.csv · shap_caso_FP.csv

# LIME
interp_lime_4casos.png             # Panel 2×2 para presentación
interp_lime_estabilidad.png
lime_estabilidad_FN.csv

# Análisis comparativo
concordancia_shap_lime.csv
shap_vs_scores_clinicos.csv
variables_sorpresa_ml.csv
```

---

## Instalación y uso

### Opción A — Google Colab (recomendada)

1. Haz clic en el badge **Open in Colab** al inicio de este README
2. Sube `CohorteFinalTesis.csv` usando `files.upload()` o monta Google Drive
3. Descomenta la línea de instalación en la Celda 1
4. Ejecuta todas las celdas en orden (`Runtime → Run all`)

### Opción B — Jupyter local

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU-USUARIO/TU-REPO.git
cd TU-REPO

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar el CSV en la carpeta raíz
# (ver data/README.md para instrucciones de acceso a MIMIC-IV)
cp /ruta/a/CohorteFinalTesis.csv .

# 4. Abrir el notebook
jupyter notebook Objetivo2_ML_SHAP_LIME.ipynb
```

> ⚠️ El notebook debe ejecutarse **de arriba hacia abajo en orden**. Cada celda depende de las anteriores. El tiempo estimado de ejecución completa es de 30–60 minutos según el hardware.

---

## Requisitos

```
Python        >= 3.10
scikit-learn  >= 1.3
xgboost       >= 2.0
shap          >= 0.44
lime          >= 0.2
pandas        >= 2.0
numpy         >= 1.24
matplotlib    >= 3.7
```

Ver [`requirements.txt`](requirements.txt) para versiones exactas.

---

## Reproducibilidad

Todo el pipeline usa `SEED = 42` de forma consistente en:

- División train/val/test (`train_test_split`)
- SMOTE personalizado (`np.random.RandomState(42)`)
- Todos los modelos sklearn (`random_state=42`)
- XGBoost nativo (`seed=42`)
- IC95% bootstrap (`np.random.RandomState(42)`)
- LIME (`random_state=42`)

El umbral de decisión óptimo se busca **siempre en el conjunto de validación** y se aplica al test sin ajuste posterior, garantizando evaluación independiente.

---

## Contexto metodológico

Este trabajo forma parte de una tesis que analiza la predicción de mortalidad en UCI desde tres enfoques complementarios:

- **Objetivo 1:** Análisis descriptivo y construcción del cohorte
- **Objetivo 2:** Modelos de ML + interpretabilidad ← *este repositorio*
- **Objetivo 3:** Validación clínica y análisis de subgrupos

Los scores clínicos SOFA y APACHE II se usan como **comparadores de referencia**, evaluados bajo el mismo esquema metodológico que los modelos ML (umbral óptimo en validación, evaluación en test, IC95% bootstrap).

---

## Licencia

El código de este repositorio se distribuye bajo la licencia **MIT**.
Los datos de MIMIC-IV están sujetos a los términos de uso de PhysioNet — ver [`data/README.md`](data/README.md).
