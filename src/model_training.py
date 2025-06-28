# src/model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import joblib
import os

from src.config import FEATURES_TO_USE, TARGET_VARIABLE, BEST_MODEL_PARAMS, MODEL_OUTPUT_PATH

def train_model(df_grouped: pd.DataFrame):
    """
    Entrena un modelo de Gradient Boosting con los hiperparámetros óptimos
    predefinidos y lo guarda en un archivo.
    
    :param df_grouped: DataFrame agrupado y listo para el modelado.
    """
    print("Iniciando el entrenamiento del modelo final...")

    # --- 1. Preparación de los datos ---
    features = [feat for feat in FEATURES_TO_USE if feat in df_grouped.columns]
    X_train = df_grouped[features]
    y_train = df_grouped[TARGET_VARIABLE]
    
    # --- 2. Transformación Logarítmica de la variable objetivo ---
    # Es crucial hacer esto, ya que los hiperparámetros se optimizaron para la variable transformada.
    y_train_log = np.log1p(y_train)
    
    # --- 3. Imputación de valores faltantes ---
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
    
    # --- 4. Definición y entrenamiento del modelo ---
    # Usamos los parámetros óptimos importados desde nuestro archivo de configuración.
    print("Creando modelo Gradient Boosting con los mejores hiperparámetros...")
    final_model = GradientBoostingRegressor(**BEST_MODEL_PARAMS)
    
    print("Entrenando el modelo...")
    # Entrenamos el modelo con los datos de entrada imputados y la variable objetivo transformada.
    final_model.fit(X_train_imputed, y_train_log)
    
    # --- 5. Guardar el pipeline completo (modelo + imputer + features) ---
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    
    model_pipeline = {
        'model': final_model,
        'imputer': imputer,
        'features': features
    }
    
    joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)
    
    print(f"✅ Modelo y pipeline de preprocesamiento guardados exitosamente en: {MODEL_OUTPUT_PATH}")