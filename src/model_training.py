# src/model_training.py

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import joblib  # Librería para guardar y cargar modelos de scikit-learn
import os

from src.config import FEATURES_TO_USE, TARGET_VARIABLE, MODEL_PARAMS, MODEL_OUTPUT_PATH

def train_model(df_grouped: pd.DataFrame):
    """
    Entrena un modelo de Gradient Boosting y lo guarda en un archivo.
    
    :param df_grouped: DataFrame agrupado y listo para el modelado.
    """
    print("Iniciando el entrenamiento del modelo...")

    # --- 1. Preparación de los datos para scikit-learn ---
    
    # Asegurarse de que solo se usen las características definidas en config
    features = [feat for feat in FEATURES_TO_USE if feat in df_grouped.columns]
    
    X_train = df_grouped[features]
    y_train = df_grouped[TARGET_VARIABLE]
    
    # --- 2. Imputación de valores faltantes (paso final de seguridad) ---
    # Aunque ya hemos limpiado, es una buena práctica tener un imputer en el pipeline
    # de entrenamiento para manejar cualquier NaN que pudiera quedar o aparecer.
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
    
    # --- 3. Definición y entrenamiento del modelo ---
    # Usamos los parámetros definidos en nuestro archivo de configuración.
    gb_model = GradientBoostingRegressor(**MODEL_PARAMS)
    
    print("Entrenando Gradient Boosting Regressor...")
    gb_model.fit(X_train_imputed, y_train)
    
    # --- 4. Guardar el modelo y el imputer ---
    # Creamos la carpeta 'models' si no existe
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    
    # Guardamos el modelo entrenado Y el imputer. Es crucial guardar el imputer
    # para poder aplicar la misma transformación a los datos nuevos en la predicción.
    model_pipeline = {
        'model': gb_model,
        'imputer': imputer,
        'features': features
    }
    
    joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)
    
    print(f"Modelo y pipeline de preprocesamiento guardados exitosamente en: {MODEL_OUTPUT_PATH}")