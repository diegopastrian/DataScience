# src/predict.py

import pandas as pd
import numpy as np
import joblib
import os

from src.config import TEST_DATA_PATH, MODEL_OUTPUT_PATH, FEATURES_TO_USE, TARGET_VARIABLE
from src.data_processing import load_dataset, clean_data, group_and_aggregate_data
from src.feature_engineering import create_domain_features

def load_model_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Carga el pipeline guardado, realiza predicciones y las revierte a su escala original.
    
    :param df: DataFrame con los datos de prueba, ya procesados y agrupados.
    :return: DataFrame con las predicciones añadidas.
    """
    # --- 1. Carga del pipeline guardado ---
    if not os.path.exists(MODEL_OUTPUT_PATH):
        print(f"Error: No se encontró el archivo del modelo en {MODEL_OUTPUT_PATH}")
        print("Por favor, ejecuta el pipeline de entrenamiento primero (ej. `python main.py --train`)")
        return pd.DataFrame()

    model_pipeline = joblib.load(MODEL_OUTPUT_PATH)

    # --- 2. Preparación de datos ---
    model = model_pipeline['model']
    imputer = model_pipeline['imputer']
    features_in_model = model_pipeline['features']
    
    X_test = df[features_in_model]
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=features_in_model)
    
    # --- 3. Predicción en escala logarítmica ---
    print("Realizando predicciones...")
    log_predictions = model.predict(X_test_imputed)
    
    # --- 4. Transformación inversa de las predicciones ---
    # Es crucial revertir la transformación logarítmica para obtener los valores reales.
    final_predictions = np.expm1(log_predictions)
    
    # --- 5. Preparación del DataFrame de resultados ---
    df_with_predictions = df.copy()
    df_with_predictions['accidentes_predichos'] = final_predictions
    
    # Aseguramos que las predicciones no sean negativas.
    df_with_predictions['accidentes_predichos'] = df_with_predictions['accidentes_predichos'].apply(lambda x: max(0, x))
    
    return df_with_predictions

def run_prediction_pipeline():
    """
    Orquesta el pipeline completo de predicción.
    """
    print("\n==========================================")
    print("=== INICIO DEL PIPELINE DE PREDICCIÓN ===")
    print("==========================================\n")

    # (El resto de esta función puede permanecer igual que la tenías, ya que la lógica de carga y
    # preprocesamiento de datos no cambia. Solo la función `load_model_and_predict` necesitaba el ajuste).
    
    # --- Carga y preprocesamiento de los datos de prueba ---
    print("Paso 1: Cargando y procesando datos de prueba...")
    # Asegúrate de que la ruta en config.py apunte a tus datos de prueba
    df_raw_test = load_dataset(file_path=TEST_DATA_PATH) 
    if df_raw_test.empty:
        return

    # Mantenemos el Idaccidente si es necesario para la agregación
    df_test_temp_id = df_raw_test[['Idaccidente']].copy()
    df_cleaned_test = clean_data(df_raw_test)
    df_features_test = create_domain_features(df_cleaned_test)
    df_features_test['Idaccidente'] = df_test_temp_id['Idaccidente']
    
    print("Datos de prueba procesados.\n")

    # --- Agregación de los datos de prueba ---
    print("Paso 2: Agrupando datos de prueba por comuna y año...")
    df_grouped_test = group_and_aggregate_data(
        df=df_features_test, 
        features=FEATURES_TO_USE, 
        target=TARGET_VARIABLE
    )
    print("Datos de prueba listos para la predicción.\n")

    # --- Realizar Predicciones ---
    print("Paso 3: Cargando modelo y realizando predicciones...")
    results_df = load_model_and_predict(df=df_grouped_test)

    if results_df.empty:
        print("Finalizando pipeline: no se pudieron realizar las predicciones.")
        return

    # --- Mostrar Resultados ---
    print("\n--- RESULTADOS DE LA PREDICCIÓN ---")
    print("\nTop 10 Comunas con Mayor Riesgo de Accidentes Estimado:")
    
    top_10_comunas = results_df.sort_values(by="accidentes_predichos", ascending=False)[
        ["comuna", "año", "accidentes_predichos"]
    ].head(10)
    
    top_10_comunas['accidentes_predichos'] = top_10_comunas['accidentes_predichos'].round(2)
    print(top_10_comunas.to_string(index=False))
    
    print("\n=======================================")
    print("=== PIPELINE DE PREDICCIÓN FINALIZADO ===")
    print("=======================================")