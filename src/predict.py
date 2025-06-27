# src/predict.py

import pandas as pd
import joblib
import os

# --- Importar funciones y constantes ---
# Reutilizamos las mismas funciones y configuraciones que en el entrenamiento
from src.config import TEST_DATA_PATH, MODEL_OUTPUT_PATH, FEATURES_TO_USE, TARGET_VARIABLE
from src.data_processing import load_dataset, clean_data, group_and_aggregate_data
from src.feature_engineering import create_domain_features

def load_model_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Carga el pipeline de modelo guardado y realiza predicciones sobre nuevos datos.
    
    :param df: DataFrame con los datos de prueba, ya procesados y agrupados.
    :return: DataFrame con las predicciones añadidas.
    """
    # --- 1. Carga del pipeline guardado ---
    # Comprobamos si el archivo del modelo existe antes de intentar cargarlo.
    if not os.path.exists(MODEL_OUTPUT_PATH):
        print(f"Error: No se encontró el archivo del modelo en {MODEL_OUTPUT_PATH}")
        print("Por favor, ejecuta el pipeline de entrenamiento primero (ej. `python main.py`)")
        return pd.DataFrame() # Devuelve un DataFrame vacío si no hay modelo

    model_pipeline = joblib.load(MODEL_OUTPUT_PATH)

    # --- 2. Preparación de datos ---
    # Extraemos los componentes guardados para asegurar la consistencia.
    model = model_pipeline['model']
    imputer = model_pipeline['imputer']
    features_in_model = model_pipeline['features']
    
    # Aseguramos que el DataFrame de entrada tenga exactamente las mismas características que el modelo espera.
    X_test = df[features_in_model]
    
    # Aplicamos la misma imputación (con las medias/modas aprendidas del entrenamiento).
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=features_in_model)
    
    # --- 3. Predicción ---
    print("Realizando predicciones...")
    predictions = model.predict(X_test_imputed)
    
    # --- 4. Preparación del DataFrame de resultados ---
    df_with_predictions = df.copy()
    df_with_predictions['accidentes_predichos'] = predictions
    
    # Nos aseguramos de que las predicciones no sean negativas (un número de accidentes no puede ser negativo).
    df_with_predictions['accidentes_predichos'] = df_with_predictions['accidentes_predichos'].apply(lambda x: max(0, x))
    
    return df_with_predictions

def run_prediction_pipeline():
    """
    Orquesta el pipeline completo de predicción.
    """
    print("\n==========================================")
    print("=== INICIO DEL PIPELINE DE PREDICCIÓN ===")
    print("==========================================\n")

    # --- Carga y preprocesamiento de los datos de prueba ---
    print("Paso 1: Cargando y procesando datos de prueba...")
    df_raw_test = load_dataset(file_path=TEST_DATA_PATH)
    if df_raw_test.empty:
        return

    # Es crucial aplicar la misma cadena de transformaciones.
    # Aquí Idaccidente se mantiene temporalmente para poder contar en la agregación.
    df_test_temp_id = df_raw_test[['Idaccidente']].copy() # Guardamos el ID para el conteo
    df_cleaned_test = clean_data(df_raw_test)
    df_features_test = create_domain_features(df_cleaned_test)
    df_features_test['Idaccidente'] = df_test_temp_id['Idaccidente'] # Lo reincorporamos
    
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
    
    # Formateamos la salida para que sea más legible
    top_10_comunas['accidentes_predichos'] = top_10_comunas['accidentes_predichos'].round(2)
    print(top_10_comunas.to_string(index=False))
    
    print("\n=======================================")
    print("=== PIPELINE DE PREDICCIÓN FINALIZADO ===")
    print("=======================================")