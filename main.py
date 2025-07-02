# main.py
import argparse
from src.model_training import train_model
from src.predict import run_prediction_pipeline
# --- Importar las funciones de nuestros módulos ---
from src.config import TRAIN_DATA_PATH, FEATURES_TO_USE, TARGET_VARIABLE
from src.data_processing import load_dataset, clean_data, group_and_aggregate_data
from src.feature_engineering import create_domain_features
from src.model_training import train_model
# --- Importar XGBoost ---
from src.model_training_xgb import train_xgb_model
from src.predict_xgb import load_xgb_model_and_predict
from src.feature_engineering_xgb import create_xgb_features

def run_training_pipeline():
    """
    Ejecuta el pipeline completo de entrenamiento del modelo.
    """
    print("=============================================")
    print("=== INICIO DEL PIPELINE DE ENTRENAMIENTO ===")
    print("=============================================\n")

    # --- Paso 1: Carga de Datos ---
    print("Paso 1: Cargando datos crudos...")
    df_raw = load_dataset(file_path=TRAIN_DATA_PATH)
    if df_raw.empty:
        print("Finalizando pipeline: no se pudieron cargar los datos.")
        return
    print(f"Datos cargados exitosamente. {df_raw.shape[0]} filas encontradas.\n")

    # --- Paso 2: Limpieza de Datos ---
    print("Paso 2: Limpiando datos...")
    df_cleaned = clean_data(df_raw)
    print("Datos limpiados.\n")

    # --- Paso 3: Ingeniería de Características ---
    print("Paso 3: Creando nuevas características...")
    df_features = create_domain_features(df_cleaned)
    print(f"Características creadas: {list(df_features.columns)}\n")

    # --- Paso 4: Agregación de Datos ---
    # Este paso prepara la tabla final para el modelo.
    print("Paso 4: Agrupando y agregando datos por comuna y año...")
    df_grouped = group_and_aggregate_data(
        df=df_features, 
        features=FEATURES_TO_USE, 
        target=TARGET_VARIABLE
    )
    print("Datos agrupados y listos para el entrenamiento.\n")

    # --- Paso 5: Entrenamiento del Modelo ---
    # Esta función entrenará el modelo y lo guardará en un archivo.
    print("Paso 5: Entrenando el modelo...")
    train_model(df_grouped=df_grouped)
    print("Modelo entrenado y guardado.\n")

    print("============================================")
    print("=== PIPELINE DE ENTRENAMIENTO FINALIZADO ===")
    print("=============================================")

def run_training_pipeline_xgb():
    """
    Ejecuta el pipeline de entrenamiento para XGBoost.
    """
    print("=============================================")
    print("=== INICIO DEL PIPELINE XGBOOST ===")
    print("=============================================")
    df_raw = load_dataset(file_path=TRAIN_DATA_PATH)
    if df_raw.empty:
        print("Finalizando pipeline: no se pudieron cargar los datos.")
        return
    df_cleaned = clean_data(df_raw)
    df_features = create_domain_features(df_cleaned)
    # Paso de agregación para obtener total_accidentes
    features_agregacion = [
        "total_vehiculos", "poblacion", "Mes", "zona_urbana", "Muertos", "Graves", "M/Grave", "Leves", "Ilesos"
    ]
    df_grouped = group_and_aggregate_data(
        df=df_features,
        features=features_agregacion,
        target="total_accidentes"
    )
    df_xgb = create_xgb_features(df_grouped)
    train_xgb_model(df_grouped=df_xgb)
    print("============================================")
    print("=== PIPELINE XGBOOST FINALIZADO ===")
    print("=============================================")

def run_prediction_pipeline_xgb():
    print("\n==========================================")
    print("=== INICIO DEL PIPELINE DE PREDICCIÓN XGB ===")
    print("==========================================\n")
    from src.config import TEST_DATA_PATH
    df_raw_test = load_dataset(file_path=TEST_DATA_PATH)
    if df_raw_test.empty:
        return
    df_cleaned_test = clean_data(df_raw_test)
    df_features_test = create_domain_features(df_cleaned_test)
    features_agregacion = [
        "total_vehiculos", "poblacion", "Mes", "zona_urbana", "Muertos", "Graves", "M/Grave", "Leves", "Ilesos"
    ]
    df_grouped_test = group_and_aggregate_data(
        df=df_features_test,
        features=features_agregacion,
        target="total_accidentes"
    )
    df_xgb_test = create_xgb_features(df_grouped_test)
    results_df = load_xgb_model_and_predict(df=df_xgb_test)
    if results_df.empty:
        print("Finalizando pipeline: no se pudieron realizar las predicciones.")
        return
    print("\n--- RESULTADOS DE LA PREDICCIÓN XGBOOST ---")
    print(results_df[["comuna", "año", "accidentes_predichos_xgb"]].sort_values(by="accidentes_predichos_xgb", ascending=False).head(10))
    print("\n=======================================" )
    print("=== PIPELINE DE PREDICCIÓN XGB FINALIZADO ===")
    print("=======================================" )

def main():
    # Usamos argparse para permitir al usuario elegir qué pipeline ejecutar
    parser = argparse.ArgumentParser(description="Pipeline de ML para predicción de accidentes.")
    parser.add_argument(
        'pipeline', 
        type=str, 
        choices=['train', 'predict', 'metrics', 'train_xgb', 'predict_xgb', 'metrics_xgb'],
        help="Elige el pipeline a ejecutar: 'train' para entrenar, 'predict' para predecir, 'metrics' para ver métricas, 'train_xgb'/'predict_xgb'/'metrics_xgb' para XGBoost."
    )
    
    args = parser.parse_args()
    
    if args.pipeline == 'train':
        run_training_pipeline()
        show_model_metrics()
    elif args.pipeline == 'predict':
        run_prediction_pipeline()
        show_model_metrics()
    elif args.pipeline == 'metrics':
        show_model_metrics()
    elif args.pipeline == 'train_xgb':
        run_training_pipeline_xgb()
        show_model_metrics_xgb()
    elif args.pipeline == 'predict_xgb':
        run_prediction_pipeline_xgb()
        show_model_metrics_xgb()
    elif args.pipeline == 'metrics_xgb':
        show_model_metrics_xgb()

# === FUNCIÓN PARA MOSTRAR MÉTRICAS ===
def show_model_metrics():
    from src.data_processing import load_dataset, clean_data, group_and_aggregate_data
    from src.feature_engineering import create_domain_features
    from src.model_training import metrics_model
    from src.config import TRAIN_DATA_PATH, FEATURES_TO_USE, TARGET_VARIABLE

    df = load_dataset(TRAIN_DATA_PATH)
    df_clean = clean_data(df)
    df_feat = create_domain_features(df_clean)
    df_grouped = group_and_aggregate_data(df_feat, FEATURES_TO_USE, TARGET_VARIABLE)
    X = df_grouped[FEATURES_TO_USE]
    y = df_grouped[TARGET_VARIABLE]

    metrics = metrics_model(X, y)
    print("\n=== MÉTRICAS DEL MODELO (Validación Cruzada) ===")
    for k, v in metrics.items():
        print(f"{k:12}: {v:.4f}" if isinstance(v, float) else f"{k:12}: {v}")

def show_model_metrics_xgb():
    # Métricas para XGBoost (entrenamiento)
    print("\n=== MÉTRICAS DEL MODELO XGBOOST ===")
    # Aquí podrías cargar el modelo y mostrar métricas adicionales si lo deseas
    print("(Las métricas principales se muestran durante el entrenamiento XGBoost)")

if __name__ == "__main__":
    main()
