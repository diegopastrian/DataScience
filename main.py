# main.py
import argparse
from src.model_training import train_model
from src.predict import run_prediction_pipeline
# --- Importar las funciones de nuestros módulos ---
from src.config import TRAIN_DATA_PATH, FEATURES_TO_USE, TARGET_VARIABLE
from src.data_processing import load_dataset, clean_data, group_and_aggregate_data
from src.feature_engineering import create_domain_features
from src.model_training import train_model

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

def main():
    # Usamos argparse para permitir al usuario elegir qué pipeline ejecutar
    parser = argparse.ArgumentParser(description="Pipeline de ML para predicción de accidentes.")
    parser.add_argument(
        'pipeline', 
        type=str, 
        choices=['train', 'predict'],
        help="Elige el pipeline a ejecutar: 'train' para entrenar o 'predict' para predecir."
    )
    
    args = parser.parse_args()
    
    if args.pipeline == 'train':
        run_training_pipeline()
    elif args.pipeline == 'predict':
        run_prediction_pipeline()

if __name__ == "__main__":
    main()
