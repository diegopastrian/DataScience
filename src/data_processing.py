# src/data_processing.py

import pandas as pd
from src.config import TARGET_VARIABLE, FEATURES_TO_USE, COLUMNAS_A_ELIMINAR

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Carga un dataset desde una ruta de archivo CSV.
    
    :param file_path: Ruta al archivo .csv
    :return: DataFrame de pandas.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {file_path}")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame eliminando columnas innecesarias e imputando valores faltantes.
    
    :param df: DataFrame de pandas crudo.
    :return: DataFrame de pandas limpio.
    """
    df_cleaned = df.copy()

    # --- Eliminar columnas no deseadas ---
    df_cleaned = df_cleaned.drop(columns=COLUMNAS_A_ELIMINAR, errors='ignore')

    # --- Imputación de valores faltantes (usando la estrategia del notebook) ---
    
    # Columnas categóricas con la moda
    columnas_moda = ['Calleuno', 'Calledos', 'Tribunal', 'Nombre Region', 'provincia']
    for col in columnas_moda:
        if col in df_cleaned.columns:
            moda = df_cleaned[col].mode()
            if not moda.empty:
                df_cleaned[col].fillna(moda[0], inplace=True)

    # Columnas numéricas con la media
    columnas_media = ['total_vehiculos', 'poblacion', 'Muertos']
    for col in columnas_media:
        if col in df_cleaned.columns:
            media = df_cleaned[col].mean()
            df_cleaned[col].fillna(media, inplace=True)
            
    # Columna 'Comuna_y' con la mediana (para ser robusto a outliers)
    if 'Comuna_y' in df_cleaned.columns:
        mediana_comuna = df_cleaned['Comuna_y'].median()
        df_cleaned['Comuna_y'].fillna(mediana_comuna, inplace=True)
        
    return df_cleaned

def group_and_aggregate_data(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """
    Agrupa los datos por comuna y año para crear el dataset de modelado.
    Calcula el promedio para las características y el conteo para la variable objetivo.
    
    :param df: DataFrame limpio y con características de ingeniería.
    :param features: Lista de columnas a usar como características.
    :param target: Nombre de la columna objetivo.
    :return: DataFrame agrupado y listo para el modelo.
    """
    # Asegurarnos de que las columnas necesarias para la agregación existan
    # Idaccidente se usa para contar el total de accidentes
    agg_dict = {feat: "mean" for feat in features if feat in df.columns}
    agg_dict["Idaccidente"] = "count" 

    df_grouped = df.groupby(["comuna", "año", "Mes"], as_index=False).agg(agg_dict)
    
    # Renombrar la columna de conteo para que sea la variable objetivo
    df_grouped.rename(columns={"Idaccidente": target}, inplace=True)
    
    return df_grouped