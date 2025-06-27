# src/feature_engineering.py

import pandas as pd
from src.config import MESES_MAPPING

def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas características basadas en el conocimiento del dominio del problema.
    
    :param df: DataFrame limpio.
    :return: DataFrame con nuevas características añadidas.
    """
    df_featured = df.copy()

    # --- Creación de 'comuna' ---
    if 'Nomcomuna' in df_featured.columns:
        df_featured['comuna'] = df_featured['Nomcomuna'].astype(str).str.lower().str.strip()

    # --- Creación de 'zona_urbana' ---
    if 'Urbano/Rural' in df_featured.columns:
        df_featured['zona_urbana'] = df_featured['Urbano/Rural'].astype(str).str.lower().str.strip().apply(
            lambda x: 1 if x == "urbano" else 0
        )

    # --- Transformación de 'Mes' ---
    if 'Mes' in df_featured.columns:
        df_featured['Mes'] = df_featured['Mes'].astype(str).str.lower().map(MESES_MAPPING).fillna(df_featured['Mes'])

    # --- Creación de 'densidad_vehicular' ---
    if 'total_vehiculos' in df_featured.columns and 'poblacion' in df_featured.columns:
        epsilon = 1e-6 # Para evitar división por cero si alguna población es 0
        df_featured['densidad_vehicular'] = df_featured['total_vehiculos'] / (df_featured['poblacion'] + epsilon)

    return df_featured