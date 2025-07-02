import numpy as np
import pandas as pd

def create_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    cols_outliers = ["Muertos", "Graves", "M/Grave", "Leves", "Ilesos"]
    # --- Features básicas ---
    df_feat["veh_pob_ratio"] = df_feat["total_vehiculos"] / (df_feat["poblacion"] + 1)
    df_feat["log_poblacion"] = np.log1p(df_feat["poblacion"])
    df_feat["log_vehiculos"] = np.log1p(df_feat["total_vehiculos"])
    df_feat["veh_zona_interaction"] = df_feat["total_vehiculos"] * df_feat["zona_urbana"]
    df_feat["mes_sin"] = np.sin(2 * np.pi * df_feat["Mes"] / 12)
    df_feat["mes_cos"] = np.cos(2 * np.pi * df_feat["Mes"] / 12)
    # --- Lags y rolling ---
    df_feat = df_feat.sort_values(["comuna", "Mes"]).copy()
    df_feat["accidentes_lag1"] = df_feat.groupby("comuna")["total_accidentes"].shift(1)
    df_feat["accidentes_rolling_mean"] = df_feat.groupby("comuna")["total_accidentes"].shift(1).rolling(window=3).mean()
    df_feat["accidentes_rolling_std"] = df_feat.groupby("comuna")["total_accidentes"].shift(1).rolling(window=3).std()
    df_feat["accidentes_lag1"].fillna(df_feat["total_accidentes"].mean(), inplace=True)
    df_feat["accidentes_rolling_mean"].fillna(df_feat["total_accidentes"].mean(), inplace=True)
    df_feat["accidentes_rolling_std"].fillna(df_feat["total_accidentes"].std(), inplace=True)
    # --- Log transform para outliers ---
    for col in cols_outliers:
        df_feat[f"log_{col.lower()}" ] = np.log1p(df_feat[col])
    return df_feat 