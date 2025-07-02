import pandas as pd
import numpy as np
import joblib
import os

MODEL_OUTPUT_PATH = "models/xgboost_regressor.pkl"
FEATURES_FINAL = [
    "zona_urbana",
    "veh_pob_ratio",
    "log_poblacion", "log_vehiculos", "veh_zona_interaction",
    "accidentes_lag1",
    "mes_cos","mes_sin",
    "log_muertos", "log_graves", "log_m/grave", "log_leves", "log_ilesos"
]

def load_xgb_model_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(MODEL_OUTPUT_PATH):
        print(f"Error: No se encontró el archivo del modelo en {MODEL_OUTPUT_PATH}")
        return pd.DataFrame()
    model_pipeline = joblib.load(MODEL_OUTPUT_PATH)
    model = model_pipeline['model']
    features_in_model = model_pipeline['features']
    X = df[features_in_model]
    print("Realizando predicciones con XGBoost...")
    log_predictions = model.predict(X)
    final_predictions = np.expm1(log_predictions)
    df_with_predictions = df.copy()
    df_with_predictions['accidentes_predichos_xgb'] = final_predictions
    df_with_predictions['accidentes_predichos_xgb'] = df_with_predictions['accidentes_predichos_xgb'].apply(lambda x: max(0, x))
    return df_with_predictions 