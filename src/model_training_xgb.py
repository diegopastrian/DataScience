import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, learning_curve, validation_curve, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, max_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import json
import xgboost as xgb
from src.config import TEST_DATA_PATH, TARGET_VARIABLE
from src.data_processing import load_dataset, clean_data, group_and_aggregate_data
from src.feature_engineering_xgb import create_xgb_features
from src.feature_engineering import create_domain_features

# --- Configuración específica para XGBoost ---
TRAIN_DATA_PATH = "data/Dataset_Train.csv"
MODEL_OUTPUT_PATH = "models/xgboost_regressor.pkl"

FEATURES_FINAL = [
    "zona_urbana",
    "veh_pob_ratio",
    "log_poblacion", "log_vehiculos", "veh_zona_interaction",
    "accidentes_lag1",
    "mes_cos","mes_sin",
    "log_muertos", "log_graves", "log_m/grave", "log_leves", "log_ilesos"
]
TARGET_VARIABLE = "total_accidentes"

PARAM_GRID = {
    "model__n_estimators": [100, 200, 400, 600],
    "model__max_depth": [3, 5, 7, 9],
    "model__learning_rate": [0.005, 0.01, 0.05, 0.1],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__gamma": [0, 0.1, 1],
    "model__reg_lambda": [0.1, 1, 10],
    "model__reg_alpha": [0, 0.1, 1]
}

def train_xgb_model(df: pd.DataFrame):
    print("Iniciando el entrenamiento del modelo XGBoost...")
    # --- Procesamiento igual que en el Colab ---
    # 1. Limpieza y feature engineering
    df["comuna"] = df["Nomcomuna"].astype(str).str.lower().str.strip()
    df["zona_urbana"] = df["Urbano/Rural"].astype(str).str.lower().str.strip().apply(lambda x: 1 if x == "urbano" else 0)
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    df["Mes"] = df["Mes"].astype(str).str.lower().map(meses).fillna(df["Mes"]).astype(float)
    df["total_vehiculos"] = df["total_vehiculos"].fillna(df["total_vehiculos"].mean())
    df["poblacion"] = df["poblacion"].fillna(df["poblacion"].mean())
    for col in ["Muertos", "Graves", "M/Grave", "Leves", "Ilesos"]:
        df[col] = df[col].fillna(df[col].median())
    # 2. Agrupación
    features = ["total_vehiculos", "poblacion", "zona_urbana", "Muertos", "Graves", "M/Grave", "Leves", "Ilesos"]
    agg_dict = {col: "mean" for col in features}
    agg_dict["Idaccidente"] = "count"
    df_grouped = df.groupby(["comuna", "Mes", "año"]).agg(agg_dict).rename(columns={"Idaccidente": "total_accidentes"}).reset_index()
    # 3. Feature engineering avanzado
    df_xgb = create_xgb_features(df_grouped)
    # 4. Selección de features finales
    X_train = df_xgb[FEATURES_FINAL]
    y_train = df_xgb[TARGET_VARIABLE]
    y_train_log = np.log1p(y_train)
    # 5. Pipeline y entrenamiento igual que Colab
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", RobustScaler()),
        ("model", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1))
    ])
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_GRID,
        n_iter=50,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train_log)
    print("\n🔍 Mejores hiperparámetros encontrados:")
    print(search.best_params_)
    best_model = search.best_estimator_
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model_pipeline = {
        'model': best_model,
        'features': FEATURES_FINAL
    }
    joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)
    print(f"✅ Modelo XGBoost guardado exitosamente en: {MODEL_OUTPUT_PATH}")
    # --- Procesar TEST igual que en Colab ---
    df_test = load_dataset(TEST_DATA_PATH)
    df_test["comuna"] = df_test["Nomcomuna"].astype(str).str.lower().str.strip()
    df_test["zona_urbana"] = df_test["Urbano/Rural"].astype(str).str.lower().str.strip().apply(lambda x: 1 if x == "urbano" else 0)
    df_test["Mes"] = df_test["Mes"].astype(str).str.lower().map(meses).fillna(df_test["Mes"]).astype(float)
    df_test["total_vehiculos"] = df_test["total_vehiculos"].fillna(df_test["total_vehiculos"].mean())
    df_test["poblacion"] = df_test["poblacion"].fillna(df_test["poblacion"].mean())
    for col in ["Muertos", "Graves", "M/Grave", "Leves", "Ilesos"]:
        df_test[col] = df_test[col].fillna(df_test[col].median())
    df_test_grouped = df_test.groupby(["comuna", "Mes", "año"]).agg(agg_dict).rename(columns={"Idaccidente": "total_accidentes"}).reset_index()
    df_test_xgb = create_xgb_features(df_test_grouped)
    X_test = df_test_xgb[FEATURES_FINAL]
    y_test = df_test_xgb[TARGET_VARIABLE]
    y_test_pred = np.expm1(best_model.predict(X_test))
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    medae = median_absolute_error(y_test, y_test_pred)
    maxerr = max_error(y_test, y_test_pred)
    mask_test = y_test > 100
    mape_test = np.mean(np.abs((y_test[mask_test] - y_test_pred[mask_test]) / y_test[mask_test])) * 100 if np.any(mask_test) else None
    metrics_dict = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MedAE": medae,
        "Max Error": maxerr,
        "MAPE (>100)": mape_test
    }
    with open("models/xgb_metrics.json", "w") as f:
        json.dump(metrics_dict, f)

    # Guardar importancia de variables (gain) en CSV con mapeo correcto de nombres
    xgb_model = best_model.named_steps['model']
    booster = xgb_model.get_booster()
    feature_names = list(X_train.columns)
    importances = booster.get_score(importance_type='gain')
    booster_feature_names = [f"f{i}" for i in range(len(feature_names))]
    mapped_importances = {}
    for idx, fname in enumerate(booster_feature_names):
        colname = feature_names[idx]
        mapped_importances[colname] = importances.get(fname, 0)
    importances_sum = sum(mapped_importances.values())
    importances_norm = {k: v / importances_sum if importances_sum > 0 else 0 for k, v in mapped_importances.items()}
    df_importance_xgb = pd.DataFrame({
        'feature': list(importances_norm.keys()),
        'importance': list(importances_norm.values())
    })
    df_importance_xgb.to_csv('models/xgb_feature_importance.csv', index=False)

    # Guardar curva de aprendizaje (learning curve)
    train_sizes = np.linspace(0.05, 1.0, 50)
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=best_model,
        X=X_train,
        y=y_train_log,
        train_sizes=train_sizes,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = -np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    df_curve = pd.DataFrame({
        "train_size": train_sizes,
        "train_mean": train_mean,
        "train_std": train_std,
        "valid_mean": valid_mean,
        "valid_std": valid_std
    })
    df_curve.to_csv("models/xgb_learning_curve.csv", index=False)

    # Guardar curva de validación (validation curve)
    param_range = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3]
    train_scores, val_scores = validation_curve(
        XGBRegressor(
            n_estimators=100,
            max_depth=7,
            subsample=0.8,
            random_state=42
        ),
        X_train,
        y_train_log,
        param_name="learning_rate",
        param_range=param_range,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1
    )
    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)
    train_mean = train_rmse.mean(axis=1)
    val_mean = val_rmse.mean(axis=1)
    train_std = train_rmse.std(axis=1)
    val_std = val_rmse.std(axis=1)
    df_val_curve = pd.DataFrame({
        "learning_rate": param_range,
        "train_mean": train_mean,
        "train_std": train_std,
        "val_mean": val_mean,
        "val_std": val_std
    })
    df_val_curve.to_csv("models/xgb_validation_curve.csv", index=False)