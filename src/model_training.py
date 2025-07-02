# src/model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import joblib
import os

from src.config import FEATURES_TO_USE, TARGET_VARIABLE, BEST_MODEL_PARAMS, MODEL_OUTPUT_PATH

# === MÉTRICAS DE VALIDACIÓN CRUZADA ===
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, max_error
)
from sklearn.model_selection import learning_curve, validation_curve

def train_model(df_grouped: pd.DataFrame):
    """
    Entrena un modelo de Gradient Boosting con los hiperparámetros óptimos
    predefinidos y lo guarda en un archivo.
    
    :param df_grouped: DataFrame agrupado y listo para el modelado.
    """
    print("Iniciando el entrenamiento del modelo final...")

    # --- 1. Preparación de los datos ---
    features = [feat for feat in FEATURES_TO_USE if feat in df_grouped.columns]
    X_train = df_grouped[features]
    y_train = df_grouped[TARGET_VARIABLE]
    
    # --- 2. Transformación Logarítmica de la variable objetivo ---
    # Es crucial hacer esto, ya que los hiperparámetros se optimizaron para la variable transformada.
    y_train_log = np.log1p(y_train)
    
    # --- 3. Imputación de valores faltantes ---
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
    
    # --- 4. Definición y entrenamiento del modelo ---
    # Usamos los parámetros óptimos importados desde nuestro archivo de configuración.
    print("Creando modelo Gradient Boosting con los mejores hiperparámetros...")
    final_model = GradientBoostingRegressor(**BEST_MODEL_PARAMS)
    
    print("Entrenando el modelo...")
    # Entrenamos el modelo con los datos de entrada imputados y la variable objetivo transformada.
    final_model.fit(X_train_imputed, y_train_log)
    
    # --- 5. Guardar el pipeline completo (modelo + imputer + features) ---
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    
    model_pipeline = {
        'model': final_model,
        'imputer': imputer,
        'features': features
    }
    
    joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)
    
    print(f"✅ Modelo y pipeline de preprocesamiento guardados exitosamente en: {MODEL_OUTPUT_PATH}")

    # Guardar curva de aprendizaje (learning curve)
    train_sizes = np.linspace(0.05, 1.0, 50)
    X_train = df_grouped[features]
    y_train = df_grouped[TARGET_VARIABLE]
    y_train_log = np.log1p(y_train)
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=final_model,
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
    df_curve.to_csv("models/gb_learning_curve.csv", index=False)

    # Guardar curva de validación (validation curve)
    param_range = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3]
    train_scores, val_scores = validation_curve(
        GradientBoostingRegressor(
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
    df_val_curve.to_csv("models/gb_validation_curve.csv", index=False)

def metrics_model(X, y, model=None, n_splits=5):

  
    if model is None:
        model = GradientBoostingRegressor(**BEST_MODEL_PARAMS)
    y_log = np.log1p(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_log = cross_val_predict(model, X, y_log, cv=kf, n_jobs=-1)
    y_pred = np.expm1(y_pred_log)
    metrics = {
        "R²": r2_score(y, y_pred),
        "MAE": mean_absolute_error(y, y_pred),
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "MedAE": median_absolute_error(y, y_pred),
        "Max Error": max_error(y, y_pred),
    }
    mask = y > 100
    if np.any(mask):
        mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
        metrics["MAPE (>100)"] = mape
    else:
        metrics["MAPE (>100)"] = None
    return metrics