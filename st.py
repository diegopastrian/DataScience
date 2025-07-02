import streamlit as st
import pandas as pd
import os
from src.data_processing import load_dataset, clean_data, group_and_aggregate_data
from src.feature_engineering import create_domain_features
from src.predict import load_model_and_predict
from src.model_training import metrics_model
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, FEATURES_TO_USE, TARGET_VARIABLE, MESES_MAPPING
import plotly.express as px
import joblib
from src.predict_xgb import load_xgb_model_and_predict
from src.feature_engineering_xgb import create_xgb_features
from src.model_training_xgb import FEATURES_FINAL as FEATURES_FINAL_XGB
import json
import plotly.graph_objects as go


# === CONFIGURACIÓN DE MODELOS ===
MODEL_CONFIG = {
    "Gradient Boosting": {
        "color_linea": "#1f77b4",
        "color_barra": "Blues",
        "color_top": "Oranges",
        "titulo": "<h3 style='color:#1f77b4;'>Predicción de Accidentes - Gradient Boosting</h3>",
        "pred_col": "accidentes_predichos",
        "features": FEATURES_TO_USE,
        "target": TARGET_VARIABLE,
        "color_train": "#1f77b4",
        "color_val": "#FF914D",
        "learning_curve_file": "models/gb_learning_curve.csv",
        "validation_curve_file": "models/gb_validation_curve.csv",
        "model_file": "models/gradient_boosting_regressor.pkl"
    },
    "XGBoost": {
        "color_linea": "#FF914D",
        "color_barra": "Oranges",
        "color_top": "Oranges",
        "titulo": "<h3 style='color:#FF914D;'>Predicción de Accidentes - XGBoost</h3>",
        "pred_col": "accidentes_predichos_xgb",
        "features": ["total_vehiculos", "poblacion", "Mes", "zona_urbana", "Muertos", "Graves", "M/Grave", "Leves", "Ilesos"],
        "target": "total_accidentes",
        "color_train": "#FF914D",
        "color_val": "#1f77b4",
        "learning_curve_file": "models/xgb_learning_curve.csv",
        "validation_curve_file": "models/xgb_validation_curve.csv",
        "metrics_file": "models/xgb_metrics.json",
        "importance_file": "models/xgb_feature_importance.csv"
    }
}


# === FUNCIONES AUXILIARES ===
def load_and_process_data(modelo_config):
    """Carga y procesa los datos según el modelo seleccionado"""
    df = load_dataset(TEST_DATA_PATH)
    df_clean = clean_data(df)
    df_feat = create_domain_features(df_clean)
    
    # Preservar Idaccidente si existe
    if 'Idaccidente' in df.columns:
        df_feat['Idaccidente'] = df['Idaccidente']
    
    df_grouped = group_and_aggregate_data(df_feat, modelo_config["features"], modelo_config["target"])
    
    return df, df_grouped


def get_predictions(modelo, df_grouped):
    """Obtiene las predicciones según el modelo seleccionado"""
    if modelo == "Gradient Boosting":
        return load_model_and_predict(df_grouped)
    else:  # XGBoost
        df_xgb = create_xgb_features(df_grouped)
        return load_xgb_model_and_predict(df_xgb)


def get_model_metrics(modelo, modelo_config):
    """Obtiene las métricas del modelo"""
    if modelo == "Gradient Boosting":
        df_train = load_dataset(TRAIN_DATA_PATH)
        df_train_clean = clean_data(df_train)
        df_train_feat = create_domain_features(df_train_clean)
        df_train_grouped = group_and_aggregate_data(df_train_feat, modelo_config["features"], modelo_config["target"])
        X = df_train_grouped[modelo_config["features"]]
        y = df_train_grouped[modelo_config["target"]]
        return metrics_model(X, y)
    else:  # XGBoost
        try:
            with open(modelo_config["metrics_file"], "r") as f:
                return json.load(f)
        except Exception:
            return None


def get_feature_importance(modelo, modelo_config):
    """Obtiene la importancia de las características"""
    if modelo == "Gradient Boosting":
        model_pipeline = joblib.load(modelo_config["model_file"])
        importances = model_pipeline['model'].feature_importances_
        features = model_pipeline['features']
        return pd.DataFrame({"feature": features, "importance": importances}).sort_values(by="importance", ascending=False)
    else:  # XGBoost
        try:
            df_importance = pd.read_csv(modelo_config["importance_file"])
            return df_importance.sort_values(by="importance", ascending=False)
        except Exception:
            return None


def plot_learning_curve(df_curve, color_train, color_val, title):
    """Crea el gráfico de curva de aprendizaje"""
    fig = go.Figure()
    
    # Entrenamiento
    fig.add_trace(go.Scatter(
        x=df_curve["train_size"], y=df_curve["train_mean"],
        mode="lines+markers", name="Error de entrenamiento", line=dict(color=color_train)
    ))
    fig.add_trace(go.Scatter(
        x=df_curve["train_size"], y=df_curve["train_mean"] - df_curve["train_std"],
        mode="lines", name="Train - std", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_curve["train_size"], y=df_curve["train_mean"] + df_curve["train_std"],
        mode="lines", name="Train + std", fill='tonexty', line=dict(width=0),
        fillcolor="rgba(0,80,158,0.15)" if color_train == "#00509E" else "rgba(255,111,0,0.15)", showlegend=False
    ))
    
    # Validación
    fig.add_trace(go.Scatter(
        x=df_curve["train_size"], y=df_curve["valid_mean"],
        mode="lines+markers", name="Error de validación", line=dict(color=color_val)
    ))
    fig.add_trace(go.Scatter(
        x=df_curve["train_size"], y=df_curve["valid_mean"] - df_curve["valid_std"],
        mode="lines", name="Valid - std", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_curve["train_size"], y=df_curve["valid_mean"] + df_curve["valid_std"],
        mode="lines", name="Valid + std", fill='tonexty', line=dict(width=0),
        fillcolor="rgba(255,111,0,0.15)" if color_val == "#FF6F00" else "rgba(0,80,158,0.15)", showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Cantidad de muestras de entrenamiento",
        yaxis_title="RMSE",
        template="plotly_white"
    )
    return fig


def plot_validation_curve(df_val_curve, color_train, color_val, title):
    """Crea el gráfico de curva de validación"""
    fig = go.Figure()
    
    # Train
    fig.add_trace(go.Scatter(
        x=df_val_curve["learning_rate"], y=df_val_curve["train_mean"],
        mode="lines+markers", name="Train RMSE", line=dict(color=color_train)
    ))
    fig.add_trace(go.Scatter(
        x=df_val_curve["learning_rate"], y=df_val_curve["train_mean"] - df_val_curve["train_std"],
        mode="lines", name="Train - std", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_val_curve["learning_rate"], y=df_val_curve["train_mean"] + df_val_curve["train_std"],
        mode="lines", name="Train + std", fill='tonexty', line=dict(width=0),
        fillcolor="rgba(0,80,158,0.15)" if color_train == "#00509E" else "rgba(255,111,0,0.15)", showlegend=False
    ))
    
    # Validation
    fig.add_trace(go.Scatter(
        x=df_val_curve["learning_rate"], y=df_val_curve["val_mean"],
        mode="lines+markers", name="Validation RMSE", line=dict(color=color_val)
    ))
    fig.add_trace(go.Scatter(
        x=df_val_curve["learning_rate"], y=df_val_curve["val_mean"] - df_val_curve["val_std"],
        mode="lines", name="Val - std", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_val_curve["learning_rate"], y=df_val_curve["val_mean"] + df_val_curve["val_std"],
        mode="lines", name="Val + std", fill='tonexty', line=dict(width=0),
        fillcolor="rgba(255,111,0,0.15)" if color_val == "#FF6F00" else "rgba(0,80,158,0.15)", showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="learning_rate",
        yaxis_title="RMSE",
        template="plotly_white",
        xaxis_type='log'
    )
    return fig


def display_top5_comunas(df_pred, anio_sel, mes_nombre_sel, pred_col, color_top):
    """Muestra el top 5 de comunas"""
    MESES_INV = {v: k.capitalize() for k, v in MESES_MAPPING.items()}
    
    if mes_nombre_sel == "Todos":
        df_filtrado = df_pred[df_pred['año'] == anio_sel]
        top5 = df_filtrado.groupby("comuna")[pred_col].sum().reset_index().sort_values(by=pred_col, ascending=False).head(5)
        st.subheader(f"Top 5 comunas con mayor predicción de accidentes ({anio_sel}, todo el año)")
    else:
        mes_sel = [k for k, v in MESES_INV.items() if v == mes_nombre_sel][0]
        df_filtrado = df_pred[(df_pred['año'] == anio_sel) & (df_pred['Mes'] == mes_sel)]
        top5 = df_filtrado.sort_values(by=pred_col, ascending=False).head(5)
        st.subheader(f"Top 5 comunas con mayor predicción de accidentes ({mes_nombre_sel} {anio_sel})")
    
    if top5.empty:
        st.info("No hay datos para ese mes y año.")
        return
    
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(top5[["comuna", pred_col]].set_index("comuna").style.background_gradient(cmap=color_top))
    with c2:
        fig = px.bar(top5, x="comuna", y=pred_col, color=pred_col,
                     color_continuous_scale=color_top, labels={pred_col: "Accidentes predichos"},
                     title="Top 5 comunas")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


def display_temporal_trend(df_pred, pred_col, color_linea):
    """Muestra la tendencia temporal"""
    MESES_INV = {v: k.capitalize() for k, v in MESES_MAPPING.items()}
    
    st.subheader("Tendencia temporal de accidentes predichos")
    trend = df_pred.groupby(["año", "Mes"])[pred_col].sum().reset_index()
    trend["Mes_nombre"] = trend["Mes"].map(MESES_INV)
    trend["Periodo"] = trend["año"].astype(str) + "-" + trend["Mes"].astype(str).str.zfill(2)
    
    fig2 = px.line(trend, x="Periodo", y=pred_col, markers=True,
                   labels={pred_col: "Accidentes predichos", "Periodo": "Año-Mes"},
                   title="Accidentes predichos por mes")
    fig2.update_traces(line_color=color_linea)
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)


def display_metrics(modelo, metrics):
    st.subheader(f"Métricas globales del modelo {modelo}")
    
    if metrics is not None:
        m1, m2, m3 = st.columns(3)
        
        if modelo == "Gradient Boosting":
            m1.metric("R²", f"{metrics['R²']:.5f}")
            m2.metric("MAE", f"{metrics['MAE']:.3f}")
            m3.metric("RMSE", f"{metrics['RMSE']:.2f}")
            st.caption("Otras métricas:")
            st.write({k: v for k, v in metrics.items() if k not in ['R²', 'MAE', 'RMSE']})
        else:  # XGBoost
            m1.metric("R²", f"{metrics['R2']:.5f}")
            m2.metric("MAE", f"{metrics['MAE']:.2f}")
            m3.metric("RMSE", f"{metrics['RMSE']:.2f}")
    else:
        st.info("Las métricas no aparecen en el modelo seleccionado")


def display_feature_importance(df_importance, color_barra, modelo):
    """Muestra la importancia de las características"""
    if df_importance is not None:
        st.subheader(f"Importancia de variables del modelo {modelo}")
        fig3 = px.bar(df_importance, x="feature", y="importance", color="importance",
                      color_continuous_scale=color_barra, title=f"Importancia de variables {modelo}")
        fig3.update_layout(template="plotly_white", yaxis_title="Importancia", xaxis_title="Variable")
        st.plotly_chart(fig3, use_container_width=True)


def compare_models():
    """Función para comparar ambos modelos lado a lado"""
    st.markdown("<h3 style='color:#2E8B57;'>Comparación de Modelos: Gradient Boosting vs XGBoost</h3>", unsafe_allow_html=True)
    
    # Procesar datos para ambos modelos
    models_data = {}
    models_metrics = {}
    models_importance = {}
    
    for model_name in ["Gradient Boosting", "XGBoost"]:
        config = MODEL_CONFIG[model_name]
        _, df_grouped = load_and_process_data(config)
        df_pred = get_predictions(model_name, df_grouped)
        metrics = get_model_metrics(model_name, config)
        importance = get_feature_importance(model_name, config)
        
        models_data[model_name] = df_pred
        models_metrics[model_name] = metrics
        models_importance[model_name] = importance
    
    # Comparación de métricas
    st.subheader("📊 Comparación de Métricas")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Gradient Boosting**")
        if models_metrics["Gradient Boosting"]:
            gb_metrics = models_metrics["Gradient Boosting"]
            st.metric("R²", f"{gb_metrics['R²']:.5f}")
            st.metric("MAE", f"{gb_metrics['MAE']:.3f}")
            st.metric("RMSE", f"{gb_metrics['RMSE']:.2f}")
    
    with col2:
        st.markdown("**XGBoost**")
        if models_metrics["XGBoost"]:
            xgb_metrics = models_metrics["XGBoost"]
            st.metric("R²", f"{xgb_metrics['R2']:.5f}")
            st.metric("MAE", f"{xgb_metrics['MAE']:.2f}")
            st.metric("RMSE", f"{xgb_metrics['RMSE']:.2f}")
    
    # Gráfico comparativo de métricas
    if models_metrics["Gradient Boosting"] and models_metrics["XGBoost"]:
        metrics_comparison = pd.DataFrame({
            'Modelo': ['Gradient Boosting', 'XGBoost'],
            'R²': [models_metrics["Gradient Boosting"]['R²'], models_metrics["XGBoost"]['R2']],
            'MAE': [models_metrics["Gradient Boosting"]['MAE'], models_metrics["XGBoost"]['MAE']],
            'RMSE': [models_metrics["Gradient Boosting"]['RMSE'], models_metrics["XGBoost"]['RMSE']]
        })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_r2 = px.bar(metrics_comparison, x='Modelo', y='R²', 
                           title='R² Score', color='Modelo',
                           color_discrete_map={'Gradient Boosting': '#1f77b4', 'XGBoost': '#FF914D'})
            fig_r2.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            fig_mae = px.bar(metrics_comparison, x='Modelo', y='MAE',
                           title='MAE ', color='Modelo',
                           color_discrete_map={'Gradient Boosting': '#1f77b4', 'XGBoost': '#FF914D'})
            fig_mae.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col3:
            fig_rmse = px.bar(metrics_comparison, x='Modelo', y='RMSE',
                            title='RMSE', color='Modelo',
                            color_discrete_map={'Gradient Boosting': '#1f77b4', 'XGBoost': '#FF914D'})
            fig_rmse.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Comparación de predicciones por año
    st.subheader("📈 Comparación de Predicciones Temporales")
    
    # Preparar datos para comparación temporal
    gb_trend = models_data["Gradient Boosting"].groupby(["año", "Mes"])[MODEL_CONFIG["Gradient Boosting"]["pred_col"]].sum().reset_index()
    xgb_trend = models_data["XGBoost"].groupby(["año", "Mes"])[MODEL_CONFIG["XGBoost"]["pred_col"]].sum().reset_index()
    
    gb_trend["Periodo"] = gb_trend["año"].astype(str) + "-" + gb_trend["Mes"].astype(str).str.zfill(2)
    xgb_trend["Periodo"] = xgb_trend["año"].astype(str) + "-" + xgb_trend["Mes"].astype(str).str.zfill(2)
    

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=gb_trend["Periodo"], 
        y=gb_trend[MODEL_CONFIG["Gradient Boosting"]["pred_col"]],
        mode='lines+markers',
        name='Gradient Boosting',
        line=dict(color='#1f77b4')
    ))
    fig_comp.add_trace(go.Scatter(
        x=xgb_trend["Periodo"], 
        y=xgb_trend[MODEL_CONFIG["XGBoost"]["pred_col"]],
        mode='lines+markers',
        name='XGBoost',
        line=dict(color='#FF914D')
    ))
    
    fig_comp.update_layout(
        title="Comparación de Predicciones Temporales",
        xaxis_title="Período",
        yaxis_title="Accidentes Predichos",
        template="plotly_white"
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Top 5 comunas por modelo
    st.subheader("🏆 Top 5 Comunas por Modelo")
    
    anio_comp = st.selectbox("Año para comparación", sorted(models_data["Gradient Boosting"]['año'].unique()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 5 - Gradient Boosting**")
        gb_top5 = models_data["Gradient Boosting"][models_data["Gradient Boosting"]['año'] == anio_comp].groupby("comuna")[MODEL_CONFIG["Gradient Boosting"]["pred_col"]].sum().reset_index().sort_values(by=MODEL_CONFIG["Gradient Boosting"]["pred_col"], ascending=False).head(5)
        st.dataframe(gb_top5.set_index("comuna"))
        
        fig_gb = px.bar(gb_top5, x="comuna", y=MODEL_CONFIG["Gradient Boosting"]["pred_col"],
                       color=MODEL_CONFIG["Gradient Boosting"]["pred_col"],
                       color_continuous_scale="Blues",
                       title="GB - Top 5 Comunas")
        fig_gb.update_layout(template="plotly_white")
        st.plotly_chart(fig_gb, use_container_width=True)
    
    with col2:
        st.markdown("**Top 5 - XGBoost**")
        xgb_top5 = models_data["XGBoost"][models_data["XGBoost"]['año'] == anio_comp].groupby("comuna")[MODEL_CONFIG["XGBoost"]["pred_col"]].sum().reset_index().sort_values(by=MODEL_CONFIG["XGBoost"]["pred_col"], ascending=False).head(5)
        st.dataframe(xgb_top5.set_index("comuna"))
        
        fig_xgb = px.bar(xgb_top5, x="comuna", y=MODEL_CONFIG["XGBoost"]["pred_col"],
                        color=MODEL_CONFIG["XGBoost"]["pred_col"],
                        color_continuous_scale="Oranges",
                        title="XGBoost - Top 5 Comunas")
        fig_xgb.update_layout(template="plotly_white")
        st.plotly_chart(fig_xgb, use_container_width=True)

    if models_importance["Gradient Boosting"] is not None and models_importance["XGBoost"] is not None:
        st.subheader("🔍 Comparación de Importancia de Variables")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gradient Boosting**")
            fig_gb_imp = px.bar(models_importance["Gradient Boosting"].head(10), 
                               x="feature", y="importance",
                               color="importance", color_continuous_scale="Blues",
                               title="Top 10 Variables - GB")
            fig_gb_imp.update_layout(template="plotly_white")
            st.plotly_chart(fig_gb_imp, use_container_width=True)
        
        with col2:
            st.markdown("**XGBoost**")
            fig_xgb_imp = px.bar(models_importance["XGBoost"].head(10), 
                                x="feature", y="importance",
                                color="importance", color_continuous_scale="Oranges",
                                title="Top 10 Variables - XGBoost")
            fig_xgb_imp.update_layout(template="plotly_white")
            st.plotly_chart(fig_xgb_imp, use_container_width=True)
    
    # Resumen y recomendaciones
    st.subheader("📝 Resumen")
    
    if models_metrics["Gradient Boosting"] and models_metrics["XGBoost"]:
        gb_r2 = models_metrics["Gradient Boosting"]['R²']
        xgb_r2 = models_metrics["XGBoost"]['R2']
        gb_rmse = models_metrics["Gradient Boosting"]['RMSE']
        xgb_rmse = models_metrics["XGBoost"]['RMSE']
        
        if gb_r2 > xgb_r2:
            mejor_r2 = "Gradient Boosting"
            diff_r2 = ((gb_r2 - xgb_r2) / xgb_r2) * 100
        else:
            mejor_r2 = "XGBoost"
            diff_r2 = ((xgb_r2 - gb_r2) / gb_r2) * 100
        
        if gb_rmse < xgb_rmse:
            mejor_rmse = "Gradient Boosting"
            diff_rmse = ((xgb_rmse - gb_rmse) / xgb_rmse) * 100
        else:
            mejor_rmse = "XGBoost"
            diff_rmse = ((gb_rmse - xgb_rmse) / gb_rmse) * 100
        
        st.info(f"""
        **Análisis Comparativo:**
        
        • **Mejor R² Score**: {mejor_r2} ({diff_r2:.2f}% superior)
        • **Mejor RMSE**: {mejor_rmse} ({diff_rmse:.2f}% mejor) """)


def display_individual_model(modelo, config):
    """Función para mostrar un modelo individual"""
    # Cargar y procesar datos
    df, df_grouped = load_and_process_data(config)
    
    # Obtener predicciones
    df_pred = get_predictions(modelo, df_grouped)
    
    # Configurar filtros en sidebar
    anios = sorted(df_pred['año'].unique())
    meses_num = sorted(df_pred['Mes'].unique())
    MESES_INV = {v: k.capitalize() for k, v in MESES_MAPPING.items()}
    meses_nombres = [MESES_INV[m] for m in meses_num]
    meses_nombres_todos = ["Todos"] + meses_nombres
    
    anio_sel = st.sidebar.selectbox("Selecciona el año", anios, key="anio")
    mes_nombre_sel = st.sidebar.selectbox("Selecciona el mes", meses_nombres_todos, key="mes")
    
    # Contenido principal
    st.markdown(config["titulo"], unsafe_allow_html=True)
    
    # Mostrar top 5 comunas
    display_top5_comunas(df_pred, anio_sel, mes_nombre_sel, config["pred_col"], config["color_top"])
    
    # Mostrar tendencia temporal
    display_temporal_trend(df_pred, config["pred_col"], config["color_linea"])
    
    # Obtener y mostrar métricas
    metrics = get_model_metrics(modelo, config)
    display_metrics(modelo, metrics)
    
    # Obtener y mostrar importancia de características
    df_importance = get_feature_importance(modelo, config)
    display_feature_importance(df_importance, config["color_barra"], modelo)
    
    # Mostrar curvas de aprendizaje
    display_learning_curves(config, modelo)


def display_learning_curves(modelo_config, modelo):
    """Muestra las curvas de aprendizaje y validación"""
    # Curva de aprendizaje
    try:
        df_curve = pd.read_csv(modelo_config["learning_curve_file"])
        fig_curve = plot_learning_curve(
            df_curve, 
            modelo_config["color_train"], 
            modelo_config["color_val"], 
            f"Curva de Aprendizaje - {modelo}"
        )
        st.subheader(f"Curva de Aprendizaje ({modelo})")
        st.plotly_chart(fig_curve, use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo cargar la curva de aprendizaje. Entrena el modelo {modelo} para generarla.")
    
    # Curva de validación
    try:
        df_val_curve = pd.read_csv(modelo_config["validation_curve_file"])
        fig_val = plot_validation_curve(
            df_val_curve, 
            modelo_config["color_train"], 
            modelo_config["color_val"], 
            "Curva de Validación"
        )
        st.subheader("Curva de Validación")
        st.plotly_chart(fig_val, use_container_width=True)
    except Exception as e:
        st.info("No se pudo cargar la curva de validación. Entrena el modelo para generarla.")


# === APLICACIÓN PRINCIPAL ===
st.set_page_config(page_title="Predicción de Accidentes", layout="wide")
st.title("Predicción de Accidentes por Comuna")

# Sidebar: Selector de modelo
st.sidebar.header("Configuración y Filtros")
modo = st.sidebar.radio("Modo de visualización", ["Modelo Individual", "Comparación de Modelos"])

if modo == "Modelo Individual":
    modelo = st.sidebar.selectbox("Selecciona modelo", ["Gradient Boosting", "XGBoost"])
    # Obtener configuración del modelo seleccionado
    config = MODEL_CONFIG[modelo]
    display_individual_model(modelo, config)
else:
    compare_models()