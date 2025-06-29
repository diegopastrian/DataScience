import streamlit as st
import pandas as pd
from src.data_processing import load_dataset, clean_data, group_and_aggregate_data
from src.feature_engineering import create_domain_features
from src.predict import load_model_and_predict
from src.model_training import metrics_model
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, FEATURES_TO_USE, TARGET_VARIABLE, MESES_MAPPING
import plotly.express as px
import joblib

st.set_page_config(page_title="Predicción de Accidentes", layout="wide")
st.title("Predicción de Accidentes por Comuna")

# --- Sidebar: Selector de modo y filtros ---
st.sidebar.header("Configuración y Filtros")
modo = st.sidebar.radio("Modo", ["Histórico (2022-2023)", "Predicción Futura (2024)"])

if modo == "Histórico (2022-2023)":
    df = load_dataset(TRAIN_DATA_PATH)
    color_linea = "#1f77b4"
    color_barra = "Blues"
    color_top = "Oranges"
    titulo = "<h3 style='color:#1f77b4;'>Histórico de Accidentes (2022-2023)</h3>"
else:
    df = load_dataset(TEST_DATA_PATH)
    color_linea = "#FF914D"
    color_barra = "Oranges"
    color_top = "Oranges"
    titulo = "<h3 style='color:#FF914D;'>Predicción Futura de Accidentes (2024)</h3>"

df_clean = clean_data(df)
df_feat = create_domain_features(df_clean)
if 'Idaccidente' in df.columns:
    df_feat['Idaccidente'] = df['Idaccidente']
df_grouped = group_and_aggregate_data(df_feat, FEATURES_TO_USE, TARGET_VARIABLE)
df_pred = load_model_and_predict(df_grouped)

anios = sorted(df_pred['año'].unique())
meses_num = sorted(df_pred['Mes'].unique())
MESES_INV = {v: k.capitalize() for k, v in MESES_MAPPING.items()}
meses_nombres = [MESES_INV[m] for m in meses_num]
meses_nombres_todos = ["Todos"] + meses_nombres

anio_sel = st.sidebar.selectbox("Selecciona el año", anios, key="anio")
mes_nombre_sel = st.sidebar.selectbox("Selecciona el mes", meses_nombres_todos, key="mes")

# --- Contenido principal ---
st.markdown(titulo, unsafe_allow_html=True)

if mes_nombre_sel == "Todos":
    df_filtrado = df_pred[df_pred['año'] == anio_sel]
    top5 = df_filtrado.groupby("comuna")["accidentes_predichos"].sum().reset_index().sort_values(by="accidentes_predichos", ascending=False).head(5)
    st.subheader(f"Top 5 comunas con mayor predicción de accidentes ({anio_sel}, todo el año)")
else:
    mes_sel = [k for k, v in MESES_INV.items() if v == mes_nombre_sel][0]
    df_filtrado = df_pred[(df_pred['año'] == anio_sel) & (df_pred['Mes'] == mes_sel)]
    top5 = df_filtrado.sort_values(by="accidentes_predichos", ascending=False).head(5)
    st.subheader(f"Top 5 comunas con mayor predicción de accidentes ({mes_nombre_sel} {anio_sel})")

c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(top5[["comuna", "accidentes_predichos"]].set_index("comuna").style.background_gradient(cmap=color_top))
with c2:
    fig = px.bar(top5, x="comuna", y="accidentes_predichos", color="accidentes_predichos",
                 color_continuous_scale=color_top, labels={"accidentes_predichos": "Accidentes predichos"},
                 title="Top 5 comunas")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
if top5.empty:
    st.info("No hay datos para ese mes y año.")

st.subheader("Tendencia temporal de accidentes predichos" + (" (proyección)" if modo != "Histórico (2022-2023)" else ""))
trend = df_pred.groupby(["año", "Mes"])["accidentes_predichos"].sum().reset_index()
trend["Mes_nombre"] = trend["Mes"].map(MESES_INV)
trend["Periodo"] = trend["año"].astype(str) + "-" + trend["Mes"].astype(str).str.zfill(2)
fig2 = px.line(trend, x="Periodo", y="accidentes_predichos", markers=True,
               labels={"accidentes_predichos": "Accidentes predichos", "Periodo": "Año-Mes"},
               title="Accidentes predichos por mes" + (" (proyección)" if modo != "Histórico (2022-2023)" else ""))
fig2.update_traces(line_color=color_linea)
fig2.update_layout(template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

if modo == "Histórico (2022-2023)":
    st.subheader("Métricas globales del modelo (validación cruzada)")
    X = df_grouped[FEATURES_TO_USE]
    y = df_grouped[TARGET_VARIABLE]
    metrics = metrics_model(X, y)
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['R²']:.3f}")
    m2.metric("MAE", f"{metrics['MAE']:.2f}")
    m3.metric("RMSE", f"{metrics['RMSE']:.2f}")
    st.caption("Otras métricas:")
    st.write({k: v for k, v in metrics.items() if k not in ['R²', 'MAE', 'RMSE']})

    st.subheader("Importancia de variables del modelo")
    model_pipeline = joblib.load("models/gradient_boosting_regressor.pkl")
    importances = model_pipeline['model'].feature_importances_
    features = model_pipeline['features']
    df_importance = pd.DataFrame({"feature": features, "importance": importances}).sort_values(by="importance", ascending=False)
    fig3 = px.bar(df_importance, x="feature", y="importance", color="importance",
                  color_continuous_scale=color_barra, title="Importancia de variables")
    fig3.update_layout(template="plotly_white", yaxis_title="Importancia", xaxis_title="Variable")
    st.plotly_chart(fig3, use_container_width=True)
