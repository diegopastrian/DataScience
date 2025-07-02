def predict_xgb():
    print("Realizando predicción con XGBoost...")
    # Procesar TEST igual que en Colab
    df_test = load_dataset(TEST_DATA_PATH)
    df_test["comuna"] = df_test["Nomcomuna"].astype(str).str.lower().str.strip()
    df_test["zona_urbana"] = df_test["Urbano/Rural"].astype(str).str.lower().str.strip().apply(lambda x: 1 if x == "urbano" else 0)
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    df_test["Mes"] = df_test["Mes"].astype(str).str.lower().map(meses).fillna(df_test["Mes"]).astype(float)
    df_test["total_vehiculos"] = df_test["total_vehiculos"].fillna(df_test["total_vehiculos"].mean())
    df_test["poblacion"] = df_test["poblacion"].fillna(df_test["poblacion"].mean())
    for col in ["Muertos", "Graves", "M/Grave", "Leves", "Ilesos"]:
        df_test[col] = df_test[col].fillna(df_test[col].median())
    # Agrupación
    features = ["total_vehiculos", "poblacion", "zona_urbana", "Muertos", "Graves", "M/Grave", "Leves", "Ilesos"]
    agg_dict = {col: "mean" for col in features}
    agg_dict["Idaccidente"] = "count"
    df_test_grouped = df_test.groupby(["comuna", "Mes", "año"]).agg(agg_dict).rename(columns={"Idaccidente": "total_accidentes"}).reset_index()
    df_test_xgb = create_xgb_features(df_test_grouped)
    X_test = df_test_xgb[FEATURES_FINAL]
    y_test = df_test_xgb[TARGET_VARIABLE]
    # Cargar modelo
    model_pipeline = joblib.load(MODEL_OUTPUT_PATH)
    model = model_pipeline['model']
    y_pred = np.expm1(model.predict(X_test))
    # Cast a int para visualización
    y_pred_int = np.round(y_pred).astype(int)
    return y_test, y_pred_int, X_test 