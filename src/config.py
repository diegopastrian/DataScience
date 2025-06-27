# src/config.py

# --- Rutas de Archivos ---
# Define dónde se encuentran los datos de entrada y dónde se guardará el modelo.
TRAIN_DATA_PATH = "data/Dataset_Train.csv"
TEST_DATA_PATH = "data/Dataset_Test.csv"
MODEL_OUTPUT_PATH = "models/gradient_boosting_regressor.pkl" # Guardaremos el modelo aquí

# --- Definición de Variables ---
# Define la variable objetivo y las características que usará el modelo.
TARGET_VARIABLE = "total_accidentes"

FEATURES_TO_USE = [
    "total_vehiculos", 
    "poblacion", 
    "densidad_vehicular", # Esta la crearemos en feature engineering
    "Mes", 
    "zona_urbana", # Esta la crearemos en feature engineering
    "Muertos", 
    "Graves", 
    "M/Grave", 
    "Leves", 
    "Ilesos"
]

# --- Parámetros Modelo ---
MODEL_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42
}

# --- Mapeos y Constantes de Procesamiento ---
# Mapeo para convertir el nombre del mes a un número.
MESES_MAPPING = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
}

# Columnas a eliminar durante la limpieza inicial.
COLUMNAS_A_ELIMINAR = [
     'Fecha', 'Provincia', 'region', 'Parte Nro.', 'comuna', 'Region', 
    'Nombre Comuna', 'Nombre Provincia', 'Región', 'Rolruta', 'Ubicacionkm', 
    'Siniestros', 'Accdte', 'Partenro', 'Mes2', 'Frentenumero', 'Ruta', 'Ubicación km',
    'Comuna_y'
]