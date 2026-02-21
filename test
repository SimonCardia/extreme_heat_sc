import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, average_precision_score)

# --- 1. SETUP & DATEN LADEN ---
# Findet Dateien im selben Ordner wie dieses Skript
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    processed_frames = {}
    city_climatologies = {}
    
    # Suche alle JSON Dateien
    all_files = os.listdir(BASE_DIR)
    json_files = sorted([f for f in all_files if f.endswith('.json')])
    parquet_lookup = {f.lower(): f for f in all_files if f.endswith('.parquet')}

    if not json_files:
        st.error(f"Keine JSON-Dateien in {BASE_DIR} gefunden!")
        st.stop()

    for j_file in json_files:
        j_path = os.path.join(BASE_DIR, j_file)
        with open(j_path, 'r') as f:
            meta = json.load(f)
        
        city_name = meta['city']
        p_file = f"{city_name.lower()}.parquet"
        
        if p_file in parquet_lookup:
            p_path = os.path.join(BASE_DIR, parquet_lookup[p_file])
            df = pd.read_parquet(p_path)
            
            # Speicher sparen
            cols = df.select_dtypes(include=['float64']).columns
            df[cols] = df[cols].astype('float32')
            
            processed_frames[city_name] = df
        else:
            st.warning(f"Parquet für {city_name} fehlt!")
            
    return processed_frames

# --- 2. FEATURE ENGINEERING LOGIK ---
TRAIN_END = '2015-12-31'
REF_START = '1990-01-01'
ZSCORE_COLS = ['temperature_2m', 't_850hPa', 't_500hPa', 'pressure_msl', 
               'relative_humidity_2m', 'soil_moisture_0_to_7cm', 'wind_speed_10m']

def compute_climatology(df, cols):
    ref = df[REF_START:TRAIN_END]
    clim = {}
    for col in cols:
        if col in df.columns:
            grp = ref.groupby([ref.index.dayofyear, ref.index.hour])[col]
            clim[col] = {'mean': grp.mean(), 'std': grp.std()}
    return clim

def apply_zscore(df, clim):
    doy = df.index.dayofyear.values
    hour = df.index.hour.values
    for col, stats in clim.items():
        mean_2d = stats['mean'].unstack(level=1).values
        std_2d = stats['std'].unstack(level=1).values
        # Schutz gegen Index-Out-of-Bounds (Schaltjahre 366)
        doy_idx = np.clip(doy - 1, 0, mean_2d.shape[0]-1)
        df[f'{col}_zscore'] = (df[col].values - mean_2d[doy_idx, hour]) / (std_2d[doy_idx, hour] + 1e-8)
    return df

def engineer_features(df, city_name):
    df = df.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)

    clim = compute_climatology(df, ZSCORE_COLS)
    df = apply_zscore(df, clim)

    # Cyclical Time
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['doy_sin']  = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['doy_cos']  = np.cos(2 * np.pi * df.index.dayofyear / 365)

    # Lags & Delta
    if 't_850hPa' in df.columns: df['lag_t850_72h'] = df['t_850hPa'].shift(72)
    if 'pressure_msl' in df.columns: 
        df['lag_press_48h'] = df['pressure_msl'].shift(48)
        df['delta_press_24h'] = df['pressure_msl'] - df['pressure_msl'].shift(24)
    if 'v_850hPa' in df.columns: 
        df['lag_v850_72h'] = df['v_850hPa'].shift(72)
        df['v850_smooth_6h'] = df['v_850hPa'].rolling(6).mean()
    
    # Heat Aridity Index
    if 't_850hPa' in df.columns and 'relative_humidity_2m' in df.columns:
        df['heat_aridity_index'] = (df['t_850hPa'] - 273.15) - df['relative_humidity_2m']

    return df, clim

# --- 3. TARGET LABELING ---
def label_heatwave(df, percentile=0.95):
    daily_max = df['temperature_2m'].resample('D').max()
    threshold = daily_max[REF_START:TRAIN_END].quantile(percentile)
    
    daily_max_hourly = daily_max.reindex(df.index, method='ffill')
    is_hot_hour = (daily_max_hourly >= threshold).astype(float)
    
    # Klement: 72h am Stück heiß
    is_klement = is_hot_hour.rolling(window=72).min()
    # Target: Klement in den nächsten 72h
    y = is_klement.shift(-72).rolling(window=72, min_periods=1).max()
    return y, threshold

# --- MAIN APP ---
st.title("☀️ Heatwave Prediction Training Pipeline")

processed_frames = load_data()
engineered_frames = {}
city_climatologies = {}

st.write("🔧 Engineering Features...")
for city, df in processed_frames.items():
    df_eng, clim = engineer_features(df, city)
    y, thresh = label_heatwave(df_eng)
    df_eng['y'] = y
    engineered_frames[city] = df_eng
    city_climatologies[city] = clim

# Mergen & Filtern (Mai-Sept)
full_df = pd.concat([df.assign(city=c) for c, df in engineered_frames.items()])
full_df = full_df[full_df.index.month.isin([5,6,7,8,9])].dropna()
full_df = pd.get_dummies(full_df, columns=['city'])

# Split
train = full_df[full_df.index <= TRAIN_END]
test = full_df[full_df.index > TRAIN_END]

X_train, y_train = train.drop('y', axis=1), train['y']
X_test, y_test = test.drop('y', axis=1), test['y']

# Gewichtung
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Training mit deinen Best-Params
st.write("🚀 Training XGBoost Model...")
model = xgb.XGBClassifier(
    n_estimators=1025,
    max_depth=3,
    learning_rate=0.064,
    subsample=0.68,
    colsample_bytree=0.77,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='aucpr',
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluation
y_pred = model.predict(X_test)
st.success("Training abgeschlossen!")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred))

# Speichern
joblib.dump(model, 'model_final.pkl')
joblib.dump(city_climatologies, 'climatologies.pkl')
st.write("✅ Modelle als .pkl gespeichert.")
