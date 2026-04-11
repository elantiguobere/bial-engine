import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="BIAL IA Bridge - Custom Path", layout="wide")

with st.sidebar:
    st.header("⚙️ Configuración")
    symbol_yahoo = st.text_input("Símbolo Yahoo Finance", value="GC=F")
    symbol_mt5 = st.text_input("Nombre en tu MT5", value="GOLD")
    # DEFINIMOS TU RUTA PERSONALIZADA
    custom_path = r"C:\Users\Usuario\Desktop\Proyecto Sebas"
    
    st.divider()
    intervalo = st.selectbox("Temporalidad", ["1d", "1h", "15m", "5m"], index=0)
    days = st.slider("Días de Histórico", 5, 720, 360)
    n_clusters = st.slider("Cantidad de Niveles (K)", 3, 12, 6)
    window = st.number_input("Ventana de Pivot", value=5)
    run = st.button("🚀 Exportar Niveles", use_container_width=True)

if run:
    df = yf.download(symbol_yahoo, period=f"{days}d", interval=intervalo)
    if not df.empty:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        pivots = []
        highs, lows = df['High'].values, df['Low'].values
        for i in range(window, len(df)-window):
            if highs[i] == max(highs[i-window:i+window+1]): pivots.append(highs[i])
            if lows[i] == min(lows[i-window:i+window+1]): pivots.append(lows[i])
        
        pivots = np.array(pivots).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(pivots)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(scaled)
        centers = np.sort(scaler.inverse_transform(km.cluster_centers_).flatten())

        # EXPORTACIÓN A RUTA ESPECÍFICA
        try:
            if not os.path.exists(custom_path):
                os.makedirs(custom_path)
            
            filename = f"ia_levels_{symbol_mt5.lower()}.csv"
            full_path = os.path.join(custom_path, filename)
            
            pd.DataFrame(centers).to_csv(full_path, index=False, header=False, float_format='%.2f')
            st.success(f"✅ Archivo guardado en Escritorio: {filename}")
            st.info(f"Ruta: {full_path}")
        except Exception as e:
            st.error(f"Error al guardar: {e}")
