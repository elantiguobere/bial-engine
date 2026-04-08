import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
from fpdf import FPDF
from pypfopt.efficient_frontier import EfficientFrontier

# ==========================================
# 1. CONFIGURACIÓN Y SEGURIDAD BIAL
# ==========================================
st.set_page_config(page_title="BIAL AI Portfolio Manager", page_icon="🏆", layout="wide")

PASSWORD_BIAL = "BIAL2026"

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        st.markdown("<h1 style='text-align: center; color: #f59e0b;'>🔐 Acceso BIAL TRADING</h1>", unsafe_allow_html=True)
        pwd = st.text_input("Ingresa la clave de acceso de alumno:", type="password")
        if st.button("Ingresar"):
            if pwd == PASSWORD_BIAL:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Clave incorrecta. Acceso denegado.")
        return False
    return True

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #f59e0b; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #f59e0b; color: black; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. FUNCIONES CORE Y AUTO-DETECCIÓN IA
# ==========================================
def analizar_con_ia(api_key, datos, rango):
    if not api_key: return "⚠️ Ingrese su API Key en la barra lateral para activar la IA."
    
    try:
        url_models = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        resp_models = requests.get(url_models)
        
        if resp_models.status_code != 200:
            return f"❌ Error de Autenticación API: {resp_models.json().get('error', {}).get('message', 'Error')}"
            
        modelos_disponibles = resp_models.json().get('models', [])
        
        modelo_elegido = None
        for m in modelos_disponibles:
            if 'generateContent' in m.get('supportedGenerationMethods', []) and 'gemini' in m.get('name', '').lower():
                if '1.5-flash' in m.get('name'):
                    modelo_elegido = m.get('name')
                    break
                elif not modelo_elegido:
                    modelo_elegido = m.get('name')

        if not modelo_elegido:
            return "❌ Tu API Key no tiene ningún modelo Gemini habilitado para generar texto."

        url_gen = f"https://generativelanguage.googleapis.com/v1beta/{modelo_elegido}:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        
        prompt = f"""
        Actúa como un Consultor Senior de Riesgos en BIAL TRADING. 
        Analiza este portafolio de trading algorítmico:
        - Rango BIAL: {rango}
        - Profit Neto: ${datos['p']:,.2f}
        - Máximo Drawdown: ${datos['dd']:,.2f}
        - Ratio Sharpe: {datos['sh']:.2f}
        - Cantidad de Estrategias: {datos['n']}
        
        Brinda un informe técnico breve (3 párrafos) sobre la robustez del portafolio, el riesgo asumido y una recomendación estratégica directa.
        """
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url_gen, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"❌ Error del Servidor al procesar: {response.text}"
            
    except Exception as e:
        return f"❌ Error crítico de Conexión: {str(e)}"

def calcular_kpis(series, capital):
    equity = capital + series.cumsum()
    net_profit = series.sum()
    max_dd = (equity - equity.cummax()).min()
    ratio = abs(net_profit / max_dd) if max_dd != 0 else 0
    sharpe = (series.mean() / series.std()) * np.sqrt(252) if series.std() != 0 else 0
    return net_profit, max_dd, ratio, sharpe

def obtener_rango_bial(series, capital, corr_matrix):
    net_p, m_dd, m_ratio, sharpe = calcular_kpis(series, capital)
    s_score = min(30, (max(0, sharpe) / 2) * 30)
    c_score = min(30, (m_ratio / 3) * 30)
    
    try:
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        if np.isnan(avg_corr): avg_corr = 0
    except:
        avg_corr = 0
        
    d_score = max(0, (1 - avg_corr) * 20)
    
    equity = series.cumsum()
    x = np.arange(len(equity))
    r_score = 0
    if len(x) > 1:
        slope, intercept = np.polyfit(x, equity, 1)
        varianza_total = np.sum((equity - equity.mean())**2)
        if varianza_total != 0:
            r_squared = 1 - (np.sum((equity - (slope * x + intercept))**2) / varianza_total)
            r_score = max(0, r_squared * 20)
    
    score = s_score + c_score + d_score + r_score
    if score >= 90: return score, "DIAMANTE", "💎", "#b9f2ff", "Nivel Institucional / Hedge Fund"
    elif score >= 80: return score, "ORO", "🥇", "#f59e0b", "Gestión Profesional"
    elif score >= 65: return score, "PLATA", "🥈", "#c0c0c0", "Cartera Robusta"
    elif score >= 50: return score, "BRONCE", "🥉", "#cd7f32", "Cartera Aceptable"
    return score, "SIN RANGO", "❌", "#ef4444", "Optimización Urgente"

class BIAL_Report(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(245, 158, 11)
        self.cell(0, 10, 'BIAL TRADING - AUDITORIA DE CARTERA', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Ingenieria Financiera para Traders Algoritmicos', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

# ==========================================
# 3. INTERFAZ Y FLUJO DE DATOS
# ==========================================
if check_password():
    st.sidebar.markdown("<h1 style='text-align: center; color: #f59e0b;'>BIAL ENGINE</h1>", unsafe_allow_html=True)
    cap_inicial = st.sidebar.number_input("Capital Inicial ($)", value=10000, step=1000)
    objetivo_opt = st.sidebar.selectbox("Optimizar para", ["Max Sharpe", "Min Volatilidad", "Max Return"])
    peso_max_ea = st.sidebar.slider("Peso Máximo por EA", 0.1, 1.0, 0.5)
    
    st.sidebar.markdown("---")
    auto_limpieza = st.sidebar.checkbox("🧹 Activar Auto-Limpieza BIAL", value=True, help="Elimina automáticamente estrategias con alta correlación (>0.75) y bajo Sharpe.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Reglas de Prop Firm")
    target_pct = st.sidebar.number_input("Target Requerido (%)", value=8.0, step=1.0)
    max_dd_pct = st.sidebar.number_input("Max Drawdown Global (%)", value=10.0, step=1.0)
    daily_dd_pct = st.sidebar.number_input("Max Daily DD (%)", value=5.0, step=1.0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Ejecución MT5")
    lotes_totales = st.sidebar.number_input("Lotes Totales a Distribuir", value=1.00, step=0.01)
    st.sidebar.markdown("---")

    api_key_gemini = st.sidebar.text_input("🔑 Gemini API Key", type="password", help="Pega aquí tu llave de Google AI Studio")

    st.title("🏆 BIAL AI Portfolio Manager Pro")
    archivos = st.file_uploader("Subí tus reportes CSV de StrategyQuant", accept_multiple_files=True, type=['csv'])

    if archivos:
        if st.button("🚀 CALCULAR ESTRATEGIA BIAL"):
            dict_ret = {}
            all_trades = []
            for f in archivos:
                df = pd.read_csv(f, sep=';')
                df['Close time'] = pd.to_datetime(df['Close time'])
                name = f.name.replace(".csv", "")
                dict_ret[name] = df.groupby(df['Close time'].dt.date)['Profit/Loss'].sum()
                
                df['EA'] = name
                simbolo = str(df['Symbol'].iloc[0]).upper() if not df['Symbol'].empty else "Otro"
                df['Asset'] = "ORO (XAUUSD)" if "XAU" in simbolo or "GOLD" in simbolo else simbolo
                all_trades.append(df[['EA', 'Asset', 'Profit/Loss', 'Close time']])
            
            df_retornos = pd.DataFrame(dict_ret).fillna(0)
            df_retornos.index = pd.to_datetime(df_retornos.index)
            df_retornos = df_retornos[df_retornos.index.notnull()]
            df_retornos = df_retornos.groupby(df_retornos.index).sum()
            df_retornos = df_retornos.astype(np.float64)

            eliminados_log = []
            if auto_limpieza and len(df_retornos.columns) > 1:
                corr_matrix_temp = df_retornos.corr()
                to_drop = set()
                columnas = corr_matrix_temp.columns
                
                for i in range(len(columnas)):
                    for j in range(i+1, len(columnas)):
                        ea1 = columnas[i]
                        ea2 = columnas[j]
                        if corr_matrix_temp.iloc[i, j] > 0.75:
                            std1 = df_retornos[ea1].std()
                            std2 = df_retornos[ea2].std()
                            sh1 = (df_retornos[ea1].mean() / std1) * np.sqrt(252) if std1 != 0 else 0
                            sh2 = (df_retornos[ea2].mean() / std2) * np.sqrt(252) if std2 != 0 else 0
                            
                            if sh1 >= sh2: to_drop.add(ea2)
                            else: to_drop.add(ea1)
                
                if to_drop:
                    df_retornos = df_retornos.drop(columns=list(to_drop))
                    eliminados_log = list(to_drop)

            df_trades = pd.concat(all_trades)
            df_trades = df_trades[df_trades['EA'].isin(df_retornos.columns)]

            df_pct = df_retornos / cap_inicial
            mu = df_pct.mean() * 252            
            S = df_pct.cov() * 252              
            
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(0.0, peso_max_ea))
                if "Sharpe" in objetivo_opt: weights = ef.max_sharpe()
                elif "Volatilidad" in objetivo_opt: weights = ef.min_volatility()
                else: weights = ef.max_sharpe()
                cleaned_weights = ef.clean_weights()
            except:
                try:
                    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, peso_max_ea))
                    weights = ef.min_volatility()
                    cleaned_weights = ef.clean_weights()
                except:
                    n_estrategias = len(df_retornos.columns)
                    peso_fijo = 1.0 / n_estrategias
                    cleaned_weights = {ea: peso_fijo for ea in df_retornos.columns}

            portfolio_series = pd.Series(0.0, index=df_retornos.index)
            for ea, w in cleaned_weights.items(): portfolio_series += df_retornos[ea] * w
                
            matriz_correlacion = df_retornos.corr()
            score_f, rango, icono, color, desc = obtener_rango_bial(portfolio_series, cap_inicial, matriz_correlacion)
            net_p, m_dd, m_ratio, sharpe_f = calcular_kpis(portfolio_series, cap_inicial)

            target_usd = cap_inicial * (target_pct / 100)
            max_dd_usd = cap_inicial * (max_dd_pct / 100)
            daily_dd_usd = cap_inicial * (daily_dd_pct / 100)

            passed_prop, failed_prop, fail_reason, days_to_pass = False, False, "", 0
            current_equity, high_water_mark, worst_daily = cap_inicial, cap_inicial, portfolio_series.min()

            for date, daily_pnl in portfolio_series.items():
                days_to_pass += 1
                current_equity += daily_pnl
                if current_equity > high_water_mark: high_water_mark = current_equity
                current_dd = high_water_mark - current_equity
                
                if daily_pnl < -daily_dd_usd:
                    failed_prop = True
                    fail_reason = f"Violación de Daily Drawdown (${abs(daily_pnl):,.2f}) el {date.strftime('%d/%m/%Y')}."
                    break
                if current_dd > max_dd_usd:
                    failed_prop = True
                    fail_reason = f"Violación de Max Drawdown (${current_dd:,.2f}) el {date.strftime('%d/%m/%Y')}."
                    break
                if (current_equity - cap_inicial) >= target_usd:
                    passed_prop = True
                    break

            eval_prop = {
                'passed': passed_prop, 'failed': failed_prop, 'reason': fail_reason,
                'days': days_to_pass, 'target_usd': target_usd, 'max_dd_usd': max_dd_usd,
                'daily_dd_usd': daily_dd_usd, 'worst_daily': worst_daily
            }

            temp_df = pd.DataFrame({'Retorno_USD': portfolio_series})
            temp_df['Year'] = temp_df.index.year
            temp_df['Month'] = temp_df.index.month
            monthly_sum = temp_df.groupby(['Year', 'Month'])['Retorno_USD'].sum().reset_index()
            monthly_sum['Retorno_Pct'] = (monthly_sum['Retorno_USD'] / cap_inicial) * 100
            
            heatmap_data = monthly_sum.pivot(index='Year', columns='Month', values='Retorno_Pct')
            heatmap_data = heatmap_data.reindex(columns=range(1, 13))
            heatmap_data.rename(columns={1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 
                                         7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}, inplace=True)
            heatmap_data['YTD'] = heatmap_data.sum(axis=1)
            heatmap_data.index = heatmap_data.index.astype(str)

            st.session_state['calculado'] = True
            st.session_state['res'] = {
                'score': score_f, 'rango': rango, 'icono': icono, 'color': color, 'desc': desc,
                'p_series': portfolio_series, 'net_p': net_p, 'm_dd': m_dd, 'sharpe': sharpe_f,
                'trades': df_trades, 'weights': cleaned_weights, 'returns': df_retornos,
                'n_archivos': len(df_retornos.columns), 'corr_matrix': matriz_correlacion,
                'eliminados': eliminados_log, 'eval_prop': eval_prop, 'heatmap_data': heatmap_data
            }

        if st.session_state.get('calculado'):
            r = st.session_state['res']
            
            if r['eliminados']:
                st.warning(f"🧹 **Filtro BIAL Activado:** Se descartaron {len(r['eliminados'])} estrategias redundantes o con mal ratio riesgo/beneficio: {', '.join(r['eliminados'])}")
            
            if r['score'] >= 80: st.balloons()
            st.markdown(f"""
                <div style="background-color: #1f2937; padding: 25px; border-radius: 15px; border: 2px solid {r['color']}; text-align: center; margin-bottom: 20px;">
                    <h1 style="color: white; margin:0; font-size: 50px;">{r['icono']} {r['rango']}</h1>
                    <h2 style="color: {r['color']}; margin:10px 0;">{r['score']:.1f} / 100</h2>
                    <p style="color: #9ca3af; font-style: italic;">{r['desc']}</p>
                </div>
            """, unsafe_allow_html=True)

            tabs = st.tabs(["📈 Análisis Visual", "🌍 Activos", "🔗 Correlación", "🎯 Prueba de Fondeo", "⚙️ Ejecución MT5", "🤖 Consultoría IA", "📥 Auditoría"])
            
            with tabs[0]:
                equity_curve = cap_inicial + r['p_series'].cumsum()
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=equity_curve.index, y=equity_curve, mode='lines', name='Valor Cartera',
                    line=dict(color='#f59e0b', width=2), fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.2)',
                    hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Valor: $%{y:,.2f}<extra></extra>'
                ))
                fig_eq.update_layout(
                    title={'text': "<b>Crecimiento Patrimonial Consolidado BIAL</b>", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=20, color='#f59e0b')},
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title='Fecha', showgrid=False, color='#9ca3af'),
                    yaxis=dict(title='Valor de la Cartera ($)', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', tickprefix="$", color='#9ca3af'),
                    showlegend=False, hovermode="x unified"
                )
                st.plotly_chart(fig_eq, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📊 Rentabilidad Mensual Histórica (%)")
                fig_hm = px.imshow(r['heatmap_data'], 
                                   text_auto=".2f", 
                                   aspect="auto",
                                   color_continuous_scale="RdYlGn", 
                                   color_continuous_midpoint=0, 
                                   labels=dict(x="Mes", y="Año", color="Retorno %"))
                fig_hm.update_layout(xaxis_title="", yaxis_title="Año", coloraxis_showscale=False)
                fig_hm.update_xaxes(side="top") 
                st.plotly_chart(fig_hm, use_container_width=True)

            with tabs[1]:
                asset_perf = r['trades'].groupby('Asset')['Profit/Loss'].sum().sort_values()
                fig_assets = px.bar(asset_perf, orientation='h', color=asset_perf.values, 
                                    color_continuous_scale='RdYlGn', title="Beneficio Neto por Instrumento")
                st.plotly_chart(fig_assets, use_container_width=True)

            with tabs[2]:
                st.subheader("🔗 Mapa de Calor de Descorrelación")
                st.markdown("Valores cercanos a **0** o **negativos (azul/verde)** indican excelente diversificación. Valores cercanos a **1 (rojo)** indican que los sistemas operan igual.")
                fig_corr = px.imshow(r['corr_matrix'], text_auto=".2f", aspect="auto",
                                     color_continuous_scale="RdBu_r", title="Correlación Diaria entre Estrategias")
                st.plotly_chart(fig_corr, use_container_width=True)

            with tabs[3]:
                e = r['eval_prop']
                st.subheader("🎯 Simulador de Challenge (Prop Firms)")
                st.markdown("Basado en los límites establecidos en la barra lateral, BIAL ENGINE simula día a día si la cartera superaría la evaluación.")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Target Objetivo", f"${e['target_usd']:,.2f}")
                col2.metric("Límite Max DD", f"${e['max_dd_usd']:,.2f}")
                col3.metric("Límite Daily DD", f"${e['daily_dd_usd']:,.2f}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if e['passed']:
                    st.success(f"✅ **¡PRUEBA SUPERADA!** El portafolio alcanzó el target en **{e['days']} días** de operativa sin violar ninguna regla de riesgo. Apto para FTMO / FundedNext.")
                    st.snow()
                elif e['failed']:
                    st.error(f"❌ **PRUEBA REPROBADA:** {e['reason']} Días operados hasta la eliminación: {e['days']}.")
                else:
                    st.warning(f"⏳ **TIEMPO AGOTADO:** El portafolio operó todo el período histórico sin romper reglas, pero no logró alcanzar el Target.")
                    
                st.markdown("---")
                st.markdown("### 📊 Métricas de Estrés de la Cartera")
                st.write(f"- **Peor Pérdida en un solo día (Worst Daily):** ${abs(e['worst_daily']):,.2f}")
                st.write(f"- **Drawdown Global Máximo:** ${r['m_dd']:,.2f}")

            with tabs[4]:
                st.subheader("⚙️ Traducir a MetaTrader 5")
                st.markdown(f"Basado en un riesgo total distribuido de **{lotes_totales} lotes**, así debés configurar el parámetro `FixedLots` (o equivalente) en cada EA de tu MetaTrader 5.")
                
                lotes_data = []
                alertas_microlotes = False
                
                for ea, w in r['weights'].items():
                    if w > 0:
                        lotes_asignados = round(w * lotes_totales, 2)
                        estado = "✅ Listo para Operar"
                        if lotes_asignados < 0.01:
                            estado = "⚠️ Muy bajo (Redondear a 0.01 o descartar)"
                            alertas_microlotes = True
                            
                        lotes_data.append({
                            "Estrategia (EA)": ea, 
                            "Peso Markowitz": f"{w*100:.2f}%", 
                            "Lotes MT5 (FixedLots)": lotes_asignados,
                            "Estado": estado
                        })
                
                df_lotes = pd.DataFrame(lotes_data)
                
                # --- PARCHE DE COMPATIBILIDAD CON PANDAS 2.1+ ---
                st.dataframe(
                    df_lotes.style.format({"Lotes MT5 (FixedLots)": "{:.2f}"})
                                  .map(lambda x: 'background-color: #ef4444' if '⚠️' in str(x) else '', subset=['Estado']),
                    use_container_width=True
                )
                # ------------------------------------------------

                st.info("💡 **Tip BIAL TRADING:** Entrá a tu VPS, abrí las propiedades del Asesor Experto (F7) y pegá exactamente el número de la columna 'Lotes MT5' en la configuración de riesgo manual.")
                if alertas_microlotes:
                    st.warning("⚠️ OJO: Tenés estrategias a las que les toca menos de 0.01 lotes. MetaTrader 5 no soporta nano-lotes. Te sugiero aumentar los 'Lotes Totales' en la barra lateral o eliminar manualmente esa estrategia.")

            with tabs[5]:
                st.subheader("🤖 Consultoría Estratégica BIAL AI")
                if st.button("Generar Informe Senior"):
                    with st.spinner("La IA está revisando los modelos disponibles y auditando tu cartera..."):
                        analisis = analizar_con_ia(api_key_gemini, 
                                                  {'p': r['net_p'], 'dd': r['m_dd'], 'sh': r['sharpe'], 'n': r['n_archivos']}, 
                                                  r['rango'])
                        st.info(analisis)
            
            with tabs[6]:
                st.subheader("📂 Reportes de Auditoría Institucional")
                detalles = []
                for ea, w in r['weights'].items():
                    if w > 0:
                        p_i, dd_i, r_i, sh_i = calcular_kpis(r['returns'][ea], cap_inicial)
                        detalles.append({"Estrategia": ea, "Asignación": f"{w*100:.2f}%", "Profit": round(p_i, 2), "MaxDD": round(dd_i, 2)})
                
                df_det = pd.DataFrame(detalles)
                st.dataframe(df_det, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    xl_out = io.BytesIO()
                    with pd.ExcelWriter(xl_out, engine='xlsxwriter') as wr: df_det.to_excel(wr, index=False)
                    st.download_button("📥 Descargar Excel", xl_out.getvalue(), f"Auditoria_BIAL_{r['rango']}.xlsx")
                with col2:
                    pdf = BIAL_Report(); pdf.add_page(); pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, f"RESULTADO: {r['rango']} ({r['score']:.1f}/100)", 0, 1)
                    pdf.set_font("Arial", '', 11)
                    pdf.cell(0, 8, f"Profit: ${r['net_p']:,.2f} | MaxDD: ${r['m_dd']:,.2f} | Sharpe: {r['sharpe']:.2f}", 0, 1)
                    pdf.ln(5)
                    pdf.set_fill_color(245, 158, 11); pdf.set_text_color(255)
                    pdf.cell(120, 10, ' Estrategia', 1, 0, 'L', True); pdf.cell(50, 10, ' Asignacion', 1, 1, 'C', True)
                    pdf.set_text_color(0)
                    for i, row in df_det.iterrows():
                        pdf.cell(120, 8, f" {row['Estrategia']}", 1); pdf.cell(50, 8, f" {row['Asignación']}", 1, 1, 'C')
                    
                    st.download_button("📥 Descargar PDF", pdf.output(dest='S').encode('latin-1', 'replace'), f"Reporte_BIAL_{r['rango']}.pdf")
else:
    st.info("👋 Leandro, cargá los archivos de StrategyQuant para que BIAL ENGINE comience su análisis.")
