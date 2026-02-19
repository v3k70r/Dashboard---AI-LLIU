import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import plotly.express as px

# 1. CONFIGURACIÓN DE PÁGINA Y ESTÉTICA MEJORADA
st.set_page_config(page_title="Dashboard AI-LLIU | Impacto Nacional", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")

# Paleta de colores más profesional y contrastante
COLOR_FONDO = "#F4F7F6"
COLOR_PRIMARIO = "#0D1B2A"
COLOR_SECUNDARIO = "#415A77"
COLOR_ACENTO = "#E0E1DD"
COLOR_GRAFICOS = ["#1B263B", "#415A77", "#778DA9", "#E0E1DD", "#00A896"]

st.markdown(f"""
    <style>
    /* Tipografía y fondo global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {COLOR_FONDO};
    }}
    h1, h2, h3 {{ 
        color: {COLOR_PRIMARIO}; 
        font-weight: 600;
        letter-spacing: -0.5px;
    }}
    /* Tarjetas de Métricas (KPIs) con efecto Hover */
    div[data-testid="metric-container"] {{
        background-color: #FFFFFF;
        border: 1px solid #EAEAEA;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        border-left: 6px solid #00A896;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    div[data-testid="metric-container"]:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }}
    /* Ajuste de separadores */
    hr {{ border-color: #D1D5DB; }}
    </style>
""", unsafe_allow_html=True)


# 2. FUNCIÓN DE NORMALIZACIÓN DE DATOS (Filtro de Ruido)
def normalizar_cargos(serie_cargos):
    """
    Toma una serie de Pandas con textos sucios y los agrupa mediante patrones Regex.
    """
    # Convertir a string, pasar a minúsculas y quitar espacios extra
    s = serie_cargos.astype(str).str.lower().str.strip()

    # Aplicar reglas de normalización (el orden importa, las más específicas primero)
    condiciones = [
        s.str.contains('utp|tecnico|técnico|curricular|evaluador', regex=True),
        s.str.contains('director|directora|rector|rectora', regex=True),
        s.str.contains('profesor|docente|educador|maestro', regex=True),
        s.str.contains('coordinador|coordinadora', regex=True),
        s.str.contains('inspector', regex=True),
        s.str.contains('psicologo|psicóloga|pie|fonoaudiólogo', regex=True)
    ]

    # Etiquetas estandarizadas correspondientes
    etiquetas = [
        'Jefatura UTP',
        'Equipo Directivo',
        'Docente de Aula',
        'Coordinación',
        'Inspectoría',
        'Equipo PIE / Apoyo'
    ]

    # Aplicar las condiciones, si no cumple ninguna, se clasifica como 'Otro'
    s_normalizada = np.select(condiciones, etiquetas, default='Otro Profesional')

    # Casos donde no había dato
    s_normalizada = np.where(s == 'nan', 'No Especificado', s_normalizada)
    s_normalizada = np.where(s == 'sin especificar', 'No Especificado', s_normalizada)

    return s_normalizada


# 3. MOTOR DE EXTRACCIÓN Y PROCESAMIENTO ETL
@st.cache_data(show_spinner="Consolidando bases de datos y normalizando perfiles...")
def load_and_process_data():
    output_file = "full_conversations.json"
    parts = sorted(glob.glob("*ConversationTable*.part*"))

    if not os.path.exists(output_file) and len(parts) > 0:
        with open(output_file, "wb") as outfile:
            for part in parts:
                with open(part, "rb") as infile:
                    outfile.write(infile.read())

    try:
        df_users = pd.read_csv("cleaned_cognito_users.csv")
        # NORMALIZACIÓN: Aplicamos el filtro a la base de usuarios
        if 'jobTitle' in df_users.columns:
            df_users['jobTitle_norm'] = normalizar_cargos(df_users['jobTitle'])
        else:
            df_users['jobTitle_norm'] = 'No Especificado'

    except FileNotFoundError:
        st.error("Error crítico: No se encuentra 'cleaned_cognito_users.csv'.")
        st.stop()

    records = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                pk = item.get("PK", "")
                sk = item.get("SK", "")

                if pk and "BOT_ALIAS" not in sk:
                    create_time = item.get("CreateTime")
                    dt = pd.to_datetime(int(create_time), unit='ms') if create_time else None

                    records.append({
                        "UserId": pk,
                        "Fecha": dt,
                        "Titulo": item.get("Title", "Interacción General"),
                        "Costo": float(item.get("TotalPrice", 0))
                    })

    df_conv = pd.DataFrame(records)
    if not df_conv.empty:
        df_conv['Fecha'] = pd.to_datetime(df_conv['Fecha'])

    if not df_conv.empty and not df_users.empty:
        df_master = pd.merge(df_conv, df_users, left_on="UserId", right_on="sub", how="left")
        df_master['region'] = df_master['region'].fillna('Desconocida')
    else:
        df_master = df_conv
        df_master['jobTitle_norm'] = 'No Especificado'
        df_master['region'] = 'Desconocida'

    return df_master, df_users


# 4. EJECUCIÓN DEL PIPELINE
df_master, df_users = load_and_process_data()

# 5. INTERFAZ VISUAL
st.markdown("<h1 style='text-align: center;'>🤖 Monitor de Impacto: Proyecto AI-LLIU</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #778DA9; font-size: 1.1rem;'>Análisis de penetración nacional e interacciones del modelo MASXXI</p>",
    unsafe_allow_html=True)
st.write("")

# --- KPIs Ejecutivos ---
usuarios_unicos = df_master['UserId'].nunique() if not df_master.empty else 0
interacciones_totales = len(df_master)
alumnos_impactados = usuarios_unicos * 35

col1, col2, col3, col4 = st.columns(4)
col1.metric("Docentes Activos", f"{usuarios_unicos:,}".replace(",", "."))
col2.metric("Salas Impactadas (Aprox)", f"{usuarios_unicos:,}".replace(",", "."))
col3.metric("Matrícula Nacional Beneficiada", f"{alumnos_impactados:,}".replace(",", "."), "+35 por docente")
col4.metric("Total Interacciones IA", f"{interacciones_totales:,}".replace(",", "."))

st.write("---")

# --- Gráficos Analíticos Mejorados ---
st.markdown("### 📊 Adopción y Alcance Territorial")
col_graf1, col_graf2 = st.columns([1.2, 1])

with col_graf1:
    # Gráfico de barras mejorado (Regiones)
    df_region = df_master['region'].value_counts().reset_index()
    df_region.columns = ['Región', 'Interacciones']
    df_region = df_region.sort_values('Interacciones', ascending=True)  # Orden para barras horizontales

    fig_region = px.bar(df_region, x='Interacciones', y='Región', orientation='h',
                        color_discrete_sequence=["#1B263B"])
    fig_region.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Número de Sesiones",
        yaxis_title="",
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="Inter", size=12, color=COLOR_PRIMARIO)
    )
    # Suavizar las líneas de la grilla
    fig_region.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#EAEAEA')
    st.plotly_chart(fig_region, use_container_width=True)

with col_graf2:
    # Gráfico de Dona (Roles Normalizados)
    df_rol = df_master['jobTitle_norm'].value_counts().reset_index()
    df_rol.columns = ['Cargo', 'Cantidad']

    fig_rol = px.pie(df_rol, names='Cargo', values='Cantidad', hole=0.5,
                     color_discrete_sequence=COLOR_GRAFICOS)
    fig_rol.update_traces(textposition='inside', textinfo='percent+label',
                          hoverinfo='label+value+percent', marker=dict(line=dict(color='#FFFFFF', width=2)))
    fig_rol.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=10, b=10),
        showlegend=False,  # Leyenda oculta para diseño más limpio, los datos están en el hover/inside
        font=dict(family="Inter", size=12)
    )
    st.plotly_chart(fig_rol, use_container_width=True)

st.write("---")

st.markdown("### 📈 Crecimiento Temporal de Uso")
if not df_master.empty:
    df_timeline = df_master.set_index('Fecha').resample('D').size().reset_index(name='Sesiones')
    # Rellenar fechas sin datos para no cortar la línea
    df_timeline['Sesiones'] = df_timeline['Sesiones'].replace(0, np.nan).interpolate().fillna(0)

    fig_time = px.area(df_timeline, x='Fecha', y='Sesiones',
                       color_discrete_sequence=["#00A896"])
    fig_time.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="",
        yaxis_title="Volumen de Sesiones MASXXI",
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified"  # Tooltip consolidado y elegante
    )
    fig_time.update_xaxes(showgrid=False)
    fig_time.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EAEAEA')
    st.plotly_chart(fig_time, use_container_width=True)