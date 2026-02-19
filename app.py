import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px

# 1. CONFIGURACIÓN DE PÁGINA Y ESTÉTICA (Ajusta los HEX a los colores exactos de AI-LLIU)
st.set_page_config(page_title="Dashboard AI-LLIU | Impacto Nacional", page_icon="📊", layout="wide")

COLOR_FONDO = "#F8F9FA"
COLOR_PRIMARIO = "#0A2540"  # Azul corporativo (Ejemplo)
COLOR_SECUNDARIO = "#635BFF"  # Morado/Azul claro (Ejemplo)
COLOR_ACENTO = "#00D4FF"  # Cian para destacar

st.markdown(f"""
    <style>
    .stApp {{ background-color: {COLOR_FONDO}; }}
    h1, h2, h3 {{ color: {COLOR_PRIMARIO}; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
    div[data-testid="metric-container"] {{
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 6px solid {COLOR_SECUNDARIO};
    }}
    </style>
""", unsafe_allow_html=True)


# 2. MOTOR DE EXTRACCIÓN Y PROCESAMIENTO ETL
@st.cache_data(show_spinner="Procesando matriz de datos de 234MB. Esto ocurrirá solo una vez...")
def load_and_process_data():
    output_file = "full_conversations.json"
    parts = sorted(glob.glob("*ConversationTable*.part*"))

    # Ensamblaje binario de fragmentos si el archivo unificado no existe
    if not os.path.exists(output_file) and len(parts) > 0:
        with open(output_file, "wb") as outfile:
            for part in parts:
                with open(part, "rb") as infile:
                    outfile.write(infile.read())

    # Carga de base de usuarios Cognito
    try:
        df_users = pd.read_csv("cleaned_cognito_users.csv")
    except FileNotFoundError:
        st.error("Error crítico: No se encuentra 'cleaned_cognito_users.csv'.")
        st.stop()

    # Extracción de métricas de DynamoDB
    records = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                pk = item.get("PK", "")
                sk = item.get("SK", "")

                # Filtramos para obtener las interacciones reales y no los metadatos del bot
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
    df_conv['Fecha'] = pd.to_datetime(df_conv['Fecha'])

    # Operación de JOIN (Cruce de IDs)
    if not df_conv.empty and not df_users.empty:
        df_master = pd.merge(df_conv, df_users, left_on="UserId", right_on="sub", how="left")
        # Rellenar valores nulos de Cognito
        df_master['region'] = df_master['region'].fillna('Desconocida')
        df_master['jobTitle'] = df_master['jobTitle'].fillna('Sin especificar')
    else:
        df_master = df_conv

    return df_master, df_users


# 3. EJECUCIÓN DEL PIPELINE
df_master, df_users = load_and_process_data()

# 4. INTERFAZ Y VISUALIZACIÓN
st.markdown("<h1 style='text-align: center;'>🤖 Monitor de Impacto: Proyecto AI-LLIU</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Análisis de penetración nacional e interacciones del modelo MASXXI</p>",
            unsafe_allow_html=True)
st.divider()

# --- KPIs Ejecutivos (High-Level) ---
usuarios_unicos = df_master['UserId'].nunique()
interacciones_totales = len(df_master)
# Cálculo estimado de impacto: Promedio estándar de 35 alumnos por aula chilena
alumnos_impactados = usuarios_unicos * 35
costo_total_api = df_master['Costo'].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Docentes Activos", f"{usuarios_unicos:,}".replace(",", "."))
col2.metric("Salas Impactadas (Aprox)", f"{usuarios_unicos:,}".replace(",", "."))
col3.metric("Matrícula Nacional Beneficiada", f"{alumnos_impactados:,}".replace(",", "."), "Est. 35 alumnos/docente")
col4.metric("Total Interacciones IA", f"{interacciones_totales:,}".replace(",", "."))

st.write("---")

# --- Gráficos Analíticos ---
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.subheader("📍 Penetración Territorial (Regiones)")
    # Agrupación por región
    df_region = df_master['region'].value_counts().reset_index()
    df_region.columns = ['Región', 'Interacciones']
    fig_region = px.bar(df_region, x='Interacciones', y='Región', orientation='h',
                        color_discrete_sequence=[COLOR_SECUNDARIO],
                        title="Uso de AI-LLIU por Región")
    fig_region.update_layout(yaxis={'categoryorder': 'total ascending'}, plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_region, use_container_width=True)

with col_graf2:
    st.subheader("🎓 Adopción por Perfil de Usuario")
    df_rol = df_master['jobTitle'].value_counts().reset_index()
    df_rol.columns = ['Cargo', 'Cantidad']
    fig_rol = px.pie(df_rol, names='Cargo', values='Cantidad', hole=0.4,
                     color_discrete_sequence=px.colors.sequential.Teal,
                     title="Distribución de Roles en la Plataforma")
    st.plotly_chart(fig_rol, use_container_width=True)

st.write("---")

st.subheader("📈 Crecimiento Temporal de Uso")
df_timeline = df_master.set_index('Fecha').resample('D').size().reset_index(name='Sesiones')
fig_time = px.line(df_timeline, x='Fecha', y='Sesiones',
                   color_discrete_sequence=[COLOR_PRIMARIO],
                   title="Interacciones Diarias")
fig_time.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Fecha", yaxis_title="N° de Sesiones MASXXI")
st.plotly_chart(fig_time, use_container_width=True)