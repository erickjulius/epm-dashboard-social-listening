import streamlit as st
import pandas as pd
import re
from openai import OpenAI
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from gtts import gTTS
import os

# Configuración de página
st.set_page_config(
    page_title="Asistente de percepción de marca con IA",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Leer prompt base
with open("prompt_base.txt", "r", encoding="utf-8") as f:
    base_prompt = f.read()

# CSS personalizado para el diseño
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Ocultar elementos por defecto de Streamlit */
    .stApp > header {visibility: hidden;}
    .stApp > footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Fondo principal */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Contenedor principal */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Logo y header */
    .header-container {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .logo-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        display: inline-block;
        margin-bottom: 20px;
    }
    
    .title-main {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 30px;
    }
    
    /* Sección de filtros */
    .filters-section {
        background: rgba(0, 0, 0, 0.3);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 30px;
    }
    
    .filters-title {
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Botón principal */
    .generate-button {
        background: linear-gradient(45deg, #ff6b6b, #ff8e53);
        color: white;
        padding: 15px 40px;
        border: none;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: block;
        margin: 30px auto;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    .generate-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
    }
    
    /* Sección de preguntas */
    .questions-section {
        background: rgba(0, 0, 0, 0.3);
        padding: 25px;
        border-radius: 15px;
        margin: 30px 0;
    }
    
    .questions-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .questions-subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Secciones de contenido */
    .content-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .section-title {
        color: #333;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 2px solid #667eea;
        padding-bottom: 5px;
    }
    
    /* Footer */
    .footer-logo {
        text-align: center;
        margin-top: 40px;
        opacity: 0.7;
    }
    
    /* Estilos para elementos de Streamlit */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #ff8e53) !important;
        color: white !important;
        padding: 15px 40px !important;
        border: none !important;
        border-radius: 50px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6) !important;
    }
    
    /* Wordcloud container */
    .wordcloud-container {
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header con logo y título
st.markdown("""
<div class="main-container">
    <div class="header-container">
        <div class="logo-box">
            <img src="https://raw.githubusercontent.com/SantiagoCardonaP/epm-dashboard-social-listening/2fdfac81f49d7c03afea8e29c0d67d96bcdcf750/logo-grupo-epm%20(1).png" style="height: 60px;">
        </div>
        <h1 class="title-main">Asistente de percepción de marca con IA</h1>
        <p class="subtitle">Te cuento cómo está nuestra percepción de marca en los territorios</p>
    </div>
""", unsafe_allow_html=True)

# Ruta del archivo predeterminado
file = "Menciones_EPM.csv"

if file:
    df = pd.read_csv(file, sep=";")

    # Sección de filtros
    st.markdown('<div class="filters-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="filters-title">Filtros de Análisis</h3>', unsafe_allow_html=True)
    
    # Filtrado en cascada
    df_filtrado_region = df.copy()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region = st.multiselect("Región", df['Region'].unique(), key="region_select")
        if region:
            df_filtrado_region = df[df['Region'].isin(region)]

    with col2:
        filiales_disponibles = df_filtrado_region['Filial'].unique()
        filial = st.multiselect("Filial", filiales_disponibles, key="filial_select")
        if filial:
            df_filtrado_filial = df_filtrado_region[df_filtrado_region['Filial'].isin(filial)]
        else:
            df_filtrado_filial = df_filtrado_region

    with col3:
        territorios_disponibles = df_filtrado_filial['Territorio_comunicacion'].unique()
        territorio = st.multiselect("Territorio de comunicación", territorios_disponibles, key="territorio_select")
        if territorio:
            df_filtrado = df_filtrado_filial[df_filtrado_filial['Territorio_comunicacion'].isin(territorio)]
        else:
            df_filtrado = df_filtrado_filial

    st.markdown('</div>', unsafe_allow_html=True)

    # Botón principal
    if st.button("¿Quieres que genere el informe de percepciones y recomendaciones?"):
        resumen = df_filtrado.groupby("Territorio_comunicacion")[["Negativo","Neutral","Positivo"]].sum().reset_index()
        resumen_str = resumen.to_string(index=False)

        prompt_informe = f"""
{base_prompt}

Estos son datos agregados por territorio de comunicación:
{resumen_str}

Genera un resumen de las percepciones con insights y recomendaciones de narrativa digital que contenga una frase de narrativa emocional y acciones puntuales con su respectiva táctica, 
basadas en estos datos. No pongas explícito en el análisis el Territorio_comunicacion No asignado.
"""

        with st.spinner("Generando informe..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_informe}],
                temperature=0.4
            )
            informe = response.choices[0].message.content
            
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">📊 Informe Generado</h3>', unsafe_allow_html=True)
            st.write(informe)
            st.markdown('</div>', unsafe_allow_html=True)

            # Audio
            texto_para_voz = re.sub(r'[^\w\s.,¡!¿?áéíóúÁÉÍÓÚñÑ]', '', informe)
            texto_para_voz = re.sub(r'\n+', '. ', texto_para_voz)

            tts = gTTS(text=texto_para_voz, lang='es')
            audio_path = "informe_audio.mp3"
            tts.save(audio_path)

            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">🎧 Audio del Informe</h3>', unsafe_allow_html=True)
            st.audio(audio_path, format='audio/mp3')
            st.markdown('</div>', unsafe_allow_html=True)

    # Sección de preguntas
    st.markdown("""
    <div class="questions-section">
        <h2 class="questions-title">¿Quieres profundizar en algo más?</h2>
        <p class="questions-subtitle">Ejemplo: ¿Qué podemos hacer para mejorar la percepción de la sostenibilidad en el territorio?</p>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_area("Escribe tu pregunta aquí:", "", height=120, key="user_question")

    if user_input:
        prompt_pregunta = f"""
{base_prompt}

Estos son ejemplos individuales de menciones:
{df_filtrado[['Mencion','Negativo','Neutral','Positivo','Territorio_comunicacion']].head(10).to_string(index=False)}

Responde de forma clara y útil:
{user_input}
"""
        with st.spinner("Generando respuesta..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_pregunta}],
                temperature=0.3
            )
            answer = response.choices[0].message.content
            
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">🤖 Respuesta de la IA</h3>', unsafe_allow_html=True)
            st.write(answer)
            st.markdown('</div>', unsafe_allow_html=True)

    # Vista general de datos
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📋 Vista General de los Datos</h3>', unsafe_allow_html=True)
    st.dataframe(df_filtrado.sample(frac=1).reset_index(drop=True), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Distribución de sentimientos
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📈 Distribución de Sentimientos</h3>', unsafe_allow_html=True)
    sentiments_df = df_filtrado[['Negativo', 'Neutral', 'Positivo']].sum().reset_index()
    sentiments_df.columns = ['Sentimiento', 'Total']
    st.bar_chart(sentiments_df.set_index('Sentimiento'), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Nube de palabras
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">☁️ Nube de Palabras (Menciones) Depurada</h3>', unsafe_allow_html=True)
    
    raw_text = " ".join(df_filtrado['Mencion'].dropna().astype(str))
    raw_text = re.sub(r'[^\w\s]', '', raw_text.lower())

    stopwords_es = set(STOPWORDS)
    stopwords_es.update([
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
        "por", "un", "para", "con", "no", "una", "su", "al", "es", "lo",
        "como", "más", "pero", "sus", "ya", "o", "este", "sí", "porque",
        "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me",
        "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante",
        "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante",
        "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo",
        "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho",
        "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas"
    ])

    if raw_text.strip():
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            stopwords=stopwords_es,
            colormap='viridis'
        ).generate(raw_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-logo">
        <img src="https://raw.githubusercontent.com/SantiagoCardonaP/epm-dashboard-social-listening/2fdfac81f49d7c03afea8e29c0d67d96bcdcf750/logo-grupo-epm%20(1).png" style="height: 40px; opacity: 0.7;">
    </div>
</div>
""", unsafe_allow_html=True)
