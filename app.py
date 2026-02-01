import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. CONFIGURATION ÉLÉGANTE
st.set_page_config(page_title="CottonAI | Global Diagnostic Platform", layout="wide")

# 2. DESIGN CSS "ULTRA-PREMIUM" AVEC CORRECTIF DE LISIBILITÉ
st.markdown("""
    <style>
    /* Fond global */
    .stApp { background: #0F0F0F !important; }
    
    /* LE TITRE DU MODULE */
    .stFileUploader label {
        color: #FFB300 !important;
        font-weight: 800 !important;
        font-size: 1.6rem !important;
    }

    /* LA ZONE DE DROP (Contours et fond) */
    [data-testid="stFileUploadDropzone"] {
        background-color: #1A1A1A !important;
        border: 2px dashed #FFB300 !important;
        border-radius: 15px !important;
        color: #FFFFFF !important;
    }

    /* LE NOM DU FICHIER CHARGÉ (Forcé en blanc pur) */
    [data-testid="stUploadedFileName"] {
        color: #FFFFFF !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }

    /* LES TEXTES D'INSTRUCTIONS (Small texts) */
    [data-testid="stMarkdownContainer"] p, .st-ae code {
        color: #EEEEEE !important;
        font-weight: 500 !important;
    }

    /* LE BOUTON BROWSE (Le transformer en bouton doré ultra-lisible) */
    button[kind="secondary"] {
        background-color: #FFB300 !important;
        color: #000000 !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        box-shadow: 0px 4px 15px rgba(255, 179, 0, 0.3) !important;
    }

    /* Suppression de l'icône de trombone grise si elle gêne */
    [data-testid="stFileUploadDropzone"] svg {
        fill: #FFB300 !important;
    }
    
    /* Style général des cartes */
    .card {
        background: #1E1E1E;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. BASE DE DONNÉES INSTRUCTIVE
CONSEILS = {
    "Alternaria Leaf Spot": {"action": "Réduire l'humidité foliaire.", "bio": "Utiliser des extraits de Neem.", "note": "Apparaît souvent après des pluies prolongées."},
    "Bacterial Blight": {"action": "Éliminer les débris infectés.", "bio": "Rotation des cultures sur 2 ans.", "note": "Se propage par le vent et l'eau."},
    "Fusarium Wilt": {"action": "Améliorer le drainage du sol.", "bio": "Apport de potasse.", "note": "Attaque le système vasculaire de la plante."},
    "Healthy Leaf": {"action": "Maintenir la surveillance.", "bio": "Engrais organique équilibré.", "note": "Vigueur optimale détectée."},
    "Verticillium Wilt": {"action": "Éviter l'excès d'azote.", "bio": "Solarisation du sol.", "note": "Favorisé par des sols frais et humides."}
}

# 4. LOGIQUE DU MODÈLE
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_lenet.h5')

model = load_my_model()
classes = list(CONSEILS.keys())

if 'history' not in st.session_state:
    st.session_state.history = []

# 5. HEADER PRO
st.markdown("<h1 style='text-align: center; color: #FFB300;'>COTTON AI <span style='color:white; font-weight:100;'>PREMIUM</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Intelligence Artificielle de Précision pour l'Agriculture Durable</p>", unsafe_allow_html=True)

# 6. ZONE DE TÉLÉCHARGEMENT
uploaded_file = st.file_uploader("📂 CHARGER UNE ANALYSE HAUTE RÉSOLUTION", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    img_array = np.array(image.resize((128, 128)))
    img_batch = np.expand_dims(img_array, axis=0).astype('float32')
    prediction = model.predict(img_batch)[0]
    idx = np.argmax(prediction)
    conf = prediction[idx] * 100
    res_nom = classes[idx]

    if st.button("💾 Enregistrer ce diagnostic pour comparaison"):
        st.session_state.history.append({"Date": datetime.now().strftime("%H:%M:%S"), "Classe": res_nom, "Confiance": f"{conf:.1f}%"})

    st.divider()

    # 7. LAYOUT PRINCIPAL
    col_img, col_info = st.columns([1, 1.5], gap="large")

    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f"<h3 style='text-align:center; color:white;'>{res_nom}</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.subheader("💡 Guide d'Intervention")
        c1, c2 = st.columns(2)
        with c1:
            st.warning(f"**Action Immédiate** : \n{CONSEILS[res_nom]['action']}")
        with c2:
            st.success(f"**Solution Bio** : \n{CONSEILS[res_nom]['bio']}")
        
        st.info(f"**Note de l'expert** : {CONSEILS[res_nom]['note']}")
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=prediction*100, theta=classes, fill='toself', line_color='#FFB300'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#444")), showlegend=False, 
                                paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=300, margin=dict(t=30, b=30))
        st.plotly_chart(fig_radar, use_container_width=True)

    # 8. TABLEAU DE COMPARAISON ANALYTIQUE
    st.write("### 📊 Comparaison Multicritères")
    df_comp = pd.DataFrame({'Pathologie': classes, 'Probabilité (%)': prediction * 100}).sort_values('Probabilité (%)', ascending=False)
    st.dataframe(df_comp, use_container_width=True)

# 9. SECTION HISTORIQUE
if st.session_state.history:
    st.write("---")
    st.subheader("📁 Historique des comparaisons de la session")

    st.table(st.session_state.history)
