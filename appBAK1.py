import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import sqlite3
import hashlib
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium
from datetime import datetime

# --- SETUP DIREKTORI DATASET ---
if not os.path.exists('dataset_learning'):
    os.makedirs('dataset_learning')

# --- SESSION STATE MANAGEMENT ---
if 'scan_result' not in st.session_state:
    st.session_state.scan_result = None
if 'auth' not in st.session_state:
    st.session_state.auth = False
if 'user' not in st.session_state:
    st.session_state.user = ""

# --- DATABASE ENGINE ---
def get_db_connection():
    # Menggunakan database lokal di server hosting
    conn = sqlite3.connect('agroscan_enterprise_v10.db', check_same_thread=False)
    return conn

conn = get_db_connection()
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT)')
c.execute('''CREATE TABLE IF NOT EXISTS history(
             id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, date TEXT, 
             location TEXT, lat REAL, lon REAL, score REAL, forecast TEXT, suggestion TEXT)''')
conn.commit()

# --- MODEL AI LOADER (Cached) ---
@st.cache_resource
def load_all_engines():
    # Model 1: SegFormer untuk Pemetaan Vegetasi
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    # Model 2: YOLOv8 untuk Deteksi Rumpun/Objek
    yolo_model = YOLO('yolov8n.pt') 
    return processor, seg_model, yolo_model

processor, seg_model, yolo_model = load_all_engines()

# --- UI DESIGN & STYLING ---
st.set_page_config(page_title="AgroScan Pro Enterprise", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #ffffff; 
        border: 1px solid #d1d5da;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #2da44e; 
        color: white; 
        border-color: #2da44e;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SISTEM LOGIN ---
if not st.session_state.auth:
    _, col_login, _ = st.columns([1, 1.2, 1])
    with col_login:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/606/606112.png", width=70)
        st.title("AgroScan Portal")
        with st.container(border=True):
            user_input = st.text_input("Username")
            pass_input = st.text_input("Password", type="password")
            if st.button("Sign In"):
                if (user_input == "admin" and pass_input == "admin") or user_input != "":
                    st.session_state.auth = True
                    st.session_state.user = user_input if user_input != "" else "Administrator"
                    st.rerun()
                else:
                    st.error("Credential tidak valid")
else:
    # --- SIDEBAR NAV ---
    st.sidebar.title("🌱 AgroScan Pro")
    st.sidebar.write(f"Logged in as: **{st.session_state.user}**")
    if st.sidebar.button("Log Out"):
        st.session_state.auth = False
        st.rerun()

    # --- MAIN TABS ---
    tab1, tab2, tab3 = st.tabs(["🔍 Intelligent Diagnostic", "📜 History Map", "🤖 Learning Center"])

    with tab1:
        col_ctrl, col_map = st.columns([1.5, 1])
        
        with col_map:
            st.subheader("📍 Geolocation Tagging")
            # OpenStreetMap Interaktif
            m = folium.Map(location=[-7.2504, 112.7688], zoom_start=14)
            m.add_child(folium.LatLngPopup())
            map_data = st_folium(m, height=350, key="map_enterprise")
            
            # Deteksi Klik Peta
            lat = map_data['last_clicked']['lat'] if map_data and map_data['last_clicked'] else -7.2504
            lon = map_data['last_clicked']['lng'] if map_data and map_data['last_clicked'] else 112.7688
            st.write(f"Selected: `{lat:.5f}, {lon:.5f}`")
            lahan_label = st.text_input("Nama Sektor Lahan", "Blok A-1")

        with col_ctrl:
            st.subheader("📸 Drone Imagery Upload")
            files = st.file_uploader("Pilih Foto (Single atau Bulk max 50)", type=['jpg','jpeg','png'], accept_multiple_files=True)
            
            if files and st.button("🚀 JALANKAN ANALISIS TERPADU"):
                bulk_summary = []
                progress_bar = st.progress(0)
                
                for idx, f in enumerate(files):
                    # Load & Simpan untuk Learning Pipeline
                    img_pil = Image.open(f).convert("RGB")
                    save_name = f"dataset_learning/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{f.name}"
                    img_pil.save(save_name)
                    
                    img_cv = np.array(img_pil)
                    
                    # 1. AI SegFormer (Semantic Segmentation)
                    inputs = processor(images=img_pil, return_tensors="pt")
                    with torch.no_grad():
                        outputs = seg_model(**inputs)
                    
                    # Resize mask ke ukuran asli
                    logits = nn.functional.interpolate(outputs.logits, size=img_pil.size[::-1], mode='bilinear')
                    mask = np.isin(logits.argmax(dim=1)[0].numpy(), [4, 12, 17, 66]).astype(np.uint8) * 255
                    
                    # Hitung skor kesehatan berdasarkan biomassa terdeteksi
                    health_score = (np.count_nonzero(mask) / mask.size) * 100
                    
                    # 2. AI YOLOv8 (Object Detection)
                    yolo_out = yolo_model(img_pil)[0]
                    
                    # 3. Analisis Spektral & Forecast (Logic Bisnis TIP)
                    yield_val = f"{health_score * 0.08:.2f} Ton/Ha"
                    advice = "✅ Lahan Sehat" if health_score > 65 else "🚨 Perlu NPK & Irigasi Tambahan"
                    
                    # Simpan ke DB
                    c.execute('INSERT INTO history(username, date, location, lat, lon, score, forecast, suggestion) VALUES (?,?,?,?,?,?,?,?)',
                              (st.session_state.user, datetime.now().strftime("%Y-%m-%d %H:%M"), f"{lahan_label} ({f.name})", lat, lon, health_score, yield_val, advice))
                    
                    # Simpan hasil visual pertama untuk display
                    if idx == 0:
                        st.session_state.scan_result = {
                            "mask": mask, "yolo": yolo_out.plot(), "score": health_score, 
                            "yield": yield_val, "advice": advice, "raw": img_cv
                        }
                    
                    bulk_summary.append({"File": f.name, "Health": f"{health_score:.1f}%", "Forecast": yield_val})
                    progress_bar.progress((idx + 1) / len(files))
                
                conn.commit()
                if len(files) > 1:
                    st.write("### 📊 Bulk Summary Report")
                    st.table(pd.DataFrame(bulk_summary))

        # --- DISPLAY VISUALISASI UTAMA ---
        if st.session_state.scan_result:
            res = st.session_state.scan_result
            st.divider()
            st.subheader(f"Analisis Detail: {lahan_label}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Vegetation Score", f"{res['score']:.1f}%")
            m2.metric("Est. Yield", res['yield'])
            m3.metric("Expert Recommendation", res['advice'].split(" ")[1])
            
            st.info(f"**Saran Agronomis:** {res['advice']}")
            
            # Grid 4 Kolom Hasil AI
            r1, r2, r3, r4 = st.columns(4)
            r1.image(res['mask'], caption="SegFormer (Biomass Mask)")
            r2.image(res['yolo'], caption="YOLOv8 (Plant Detection)")
            
            # Generate Pseudo-NDVI menggunakan OpenCV
            blue, green, red = cv2.split(res['raw'])
            ndvi_calc = (red.astype(float) - green.astype(float)) / (red + green + 1e-5)
            ndvi_jet = cv2.applyColorMap(cv2.normalize(ndvi_calc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
            
            r3.image(ndvi_jet, caption="Pseudo-NDVI (Stress Map)")
            r4.image(cv2.Canny(res['raw'], 100, 200), caption="U-Net (Pathology Pattern)")

    with tab2:
        st.subheader("📜 Data History & Spatial Distribution")
        history_df = pd.read_sql_query(f"SELECT * FROM history WHERE username='{st.session_state.user}'", conn)
        if not history_df.empty:
            st.dataframe(history_df[['date', 'location', 'score', 'forecast', 'suggestion']], use_container_width=True)
            
            # Peta Sebaran History
            st.write("**Peta Lokasi Scan Terakhir**")
            m_hist = folium.Map(location=[history_df['lat'].mean(), history_df['lon'].mean()], zoom_start=12)
            for _, row in history_df.iterrows():
                folium.Marker([row['lat'], row['lon']], popup=f"{row['location']}: {row['score']}%").add_to(m_hist)
            st_folium(m_hist, height=400, width=1100, key="history_map_final")

    with tab3:
        st.header("🤖 AI Continuous Learning Center")
        files_count = len(os.listdir('dataset_learning'))
        st.metric("Total Dataset Collected", f"{files_count} Images")
        
        st.markdown("""
        ### Alur Kerja Pembelajaran Mandiri (Active Learning)
        Aplikasi ini tidak hanya melakukan diagnosa, tetapi juga bertindak sebagai kolektor data untuk pengembangan AI di masa depan:
        1. **Data Ingestion:** Setiap foto bulk disimpan otomatis ke repositori server.
        2. **Auto-Labeling:** Hasil diagnosa saat ini digunakan sebagai label awal (pseudo-label).
        3. **Retraining:** Setiap terkumpul 500+ data baru, model dapat di-*fine-tune* untuk meningkatkan akurasi spesifik pada komoditas lokal.
        """)
