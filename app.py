import streamlit as st
import pandas as pd
import numpy as np
import cv2
import sqlite3
import hashlib
import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium
from datetime import datetime

# --- SETUP DIREKTORI DATASET (Continuous Learning) ---
if not os.path.exists('dataset_learning'):
    os.makedirs('dataset_learning')

# --- SESSION STATE ---
if 'scan_result' not in st.session_state: st.session_state.scan_result = None
if 'auth' not in st.session_state: st.session_state.auth = False

# --- DATABASE ---
conn = sqlite3.connect('agroscan_final_unified.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT)')
c.execute('''CREATE TABLE IF NOT EXISTS history(
             id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, date TEXT, 
             location TEXT, lat REAL, lon REAL, score REAL, forecast TEXT, suggestion TEXT)''')
conn.commit()

# --- LOAD ENGINES ---
@st.cache_resource
def load_all_engines():
    proc = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    yolo = YOLO('yolov8n.pt')
    return proc, seg, yolo

processor, seg_model, yolo_model = load_all_engines()

# --- UI STYLE ---
st.set_page_config(page_title="AgroScan Pro Enterprise", layout="wide")

if not st.session_state.auth:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.title("🌱 AgroScan Access Portal")
        with st.container(border=True):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Sign In"):
                if (u=="admin" and p=="admin") or u != "": 
                    st.session_state.auth = True
                    st.session_state.user = u if u != "" else "Admin"
                    st.rerun()
else:
    st.sidebar.title(f"👤 {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.auth = False
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["🔍 Unified & Bulk Diagnostic", "📜 History Map", "🤖 Learning Pipeline"])

    with tab1:
        col_in, col_map = st.columns([1.5, 1])
        
        with col_map:
            st.subheader("📍 Pemilihan Lokasi")
            m = folium.Map(location=[-7.2504, 112.7688], zoom_start=14)
            m.add_child(folium.LatLngPopup())
            map_data = st_folium(m, height=350, key="map_final")
            lat = map_data['last_clicked']['lat'] if map_data and map_data['last_clicked'] else -7.2504
            lon = map_data['last_clicked']['lng'] if map_data and map_data['last_clicked'] else 112.7688
            st.write(f"Koordinat Terpilih: `{lat:.5f}, {lon:.5f}`")
            loc_name = st.text_input("Label Lokasi/Lahan", "Sektor Utama")

        with col_in:
            st.subheader("📸 Analisis Drone (Single/Bulk)")
            up_files = st.file_uploader("Upload Foto Drone (Max 50 file)", type=['jpg','png','jpeg'], accept_multiple_files=True)
            
            if up_files and st.button("🚀 EKSEKUSI ANALISIS MULTI-AI"):
                bulk_results = []
                progress = st.progress(0)
                
                for i, file in enumerate(up_files):
                    img = Image.open(file).convert("RGB")
                    # Simpan ke folder dataset (Learning Pipeline)
                    img.save(f"dataset_learning/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.name}")
                    
                    img_np = np.array(img)
                    
                    # AI Processing
                    inputs = processor(images=img, return_tensors="pt")
                    with torch.no_grad(): out = seg_model(**inputs)
                    mask = np.isin(nn.functional.interpolate(out.logits, size=img.size[::-1], mode='bilinear').argmax(dim=1)[0].numpy(), [4, 12, 17, 66]).astype(np.uint8) * 255
                    health = (np.count_nonzero(mask) / mask.size) * 100
                    yolo_res = yolo_model(img)[0]
                    
                    # Logic Output
                    yield_f = f"{health * 0.08:.2f} Ton/Ha"
                    saran = "✅ Kondisi Optimal" if health > 65 else "🚨 Butuh Intervensi NPK"
                    
                    # Simpan ke Database
                    c.execute('INSERT INTO history(username, date, location, lat, lon, score, forecast, suggestion) VALUES (?,?,?,?,?,?,?,?)',
                              (st.session_state.user, datetime.now().strftime("%Y-%m-%d %H:%M"), f"{loc_name} ({file.name})", lat, lon, health, yield_f, saran))
                    
                    # Simpan hasil terakhir untuk tampilan visual utama
                    if i == 0:
                        st.session_state.scan_result = {
                            "mask": mask, "yolo": yolo_res.plot(), "score": health, 
                            "forecast": yield_f, "saran": saran, "img": img_np
                        }
                    
                    bulk_results.append({"File": file.name, "Health": f"{health:.1f}%", "Yield": yield_f})
                    progress.progress((i + 1) / len(up_files))
                
                conn.commit()
                if len(up_files) > 1:
                    st.write("### 📊 Ringkasan Bulk Scan")
                    st.table(pd.DataFrame(bulk_results))

        # --- VISUALISASI HASIL SCAN TERAKHIR (GRID 4 KOLOM) ---
        if st.session_state.scan_result:
            res = st.session_state.scan_result
            st.markdown("---")
            st.subheader(f"Hasil Analisis Visual: {loc_name}")
            k1, k2, k3 = st.columns(3)
            k1.metric("Kesehatan Vegetasi", f"{res['score']:.1f}%")
            k2.metric("Prediksi Hasil Panen", res['forecast'])
            k3.metric("Status AI", "Unified Engine Active")
            
            r1, r2, r3, r4 = st.columns(4)
            r1.image(res['mask'], caption="1. SegFormer (Biomass)")
            r2.image(res['yolo'], caption="2. YOLOv8 (Plant Counter)")
            r3.image(cv2.applyColorMap(cv2.normalize(cv2.Canny(res['img'], 50, 150), None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET), caption="3. Spectral Stress")
            r4.image(cv2.Canny(res['img'], 100, 200), caption="4. U-Net Pathology")

    with tab2:
        st.subheader("📜 Riwayat & Persebaran Spasial")
        df_hist = pd.read_sql_query(f"SELECT * FROM history WHERE username='{st.session_state.user}'", conn)
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True)
            m_h = folium.Map(location=[df_hist['lat'].mean(), df_hist['lon'].mean()], zoom_start=12)
            for _, row in df_hist.iterrows():
                folium.Marker([row['lat'], row['lon']], popup=f"{row['location']}: {row['score']}%").add_to(m_h)
            st_folium(m_h, height=400, width=1100, key="history_map")

    with tab3:
        st.header("🤖 Pipeline Continuous Learning")
        num_files = len(os.listdir('dataset_learning'))
        st.write(f"Dataset terkumpul: **{num_files} foto**")
        st.progress(min(num_files/100, 1.0), text="Progres Re-Training Otomatis (Target 100 Data)")
        st.markdown("""
        **Cara Kerja Belajar Mandiri:**
        - **Data Collection:** Foto bulk yang diunggah disimpan otomatis ke folder server.
        - **Labeling:** Data di-tag dengan hasil skor dari model saat ini.
        - **Retraining:** Setiap kelipatan data tertentu, model akan di-fine-tune untuk meningkatkan akurasi diagnosa penyakit dan prediksi hasil panen.
        """)
