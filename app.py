import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import sqlite3
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium
from datetime import datetime

# --- SETUP DIREKTORI ---
if not os.path.exists('dataset_learning'):
    os.makedirs('dataset_learning')

# --- SESSION STATE ---
if 'scan_result' not in st.session_state: st.session_state.scan_result = None
if 'auth' not in st.session_state: st.session_state.auth = False

# --- DATABASE ---
def get_db():
    conn = sqlite3.connect('agroscan_v11_expert.db', check_same_thread=False)
    return conn

conn = get_db()
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, date TEXT, location TEXT, lat REAL, lon REAL, score REAL, forecast TEXT, suggestion TEXT, issue TEXT)')
conn.commit()

# --- MODEL LOADER ---
@st.cache_resource
def load_all_engines():
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    yolo_model = YOLO('yolov8n.pt') 
    return processor, seg_model, yolo_model

processor, seg_model, yolo_model = load_all_engines()

# --- UI CONFIG ---
st.set_page_config(page_title="AgriDrone Insight", layout="wide")

# --- LOGIN LOGIC ---
if not st.session_state.auth:
    _, col_login, _ = st.columns([1, 1.2, 1])
    with col_login:
        # --- INTEGRASI LOGO DI HALAMAN LOGIN ---
        if os.path.exists("logo.png"):
            st.image("logo.png", width=200)
            
        st.title("AgriDrone Insight Enterprise")
        with st.container(border=True):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                if u != "": 
                    st.session_state.auth = True
                    st.session_state.user = u
                    st.rerun()
else:
    # --- INTEGRASI LOGO DI SIDEBAR ---
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", use_container_width=True)
        
    st.sidebar.title(f"👤 {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.auth = False
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["Diagnostic Citra", "Riwayat Peta", "AI Roadmap & NPK Reference"])

    with tab1:
        col_ctrl, col_map = st.columns([1.5, 1])
        
        with col_map:
            st.subheader("Informasi Lokasi")
            # Default lokasi ke koordinat yang Anda berikan
            m = folium.Map(location=[-7.132033, 110.405796], zoom_start=14)
            m.add_child(folium.LatLngPopup())
            map_data = st_folium(m, height=350, key="map_v11")
            lat = map_data['last_clicked']['lat'] if map_data and map_data['last_clicked'] else -7.132033
            lon = map_data['last_clicked']['lng'] if map_data and map_data['last_clicked'] else 110.405796
            lahan_label = st.text_input("Label Lahan", "Blok A")

        with col_ctrl:
            st.subheader("Diagnostic Citra")
            files = st.file_uploader("Upload Drone Imagery (Max 50)", type=['jpg','jpeg','png'], accept_multiple_files=True)
            
            if files and st.button("JALANKAN DIAGNOSTIC AI"):
                results = []
                bar = st.progress(0)
                
                for idx, f in enumerate(files):
                    img_pil = Image.open(f).convert("RGB")
                    img_np = np.array(img_pil)
                    
                    # 1. AI SEGMENTATION (Health Score)
                    inputs = processor(images=img_pil, return_tensors="pt")
                    with torch.no_grad():
                        out = seg_model(**inputs)
                    logits = nn.functional.interpolate(out.logits, size=img_pil.size[::-1], mode='bilinear')
                    mask = np.isin(logits.argmax(dim=1)[0].numpy(), [4, 12, 17, 66]).astype(np.uint8) * 255
                    health = (np.count_nonzero(mask) / mask.size) * 100
                    
                    # 2. AI EXPERT SYSTEM (Nutrisi & Hama)
                    b, g, r = cv2.split(img_np)
                    # Deteksi Nutrisi (Color Ratio Analysis)
                    yellow_factor = np.mean(r) / (np.mean(g) + 1e-5)
                    if yellow_factor > 0.95:
                        nutrisi_issue = "Defisiensi Nitrogen (Kuning)"
                        advice = "Saran: Aplikasi Urea/ZA segera."
                    else:
                        nutrisi_issue = "Nutrisi Optimal"
                        advice = "Saran: Pemeliharaan rutin."
                    
                    # Deteksi Hama (Texture Roughness Analysis)
                    edges = cv2.Canny(img_np, 100, 200)
                    roughness = np.count_nonzero(edges) / edges.size
                    hama_issue = "Peringatan Hama! Segera periksa tanaman budidaya!" if roughness > 0.06 else "Aman dari Hama"
                    
                    # 3. YOLO DETECTOR
                    yolo_res = yolo_model(img_pil)[0]
                    
                    # Save to DB
                    c.execute('INSERT INTO history(username, date, location, lat, lon, score, forecast, suggestion, issue) VALUES (?,?,?,?,?,?,?,?,?)',
                              (st.session_state.user, datetime.now().strftime("%Y-%m-%d"), lahan_label, lat, lon, health, f"{health*0.08:.1f} Ton", advice, f"{nutrisi_issue} | {hama_issue}"))
                    
                    if idx == 0:
                        st.session_state.scan_result = {
                            "mask": mask, "yolo": yolo_res.plot(), "score": health, 
                            "yield": f"{health*0.08:.1f} Ton/Ha", "advice": advice, 
                            "issue": f"{nutrisi
