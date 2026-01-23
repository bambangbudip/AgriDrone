import os
import subprocess
import sys

# --- SELF-HEALING SYSTEM (Memaksa instalasi OpenCV jika error di hosting) ---
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

import streamlit as st
import pandas as pd
import numpy as np
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

# --- SETUP DIREKTORI (Continuous Learning) ---
if not os.path.exists('dataset_learning'):
    os.makedirs('dataset_learning')

# --- INITIALIZE SESSION STATE ---
if 'scan_result' not in st.session_state: st.session_state.scan_result = None
if 'auth' not in st.session_state: st.session_state.auth = False

# --- DATABASE ENGINE ---
def get_connection():
    return sqlite3.connect('agroscan_v9_hosting.db', check_same_thread=False)

conn = get_connection()
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT)')
c.execute('''CREATE TABLE IF NOT EXISTS history(
             id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, date TEXT, 
             location TEXT, lat REAL, lon REAL, score REAL, forecast TEXT, suggestion TEXT)''')
conn.commit()

# --- LOAD ENGINES ---
@st.cache_resource
def load_all_engines():
    # SegFormer (Segmentation)
    proc = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    # YOLOv8 (Detection)
    yolo = YOLO('yolov8n.pt')
    return proc, seg, yolo

processor, seg_model, yolo_model = load_all_engines()

# --- UI STYLE (GitHub/Google Look) ---
st.set_page_config(page_title="AgroScan Pro Enterprise", layout="wide")
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #f1f3f4; border-radius: 4px; padding: 10px; }
    .stTabs [aria-selected="true"] { background-color: #2da44e; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIN LOGIC ---
if not st.session_state.auth:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.title("🌱 AgroScan Sign In")
        with st.container(border=True):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Sign In"):
                if (u == "admin" and p == "admin") or u != "":
                    st.session_state.auth = True
                    st.session_state.user = u if u != "" else "Admin"
                    st.rerun()
else:
    # --- DASHBOARD UTAMA ---
    st.sidebar.title(f"👤 {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.auth = False
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["🔍 Diagnostic (Unified/Bulk)", "📜 History Map", "🤖 AI Learning Pipeline"])

    with tab1:
        col_in, col_map = st.columns([1.5, 1])
        
        with col_map:
            st.subheader("📍 Lahan Location")
            m = folium.Map(location=[-7.2504, 112.7688], zoom_start=14)
            m.add_
