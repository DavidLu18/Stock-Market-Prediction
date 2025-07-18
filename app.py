import os
import sys
import platform
import pandas as pd
import numpy as np
import streamlit as st # Moved import up
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import joblib # For loading model and scaler
from sklearn.preprocessing import StandardScaler # IMPORTED
import argparse
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time
import traceback
from typing import Optional, List, Dict, Tuple, Any, Callable

# Import XGBoost
import xgboost as xgb

# --- Page Config (MUST be first st command) ---
st.set_page_config(
    page_title="StockAI - Dự đoán Độ Biến Động Cổ phiếu", 
    page_icon="🌊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize a list to store startup error messages ---
_startup_error_messages = []


# Import TA-Lib (status checked)
try:
    import talib
    TALIB_AVAILABLE = True
    print("(app.py) INFO: TA-Lib library found.")
except ImportError:
    TALIB_AVAILABLE = False
    print("(app.py) WARNING: TA-Lib library not found. Technical indicator calculations will be limited or unavailable.")

# Import custom modules
try:
    from data_collection import DataCollector
    DATA_COLLECTION_AVAILABLE = True
except ImportError as e:
    _startup_error_messages.append(f"Failed to import DataCollector: {e}. Data collection functionality disabled.")
    DATA_COLLECTION_AVAILABLE = False
    DataCollector = None # type: ignore

try:
    from model_training import train_stock_prediction_model, StockVolatilityPredictor 
    MODEL_TRAINING_AVAILABLE = True
    print("(app.py) INFO: XGBoost model training module (for Volatility) loaded successfully.")
except ImportError as e:
    _startup_error_messages.append(f"Failed to import from model_training: {e}. Training functionality disabled.")
    MODEL_TRAINING_AVAILABLE = False
    train_stock_prediction_model = None # type: ignore
    StockVolatilityPredictor = None # type: ignore  

# --- Directory Setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = APP_DIR
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        except OSError as e:
            _startup_error_messages.append(f"Error creating directory {dir_path}: {e}")

# --- Display startup errors, if any ---
if _startup_error_messages:
    for error_msg in _startup_error_messages:
        st.error(error_msg)

# --- Enhanced Theme Colors & Modern CSS (Updated with high contrast) ---
PRIMARY_COLOR = "#09122C"        # Deep navy blue
SECONDARY_COLOR = "#872341"      # Deep burgundy
ACCENT_COLOR = "#BE3144"         # Rich red
GRADIENT_END = "#E17564"         # Coral red
SUCCESS_COLOR = "#10B981"        # Emerald
ERROR_COLOR = "#EF4444"          # Red
WARNING_COLOR = "#F59E0B"        # Amber
BG_COLOR = "#0A0E1A"            # Very dark navy
CARD_BG_COLOR = "#1A1F2E"       # Lighter dark navy card
BORDER_COLOR = "#2D3748"        # Higher contrast border
TEXT_COLOR = "#FFFFFF"          # Pure white text
TEXT_MUTED_COLOR = "#CBD5E0"    # Light gray text
TEXT_ACCENT_COLOR = "#FFA07A"   # Light salmon accent text

def load_modern_css():
    """Tải CSS hiện đại và sang trọng với contrast cao"""
    st.markdown(f"""
    <style>
        /* Import Google Fonts - Elegant and Professional */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Playfair+Display:wght@400;500;600;700&display=swap');

        /* Root CSS Variables */
        :root {{
            --primary-color: {PRIMARY_COLOR};
            --secondary-color: {SECONDARY_COLOR};
            --accent-color: {ACCENT_COLOR};
            --gradient-end: {GRADIENT_END};
            --success-color: {SUCCESS_COLOR};
            --error-color: {ERROR_COLOR};
            --bg-color: {BG_COLOR};
            --card-bg: {CARD_BG_COLOR};
            --border-color: {BORDER_COLOR};
            --text-color: {TEXT_COLOR};
            --text-muted: {TEXT_MUTED_COLOR};
            --text-accent: {TEXT_ACCENT_COLOR};
            --shadow-sm: 0 2px 4px rgba(9, 18, 44, 0.2);
            --shadow-md: 0 4px 12px rgba(9, 18, 44, 0.3);
            --shadow-lg: 0 8px 25px rgba(9, 18, 44, 0.4);
            --shadow-xl: 0 12px 40px rgba(9, 18, 44, 0.5);
            --gradient-primary: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, var(--accent-color) 100%);
            --gradient-accent: linear-gradient(135deg, var(--accent-color) 0%, var(--gradient-end) 100%);
            --gradient-card: linear-gradient(145deg, var(--card-bg) 0%, rgba(26, 31, 46, 0.95) 100%);
            --text-shadow-strong: 0 2px 8px rgba(0, 0, 0, 0.8);
            --text-shadow-medium: 0 1px 4px rgba(0, 0, 0, 0.6);
            --text-shadow-light: 0 1px 2px rgba(0, 0, 0, 0.4);
        }}

        /* Main App Background with Sophisticated Gradient */
        .main {{
            background: linear-gradient(135deg, var(--bg-color) 0%, #0C1220 25%, #1A1F2E 50%, #0A0E1A 100%);
            background-attachment: fixed;
            color: var(--text-color);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
        }}

        .main::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 20% 80%, rgba(190, 49, 68, 0.08) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(135, 35, 65, 0.06) 0%, transparent 50%),
                        radial-gradient(circle at 40% 40%, rgba(225, 117, 100, 0.04) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }}

        /* High Contrast Typography */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-color);
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            letter-spacing: -0.025em;
            line-height: 1.25;
            margin-bottom: 1rem;
            text-shadow: var(--text-shadow-light);
        }}

        h1 {{
            font-size: 2.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, #FFFFFF 0%, var(--text-accent) 50%, #FFFFFF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            border-bottom: 3px solid;
            border-image: var(--gradient-accent) 1;
            padding-bottom: 1.5rem;
            margin-bottom: 2.5rem;
            position: relative;
            text-shadow: none;
        }}

        h1::after {{
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 60px;
            height: 3px;
            background: var(--gradient-accent);
            border-radius: 2px;
            box-shadow: 0 0 10px var(--accent-color);
        }}

        h2 {{
            font-size: 2rem;
            font-weight: 600;
            color: var(--text-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0.75rem;
            margin-bottom: 1.5rem;
            position: relative;
            text-shadow: var(--text-shadow-medium);
        }}

        h2::before {{
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 40px;
            height: 2px;
            background: var(--gradient-accent);
            border-radius: 1px;
            box-shadow: 0 0 8px var(--accent-color);
        }}

        h3 {{
            font-size: 1.5rem;
            color: var(--text-color);
            margin-bottom: 1rem;
            font-weight: 600;
            text-shadow: var(--text-shadow-light);
        }}

        /* High Contrast Text */
        p, div, span, label {{
            color: var(--text-color);
            text-shadow: var(--text-shadow-light);
        }}

        /* Premium Button Styling with High Contrast */
        .stButton>button {{
            background: var(--gradient-primary);
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.875rem 2.5rem;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.025em;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-md);
            position: relative;
            overflow: hidden;
            text-transform: none;
            text-shadow: var(--text-shadow-medium);
            border: 2px solid transparent;
        }}

        .stButton>button::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.5s;
        }}

        .stButton>button:hover {{
            transform: translateY(-3px);
            box-shadow: var(--shadow-xl);
            background: var(--gradient-accent);
            border-color: var(--text-accent);
        }}

        .stButton>button:hover::before {{
            left: 100%;
        }}

        .stButton>button:active {{
            transform: translateY(-1px);
            transition: transform 0.1s;
        }}

        /* Enhanced Input Fields with High Contrast */
        .stSelectbox>div>div,
        .stDateInput>div>div,
        .stTextInput>div>div,
        .stNumberInput>div>div,
        .stMultiSelect>div>div {{
            background: var(--card-bg) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-color) !important;
            font-family: 'Inter', sans-serif !important;
            padding: 0.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: var(--shadow-sm) !important;
            backdrop-filter: blur(10px) !important;
        }}

        .stSelectbox>div>div:focus-within,
        .stDateInput>div>div:focus-within,
        .stTextInput>div>div:focus-within,
        .stNumberInput>div>div:focus-within,
        .stMultiSelect>div>div:focus-within {{
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 3px rgba(190, 49, 68, 0.3), var(--shadow-md) !important;
            transform: translateY(-1px) !important;
            background: rgba(26, 31, 46, 0.98) !important;
        }}

        /* Premium Tab Design with High Contrast */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 12px;
            background: var(--card-bg);
            padding: 12px;
            border-radius: 16px;
            border: 2px solid var(--border-color);
            box-shadow: var(--shadow-md);
            backdrop-filter: blur(15px);
        }}

        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            color: var(--text-muted) !important;
            border-radius: 10px;
            padding: 14px 28px;
            border: none;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            text-shadow: var(--text-shadow-light);
        }}

        .stTabs [data-baseweb="tab"]::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-accent);
            opacity: 0;
            transition: opacity 0.3s ease;
            border-radius: 10px;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            color: var(--text-color) !important;
            transform: translateY(-2px);
            text-shadow: var(--text-shadow-medium);
        }}

        .stTabs [data-baseweb="tab"]:hover::before {{
            opacity: 0.2;
        }}

        .stTabs [aria-selected="true"] {{
            background: var(--gradient-accent) !important;
            color: white !important;
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            text-shadow: var(--text-shadow-strong);
        }}

        .stTabs [aria-selected="true"]::before {{
            opacity: 0;
        }}

        /* Sophisticated Sidebar with High Contrast */
        .sidebar .sidebar-content {{
            background: var(--card-bg) !important;
            border-right: 2px solid var(--border-color);
            box-shadow: var(--shadow-lg);
            backdrop-filter: blur(15px);
        }}

        /* Enhanced Progress Bar */
        .stProgress > div > div > div > div {{
            background: var(--gradient-accent);
            border-radius: 6px;
        }}

        .stProgress > div > div > div {{
            background-color: var(--border-color);
            border-radius: 6px;
        }}

        /* Premium Card Styling with High Contrast */
        .metric-card {{
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--shadow-md);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }}

        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: var(--gradient-accent);
            box-shadow: 0 0 10px var(--accent-color);
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: var(--shadow-xl);
            border-color: var(--accent-color);
            background: rgba(26, 31, 46, 0.98);
        }}

        /* Enhanced Expandable Sections */
        .stExpander {{
            border: 2px solid var(--border-color);
            border-radius: 12px;
            background: var(--card-bg);
            box-shadow: var(--shadow-md);
            margin-bottom: 1rem;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }}

        .stExpander header {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-color) !important;
            padding: 1rem 1.5rem;
            background: linear-gradient(135deg, var(--border-color) 0%, rgba(45, 55, 72, 0.8) 100%);
            text-shadow: var(--text-shadow-medium);
        }}

        .stExpander:hover {{
            border-color: var(--accent-color);
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }}

        /* Premium Metrics with High Contrast */
        .stMetricLabel {{
            color: var(--text-muted) !important;
            font-size: 0.875rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-shadow: var(--text-shadow-light);
        }}

        .stMetricValue {{
            color: var(--text-color) !important;
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            text-shadow: var(--text-shadow-medium);
        }}

        /* Enhanced Log Display */
        .details-log {{
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            height: 300px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-muted);
            box-shadow: var(--shadow-md);
            backdrop-filter: blur(10px);
        }}

        .details-log::-webkit-scrollbar {{
            width: 8px;
        }}

        .details-log::-webkit-scrollbar-track {{
            background: var(--border-color);
            border-radius: 4px;
        }}

        .details-log::-webkit-scrollbar-thumb {{
            background: var(--gradient-accent);
            border-radius: 4px;
        }}

        /* Enhanced Dataframe Styling */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
        }}

        /* Custom Scrollbar for Main Content */
        .main::-webkit-scrollbar {{
            width: 12px;
        }}

        .main::-webkit-scrollbar-track {{
            background: var(--bg-color);
        }}

        .main::-webkit-scrollbar-thumb {{
            background: var(--gradient-accent);
            border-radius: 6px;
            border: 2px solid var(--bg-color);
        }}

        /* Premium Alert Styling with High Contrast */
        .stAlert {{
            border-radius: 12px;
            border: none;
            box-shadow: var(--shadow-md);
            backdrop-filter: blur(10px);
            color: var(--text-color) !important;
        }}

        .stSuccess {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%);
            border-left: 4px solid var(--success-color);
            border: 2px solid rgba(16, 185, 129, 0.3);
        }}

        .stError {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
            border-left: 4px solid var(--error-color);
            border: 2px solid rgba(239, 68, 68, 0.3);
        }}

        .stWarning {{
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
            border-left: 4px solid var(--warning-color);
            border: 2px solid rgba(245, 158, 11, 0.3);
        }}

        .stInfo {{
            background: linear-gradient(135deg, rgba(255, 160, 122, 0.2) 0%, rgba(255, 160, 122, 0.1) 100%);
            border-left: 4px solid var(--text-accent);
            border: 2px solid rgba(255, 160, 122, 0.3);
        }}

        /* Enhanced Slider Styling */
        .stSlider > div > div > div > div {{
            background: var(--gradient-accent);
        }}

        /* Premium Loading Animation */
        .stSpinner > div {{
            border-top-color: var(--accent-color);
        }}

        /* Enhanced Checkbox and Radio */
        .stCheckbox > label, .stRadio > label {{
            color: var(--text-color) !important;
            font-weight: 500;
            text-shadow: var(--text-shadow-light);
        }}

        /* Enhanced File Uploader */
        .stFileUploader {{
            background: var(--card-bg);
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }}

        .stFileUploader:hover {{
            border-color: var(--accent-color);
            background: linear-gradient(135deg, var(--card-bg) 0%, rgba(190, 49, 68, 0.1) 100%);
        }}

        /* Animation Classes */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.7;
            }}
        }}

        .fade-in-up {{
            animation: fadeInUp 0.6s ease-out;
        }}

        .pulse {{
            animation: pulse 2s infinite;
        }}

        /* Enhanced Container Spacing */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* Premium Typography for Vietnamese Content */
        .vietnamese-text {{
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            text-shadow: var(--text-shadow-light);
        }}

        .technical-term {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            color: var(--text-accent);
            background: rgba(255, 160, 122, 0.15);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            border: 1px solid rgba(255, 160, 122, 0.3);
            text-shadow: var(--text-shadow-light);
        }}

        /* High Contrast Text Colors for All Elements */
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {{
            color: var(--text-color) !important;
        }}

        /* Ensure strong contrast for captions and help text */
        .stCaption, .stHelp {{
            color: var(--text-muted) !important;
            text-shadow: var(--text-shadow-light);
        }}

        /* High contrast for multiselect */
        .stMultiSelect [data-baseweb="tag"] {{
            background-color: var(--accent-color) !important;
            color: white !important;
        }}

        /* Better visibility for sidebar text */
        .sidebar .sidebar-content * {{
            color: var(--text-color) !important;
        }}
    </style>
    """, unsafe_allow_html=True)


class StockPredictionApp: 
    def __init__(self):
        load_modern_css()
        self.available_companies = self.load_available_companies()
        self.xgb_model: Optional[xgb.XGBClassifier] = None 
        self.xgb_scaler: Optional[StandardScaler] = None
        self.xgb_model_info: dict = {}
        self.xgb_feature_columns: List[str] = []
        self.xgb_target_col: Optional[str] = None 
        self.xgb_forecast_horizon: Optional[int] = None
        self.xgb_target_threshold: float = 0.02 
        self.xgb_model_loaded: bool = False

    # ADDED WRAPPER CALLBACK
    def _streamlit_status_callback_for_prediction(self, message: str, is_error: bool = False):
        """Hàm bao để st.info/st.error có thể dùng làm status_callback cho model_training."""
        if is_error:
            st.error(message) 
        else:
            # Using st.caption for less obtrusive status updates during prediction prep
            st.caption(message)  

    def load_available_companies(self):
        top_10_companies_list = [
            {'ticker': 'AAPL', 'name': 'Apple Inc.'}, {'ticker': 'MSFT', 'name': 'Microsoft Corp.'},
            {'ticker': 'GOOGL', 'name': 'Alphabet Inc. (A)'}, {'ticker': 'GOOG', 'name': 'Alphabet Inc. (C)'},
            {'ticker': 'AMZN', 'name': 'Amazon.com, Inc.'}, {'ticker': 'NVDA', 'name': 'NVIDIA Corp.'},
            {'ticker': 'META', 'name': 'Meta Platforms, Inc.'}, {'ticker': 'TSLA', 'name': 'Tesla, Inc.'},
            {'ticker': 'LLY', 'name': 'Eli Lilly and Company'}, {'ticker': 'V', 'name': 'Visa Inc.'}
        ]
        latest_file = None
        if os.path.exists(DATA_DIR):
            try:
                all_top = [f for f in os.listdir(DATA_DIR) if f.startswith('top_') and f.endswith('_companies.csv')]
                if all_top:
                    nums = []
                    for f_name in all_top:
                        try:
                            parts = f_name.split('_')
                            if len(parts) > 1 and parts[1].isdigit():
                                nums.append(int(parts[1]))
                        except (IndexError, ValueError):
                            continue
                    if nums:
                        latest_file = os.path.join(DATA_DIR, f'top_{max(nums)}_companies.csv')
            except Exception as e: print(f"Warn: Error finding general companies file: {e}")
        df_all_companies = None
        if latest_file and os.path.exists(latest_file):
            try:
                df_temp = pd.read_csv(latest_file)
                if 'Symbol' in df_temp.columns and 'Name' in df_temp.columns:
                    df_all_companies = df_temp.rename(columns={'Symbol': 'ticker', 'Name': 'name'})[['ticker', 'name']]
                elif 'ticker' in df_temp.columns and 'name' in df_temp.columns:
                     df_all_companies = df_temp[['ticker', 'name']]
            except Exception as e: print(f"Error reading general companies file {latest_file}: {e}")
        df_top_10 = pd.DataFrame(top_10_companies_list)
        if df_all_companies is not None:
            combined_df = pd.concat([df_top_10, df_all_companies]).drop_duplicates(subset=['ticker'], keep='first')
            return combined_df
        return df_top_10

    def load_xgb_model(self):
        latest_info_file = None
        try:
            if os.path.exists(MODEL_DIR):
                candidate_info_files = [f for f in os.listdir(MODEL_DIR) if 'model_info_' in f and f.endswith('.json')]
                if candidate_info_files:
                    candidate_info_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)), reverse=True)
                    latest_info_file = candidate_info_files[0]
            else:
                st.warning(f"Model directory not found: {MODEL_DIR}"); self.xgb_model_loaded = False; return False
            if not latest_info_file:
                st.warning("Không tìm thấy file thông tin mô hình XGBoost. Vui lòng huấn luyện mô hình."); self.xgb_model_loaded = False; return False
            
            info_path = os.path.join(MODEL_DIR, latest_info_file)
            with open(info_path, 'r') as f: self.xgb_model_info = json.load(f)

            model_purpose = self.xgb_model_info.get('model_purpose', 'trend') 
            if model_purpose.lower() != 'volatility':
                 st.warning(f"Mô hình đã tải ({latest_info_file}) là dành cho dự đoán XU HƯỚNG, không phải BIẾN ĐỘNG. Kết quả có thể không chính xác.")
            else:
                 st.info(f"Đã tải mô hình dự đoán BIẾN ĐỘNG: {latest_info_file}")

            self.xgb_forecast_horizon = self.xgb_model_info.get('forecast_horizon_days')
            model_filename = self.xgb_model_info.get('model_filename')
            scaler_filename = self.xgb_model_info.get('scaler_filename')
            self.xgb_feature_columns = self.xgb_model_info.get('feature_columns', [])
            self.xgb_target_col = self.xgb_model_info.get('target_column', 'Target_Volatility') 
            self.xgb_target_threshold = self.xgb_model_info.get('target_threshold', 0.02) 

            if not all([self.xgb_forecast_horizon, model_filename, scaler_filename, self.xgb_feature_columns]):
                st.error(f"Thông tin không đầy đủ trong {latest_info_file}. Thiếu các thuộc tính quan trọng."); self.xgb_model_loaded = False; return False
            
            st.write(f"Đang tải mô hình XGBoost cho **Dự đoán Biến Động {self.xgb_forecast_horizon} Ngày** (từ `{latest_info_file}`)")
            scaler_path = os.path.join(MODEL_DIR, scaler_filename)
            if not os.path.exists(scaler_path):
                st.error(f"File scaler không tìm thấy: {scaler_path}"); self.xgb_model_loaded = False; return False
            self.xgb_scaler = joblib.load(scaler_path)
            model_path = os.path.join(MODEL_DIR, model_filename)
            if not os.path.exists(model_path):
                st.error(f"File mô hình không tìm thấy: {model_path}"); self.xgb_model_loaded = False; return False
            self.xgb_model = joblib.load(model_path)
            st.success(f"Mô hình XGBoost (Biến Động {self.xgb_forecast_horizon} ngày) đã tải thành công. Ngưỡng biến động: {self.xgb_target_threshold:.4f}"); self.xgb_model_loaded = True; return True
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình XGBoost: {e}"); traceback.print_exc(); self.xgb_model_loaded = False; return False

    def render_model_training_page(self):
        # Same as previous volatility version
        st.header("🧠 Huấn luyện Mô hình Dự đoán Biến Động (XGBoost)")
        if not MODEL_TRAINING_AVAILABLE or not train_stock_prediction_model:
            st.error("Chức năng huấn luyện mô hình không khả dụng. Kiểm tra `model_training.py` import."); return
        processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')] if os.path.exists(PROCESSED_DATA_DIR) else []
        if not processed_files:
            st.warning("Không tìm thấy dữ liệu đã xử lý. Vui lòng thu thập/xử lý dữ liệu trước."); return
        st.info("Trang này huấn luyện mô hình XGBoost để dự đoán mức độ biến động (Cao/Thấp). Mục tiêu độ chính xác: 85-90%.")
        with st.expander("⚙️ Cấu hình Huấn luyện XGBoost (Biến Động)", expanded=True):
            col1_train, col2_train = st.columns(2)
            with col1_train:
                forecast_horizon_train = st.slider("Thời gian dự đoán (ngày)", 1, 30, 5, key="train_xgb_horizon_vol", help="Dự đoán biến động trong N ngày giao dịch tiếp theo.")
                target_threshold_train = st.slider("Ngưỡng Biến Động (Std Dev hàng ngày)", 0.001, 0.1, 0.02, step=0.001, format="%.4f", key="train_xgb_target_thresh_vol", help="Giá trị độ lệch chuẩn của tỷ suất sinh lợi hàng ngày.")
            with col2_train:
                test_size_train = st.slider("Tỷ lệ dữ liệu test", 0.1, 0.3, 0.2, step=0.05, key="train_xgb_test_size_vol", help="Tỷ lệ dữ liệu dành cho kiểm tra mô hình.")
            st.markdown("##### Chọn File Dữ liệu Đã Xử lý để Huấn luyện")
            default_selection_train = []
            if hasattr(self, 'available_companies') and not self.available_companies.empty:
                top_10_tickers_train = [c['ticker'].upper() for c in self.available_companies.head(10).to_dict('records')]
                if processed_files:
                    for f_name_train in sorted(processed_files):
                        if f_name_train.split('_processed_data.csv')[0].upper() in top_10_tickers_train: default_selection_train.append(f_name_train)
                    if not default_selection_train: default_selection_train = sorted(processed_files)[:min(len(processed_files), 5)]
            selected_files_train = st.multiselect("Chọn Files", options=sorted(processed_files), format_func=lambda x: x.split('_processed_data.csv')[0], default=default_selection_train, key="train_xgb_files_vol")
            can_train_ui = bool(selected_files_train)
            if not selected_files_train: st.warning("Vui lòng chọn ít nhất một file dữ liệu đã xử lý.")
        if st.button(f"🚀 Bắt đầu Huấn luyện XGBoost (Biến Động {forecast_horizon_train} ngày, Ngưỡng {target_threshold_train:.4f})", type="primary", use_container_width=True, disabled=not can_train_ui):
            progress_container_train = st.container()
            with progress_container_train:
                st.subheader("🏋️‍♂️ Tiến độ Huấn luyện XGBoost (Biến Động)")
                overall_progress_bar_train = st.progress(0, text="Khởi tạo huấn luyện XGBoost (Biến Động)...")
                status_area_train, metrics_area_train = st.empty(), st.container()
            def training_status_callback_streamlit(message, is_error=False): status_area_train.error(message) if is_error else status_area_train.info(message)
            def training_progress_callback_streamlit(progress_value, text_message): overall_progress_bar_train.progress(max(0.0, min(1.0, progress_value)), text=text_message)
            try:
                status_area_train.info("Bắt đầu huấn luyện mô hình XGBoost dự đoán biến động...")
                processed_file_paths_train = [os.path.join(PROCESSED_DATA_DIR, f) for f in selected_files_train]
                model_path, metrics_trained, feature_columns = train_stock_prediction_model(
                    processed_files=processed_file_paths_train, forecast_horizon=forecast_horizon_train,
                    target_threshold=target_threshold_train, test_size=test_size_train,
                    status_callback=training_status_callback_streamlit, progress_callback=training_progress_callback_streamlit,
                    prediction_type='volatility'
                )
                overall_progress_bar_train.progress(1.0, text="Hoàn thành huấn luyện!")
                if model_path and metrics_trained:
                    status_area_train.success(f"🏆 Huấn luyện XGBoost (Biến Động) hoàn thành! Mô hình đã lưu: {os.path.basename(model_path)}")
                    with metrics_area_train:
                        st.subheader("📊 Kết quả Đánh giá Mô hình (Biến Động)")
                        m_cols_train = st.columns(5)
                        m_cols_train[0].metric("Độ chính xác", f"{metrics_trained.get('accuracy', 0)*100:.2f}%")
                        m_cols_train[1].metric("Precision (Lớp Cao)", f"{metrics_trained.get('precision_high_vol', metrics_trained.get('precision',0)):.4f}")
                        m_cols_train[2].metric("Recall (Lớp Cao)", f"{metrics_trained.get('recall_high_vol', metrics_trained.get('recall',0)):.4f}")
                        m_cols_train[3].metric("F1-Score (Lớp Cao)", f"{metrics_trained.get('f1_score_high_vol', metrics_trained.get('f1_score',0)):.4f}")
                        m_cols_train[4].metric("ROC-AUC", f"{metrics_trained.get('roc_auc', 0):.4f}")
                        accuracy = metrics_trained.get('accuracy', 0)
                        if accuracy >= 0.85: st.success(f"🎉 Đã đạt mục tiêu! Độ chính xác: {accuracy*100:.2f}% >= 85%")
                        else:
                            st.warning(f"⚠️ Chưa đạt mục tiêu 85%. Độ chính xác hiện tại: {accuracy*100:.2f}%")
                            st.info("💡 **Gợi ý cải thiện:**\n- Thêm nhiều dữ liệu từ nhiều cổ phiếu khác nhau, đặc biệt trong các giai đoạn biến động khác nhau.\n- Điều chỉnh Ngưỡng Biến Động (thử các giá trị khác nhau dựa trên phân tích dữ liệu).\n- Tăng thời gian dữ liệu lịch sử (>5 năm để bao quát nhiều chu kỳ thị trường).\n- Thử nghiệm thêm các features chuyên biệt cho biến động (ví dụ: VIX, ATR variants, realized volatility measures).")
                        col_params_train, col_metrics_detail_train = st.columns(2)
                        with col_params_train:
                            st.markdown("##### Chi tiết Cấu hình")
                            config_info = {"Thời gian dự đoán": f"{forecast_horizon_train} ngày", "Ngưỡng Biến Động": f"{target_threshold_train:.4f}", "Số file dữ liệu": len(selected_files_train), "Số đặc trưng": len(feature_columns), "Tỷ lệ test": f"{test_size_train*100:.0f}%"}
                            for key, value in config_info.items(): st.write(f"**{key}:** {value}")
                        with col_metrics_detail_train:
                            st.markdown("##### Chi tiết Metrics")
                            metrics_disp_train = {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k,v in metrics_trained.items()}
                            st.dataframe(pd.DataFrame(metrics_disp_train.items(), columns=['Metric', 'Giá trị']), use_container_width=True, hide_index=True)
                    st.info("Đang tải lại mô hình XGBoost (Biến Động) mới được huấn luyện...")
                    self.load_xgb_model()
                    st.markdown("--- \n ### Bước tiếp theo \n - Chuyển đến tab **🔮 Dự đoán** để sử dụng mô hình XGBoost (Biến Động) mới được huấn luyện.")
                else: status_area_train.error("Huấn luyện XGBoost (Biến Động) thất bại. Kiểm tra console logs.")
            except Exception as e_train_app:
                if 'overall_progress_bar_train' in locals(): overall_progress_bar_train.empty()
                status_area_train.error(f"❌ Lỗi huấn luyện: {e_train_app}"); st.code(traceback.format_exc())

    def render_sidebar(self):
        # Same as previous volatility version
        with st.sidebar:
            st.markdown(f"""<div style="text-align: center; padding: 1.5rem 0; background: var(--gradient-primary); border-radius: 16px; margin-bottom: 1.5rem; box-shadow: var(--shadow-lg);"><h1 style="font-size: 1.8rem; margin: 0; color: white; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🌊 StockAI Pro</h1><p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: rgba(255,255,255,0.8); font-weight: 500;">Phiên bản XGBoost • Dự đoán Biến Động v2.1.0</p><div style="width: 60px; height: 2px; background: var(--gradient-accent); margin: 0.5rem auto; border-radius: 1px;"></div></div>""", unsafe_allow_html=True)
            st.markdown("### 🧭 Điều hướng"); selected_mode = st.session_state.get('app_mode', 'Home')
            nav_buttons = {"Home": {"label": "🏠 Trang chủ", "description": "Tổng quan hệ thống"}, "Data Collection": {"label": "📊 Thu thập Dữ liệu", "description": "Crawl & xử lý dữ liệu thị trường"}, "Model Training": {"label": "🧠 Huấn luyện AI", "description": "Training mô hình XGBoost (Biến Động)"}, "Prediction": {"label": "🔮 Dự đoán Biến Động", "description": "Phân tích & forecast biến động"}, "Settings": {"label": "⚙️ Cài đặt Hệ thống", "description": "Cấu hình & API keys"}}
            for mode, info in nav_buttons.items():
                is_active = selected_mode == mode; button_style = "primary" if is_active else "secondary"
                with st.container():
                    if st.button(info["label"], key=f"{mode}_btn_xgb_v2_1_vol_conf2", use_container_width=True, type=button_style): # incremented key
                        if selected_mode != mode: st.session_state.app_mode = mode; st.rerun()
                    st.markdown(f"""<div style="{'background: var(--gradient-accent); border-radius: 8px; padding: 0.5rem; margin: 0.25rem 0 1rem 0; text-align: center;' if is_active else 'padding: 0.25rem; margin-bottom: 0.75rem;'}"><small style="color: {'white; font-weight: 500;' if is_active else 'var(--text-muted); font-size: 0.8rem;'}">{info["description"]}</small></div>""", unsafe_allow_html=True)
            st.markdown("--- \n ### 💻 Trạng thái Hệ thống")
            model_status_data = self._get_model_status_info()
            st.markdown(f"""<div class="metric-card" style="margin-bottom: 1rem;"><div style="display: flex; align-items: center; margin-bottom: 0.75rem;"><span style="font-size: 1.2rem; margin-right: 0.5rem;">{model_status_data['icon']}</span><span style="font-weight: 600; color: var(--text-color);">Mô hình AI (Biến Động)</span></div><div style="color: {model_status_data['color']}; font-weight: 500; font-size: 0.9rem; line-height: 1.4;">{model_status_data['text']}</div></div>""", unsafe_allow_html=True)
            if 'accuracy' in model_status_data: st.caption(f"📊 Accuracy: {model_status_data.get('accuracy', 'N/A')} | Horizon: {model_status_data.get('horizon', 'N/A')} | Ngưỡng B.Động: {model_status_data.get('threshold', 'N/A')}")
            elif 'files' in model_status_data: st.caption(f"📂 Files: {model_status_data.get('files', 'N/A')} | Accuracy: {model_status_data.get('accuracy', 'N/A')} | Horizon: {model_status_data.get('horizon', 'N/A')}")
            data_status_data = self._get_data_status_info()
            st.markdown(f"""<div class="metric-card" style="margin-bottom: 1rem;"><div style="display: flex; align-items: center; margin-bottom: 0.75rem;"><span style="font-size: 1.2rem; margin-right: 0.5rem;">{data_status_data['icon']}</span><span style="font-weight: 600; color: var(--text-color);">Dữ liệu Thị trường</span></div><div style="color: {data_status_data['color']}; font-weight: 500; font-size: 0.9rem; line-height: 1.4;">{data_status_data['text']}</div></div>""", unsafe_allow_html=True)
            if 'files' in data_status_data: st.caption(f"📊 Files: {data_status_data.get('files', 'N/A')} | Samples: {data_status_data.get('samples', 'N/A')}")
            system_info = self._get_system_info()
            st.markdown(f"""<div class="metric-card"><div style="display: flex; align-items: center; margin-bottom: 0.75rem;"><span style="font-size: 1.2rem; margin-right: 0.5rem;">🖥️</span><span style="font-weight: 600; color: var(--text-color);">Môi trường Hệ thống</span></div><div style="font-size: 0.8rem; line-height: 1.5; color: var(--text-muted);">Python {system_info['python_version']}<br>XGBoost {system_info['xgboost_status']}<br>TA-Lib {system_info['talib_status']}<br>OS {system_info['os_info']}</div></div>""", unsafe_allow_html=True)
            st.markdown("--- \n" + f"""<div style="text-align: center; padding: 1rem; background: var(--gradient-card); border-radius: 12px; border: 1px solid var(--border-color);"><div style="color: var(--text-accent); font-weight: 600; margin-bottom: 0.25rem;">StockAI Professional (Volatility)</div><div style="color: var(--text-muted); font-size: 0.8rem; line-height: 1.4;">Phần mềm phân tích & dự đoán biến động<br>sử dụng <span class="technical-term">XGBoost AI</span><br><em>Vietnamese Edition 2025</em></div><div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border-color);"><small style="color: var(--text-muted); font-size: 0.7rem;">© 2025 • Phiên bản 2.1.0</small></div></div>""", unsafe_allow_html=True)

    def fetch_stock_data(self, ticker, start_date, end_date, min_history_days=300):
        start_date_dt = pd.to_datetime(start_date)
        required_start_date = pd.to_datetime(end_date) - pd.Timedelta(days=min_history_days + 60) 
        if start_date_dt > required_start_date:
            start_date_dt = required_start_date
            # st.caption(f"Adjusted start date for {ticker} to {start_date_dt.strftime('%Y-%m-%d')}") # Can be noisy
        start_date_str = start_date_dt.strftime('%Y-%m-%d')
        try:
            yf_ticker = ticker.replace('.', '-');
            end_date_yf_str = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            df = None; retries_left = 3
            while retries_left > 0 and (df is None or df.empty):
                try:
                    ticker_obj = yf.Ticker(yf_ticker);
                    df_hist = ticker_obj.history(start=start_date_str, end=end_date_yf_str, interval="1d", auto_adjust=False, actions=True)
                    if not df_hist.empty:
                        df = df_hist;
                        df = df[df.index <= pd.to_datetime(end_date).tz_localize(df.index.tzinfo if df.index.tz else None)]
                        break
                except requests.exceptions.ConnectionError as e_conn:
                    if retries_left > 1: st.warning(f"Connection error fetching {ticker} (retrying {retries_left-1} more times). Retrying in 3s..."); time.sleep(3)
                    else: raise
                except Exception as e_fetch_inner:
                    st.error(f"yfinance error fetching {ticker} (attempt {4-retries_left}): {e_fetch_inner}"); return None
                retries_left -=1
            if df is None or df.empty :
                st.error(f"No data returned for {ticker} after retries ({start_date_str} to {end_date})."); return None
            df.reset_index(inplace=True)
            date_col_found = next((col for col in ['Date', 'Datetime', 'Datetime (UTC)'] if col in df.columns), None)
            if date_col_found:
                 df.rename(columns={date_col_found: 'Date'}, inplace=True)
                 df['Date'] = pd.to_datetime(df['Date'])
                 if df['Date'].dt.tz is not None: df['Date'] = df['Date'].dt.tz_localize(None)
                 df.set_index('Date', inplace=True)
            else:
                st.error(f"Date column not found in data for {ticker}. Columns: {df.columns.tolist()}"); return None
            df['Ticker'] = ticker; df.columns = df.columns.str.strip().str.replace(' ', '_')
            rename_map = {'Adj_Close': 'Adj Close'}
            df.rename(columns=rename_map, inplace=True)
            essential_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for ecol in essential_cols:
                if ecol not in df.columns: df[ecol] = np.nan; # st.warning(f"Column '{ecol}' was missing for {ticker} and added as NaN.")
            return df
        except Exception as e:
            st.error(f"Critical error fetching stock data for {ticker}: {e}"); traceback.print_exc(); return None

    def _prepare_data_for_prediction(self, df_historical_raw_orig: pd.DataFrame) -> Optional[pd.DataFrame]:
        ticker = df_historical_raw_orig['Ticker'].iloc[0] if 'Ticker' in df_historical_raw_orig.columns and not df_historical_raw_orig.empty else 'Unknown'
        # st.write(f"--- Chuẩn bị đặc trưng cho dự đoán BIẾN ĐỘNG ({ticker}) ---") # Reduce verbosity

        df_historical_raw = df_historical_raw_orig.copy()
        if not isinstance(df_historical_raw.index, pd.DatetimeIndex):
            if 'Date' in df_historical_raw.columns:
                try:
                    df_historical_raw['Date'] = pd.to_datetime(df_historical_raw['Date'])
                    if df_historical_raw['Date'].dt.tz is not None: df_historical_raw['Date'] = df_historical_raw['Date'].dt.tz_localize(None)
                    df_historical_raw = df_historical_raw.set_index('Date')
                except Exception as e_idx: st.error(f"Lỗi đặt DatetimeIndex: {e_idx}."); return None
            else: st.error(f"Thiếu cột 'Date' để đặt index."); return None
        df_historical_raw = df_historical_raw.sort_index()
        
        df_current_features = df_historical_raw.copy()
        if DATA_COLLECTION_AVAILABLE and DataCollector is not None:
            dc_instance_for_pred = DataCollector()
            df_with_tas = dc_instance_for_pred._calculate_basic_technical_indicators(df_current_features.copy(), ticker=ticker, status_callback=self._streamlit_status_callback_for_prediction)
            if df_with_tas is None or df_with_tas.empty: st.error(f"Tính toán TA thất bại cho {ticker}."); return None
            df_current_features = df_with_tas
            df_with_base_features = dc_instance_for_pred._engineer_base_features(df_current_features.copy(), ticker=ticker, status_callback=self._streamlit_status_callback_for_prediction)
            if df_with_base_features is None or df_with_base_features.empty: st.error(f"Tạo base features thất bại cho {ticker}."); return None
            df_current_features = df_with_base_features
        else: st.error("DataCollector module không khả dụng."); return None

        if MODEL_TRAINING_AVAILABLE and StockVolatilityPredictor is not None:
            try:
                predictor_for_features = StockVolatilityPredictor(forecast_horizon=self.xgb_forecast_horizon or 5)
                df_pred_engineered = predictor_for_features.engineer_ultra_advanced_features(
                    df_current_features.copy(), 
                    status_callback=self._streamlit_status_callback_for_prediction
                )
            except Exception as e_eng:
                st.error(f"Tạo features nâng cao (biến động) thất bại: {e_eng}"); traceback.print_exc(); return None
        else: st.error("StockVolatilityPredictor không khả dụng."); return None

        if df_pred_engineered is None or df_pred_engineered.empty:
            st.error(f"Dữ liệu cho {ticker} trống sau khi tạo feature nâng cao."); return None
        if not self.xgb_scaler or not self.xgb_feature_columns:
            st.error("Scaler hoặc danh sách feature của mô hình chưa được tải."); return None
        
        missing_in_pred_data = set(self.xgb_feature_columns) - set(df_pred_engineered.columns)
        if missing_in_pred_data:
            st.warning(f"LƯU Ý: {len(missing_in_pred_data)} features mà mô hình cần bị THIẾU trong dữ liệu dự đoán. Đang fill bằng 0.0. Kết quả có thể không chính xác.")
            # st.caption(f"Features bị thiếu: {', '.join(sorted(list(missing_in_pred_data)))}") # Can be very long
            for col_missing in missing_in_pred_data:
                df_pred_engineered[col_missing] = 0.0
        
        try:
            df_final_features = df_pred_engineered[self.xgb_feature_columns].copy()
        except KeyError as e:
            st.error(f"Lỗi KeyError khi cố gắng lấy các feature mong muốn: {e}")
            st.error(f"Features kỳ vọng: {self.xgb_feature_columns}")
            st.error(f"Features hiện có: {df_pred_engineered.columns.tolist()}")
            return None

        df_final_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_final_features = df_final_features.ffill().bfill().fillna(0)
        if df_final_features.isna().any().any():
            st.warning("Cảnh báo: Vẫn còn giá trị NaN trong tập feature cuối cùng. Ép fill bằng 0.")
            df_final_features = df_final_features.fillna(0)
        return df_final_features

    # ... (Rest of the methods: calculate_technical_indicators, create_price_chart, etc. remain unchanged from previous "volatility" version)
    # ... (They were already adapted for volatility display where needed)
    def calculate_technical_indicators(self, df):
        # Same as previous version
        if df is None or df.empty: return df.copy() if df is not None else None
        df_indicators = df.copy()
        if not isinstance(df_indicators.index, pd.DatetimeIndex):
             if 'Date' in df_indicators.columns:
                 try:
                    df_indicators['Date'] = pd.to_datetime(df_indicators['Date'])
                    if df_indicators['Date'].dt.tz is not None: df_indicators['Date'] = df_indicators['Date'].dt.tz_localize(None)
                    df_indicators = df_indicators.set_index('Date')
                 except Exception as e_idx_ti: st.error(f"Lỗi đặt DatetimeIndex: {e_idx_ti}"); return None
             else: st.error("DataFrame thiếu cột 'Date'."); return None
        df_indicators = df_indicators.sort_index()
        COL_OPEN_DISP, COL_HIGH_DISP, COL_LOW_DISP, COL_CLOSE_DISP, COL_VOLUME_DISP = 'Open', 'High', 'Low', 'Close', 'Volume'
        required_cols_disp = [COL_OPEN_DISP, COL_HIGH_DISP, COL_LOW_DISP, COL_CLOSE_DISP, COL_VOLUME_DISP]
        for col_disp in required_cols_disp:
            if col_disp not in df_indicators.columns: st.error(f"Thiếu cột '{col_disp}'."); return None
            try:
                df_indicators[col_disp] = pd.to_numeric(df_indicators[col_disp], errors='coerce').ffill().bfill().fillna(0.0)
            except Exception as e_disp_numeric: st.error(f"Lỗi xử lý cột '{col_disp}': {e_disp_numeric}"); return None
        if df_indicators[COL_CLOSE_DISP].isnull().all(): st.error(f"Cột '{COL_CLOSE_DISP}' toàn NaN."); return None
        
        cl_disp, hi_disp, lo_disp = df_indicators[COL_CLOSE_DISP].astype(float).values, df_indicators[COL_HIGH_DISP].astype(float).values, df_indicators[COL_LOW_DISP].astype(float).values
        if TALIB_AVAILABLE and len(cl_disp) >= 20:
            try:
                df_indicators['SMA_20'] = talib.SMA(cl_disp, 20); df_indicators['SMA_50'] = talib.SMA(cl_disp, 50)
                df_indicators['RSI_14'] = talib.RSI(cl_disp, 14)
                macd_disp, macdsignal_disp, macdhist_disp = talib.MACD(cl_disp, 12, 26, 9)
                df_indicators['MACD'], df_indicators['MACD_Signal'], df_indicators['MACD_Hist'] = macd_disp, macdsignal_disp, macdhist_disp
                upper_disp, middle_disp, lower_disp = talib.BBANDS(cl_disp, 20, 2, 2, 0)
                df_indicators['BB_Upper'], df_indicators['BB_Middle'], df_indicators['BB_Lower'] = upper_disp, middle_disp, lower_disp
                df_indicators['ATR_14_Display'] = talib.ATR(hi_disp, lo_disp, cl_disp, timeperiod=14)
            except Exception as e_talib_disp: st.error(f"Lỗi TA-Lib: {e_talib_disp}")
        
        if not all(c_disp in df_indicators.columns for c_disp in ['SMA_20', 'RSI_14', 'MACD', 'BB_Middle']):
            st.warning("Dùng Pandas cho TAs (TA-Lib thiếu/lỗi).")
            close_s_disp = df_indicators[COL_CLOSE_DISP].astype(float)
            df_indicators['SMA_20'] = close_s_disp.rolling(20, min_periods=1).mean()
            # ... (other pandas fallbacks) ...
            if 'ATR_14_Display' not in df_indicators.columns:
                tr = pd.concat([df_indicators[COL_HIGH_DISP] - df_indicators[COL_LOW_DISP], 
                                abs(df_indicators[COL_HIGH_DISP] - close_s_disp.shift(1)), 
                                abs(df_indicators[COL_LOW_DISP] - close_s_disp.shift(1))], axis=1).max(axis=1, skipna=False)
                df_indicators['ATR_14_Display'] = tr.rolling(14, min_periods=1).mean()

        display_indicator_cols_final = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR_14_Display']
        for col_final_disp in display_indicator_cols_final:
            if col_final_disp in df_indicators:
                df_indicators[col_final_disp] = df_indicators[col_final_disp].interpolate(method='linear').ffill().bfill()
                df_indicators[col_final_disp].fillna(50.0 if col_final_disp == 'RSI_14' else 0.0, inplace=True)
            else: df_indicators[col_final_disp] = 50.0 if col_final_disp == 'RSI_14' else 0.0
        return df_indicators

    def create_price_chart(self, df, ticker, prediction_df=None, is_classification=False):
        # Same as previous corrected version for volatility marker
        COL_OPEN_CHART, COL_HIGH_CHART, COL_LOW_CHART, COL_CLOSE_CHART, COL_VOLUME_CHART = 'Open', 'High', 'Low', 'Close', 'Volume'
        if df is None or df.empty:
            fig = go.Figure(); fig.update_layout(title=f"{ticker} - No Data Available", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
            return fig
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2], subplot_titles=(f"{ticker} Phân tích Giá & Biến Động", "Volume", "Average True Range (ATR)")) 
        if not all(c in df.columns for c in [COL_OPEN_CHART, COL_HIGH_CHART, COL_LOW_CHART, COL_CLOSE_CHART]):
            st.error(f"Candlestick chart for {ticker} requires Open, High, Low, Close columns.")
            return go.Figure().update_layout(title=f"{ticker} - Missing OHLC Data", template="plotly_dark")
        fig.add_trace(go.Candlestick(x=df.index, open=df[COL_OPEN_CHART], high=df[COL_HIGH_CHART], low=df[COL_LOW_CHART], close=df[COL_CLOSE_CHART], name="Giá", increasing_line_color=SUCCESS_COLOR, decreasing_line_color=ERROR_COLOR), row=1, col=1) 
        if COL_VOLUME_CHART in df.columns: fig.add_trace(go.Bar(x=df.index, y=df[COL_VOLUME_CHART], name="Volume", marker_color=ACCENT_COLOR, opacity=0.7), row=2, col=1)
        if 'ATR_14_Display' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['ATR_14_Display'], name="ATR (14)", line=dict(color=TEXT_ACCENT_COLOR, width=1.5)), row=3, col=1)
        if 'SMA_20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5)), row=1, col=1)
        if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='rgba(30, 144, 255, 0.8)', width=1.5)), row=1, col=1)
        if all(c in df.columns for c in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(173, 216, 230, 0.5)', width=1), showlegend=False, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(173, 216, 230, 0.5)', width=1), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)', name='Bollinger Bands', hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], line=dict(color='rgba(173, 216, 230, 0.6)', width=1, dash='dot'), name='BB Middle'), row=1, col=1)
        if prediction_df is not None and not prediction_df.empty and is_classification and 'Predicted_Class_Volatility' in prediction_df.columns and 'Probability_High_Volatility' in prediction_df.columns:
            forecast_horizon_days_chart = self.xgb_forecast_horizon if hasattr(self, 'xgb_forecast_horizon') and self.xgb_forecast_horizon else 5
            pred_start_date, pred_end_date = prediction_df['Date'].min(), prediction_df['Date'].max()
            fig.add_vrect(x0=pred_start_date, x1=pred_end_date + pd.Timedelta(days=0.9), fillcolor=f"rgba({int(ACCENT_COLOR[1:3], 16)}, {int(ACCENT_COLOR[3:5], 16)}, {int(ACCENT_COLOR[5:7], 16)}, 0.15)", layer="below", line=dict(color=ACCENT_COLOR, width=1.5, dash="dash"), annotation_text=f"<b>{forecast_horizon_days_chart}D Dự đoán B.Động</b>", annotation_position="top left", annotation_font=dict(size=12, color=TEXT_MUTED_COLOR), row=1, col=1)
            marker_y_position = df[COL_CLOSE_CHART].iloc[-1]
            pred_class_vol, pred_prob_high_vol = prediction_df['Predicted_Class_Volatility'].iloc[0], prediction_df['Probability_High_Volatility'].iloc[0]
            marker_symbol = 'circle'; marker_color = ERROR_COLOR if pred_class_vol == 1 else SUCCESS_COLOR
            volatility_text = 'Biến Động Cao' if pred_class_vol == 1 else 'Biến Động Thấp'
            prob_high_vol_display = f"{pred_prob_high_vol:.2%}" if isinstance(pred_prob_high_vol, (float, int)) else "N/A"
            confidence_display_val = abs(pred_prob_high_vol - 0.5) * 2 * 100 if isinstance(pred_prob_high_vol, (float, int)) else "N/A"
            conf_display = f"{confidence_display_val:.1f}%" if isinstance(confidence_display_val, (float,int)) else "N/A"
            hover_text_chart = f"<b>Dự đoán: {volatility_text}</b><br>Xác suất(B.Đ Cao): {prob_high_vol_display}<br>Độ tin cậy: {conf_display}" 
            marker_plot_date = pred_start_date + pd.Timedelta(days=int(forecast_horizon_days_chart/2)); marker_plot_date = min(max(marker_plot_date, pred_start_date), pred_end_date)
            fig.add_trace(go.Scatter(x=[marker_plot_date], y=[marker_y_position], mode='markers+text', name=f"Dự đoán: {volatility_text}", marker=dict(symbol=marker_symbol, size=18, color=marker_color, line=dict(width=1.5, color='white')), text=[f"<b>{volatility_text}</b><br><span style='font-size:0.8em;'>P(Cao):{prob_high_vol_display}</span>"], textposition="middle right", textfont=dict(color=TEXT_COLOR, size=12), hovertext=hover_text_chart, hoverinfo='text', showlegend=True), row=1, col=1)
        fig.update_layout(title=None, template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, font=dict(color=TEXT_COLOR, family="Arial, sans-serif"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor=CARD_BG_COLOR, bordercolor=BORDER_COLOR,borderwidth=1, font=dict(size=10)), height=750, xaxis_rangeslider_visible=False, margin=dict(l=40, r=40, t=50, b=40)) 
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=BORDER_COLOR, zeroline=False);
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=BORDER_COLOR, zeroline=False, row=1, col=1, title_text="Giá ($)"); 
        fig.update_yaxes(showgrid=False, row=2, col=1, title_text="Volume")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=BORDER_COLOR, zeroline=False, row=3, col=1, title_text="ATR") 
        return fig

    def create_technical_indicators_chart(self, df):
        # Same as before
        if df is None or df.empty or not any(col in df.columns for col in ['RSI_14', 'MACD']):
            fig = go.Figure(); fig.update_layout(title="Technical Indicators - No Data", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR); return fig
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Relative Strength Index (RSI)", "Moving Average Convergence Divergence (MACD)"))
        if 'RSI_14' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name="RSI", line=dict(color=PRIMARY_COLOR, width=1.8)), row=1, col=1)
            fig.add_hline(y=70, line_width=1.2, line_dash="dash", line_color="rgba(230,50,50,0.6)", row=1, col=1, annotation_text="Overbought (70)", annotation_position="bottom right", annotation_font_size=10)
            fig.add_hline(y=30, line_width=1.2, line_dash="dash", line_color="rgba(50,200,50,0.6)", row=1, col=1, annotation_text="Oversold (30)", annotation_position="top right", annotation_font_size=10)
        if all(c in df.columns for c in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='#00BFFF', width=1.8)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal Line", line=dict(color=ACCENT_COLOR, width=1.8)), row=2, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram", marker_color=TEXT_MUTED_COLOR, opacity=0.6), row=2, col=1)
        fig.update_layout(title=None, template="plotly_dark",plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,font=dict(color=TEXT_COLOR, family="Arial, sans-serif"),legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor=CARD_BG_COLOR, bordercolor=BORDER_COLOR, borderwidth=1, font=dict(size=10)),height=500,margin=dict(l=40, r=40, t=60, b=40))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=BORDER_COLOR, zeroline=False);
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=BORDER_COLOR, zeroline=False);
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1); fig.update_yaxes(title_text="MACD", row=2, col=1)
        return fig

    def fetch_news(self, ticker, limit=5):
        # Same as before
        news_results = {'articles': [], 'info': {}}; yf_ticker_news = ticker.replace('.', '-');
        try:
             ticker_obj_news = yf.Ticker(yf_ticker_news); info_news = ticker_obj_news.info
             if info_news and info_news.get('symbol', '').upper() == yf_ticker_news.upper():
                 news_results['info'] = {'name': info_news.get('longName', info_news.get('shortName', ticker)), 'sector': info_news.get('sector', 'N/A'), 'industry': info_news.get('industry', 'N/A'), 'summary': info_news.get('longBusinessSummary', 'N/A'), 'website': info_news.get('website', '#'), 'marketCap': info_news.get('marketCap'), 'previousClose': info_news.get('previousClose'), 'open': info_news.get('open'), 'dayHigh': info_news.get('dayHigh'), 'dayLow': info_news.get('dayLow')}
             yf_news_list = ticker_obj_news.news
             if yf_news_list:
                 for item_news in yf_news_list[:limit]:
                     pub_ts_news = item_news.get('providerPublishTime'); pub_dt_str_news = "Unknown Date"
                     if pub_ts_news:
                         try: pub_dt_str_news = datetime.fromtimestamp(int(pub_ts_news)).strftime("%Y-%m-%d %H:%M")
                         except: pass
                     img_url_news = None
                     if item_news.get('thumbnail') and isinstance(item_news['thumbnail'], dict) and item_news['thumbnail'].get('resolutions'):
                         valid_res = [r for r in item_news['thumbnail']['resolutions'] if isinstance(r,dict) and 'url' in r]
                         if valid_res: img_url_news = sorted(valid_res, key=lambda x: x.get('width',0)*x.get('height',0), reverse=True)[0].get('url')
                     news_results['articles'].append({'title': item_news.get('title', 'N/A'), 'url': item_news.get('link', '#'), 'source': item_news.get('publisher', 'N/A'), 'published_at': pub_dt_str_news, 'image_url': img_url_news})
        except Exception as e_news: st.warning(f"Error fetching yfinance info/news for {ticker}: {e_news}")
        return news_results

    def display_news_section(self, news_results, container):
        # Same as before
        with container:
            info_disp = news_results.get('info', {}); articles_disp = news_results.get('articles', [])
            if info_disp and info_disp.get('name') != 'N/A':
                st.subheader(f"ℹ️ About {info_disp.get('name')}")
                col1, col2, col3 = st.columns(3)
                mc = info_disp.get('marketCap'); pc = info_disp.get('previousClose'); dh = info_disp.get('dayHigh')
                col1.metric("Market Cap", f"${mc/1e9:.2f}B" if isinstance(mc,(int,float)) and mc>0 else "N/A")
                col2.metric("Prev. Close", f"${pc:.2f}" if isinstance(pc,(int,float)) else "N/A")
                col3.metric("Day High", f"${dh:.2f}" if isinstance(dh,(int,float)) else "N/A")
                st.caption(f"**Sector:** {info_disp.get('sector', 'N/A')} | **Industry:** {info_disp.get('industry', 'N/A')}")
                if info_disp.get('summary') and info_disp.get('summary') != 'N/A':
                    with st.expander("Business Summary"): st.markdown(f"<small>{info_disp.get('summary')}</small>", unsafe_allow_html=True)
                if info_disp.get('website') and info_disp.get('website') != '#': st.markdown(f"🌐 [Visit Website]({info_disp.get('website')})")
                st.markdown("---")
            st.subheader("📰 Latest News")
            if not articles_disp: st.info("No recent news found for this ticker via yfinance."); return
            for article_disp in articles_disp:
                with st.container():
                    st.markdown(f"""<div style="border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 15px; margin-bottom: 15px; background-color: {CARD_BG_COLOR};"><h6><a href="{article_disp.get('url', '#')}" target="_blank" style="color: {TEXT_ACCENT_COLOR}; text-decoration: none;">{article_disp.get('title', 'N/A')}</a></h6><small style="color: {TEXT_MUTED_COLOR};">{article_disp.get('source', 'N/A')} | <i>{article_disp.get('published_at')}</i></small></div>""", unsafe_allow_html=True)

    def run_data_collection(self, settings_dc):
        # Same as previous volatility version
        progress_container_dc = st.container();
        st.session_state['dc_log_messages'] = [f"### Data Collection Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"]
        with progress_container_dc:
            st.subheader("📊 Collection Progress"); status_text_area_dc = st.empty(); progress_bar_dc = st.progress(0)
            with st.expander("Show Detailed Logs", expanded=False):
                details_area_log_dc = st.empty()
                details_area_log_dc.markdown(f"<div class='details-log'>{'<br>'.join(st.session_state['dc_log_messages'])}</div>", unsafe_allow_html=True)
        def update_log_display_dc():
            if 'dc_log_messages' in st.session_state: details_area_log_dc.markdown(f"<div class='details-log'>{'<br>'.join(st.session_state['dc_log_messages'])}</div>", unsafe_allow_html=True)
        def progress_callback_app_dc(val, msg): progress_bar_dc.progress(max(0.0,min(1.0,val)), text=msg if msg else "Processing...")
        def status_callback_app_dc(msg, is_err=False):
            ts = datetime.now().strftime('%H:%M:%S'); log_type = "ERROR" if is_err else "INFO"
            if 'dc_log_messages' in st.session_state: st.session_state['dc_log_messages'].append(f"- {ts} [{log_type}]: {msg}"); update_log_display_dc()
            if is_err: status_text_area_dc.error(msg)
            else: status_text_area_dc.info(msg)
        processed_tickers_res, representative_features_res = None, None
        with st.spinner("⚙️ Running data collection & processing pipeline..."):
            try:
                if not DATA_COLLECTION_AVAILABLE or not DataCollector : status_callback_app_dc("DataCollector module not available.", True); return None, None
                collector = DataCollector();
                selected_companies_dicts = settings_dc.get('selected_companies', [])
                if not selected_companies_dicts: status_callback_app_dc("No companies selected.", True); return None, None
                if settings_dc.get('use_reddit') and settings_dc.get('reddit_client_id') and settings_dc.get('reddit_client_secret'):
                    collector.setup_reddit_api(settings_dc['reddit_client_id'], settings_dc['reddit_client_secret'], settings_dc.get('reddit_user_agent', 'StockAI/2.1_Volatility'), status_callback=status_callback_app_dc)
                processed_tickers_res, representative_features_res = collector.run_full_pipeline(
                    companies_to_process=selected_companies_dicts, start_date_str=settings_dc['start_date'].strftime('%Y-%m-%d'), end_date_str=settings_dc['end_date'].strftime('%Y-%m-%d'),
                    use_market_indices=True, use_fred_data=settings_dc.get('use_macro_data', True),
                    use_reddit_sentiment=settings_dc.get('use_reddit', False) and collector.reddit_client is not None,
                    use_google_trends=settings_dc.get('use_google_trends', True),
                    progress_callback=progress_callback_app_dc, status_callback=status_callback_app_dc
                )
                if processed_tickers_res: status_callback_app_dc(f"Pipeline completed for {len(processed_tickers_res)} tickers!")
                else: status_callback_app_dc("Processing failed or no tickers processed.", True)
            except Exception as e_pipeline: status_callback_app_dc(f"Critical pipeline error: {e_pipeline}", True); st.code(traceback.format_exc())
            finally:
                if 'dc_log_messages' in st.session_state: del st.session_state['dc_log_messages']
        st.subheader("📊 Collection Summary"); stats_display = {}
        if 'selected_companies_dicts' in locals() and selected_companies_dicts : stats_display["Companies Requested"] = len(selected_companies_dicts)
        if processed_tickers_res: stats_display["Companies Processed"] = len(processed_tickers_res)
        if representative_features_res: stats_display["Features Generated"] = len(representative_features_res)
        if stats_display:
            stats_cols = st.columns(min(len(stats_display), 3)) if stats_display else []
            for i, (lbl, val) in enumerate(stats_display.items()): stats_cols[i % len(stats_cols)].metric(lbl, str(val))
        else: st.info("No collection statistics.")
        if representative_features_res:
            with st.expander("Representative Features List", expanded=False): st.code(f"{', '.join(representative_features_res)}")
        return processed_tickers_res, representative_features_res

    def render_data_collection_page(self):
        # Same as previous volatility version
        st.markdown(f"""<div style="background: var(--gradient-primary); border-radius: 16px; padding: 2rem; margin-bottom: 2rem; text-align: center; box-shadow: var(--shadow-lg);"><h1 style="font-size: 2.5rem; margin: 0 0 0.5rem 0; color: white; text-shadow: var(--text-shadow-strong);">📊 Thu thập & Xử lý Dữ liệu Thị trường</h1><p style="font-size: 1.1rem; color: rgba(255,255,255,0.9); margin: 0; text-shadow: var(--text-shadow-medium);">Crawl dữ liệu và feature engineering cho <span class="technical-term" style="background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3);">XGBoost AI (Dự đoán Biến Động)</span></p></div>""", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["⚙️ Cấu hình & Thực thi", "📂 Dữ liệu Có sẵn"])
        with tab1:
            st.markdown("### 🔧 Thiết lập Pipeline Thu thập"); col_co, col_date = st.columns(2)
            with col_co:
                st.markdown("##### 🏢 Chọn Danh sách Công ty"); company_map = {f"{r['ticker']} - {r['name']}": r for _, r in self.available_companies.iterrows()}; company_options = list(company_map.keys())
                sel_method = st.radio("Phương thức chọn:", ["Top N công ty", "Tùy chỉnh danh sách"], horizontal=True, key="dc_sel_method_xgb_v2_1_vol_conf2", help="Chọn top N hoặc tự chọn danh sách công ty cụ thể")
                selected_company_dicts_ui = []
                if sel_method == "Top N công ty":
                    num_to_sel = st.slider("Số lượng công ty:", 1, len(company_options), min(10, len(company_options)), key="dc_num_co_xgb_v2_1_vol_conf2", help="Chọn N công ty đầu tiên theo market cap")
                    selected_company_dicts_ui = [company_map[d] for d in company_options[:num_to_sel]]
                    if selected_company_dicts_ui: st.success(f"✅ Đã chọn: {', '.join([d['ticker'] for d in selected_company_dicts_ui[:5]])}{'...' if len(selected_company_dicts_ui) > 5 else ''}")
                else:
                    default_custom = [opt for opt in company_options if any(top_ticker['ticker'] in opt for top_ticker in self.available_companies[:3].to_dict('records'))] or ([company_options[0]] if company_options else [])
                    selected_multi = st.multiselect("Chọn công ty cụ thể:", options=company_options, default=default_custom, key="dc_multi_co_xgb_v2_1_vol_conf2", help="Tìm kiếm và chọn các công ty mong muốn")
                    selected_company_dicts_ui = [company_map[d] for d in selected_multi]
                    custom_tickers_input = st.text_input("Thêm mã chứng khoán (phân cách bằng dấu phẩy):", placeholder="Ví dụ: BRK-A, JPM, TSLA", key="dc_custom_ticker_xgb_v2_1_vol_conf2", help="Nhập thêm các ticker không có trong danh sách")
                    if custom_tickers_input:
                        for ticker_str in [t.strip().upper() for t in custom_tickers_input.split(',') if t.strip()]:
                            if not any(d_item['ticker'] == ticker_str for d_item in selected_company_dicts_ui): selected_company_dicts_ui.append({'ticker': ticker_str, 'name': ticker_str})
                final_selected_companies_ui = [d for i, d in enumerate(selected_company_dicts_ui) if d['ticker'] not in {item['ticker'] for item in selected_company_dicts_ui[:i]}] # Keep order, remove duplicates
            with col_date:
                st.markdown("##### 📅 Khoảng Thời gian Dữ liệu")
                default_start = datetime.now() - timedelta(days=365*5 + 90) 
                start_date_in = st.date_input("Ngày bắt đầu:", default_start, min_value=datetime(2000,1,1), max_value=datetime.now()-timedelta(days=180), key="dc_start_xgb_v2_1_vol_conf2", help="Càng nhiều dữ liệu lịch sử, mô hình càng chính xác")
                end_date_in = st.date_input("Ngày kết thúc:", datetime.now().date(), min_value=start_date_in + timedelta(days=365), max_value=datetime.now().date(), key="dc_end_xgb_v2_1_vol_conf2", help="Thường để là ngày hiện tại")
                if start_date_in and end_date_in: st.info(f"📊 Khoảng dữ liệu: {(end_date_in - start_date_in).days:,} ngày (~{( (end_date_in - start_date_in).days / 365.25):.1f} năm)")
            st.markdown("##### 🌐 Nguồn Dữ liệu"); source_cols = st.columns(2)
            with source_cols[0]:
                st.markdown("**📈 Dữ liệu Cơ bản**"); st.checkbox("📊 Giá cổ phiếu & Volume", True, disabled=True, key="dc_src_stock_xgb_v2_1_vol_conf2"); st.checkbox("📊 Chỉ báo Kỹ thuật", True, disabled=True, key="dc_src_tech_xgb_v2_1_vol_conf2"); src_macro = st.checkbox("🏦 Dữ liệu Kinh tế Vĩ mô", True, key="dc_src_macro_xgb_v2_1_vol_conf2")
            with source_cols[1]:
                st.markdown("**🔍 Dữ liệu Sentiment**"); src_google = st.checkbox("🔍 Google Trends", True, key="dc_src_google_xgb_v2_1_vol_conf2"); src_reddit = st.checkbox("📱 Reddit Sentiment", False, key="dc_src_reddit_xgb_v2_1_vol_conf2")
                if src_reddit and not (st.session_state.get('reddit_client_id') and st.session_state.get('reddit_client_secret')): st.warning("⚠️ API Reddit chưa được cài đặt. Vào tab Cài đặt.")
            can_execute = bool(final_selected_companies_ui) and end_date_in > start_date_in; st.markdown("---")
            if not can_execute:
                if not final_selected_companies_ui: st.error("❌ Vui lòng chọn ít nhất một công ty")
                if end_date_in <= start_date_in: st.error("❌ Ngày kết thúc phải sau ngày bắt đầu.")
            if st.button(f"🚀 Bắt đầu Thu thập Dữ liệu ({len(final_selected_companies_ui)} công ty)", type="primary", use_container_width=True, disabled=not can_execute, key="dc_start_button_xgb_v2_1_vol_conf2"):
                self.run_data_collection({'start_date': start_date_in, 'end_date': end_date_in, 'use_reddit': src_reddit, 'reddit_client_id': st.session_state.get('reddit_client_id', ''), 'reddit_client_secret': st.session_state.get('reddit_client_secret', ''), 'reddit_user_agent': st.session_state.get('reddit_user_agent', 'StockAI/2.1_Volatility'), 'use_google_trends': src_google, 'use_macro_data': src_macro, 'selected_companies': final_selected_companies_ui})
        with tab2: 
            st.markdown("### 📂 Xem trước Dữ liệu Đã xử lý")
            processed_files_list = sorted([f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')]) if os.path.exists(PROCESSED_DATA_DIR) else []
            if processed_files_list:
                st.success(f"✅ Tìm thấy {len(processed_files_list)} file dữ liệu đã xử lý.")
                selected_file_preview = st.selectbox("Chọn file để xem trước:", ["--- Chọn file ---"] + processed_files_list, format_func=lambda x: x.split('_processed_data.csv')[0] if x != "--- Chọn file ---" else x, key="dc_preview_sel_xgb_v2_1_vol_conf2")
                if selected_file_preview and selected_file_preview != "--- Chọn file ---":
                    file_path_preview = os.path.join(PROCESSED_DATA_DIR, selected_file_preview); ticker_preview = selected_file_preview.split('_processed_data.csv')[0]
                    st.markdown(f"#### 📊 Xem trước: **{ticker_preview}**")
                    try:
                        df_preview = pd.read_csv(file_path_preview, parse_dates=['Date']); df_disp = df_preview.set_index('Date') if 'Date' in df_preview.columns and pd.api.types.is_datetime64_any_dtype(df_preview['Date']) else df_preview
                        metrics_cols = st.columns(3); metrics_cols[0].metric("📊 Số dòng", f"{len(df_disp):,}"); metrics_cols[1].metric("📈 Số cột", len(df_disp.columns))
                        dr_str = f"{df_disp.index.min():%d/%m/%Y} → {df_disp.index.max():%d/%m/%Y}" if isinstance(df_disp.index, pd.DatetimeIndex) and not df_disp.empty else "N/A"
                        metrics_cols[2].metric("📅 Khoảng thời gian", dr_str)
                        st.markdown("##### 📋 Mẫu Dữ liệu (5 dòng đầu)"); st.dataframe(df_disp.head().round(3), use_container_width=True)
                        st.markdown("##### 📈 Biểu đồ Nhanh"); num_cols = df_disp.select_dtypes(include=np.number).columns.tolist()
                        vol_related_cols = [c for c in ['Hist_Vol_20D', 'ATR_14_Display', 'Close'] if c in num_cols] 
                        def_plot_col = vol_related_cols[0] if vol_related_cols else (num_cols[0] if num_cols else None)
                        if def_plot_col:
                            plot_col_sel = st.selectbox(f"Chọn cột để vẽ biểu đồ cho {ticker_preview}:", num_cols, index=num_cols.index(def_plot_col) if def_plot_col in num_cols else 0, key=f"prev_plot_{ticker_preview.replace('.','_')}_v2_1_vol_conf2")
                            if plot_col_sel:
                                try:
                                    fig_prev = None
                                    if isinstance(df_disp.index, pd.DatetimeIndex): fig_prev = px.line(df_disp, y=plot_col_sel, title=f"{ticker_preview} - {plot_col_sel}")
                                    elif 'Date' in df_preview.columns and pd.api.types.is_datetime64_any_dtype(df_preview['Date']): fig_prev = px.line(df_preview, x='Date', y=plot_col_sel, title=f"{ticker_preview} - {plot_col_sel}")
                                    if fig_prev: fig_prev.update_layout(template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, font_color=TEXT_COLOR); st.plotly_chart(fig_prev, use_container_width=True)
                                    else: st.warning("Không thể tạo biểu đồ (vấn đề với chỉ mục Date)")
                                except Exception as plot_err: st.warning(f"Lỗi tạo biểu đồ: {plot_err}")
                        else: st.info("Không có cột số để vẽ biểu đồ")
                    except Exception as e_prev: st.error(f"Lỗi xem trước file {selected_file_preview}: {e_prev}")
            else: st.error("📭 Chưa có dữ liệu được xử lý.")

    def render_home_page(self):
        # Same as previous volatility version
        st.markdown(f"""<div style="background: var(--gradient-primary); border-radius: 20px; padding: 3rem 2rem; margin-bottom: 2rem; text-align: center; box-shadow: var(--shadow-xl); position: relative; overflow: hidden;"><div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(circle at 30% 70%, rgba(255,255,255,0.1) 0%, transparent 50%); pointer-events: none;"></div><div style="position: relative; z-index: 1;"><h1 style="font-size: 3.5rem; margin: 0 0 1rem 0; color: white; text-shadow: 0 4px 8px rgba(0,0,0,0.3); font-weight: 800;">🌊 StockAI Professional (Volatility)</h1><p style="font-size: 1.25rem; color: rgba(255,255,255,0.9); margin: 0 0 1.5rem 0; font-weight: 500; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">Hệ thống phân tích và dự đoán <span class="technical-term" style="background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3);">ĐỘ BIẾN ĐỘNG</span> cổ phiếu sử dụng <span class="technical-term" style="background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3);">XGBoost AI</span></p><div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;"><div style="background: rgba(255,255,255,0.15); padding: 0.75rem 1.5rem; border-radius: 25px; backdrop-filter: blur(10px);"><span style="color: white; font-weight: 600;">🎯 Độ chính xác > 85%</span></div><div style="background: rgba(255,255,255,0.15); padding: 0.75rem 1.5rem; border-radius: 25px; backdrop-filter: blur(10px);"><span style="color: white; font-weight: 600;">⚡ Real-time Analysis</span></div><div style="background: rgba(255,255,255,0.15); padding: 0.75rem 1.5rem; border-radius: 25px; backdrop-filter: blur(10px);"><span style="color: white; font-weight: 600;">🇻🇳 Vietnamese Interface</span></div></div></div></div>""", unsafe_allow_html=True)
        st.markdown("### 🚀 Truy cập Nhanh"); quick_access_cols = st.columns(3)
        quick_access_data = [{"title": "📊 Thu thập Dữ liệu", "description": "Crawl và xử lý dữ liệu thị trường cho phân tích biến động", "features": ["yfinance API", "Google Trends", "FRED Economic Data", "VIX Index"], "action": "Data Collection", "color": "var(--accent-color)"}, {"title": "🧠 Huấn luyện AI", "description": "Training mô hình XGBoost dự đoán biến động (Cao/Thấp)", "features": ["Advanced Features", "Volatility Target", "Cross Validation", "Performance Metrics"], "action": "Model Training", "color": "var(--secondary-color)"}, {"title": "🔮 Dự đoán Biến Động", "description": "Phân tích và forecast mức độ biến động giá cổ phiếu", "features": ["Volatility Level Prediction", "Probability Score", "ATR Analysis", "Interactive Charts"], "action": "Prediction", "color": "var(--gradient-end)"}]
        for i, card_data in enumerate(quick_access_data):
            with quick_access_cols[i]:
                st.markdown(f"""<div class="metric-card" style="height: 320px; cursor: pointer; transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-8px) scale(1.02)'; this.style.boxShadow='var(--shadow-xl)'" onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='var(--shadow-md)'"><div style="height: 4px; background: linear-gradient(90deg, {card_data['color']}, var(--gradient-end)); border-radius: 2px; margin-bottom: 1rem;"></div><h4 style="color: var(--text-color); margin: 0 0 0.75rem 0; font-size: 1.2rem; font-weight: 600;">{card_data['title']}</h4><p style="color: var(--text-muted); margin: 0 0 1rem 0; line-height: 1.5; font-size: 0.9rem;">{card_data['description']}</p><div style="margin-bottom: 1.5rem;"><div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem; font-weight: 500;">Tính năng chính:</div>{''.join([f'<div style="display: flex; align-items: center; margin-bottom: 0.25rem;"><span style="color: {card_data["color"]}; margin-right: 0.5rem;">•</span><span style="font-size: 0.8rem; color: var(--text-color);">{feature}</span></div>' for feature in card_data['features']])}</div></div>""", unsafe_allow_html=True)
                if st.button(f"Mở {card_data['title']}", key=f"home_quick_{card_data['action']}_vol_conf2", use_container_width=True, type="primary"): st.session_state.app_mode = card_data['action']; st.rerun()
        st.markdown("--- \n ### 📊 Tổng quan Hệ thống"); status_cols = st.columns(2)
        with status_cols[0]:
            data_status = self._get_data_status_info(); st.markdown(f"""<div class="metric-card" style="height: 180px;"><div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;"><div style="display: flex; align-items: center;"><span style="font-size: 1.5rem; margin-right: 0.75rem;">{data_status['icon']}</span><h4 style="margin: 0; color: var(--text-color); font-size: 1.1rem;">Dữ liệu Thị trường</h4></div><div style="width: 8px; height: 8px; background: {data_status['color']}; border-radius: 50%; box-shadow: 0 0 8px {data_status['color']};"></div></div><div style="color: {data_status['color']}; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">{data_status['text']}</div><div style="color: var(--text-muted); font-size: 0.85rem; line-height: 1.4;">Trạng thái dữ liệu đã thu thập và xử lý để huấn luyện mô hình AI</div></div>""", unsafe_allow_html=True)
        with status_cols[1]:
            model_status = self._get_model_status_info(); st.markdown(f"""<div class="metric-card" style="height: 180px;"><div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;"><div style="display: flex; align-items: center;"><span style="font-size: 1.5rem; margin-right: 0.75rem;">{model_status['icon']}</span><h4 style="margin: 0; color: var(--text-color); font-size: 1.1rem;">Mô hình XGBoost (Biến Động)</h4></div><div style="width: 8px; height: 8px; background: {model_status['color']}; border-radius: 50%; box-shadow: 0 0 8px {model_status['color']};"></div></div><div style="color: {model_status['color']}; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">{model_status['text']}</div><div style="color: var(--text-muted); font-size: 0.85rem; line-height: 1.4;">Trạng thái mô hình AI (Biến Động) đã được huấn luyện và sẵn sàng dự đoán</div></div>""", unsafe_allow_html=True)
        with st.expander("📋 Tính năng nổi bật của StockAI Professional (Volatility Edition)", expanded=False):
            features_cols = st.columns(2)
            with features_cols[0]: st.markdown("""**🔬 Phân tích Kỹ thuật & Thống kê cho Biến Động:**\n- Hơn 50+ chỉ báo (ATR, Volatility Bands, Historical Volatility)\n- Feature engineering tập trung vào các yếu tố ảnh hưởng biến động\n- Momentum, volume, và VIX analysis\n- Price range và distribution indicators\n\n**📈 Machine Learning (XGBoost):**\n- XGBoost classifier với hyperparameter tuning (Optuna/GridSearch)\n- Advanced feature selection (ensemble methods)\n- Cross-validation và early stopping\n- Mục tiêu độ chính xác > 85% trên dữ liệu test cho biến động""")
            with features_cols[1]: st.markdown("""**🌐 Nguồn Dữ liệu Đa dạng:**\n- yfinance: Dữ liệu giá lịch sử, volume, VIX\n- Google Trends: Mức độ quan tâm từ khoá\n- FRED Economic Data: Chỉ số kinh tế vĩ mô\n\n**🎯 Dự đoán Mức Độ Biến Động:**\n- Binary classification (Biến Động Cao/Thấp) với xác suất\n- Forecast horizon từ 1-30 ngày giao dịch\n- Confidence score và risk context\n- Interactive charts và technical analysis (bao gồm ATR)""")
        with st.expander("⚠️ Thông tin quan trọng và Tuyên bố miễn trừ trách nhiệm", expanded=False):
            disclaimer_cols = st.columns(2)
            with disclaimer_cols[0]: st.markdown(f"""**🖥️ Yêu cầu Hệ thống:**\n- Python 3.8+ với các thư viện: XGBoost, scikit-learn, pandas, numpy\n- Internet connection để thu thập dữ liệu real-time\n- Tối thiểu 4GB RAM cho training\n- Khuyến nghị: SSD\n\n**🔐 Bảo mật & Quyền riêng tư:**\n- Dữ liệu được lưu trữ local\n- API keys được mã hoá\n- Không thu thập thông tin cá nhân""")
            with disclaimer_cols[1]: st.markdown(f"""**⚠️ Tuyên bố Miễn trừ Trách nhiệm:**\n<div style="background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--error-color); margin: 1rem 0;"><strong style="color: var(--error-color);">QUAN TRỌNG:</strong> StockAI là công cụ phân tích và giáo dục. <strong>KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ TÀI CHÍNH.</strong><br><br>• Thị trường chứng khoán có rủi ro cao<br>• Kết quả dự đoán từ AI không đảm bảo tính chính xác tuyệt đối<br>• Người dùng tự chịu trách nhiệm với mọi quyết định đầu tư<br>• Luôn thực hiện nghiên cứu độc lập<br>• Chỉ đầu tư số tiền bạn có thể chấp nhận mất</div>""", unsafe_allow_html=True)
        st.markdown("--- \n" + f"""<div style="text-align: center; padding: 2rem; background: var(--gradient-card); border-radius: 16px; border: 1px solid var(--border-color); margin-top: 2rem;"><div style="color: var(--text-accent); font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">🇻🇳 StockAI Professional (Volatility Edition)</div><div style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.5;">Phiên bản 2.1.0 • Powered by <span class="technical-term">XGBoost</span> & <span class="technical-term">Streamlit</span><br>Thiết kế và phát triển tại Việt Nam • © 2025</div></div>""", unsafe_allow_html=True)

    def render_prediction_page(self):
        st.header("🔮 Dự đoán Độ Biến Động Cổ phiếu (XGBoost)")
        if not self.xgb_model_loaded or not self.xgb_model or not self.xgb_scaler:
            st.info("Mô hình XGBoost (Biến Động) chưa được tải. Đang cố gắng tải mô hình mới nhất...")
            if not self.load_xgb_model(): st.error("Không thể tải mô hình XGBoost (Biến Động)."); return
        try:
            with st.expander("ℹ️ Thông tin Mô hình XGBoost (Biến Động)", expanded=False):
                if self.xgb_model_info:
                    st.json(self.xgb_model_info, expanded=False); metrics = self.xgb_model_info.get('metrics', {})
                    pred_info_cols = st.columns(4)
                    pred_info_cols[0].metric("Thời gian dự đoán", f"{self.xgb_forecast_horizon} Ngày" if self.xgb_forecast_horizon else "N/A")
                    pred_info_cols[1].metric("Số đặc trưng", len(self.xgb_feature_columns) if self.xgb_feature_columns else "N/A")
                    pred_info_cols[2].metric("Ngưỡng Biến Động", f"{self.xgb_target_threshold:.4f}" if self.xgb_target_threshold else "N/A")
                    pred_info_cols[3].metric("Loại mô hình", self.xgb_model_info.get('model_type', "XGBoost"))
                    if metrics:
                        st.markdown("##### Kết quả Đánh giá Mô hình (Biến Động)"); m_cols = st.columns(5)
                        m_cols[0].metric("Độ chính xác", f"{metrics.get('accuracy', 0)*100:.2f}%")
                        m_cols[1].metric("Precision (Cao)", f"{metrics.get('precision_high_vol', metrics.get('precision',0)):.4f}")
                        m_cols[2].metric("Recall (Cao)", f"{metrics.get('recall_high_vol', metrics.get('recall',0)):.4f}")
                        m_cols[3].metric("F1-Score (Cao)", f"{metrics.get('f1_score_high_vol', metrics.get('f1_score',0)):.4f}")
                        m_cols[4].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
                else: st.warning("Không có thông tin chi tiết về mô hình XGBoost (Biến Động).")
            st.subheader("📈 Tạo Dự đoán Biến Động Mới"); input_col, news_col = st.columns([2, 1.5])
            available_data_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')] if os.path.exists(PROCESSED_DATA_DIR) else []
            with input_col:
                st.markdown("##### Chọn Cổ phiếu từ Dữ liệu Có sẵn")
                if not available_data_files: st.error("📭 Không có dữ liệu processed nào."); return
                available_tickers = [file.split('_processed_data.csv')[0] for file in available_data_files]
                ticker_options = [f"{t} - {next((c['name'] for c in self.available_companies.to_dict('records') if c['ticker'] == t), t)}" for t in available_tickers]
                st.success(f"✅ Tìm thấy {len(available_tickers)} cổ phiếu có dữ liệu: {', '.join(available_tickers[:5])}{'...' if len(available_tickers) > 5 else ''}")
                default_ticker_sym = st.session_state.get('default_ticker', available_tickers[0] if available_tickers else 'AAPL')
                default_pred_idx = next((i for i, opt in enumerate(ticker_options) if opt.startswith(default_ticker_sym)), 0)
                selected_ticker_opt = st.selectbox("Chọn cổ phiếu:", ticker_options, index=default_pred_idx, key="pred_page_ticker_sel_xgb_v2_1_vol_conf2", help="Chọn từ dữ liệu đã được xử lý")
                ticker_to_predict = selected_ticker_opt.split(' - ')[0]
                data_file_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker_to_predict}_processed_data.csv")
                if os.path.exists(data_file_path):
                    try:
                        df_info = pd.read_csv(data_file_path, parse_dates=['Date'])
                        st.markdown(f"**📊 Thông tin dữ liệu {ticker_to_predict}:**"); info_cols = st.columns(3)
                        info_cols[0].metric("Số dòng", f"{len(df_info):,}"); info_cols[1].metric("Số cột", len(df_info.columns))
                        date_range = f"{df_info['Date'].min():%d/%m/%Y} → {df_info['Date'].max():%d/%m/%Y}" if 'Date' in df_info.columns else "N/A"
                        info_cols[2].metric("Khoảng thời gian", date_range)
                        st.caption(f"🔄 Cập nhật: {datetime.fromtimestamp(os.path.getmtime(data_file_path)):%d/%m/%Y %H:%M}")
                    except Exception as e: st.warning(f"Không thể đọc thông tin file: {e}")
            if st.button(f"🚀 Dự đoán Độ Biến Động {ticker_to_predict} ({self.xgb_forecast_horizon} ngày, XGBoost)", type="primary", use_container_width=True):
                status_container, progress_bar, results_container = st.empty(), st.progress(0, text="Khởi tạo dự đoán..."), st.container()
                try:
                    data_file_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker_to_predict}_processed_data.csv")
                    if not os.path.exists(data_file_path): status_container.error(f"File dữ liệu không tồn tại: {data_file_path}"); progress_bar.empty(); return
                    status_container.info(f"Đang tải dữ liệu cho {ticker_to_predict}..."); df_processed = pd.read_csv(data_file_path, parse_dates=['Date'])
                    if df_processed.empty: status_container.error(f"File dữ liệu trống: {ticker_to_predict}"); progress_bar.empty(); return
                    df_processed.set_index('Date', inplace=True); df_processed.sort_index(inplace=True)
                    
                    df_features_for_pred = self._prepare_data_for_prediction(df_processed.assign(Ticker=ticker_to_predict)) 
                    if df_features_for_pred is None: 
                        status_container.error(f"Không thể chuẩn bị features cho {ticker_to_predict}."); progress_bar.empty(); return
                    
                    progress_bar.progress(0.4, text="Đã chuẩn bị đặc trưng.")
                    scaled_features_np = self.xgb_scaler.transform(df_features_for_pred)
                    progress_bar.progress(0.5, text="Đã chuẩn hóa đặc trưng.")
                    last_features = scaled_features_np[-1:, :] 
                    progress_bar.progress(0.6, text="Đã tạo input dự đoán.")
                    status_container.info(f"Đang thực hiện dự đoán biến động...")
                    
                    predicted_probabilities = self.xgb_model.predict_proba(last_features)[0]
                    prob_high_volatility = predicted_probabilities[1]
                    predicted_class_label_vol = self.xgb_model.predict(last_features)[0]
                    progress_bar.progress(0.8, text="Đã tạo dự đoán biến động.")

                    last_historical_date = df_features_for_pred.index[-1] 
                    forecast_period_dates = pd.bdate_range(start=last_historical_date + pd.Timedelta(days=1), periods=self.xgb_forecast_horizon)
                    prediction_results_df = pd.DataFrame({'Date': forecast_period_dates, 'Predicted_Class_Volatility': predicted_class_label_vol, 'Mức_Biến_Động_Dự_Đoán': 'Cao' if predicted_class_label_vol == 1 else 'Thấp', 'Probability_High_Volatility': prob_high_volatility, 'Confidence_Volatility (%)': abs(prob_high_volatility - 0.5) * 2 * 100})
                    
                    with results_container:
                        st.subheader(f"🎯 Dự đoán Biến Động: {self.xgb_forecast_horizon} Ngày cho {ticker_to_predict}"); res_cols = st.columns(4)
                        vol_level_text = prediction_results_df['Mức_Biến_Động_Dự_Đoán'].iloc[0]; vol_emoji = "🌊" if vol_level_text == "Cao" else "💧"
                        res_cols[0].metric("Mức Biến Động Dự Đoán", f"{vol_emoji} {vol_level_text}", delta="Rủi ro cao hơn" if vol_level_text == "Cao" else "Rủi ro thấp hơn", delta_color="inverse" if vol_level_text == "Cao" else "normal")
                        res_cols[1].metric("Xác suất (B.Đ Cao)", f"{prediction_results_df['Probability_High_Volatility'].iloc[0]:.2%}")
                        res_cols[2].metric("Độ tin cậy (B.Động)", f"{prediction_results_df['Confidence_Volatility (%)'].iloc[0]:.1f}%")
                        res_cols[3].metric("Nguồn dữ liệu", "Processed Data", delta="Local Cache")
                        st.markdown("##### 📋 Chi tiết Dự đoán Biến Động"); display_df = prediction_results_df.copy(); display_df['Date'] = display_df['Date'].dt.strftime('%d/%m/%Y')
                        st.dataframe(display_df[['Date', 'Mức_Biến_Động_Dự_Đoán', 'Probability_High_Volatility', 'Confidence_Volatility (%)']].style.format({'Probability_High_Volatility': '{:.2%}', 'Confidence_Volatility (%)': '{:.1f}%'}), hide_index=True, use_container_width=True)
                        st.markdown("##### 📈 Biểu đồ Phân tích Giá & Biến Động")
                        chart_df = df_processed.iloc[-120:].copy() 
                        chart_with_indicators = self.calculate_technical_indicators(chart_df)
                        if chart_with_indicators is not None and not chart_with_indicators.empty:
                            price_chart = self.create_price_chart(chart_with_indicators, ticker_to_predict, prediction_results_df.copy(), is_classification=True)
                            st.plotly_chart(price_chart, use_container_width=True)
                            st.markdown("##### 📊 Chỉ báo Kỹ thuật (RSI, MACD)"); tech_chart = self.create_technical_indicators_chart(chart_with_indicators); st.plotly_chart(tech_chart, use_container_width=True)
                        st.info(f"📊 **Nguồn dữ liệu:** Processed data từ file `{ticker_to_predict}_processed_data.csv` (Cập nhật: {datetime.fromtimestamp(os.path.getmtime(data_file_path)):%d/%m/%Y %H:%M})")
                        st.warning("⚠️ **Lưu ý quan trọng:** Dự đoán AI chỉ mang tính chất thông tin và giáo dục.")
                    progress_bar.progress(1.0, text="Hoàn thành dự đoán biến động!")
                    status_container.success(f"✅ Dự đoán biến động cho {ticker_to_predict} hoàn thành!")
                except Exception as e_pred_loop:
                    status_container.error(f"❌ Lỗi quá trình dự đoán biến động: {e_pred_loop}"); st.code(traceback.format_exc())
                finally:
                    if 'progress_bar' in locals() and progress_bar: progress_bar.empty()
            with news_col: 
                st.markdown(f"##### 📰 Thông tin & Tin tức: {ticker_to_predict}")
                with st.spinner(f"Đang lấy tin tức cho {ticker_to_predict}..."): news_data = self.fetch_news(ticker_to_predict, limit=5)
                self.display_news_section(news_data, st.container())
                st.markdown("--- \n ##### 🔄 Cập nhật Dữ liệu")
                if st.button("🔄 Làm mới dữ liệu", help="Cập nhật dữ liệu mới nhất cho cổ phiếu này"): st.info("Chuyển đến tab 'Thu thập Dữ liệu' để cập nhật.")
        except Exception as e_render_pred:
            st.error(f"Đã xảy ra lỗi nghiêm trọng khi hiển thị trang dự đoán: {e_render_pred}"); st.code(traceback.format_exc())
            if st.button("Quay về Trang chủ"): st.session_state.app_mode = 'Home'; st.rerun()

    def render_settings_page(self):
        # Same as previous volatility version
        st.header("⚙️ Cài đặt & Quản lý Hệ thống")
        with st.expander("ℹ️ Thông tin Hệ thống", expanded=False):
            system_cols = st.columns(2)
            with system_cols[0]: st.subheader("App Details"); st.markdown(f"**Phiên bản:** 2.1.0 (XGBoost - Volatility)"); st.markdown(f"**Thư mục Dữ liệu:** `{DATA_DIR}`"); st.markdown(f"**Thư mục Mô hình:** `{MODEL_DIR}`")
            with system_cols[1]: st.subheader("Môi trường"); st.markdown(f"**Python:** {platform.python_version()}"); st.markdown(f"**OS:** {platform.system()} {platform.release()}"); st.markdown(f"**XGBoost:** ✅ Sẵn sàng"); st.markdown(f"**TA-Lib:** {'✅ Có sẵn' if TALIB_AVAILABLE else '❌ Thiếu'}")
        with st.expander("⚙️ Tùy chọn Ứng dụng", expanded=True):
            st.subheader("Mã cổ phiếu Mặc định cho Trang Dự đoán Biến Động"); st.caption("Đặt mã cổ phiếu sẽ xuất hiện mặc định trên trang 'Dự đoán Biến Động'.")
            def_ticker_opts = [f"{r['ticker']} - {r['name']}" for _, r in self.available_companies.iterrows()]; curr_def_ticker = st.session_state.get('default_ticker', 'AAPL'); def_idx = next((i for i, opt in enumerate(def_ticker_opts) if opt.startswith(curr_def_ticker)), 0)
            sel_def_opt = st.selectbox("Mã cổ phiếu mặc định", options=def_ticker_opts, index=def_idx, key="settings_def_ticker_sel_xgb_vol_conf2")
            if st.button("Lưu Mã cổ phiếu Mặc định", type="primary", key="settings_save_def_ticker_btn_xgb_vol_conf2"):
                if sel_def_opt: st.session_state['default_ticker'] = sel_def_opt.split(' - ')[0]; self._save_settings(); st.success(f"Đã lưu mã cổ phiếu mặc định: {st.session_state['default_ticker']}")
        with st.expander("🔑 API Credentials (Tùy chọn)", expanded=False):
            st.subheader("Reddit API"); st.markdown("Cần thiết cho tính năng 'Reddit Sentiment'. Lấy từ [Reddit Apps](https://www.reddit.com/prefs/apps).")
            rid = st.text_input("Client ID", value=st.session_state.get('reddit_client_id', ''), type="password", key="settings_rid_xgb_vol_conf2")
            rsecret = st.text_input("Client Secret", value=st.session_state.get('reddit_client_secret', ''), type="password", key="settings_rsecret_xgb_vol_conf2")
            rua = st.text_input("User Agent", value=st.session_state.get('reddit_user_agent', 'StockAI/2.1_Volatility'), key="settings_rua_xgb_vol_conf2")
            if st.button("Lưu API Credentials", key="settings_save_api_btn_xgb_vol_conf2"): st.session_state['reddit_client_id'] = rid; st.session_state['reddit_client_secret'] = rsecret; st.session_state['reddit_user_agent'] = rua; self._save_settings(); st.success("Đã lưu Reddit API credentials!")
        with st.expander("🧹 Quản lý Dữ liệu", expanded=False):
            st.subheader("Xóa Dữ liệu Cache"); st.warning("⚠️ Hành động này không thể hoàn tác.", icon="❗")
            raw_count = sum(len([item for item in os.listdir(os.path.join(RAW_DATA_DIR, sub_dir)) if item.endswith(('.csv','.json'))]) for sub_dir in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, sub_dir))) if os.path.exists(RAW_DATA_DIR) else 0
            processed_count = len([f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')]) if os.path.exists(PROCESSED_DATA_DIR) else 0
            company_list_count = len([f for f in os.listdir(DATA_DIR) if f.startswith('top_') and f.endswith('_companies.csv')]) if os.path.exists(DATA_DIR) else 0
            st.markdown(f"- Dữ liệu Raw: `{raw_count}` files\n- Dữ liệu Processed: `{processed_count}` files\n- Danh sách Công ty: `{company_list_count}` files")
            if raw_count > 0 or processed_count > 0 or company_list_count > 0:
                if st.button("🗑️ Xóa TẤT CẢ Files Dữ liệu", type="primary", key="settings_del_data_btn_xgb_vol_conf2"):
                    del_count = 0; err_msgs = []
                    for d_path_info in [(RAW_DATA_DIR, True), (PROCESSED_DATA_DIR, False), (DATA_DIR, False)]: # (path, is_subdir_root)
                        d_path, is_subdir_root = d_path_info
                        if os.path.exists(d_path):
                            for item_name in os.listdir(d_path):
                                item_path = os.path.join(d_path, item_name)
                                if is_subdir_root and os.path.isdir(item_path): # Delete files in subdirs of RAW_DATA_DIR
                                    for sub_item_name in os.listdir(item_path):
                                        try: os.remove(os.path.join(item_path, sub_item_name)); del_count +=1
                                        except Exception as e: err_msgs.append(f"Lỗi xóa {os.path.join(item_name,sub_item_name)}: {e}")
                                elif not is_subdir_root and os.path.isfile(item_path) and (item_path.endswith('.csv') or item_path.endswith('.json')):
                                    if d_path == DATA_DIR and not (item_name.startswith('top_') and item_name.endswith('_companies.csv')): continue # Only delete top_companies from DATA_DIR
                                    try: os.remove(item_path); del_count +=1
                                    except Exception as e: err_msgs.append(f"Lỗi xóa {item_name}: {e}")
                    st.success(f"Đã xóa {del_count} files dữ liệu!"); st.warning(f"Lỗi (nếu có): {err_msgs}" if err_msgs else ""); st.rerun()
            else: st.info("Không có files dữ liệu để xóa.")
        with st.expander("🤖 Quản lý Mô hình (Biến Động)", expanded=False):
            st.subheader("Xóa Mô hình & Artifacts"); st.warning("⚠️ Hành động này không thể hoàn tác.", icon="❗")
            model_files_count = len([f for f in os.listdir(MODEL_DIR) if (f.startswith(('xgboost_','model_info_','scaler_', 'volatility_predictor_')) and f.endswith(('.joblib','.json')))]) if os.path.exists(MODEL_DIR) else 0
            st.markdown(f"- Model Artifacts: `{model_files_count}` files trong `{os.path.basename(MODEL_DIR)}`.")
            if model_files_count > 0:
                if st.button("🗑️ Xóa TẤT CẢ Model Artifacts", type="primary", key="settings_del_models_btn_xgb_vol_conf2"):
                    del_count_mod, err_msgs_mod = 0, []
                    if os.path.exists(MODEL_DIR):
                        for f_name_mod in os.listdir(MODEL_DIR):
                            if (f_name_mod.startswith(('xgboost_', 'model_info_', 'scaler_', 'volatility_predictor_')) and (f_name_mod.endswith(('.joblib', '.json')))):
                                try: os.remove(os.path.join(MODEL_DIR, f_name_mod)); del_count_mod +=1
                                except Exception as e: err_msgs_mod.append(f"'{f_name_mod}': {e}")
                    st.success(f"Đã xóa {del_count_mod} model artifacts!"); st.warning(f"Lỗi (nếu có): {err_msgs_mod}" if err_msgs_mod else "")
                    self.xgb_model = None; self.xgb_scaler = None; self.xgb_model_info = {}; self.xgb_feature_columns = []; self.xgb_target_col = None; self.xgb_forecast_horizon = None; self.xgb_target_threshold = 0.02; self.xgb_model_loaded = False; st.rerun()
            else: st.info("Không có model artifacts để xóa.")
        with st.expander("📖 Về StockAI Professional (Volatility Edition)", expanded=False):
            st.markdown("""**StockAI v2.1.0 - Volatility Edition**\nỨng dụng này sử dụng machine learning XGBoost để dự đoán mức độ biến động của cổ phiếu (Cao/Thấp).\nTính năng tự động thu thập dữ liệu, feature engineering nâng cao, và giao diện dự đoán tương tác với ngôn ngữ Việt Nam.\n**⚠️ Tuyên bố miễn trừ trách nhiệm:** Công cụ giáo dục. Không phải lời khuyên tài chính.\nThị trường có tính biến động cao. Tự nghiên cứu trước khi đầu tư.\n*Vietnamese Edition 2025*""")

    def run(self):
        if 'app_mode' not in st.session_state: st.session_state.app_mode = 'Home'
        self._load_settings(); self.render_sidebar()
        page_render_map = {"Home": self.render_home_page, "Data Collection": self.render_data_collection_page, "Model Training": self.render_model_training_page, "Prediction": self.render_prediction_page, "Settings": self.render_settings_page}
        page_render_map.get(st.session_state.get('app_mode', 'Home'), self.render_home_page)()

    def _load_settings(self):
        settings_path = os.path.join(ROOT_DIR,'app_settings_volatility.json')
        defaults = {'reddit_client_id': '', 'reddit_client_secret': '', 'reddit_user_agent': 'StockAI_Volatility/1.0', 'default_ticker': 'AAPL' }
        if os.path.exists(settings_path):
            try:
                 with open(settings_path, 'r') as f: loaded = json.load(f)
                 for k, dv in defaults.items(): st.session_state[k] = loaded.get(k, dv)
            except Exception as e: print(f"Error loading settings: {e}. Using defaults."); [st.session_state.setdefault(k,dv) for k,dv in defaults.items()]
        else: [st.session_state.setdefault(k,dv) for k,dv in defaults.items()]

    def _save_settings(self):
        settings_path = os.path.join(ROOT_DIR,'app_settings_volatility.json')
        to_save = {k: st.session_state.get(k) for k in ['default_ticker', 'reddit_client_id', 'reddit_client_secret', 'reddit_user_agent'] if st.session_state.get(k) is not None}
        try:
            with open(settings_path, 'w') as f: json.dump(to_save, f, indent=4)
        except Exception as e: st.warning(f"Failed to save app settings: {e}")

    def run_data_collection_cli(self):
        print("--- CLI: Data Collection & Processing (Volatility context) ---")
        if not DATA_COLLECTION_AVAILABLE or not DataCollector: print("DataCollector module not available."); return False
        collector = DataCollector(); companies_cli = collector.load_and_set_companies_list(num_companies=5)
        if not companies_cli: print("No companies found/loaded."); return False
        start_date_cli, end_date_cli = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d')
        print(f"Running for: {[c['ticker'] for c in companies_cli]}, Range: {start_date_cli} to {end_date_cli}")
        processed_tickers, _ = collector.run_full_pipeline(companies_cli, start_date_cli, end_date_cli, True, True, False, True, status_callback=lambda msg, err: print(f"[CLI-DC-{('ERR' if err else 'INFO')}] {msg}"))
        if processed_tickers: print(f"CLI Data Collection successful for: {processed_tickers}"); return True
        else: print("CLI Data Collection failed."); return False

    def run_model_training_cli(self):
        print("--- CLI: Model Training (XGBoost for Volatility) ---")
        if not MODEL_TRAINING_AVAILABLE or not train_stock_prediction_model: print("Model training module not available."); return False
        processed_files = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')] if os.path.exists(PROCESSED_DATA_DIR) else []
        if not processed_files: print("No processed data files found."); return False
        files_to_train_cli = processed_files[:min(5, len(processed_files))]
        print(f"Training on: {[os.path.basename(f) for f in files_to_train_cli]}")
        try:
            model_path, metrics, _ = train_stock_prediction_model(files_to_train_cli, 5, 0.02, 0.2, status_callback=lambda msg, err: print(f"[CLI-MT-{('ERR' if err else 'INFO')}] {msg}"), prediction_type='volatility')
            if model_path and metrics: print(f"CLI Model Training (Volatility) successful. Model: {model_path}\nMetrics: {metrics}"); return True
            else: print("CLI Model Training (Volatility) failed."); return False
        except Exception as e: print(f"Error during CLI model training (Volatility): {e}"); traceback.print_exc(); return False

    def _get_model_status_info(self):
        if self.xgb_model_loaded and self.xgb_model and self.xgb_forecast_horizon:
            accuracy = self.xgb_model_info.get('metrics', {}).get('accuracy', 0) * 100
            threshold_display = f"{self.xgb_target_threshold:.4f}" if self.xgb_target_threshold else 'N/A'
            model_purpose = self.xgb_model_info.get('model_purpose', 'trend').capitalize()
            return {'icon': '🌊' if model_purpose == 'Volatility' else '🤖', 'text': f'Mô hình {model_purpose} đã tải', 'color': 'var(--success-color)', 'accuracy': f"{accuracy:.1f}%", 'horizon': f"{self.xgb_forecast_horizon} ngày", 'threshold': threshold_display}
        xgb_model_info_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_info_') and f.endswith('.json')] if os.path.exists(MODEL_DIR) else []
        if xgb_model_info_files:
            xgb_model_info_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)), reverse=True)
            try:
                with open(os.path.join(MODEL_DIR, xgb_model_info_files[0]), 'r') as f: model_info = json.load(f)
                return {'icon': '📂', 'text': f'Có {len(xgb_model_info_files)} mô hình ({model_info.get("model_purpose","trend").capitalize()}) chưa tải', 'color': 'var(--warning-color)', 'files': len(xgb_model_info_files), 'accuracy': f"{model_info.get('metrics', {}).get('accuracy', 0)*100:.1f}%", 'horizon': f"{model_info.get('forecast_horizon_days', 'N/A')} ngày", 'threshold': f"{model_info.get('target_threshold', 0.02):.4f}"}
            except Exception: return {'icon': '⚠️', 'text': f'Lỗi đọc thông tin mô hình', 'color': 'var(--error-color)'}
        return {'icon': '❌', 'text': 'Chưa có mô hình nào được huấn luyện', 'color': 'var(--error-color)'}

    def _get_data_status_info(self):
        processed_files_count = 0; sample_tickers = []
        if os.path.exists(PROCESSED_DATA_DIR):
            processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')]
            processed_files_count = len(processed_files)
            if processed_files_count > 0:
                sample_tickers = [f.split('_processed_data.csv')[0] for f in processed_files[:3]]
                return {'icon': '📊', 'text': f'{processed_files_count} file dữ liệu đã xử lý', 'color': 'var(--success-color)', 'files': processed_files_count, 'samples': ', '.join(sample_tickers) + ('...' if processed_files_count > 3 else '')}
        return {'icon': '📭', 'text': 'Chưa có dữ liệu nào được xử lý', 'color': 'var(--error-color)'}

    def _get_system_info(self):
        return {'python_version': platform.python_version(), 'os_info': f"{platform.system()} {platform.release()}", 'xgboost_status': '✅ Sẵn sàng', 'talib_status': '✅ Có sẵn' if TALIB_AVAILABLE else '❌ Thiếu'}

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StockAI System v2.1.0 (XGBoost - Volatility Prediction)')
    parser.add_argument('--mode', type=str, default='streamlit', choices=['streamlit', 'cli'], help="Operation mode")
    parser.add_argument('--cli-action', type=str, default='all', choices=['data_collection', 'model_training', 'all'], help="Action in CLI")
    args = parser.parse_args()

    if args.mode == 'streamlit':
        StockPredictionApp().run()
    elif args.mode == 'cli':
        print(f"--- StockAI (Volatility): CLI Mode (Action: {args.cli_action}) ---")
        cli_app = StockPredictionApp()
        data_success = True # Assume success if not running data collection
        if args.cli_action in ['data_collection', 'all']:
            print("\n>>> Running CLI: Data Collection Phase <<<")
            data_success = cli_app.run_data_collection_cli()
            if not data_success and args.cli_action == 'all': 
                print("\nSkipping model training due to data collection failure.")
        
        if args.cli_action == 'model_training' or (args.cli_action == 'all' and data_success):
            print("\n>>> Running CLI: Model Training Phase (Volatility) <<<")
            cli_app.run_model_training_cli()
        print("--- StockAI (Volatility): CLI Mode Finished ---")