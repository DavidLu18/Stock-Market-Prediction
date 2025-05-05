import os
import sys
import platform
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# Removed torch import (no longer needed for model)
import json
import joblib # For loading LGBM model and scaler
import argparse
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time
# Removed PIL, Matplotlib - Not used directly for display now
import traceback
# Import metrics potentially for display later
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Import custom modules using relative paths (assuming standard project structure)
try:
    from data_collection import DataCollector
    DATA_COLLECTION_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import DataCollector: {e}. Data collection functionality disabled.")
    DATA_COLLECTION_AVAILABLE = False
    DataCollector = None

try:
    # <<< SỬA ĐỔI: Chỉ cần import hàm chạy Optuna từ model_training >>>
    from model_training import run_optuna_optimization_lgbm # No need for engineer_features here
    MODEL_TRAINING_AVAILABLE = True
    import lightgbm as lgb # Import for potential type checking
except ImportError as e:
    st.error(f"Failed to import from model_training: {e}. Training functionality disabled.")
    MODEL_TRAINING_AVAILABLE = False
    run_optuna_optimization_lgbm = None
    lgb = None


# --- Page Config (MUST be first st command) ---
st.set_page_config(page_title="StockAI System", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- Directory Setup ---
# Get the directory of the current script (app.py)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = APP_DIR # Assume app.py is at the root, adjust if needed
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    if not os.path.exists(dir_path):
        try: os.makedirs(dir_path); print(f"Created directory: {dir_path}")
        except OSError as e: st.error(f"Error creating directory {dir_path}: {e}")

# Define custom theme colors (kept as is)
PRIMARY_COLOR = "#ff1493"; BG_COLOR = "#0e1117"; TEXT_COLOR = "#ffffff"; SECONDARY_COLOR = "#ff69b4"

# --- Custom CSS ---
def load_css():
    st.markdown("""
    <style>
        /* Main background and text */
        .main { background-color: #0e1117; color: #ffffff; }
        /* Headers */
        h1, h2, h3, h4, h5, h6 { color: #ff1493; }
        /* Buttons */
        .stButton>button { background-color: #ff1493; color: white; border-radius: 5px; border: none; padding: 10px 24px; transition: all 0.3s ease; }
        .stButton>button:hover { background-color: #ff69b4; color: white; }
        /* Specific Button Types (Ensure primary stays pink) */
        .stButton button[kind="primary"] { background-color: #ff1493; }
        .stButton button[kind="primary"]:hover { background-color: #ff69b4; }
        .stButton button[kind="secondary"] { background-color: #1e1e1e; border: 1px solid #ff1493; } /* Adjust secondary */
        .stButton button[kind="secondary"]:hover { background-color: #333; border: 1px solid #ff69b4; }
        /* Inputs */
        .stSelectbox>div>div, .stDateInput>div>div, .stTextInput>div>div, .stNumberInput>div>div { background-color: #1e1e1e; color: white; border: 1px solid #ff1493 !important; } /* Added !important */
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { background-color: #1e1e1e; color: white; border-radius: 4px 4px 0px 0px; padding: 10px 20px; border: none; }
        .stTabs [aria-selected="true"] { background-color: #ff1493; color: white; }
        /* Sidebar */
        .sidebar .sidebar-content { background-color: #0e1117; }
        /* Progress Bar */
        .stProgress > div > div > div > div { background-color: #ff1493; }
        /* Log Details Area */
        .details-log { background-color: #1e1e1e; border: 1px solid #444; border-radius: 5px; padding: 10px; height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.9em; color: #ccc; }
        /* Adjust metric labels for better visibility */
        .stMetricLabel { color: #aaa; }
    </style>
    """, unsafe_allow_html=True)

class StockPredictionApp:
    def __init__(self):
        load_css() # Apply CSS
        self.available_companies = self.load_available_companies()
        # LGBM Model state
        self.lgbm_model = None
        self.lgbm_scaler = None
        self.lgbm_model_info = {}
        self.lgbm_features = []
        self.lgbm_target_col = None
        self.lgbm_horizon = None
        self.lgbm_threshold = 0.5 # Default threshold
        self.lgbm_loaded = False

    def load_available_companies(self, default_num=50):
        """Load available companies list."""
        # (Giữ nguyên logic - đọc file top_X_companies.csv hoặc dùng default)
        latest_file = None; latest_num = 0
        if os.path.exists(DATA_DIR):
            try:
                all_top = [f for f in os.listdir(DATA_DIR) if f.startswith('top_') and f.endswith('_companies.csv')]
                if all_top: nums = [int(f.split('_')[1]) for f in all_top if f.split('_')[1].isdigit()]; latest_num = max(nums) if nums else 0
                if latest_num > 0: latest_file = os.path.join(DATA_DIR, f'top_{latest_num}_companies.csv')
            except Exception as e: print(f"Warn: Error finding companies file: {e}") # Use print in init

        if latest_file and os.path.exists(latest_file):
            try:
                df = pd.read_csv(latest_file); df.rename(columns={'Symbol': 'ticker', 'Name': 'name'}, inplace=True) # Standardize
                if 'ticker' in df.columns and 'name' in df.columns:
                     print(f"Loaded {len(df)} companies from {os.path.basename(latest_file)}")
                     return df[['ticker', 'name']]
            except Exception as e: print(f"Error reading {os.path.basename(latest_file)}: {e}")

        print("Warn: Using default company list.") # Use print in init
        default_companies = [
             {'ticker': 'AAPL', 'name': 'Apple Inc.'}, {'ticker': 'MSFT', 'name': 'Microsoft Corp.'},
             {'ticker': 'GOOGL', 'name': 'Alphabet Inc. (A)'},{'ticker': 'GOOG', 'name': 'Alphabet Inc. (C)'},
             {'ticker': 'AMZN', 'name': 'Amazon.com, Inc.'}, {'ticker': 'NVDA', 'name': 'NVIDIA Corp.'},
             {'ticker': 'META', 'name': 'Meta Platforms, Inc.'}, {'ticker': 'TSLA', 'name': 'Tesla, Inc.'},
             {'ticker': 'LLY', 'name': 'Eli Lilly & Co.'}, {'ticker': 'V', 'name': 'Visa Inc.'},
             # Add more if desired
            ]
        return pd.DataFrame(default_companies, columns=['ticker', 'name'])


    # <<< SỬA ĐỔI: Cập nhật load_lgbm_model >>>
    def load_lgbm_model(self):
        """Load the LATEST trained LightGBM model, scaler, info, and threshold."""
        st.info("Attempting to load latest LightGBM model artifacts...")
        latest_info_file = None
        latest_timestamp = 0
        selected_horizon = None

        try:
            # Find the latest model info file based on modification time
            if os.path.exists(MODEL_DIR):
                all_info_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('lgbm_model_info_') and f.endswith('d.json')]
                if all_info_files:
                    latest_info_file = max(all_info_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)))
                    latest_timestamp = os.path.getmtime(os.path.join(MODEL_DIR, latest_info_file))
            else:
                st.warning(f"Model directory not found: {MODEL_DIR}")
                self.lgbm_loaded = False; return False

            if not latest_info_file:
                st.warning("No LightGBM model info file found ('lgbm_model_info_*d.json'). Please train a model.")
                self.lgbm_loaded = False; return False

            info_path = os.path.join(MODEL_DIR, latest_info_file)

            # Load model info FIRST to get filenames and horizon
            with open(info_path, 'r') as f:
                self.lgbm_model_info = json.load(f)

            self.lgbm_horizon = self.lgbm_model_info.get('forecast_horizon_days')
            model_filename_from_info = self.lgbm_model_info.get('model_filename')
            scaler_filename_from_info = self.lgbm_model_info.get('scaler_filename')

            if not all([self.lgbm_horizon, model_filename_from_info, scaler_filename_from_info]):
                st.error(f"Incomplete info in {latest_info_file} (missing horizon, model, or scaler filename).")
                self.lgbm_loaded = False; return False

            st.info(f"Loading artifacts for {self.lgbm_horizon}-Day Horizon model (from {latest_info_file})")

            # Construct full paths
            model_path = os.path.join(MODEL_DIR, model_filename_from_info)
            scaler_path = os.path.join(MODEL_DIR, scaler_filename_from_info)
            threshold_filename = f'lgbm_threshold_info_{self.lgbm_horizon}d.json'
            threshold_path = os.path.join(MODEL_DIR, threshold_filename)

            # Check existence
            if not os.path.exists(model_path): st.error(f"Model file not found: {model_path}"); self.lgbm_loaded = False; return False
            if not os.path.exists(scaler_path): st.error(f"Scaler file not found: {scaler_path}"); self.lgbm_loaded = False; return False

            # Load components
            self.lgbm_features = self.lgbm_model_info.get('feature_columns')
            self.lgbm_target_col = self.lgbm_model_info.get('target_variable') # Get target col name
            if not self.lgbm_features or not self.lgbm_target_col:
                 st.error("Model info missing feature list or target variable name."); self.lgbm_loaded = False; return False

            self.lgbm_scaler = joblib.load(scaler_path)
            st.info(f"Scaler loaded: {scaler_filename_from_info}")
            self.lgbm_model = joblib.load(model_path)
            st.info(f"Model loaded: {model_filename_from_info}")

            # Load threshold (prioritize F1, then Acc, then default)
            self.lgbm_threshold = 0.5
            if os.path.exists(threshold_path):
                try:
                    with open(threshold_path, 'r') as f: threshold_info = json.load(f)
                    th_f1 = threshold_info.get('best_threshold_f1')
                    th_acc = threshold_info.get('best_threshold_accuracy')
                    chosen_th = th_f1 if th_f1 is not None else (th_acc if th_acc is not None else 0.5)
                    # Basic validation
                    if 0.05 < chosen_th < 0.95: self.lgbm_threshold = chosen_th
                    else: st.warning(f"Loaded threshold ({chosen_th}) out of range (0.05-0.95). Using default 0.5.")
                    st.info(f"Threshold loaded ({threshold_filename}). Using: {self.lgbm_threshold:.4f}")
                except Exception as e: st.warning(f"Error loading threshold ({threshold_filename}): {e}. Using 0.5.")
            else: st.warning(f"Threshold file not found ({threshold_filename}). Using 0.5.")

            self.lgbm_loaded = True
            st.success(f"LightGBM Model ({self.lgbm_horizon}d Horizon) loaded successfully.")
            return True

        except Exception as e:
            st.error(f"Error loading LightGBM model: {e}")
            traceback.print_exc()
            self.lgbm_loaded = False; return False

    # <<< SỬA ĐỔI: render_model_training_page >>>
    def render_model_training_page(self):
        """Render the model training page (using LightGBM). NO LONGER TAKES BASE FEATURES."""
        st.title("🧠 Model Training (LightGBM - N-Day Trend)")

        if not MODEL_TRAINING_AVAILABLE or not run_optuna_optimization_lgbm:
            st.error("Model training unavailable. Check `model_training.py` import and dependencies.")
            return

        processed_files = []
        if os.path.exists(PROCESSED_DATA_DIR):
            processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')]
        if not processed_files:
            st.warning("No processed data found in 'data/processed/'. Collect/process data first.")
            return

        st.info("This page trains a LightGBM model using **all features** found in the selected processed data files.")

        # --- Training Settings ---
        st.subheader("Training Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            # Forecast horizon
            forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 5, key="train_horizon", help="Predict trend over the next N days.")
        with col2:
            # Optuna trials
            n_trials = st.slider("Optuna Trials", 10, 300, 100, key="train_trials", help="Number of hyperparameter sets Optuna will test.")
        with col3:
            # Walk-forward splits
            num_splits = st.slider("Walk-Forward Splits", 3, 10, 5, key="train_splits", help="Number of validation folds per Optuna trial.")

        # File Selection
        st.subheader("Select Processed Data Files")
        st.markdown("Select the `_processed_data.csv` files generated by the 'Data Collection' tab to use for training.")
        selected_files = st.multiselect("Select Files", options=processed_files,
                                        format_func=lambda x: x.split('_processed_data.csv')[0],
                                        default=processed_files if processed_files else None,
                                        key="train_files")

        if not selected_files:
            st.warning("Please select at least one processed data file.")
            can_train = False
        else:
            can_train = True

        # --- Start Training Button ---
        if st.button(f"🚀 Start LGBM Training ({forecast_horizon}-Day Horizon)", type="primary", use_container_width=True, disabled=not can_train):
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0, text="Initializing Training...")
                status_area = st.empty()
                metrics_area = st.empty()

            try:
                status_area.info("Starting LightGBM training with Optuna optimization...")
                processed_file_paths = [os.path.join(PROCESSED_DATA_DIR, f) for f in selected_files]

                # --- Call the updated LGBM optimization function ---
                # Note: We NO LONGER pass 'feature_columns' argument
                # The function now determines features based on the PRE_EXISTING_FEATURES_LIST internally
                # We only pass the file paths, horizon, trials, and splits.
                final_model, final_scaler, best_params, best_avg_metrics, target_column = run_optuna_optimization_lgbm(
                    processed_files=processed_file_paths,
                    forecast_horizon=forecast_horizon,
                    n_trials=n_trials,
                    num_splits=num_splits,
                    # use_gpu defaults to True if available, no need to pass explicitly unless overriding
                )

                progress_bar.progress(1.0, text="Training Complete!")

                # --- Process results ---
                if final_model and best_params and best_avg_metrics:
                    status_area.success(f"Training complete! Best {forecast_horizon}-Day model saved.")
                    with metrics_area.container():
                        st.subheader("Best Parameters (Optuna)")
                        st.json(best_params)
                        st.subheader("Average Validation Metrics (Best Trial)")
                        # Format metrics nicely
                        metrics_disp = {k: f"{v:.4f}" if isinstance(v, float) else v for k,v in best_avg_metrics.items() if k != 'confusion_matrix'}
                        st.dataframe(pd.DataFrame(metrics_disp.items(), columns=['Metric', 'Avg Value']), use_container_width=True)
                        m_cols = st.columns(4)
                        m_cols[0].metric("Avg Accuracy", f"{best_avg_metrics.get('accuracy', 0):.4f}")
                        m_cols[1].metric("Avg F1", f"{best_avg_metrics.get('f1_positive', 0):.4f}")
                        m_cols[2].metric("Avg AUC", f"{best_avg_metrics.get('roc_auc', 0):.4f}")
                        m_cols[3].metric("Avg Recall", f"{best_avg_metrics.get('recall_positive', 0):.4f}")

                    # Automatically reload the newly trained model
                    st.info("Reloading newly trained model...")
                    self.load_lgbm_model()

                    with st.expander("Saved Model Summary", expanded=True):
                         try:
                             info_path = os.path.join(MODEL_DIR, f'lgbm_model_info_{forecast_horizon}d.json')
                             if os.path.exists(info_path):
                                 with open(info_path, 'r') as f: model_info_disp = json.load(f)
                                 st.json(model_info_disp)
                                 plot_path = os.path.join(MODEL_DIR, f'lgbm_feature_importance_{forecast_horizon}d.png')
                                 if os.path.exists(plot_path): st.image(plot_path, caption="Feature Importance")
                             else: st.warning(f"Could not find info file: {os.path.basename(info_path)}")
                         except Exception as e: st.warning(f"Could not display model info: {e}")
                         st.markdown("### Next Steps\n- Go to **Prediction** tab...")

                else: status_area.error("Training failed or did not find a suitable model.")
            except Exception as e:
                progress_bar.empty(); status_area.error(f"❌ Training Error: {e}"); st.code(traceback.format_exc())


    def render_sidebar(self):
        """Render the sidebar with navigation and status"""
        # (Giữ nguyên logic - đã ổn định)
        with st.sidebar:
            st.markdown("<h1 style='text-align: center; color: #FF69B4;'>📈 StockAI</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>v3 - LGBM Enhanced</p>", unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("📌 Navigation")
            selected_mode = st.session_state.get('app_mode', 'Home')
            nav_buttons = {"Home": "🏠 Home", "Data Collection": "📊 Data", "Model Training": "🧠 Train", "Prediction": "🔮 Predict", "Settings": "⚙️ Settings"}
            for mode, label in nav_buttons.items():
                is_selected = selected_mode == mode
                button_type = "primary" if is_selected else "secondary"
                if st.button(label, key=f"{mode}_btn", use_container_width=True, type=button_type):
                    if not is_selected: st.session_state.app_mode = mode; st.rerun()
            st.markdown("---")
            st.subheader("💻 System Status")
            lgbm_model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('lgbm_model_') and f.endswith('.joblib')] if os.path.exists(MODEL_DIR) else []
            model_status = f"✅ {len(lgbm_model_files)} LGBM models" if lgbm_model_files else "⚠️ No LGBM model"
            st.metric("Model Status", model_status)
            processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')] if os.path.exists(PROCESSED_DATA_DIR) else []
            data_status = f"✅ {len(processed_files)} processed files" if processed_files else "⚠️ No processed data"
            st.metric("Data Status", data_status)
            st.markdown("---")
            st.caption("StockAI Prediction System v3.0")
            st.caption("Hope Project 2025")

    def fetch_stock_data(self, ticker, start_date, end_date):
        """Fetch stock data using yfinance"""
        # (Giữ nguyên logic - đã ổn định)
        try:
            yf_ticker = ticker.replace('.', '-')
            end_date_yf = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            ticker_obj = yf.Ticker(yf_ticker)
            df = ticker_obj.history(start=start_date, end=end_date_yf, interval="1d", auto_adjust=False, actions=True)
            if df.empty: st.error(f"No data for {ticker} ({start_date} to {end_date})."); return None
            df.reset_index(inplace=True)
            date_col = next((col for col in ['Date', 'Datetime'] if col in df.columns), None)
            if date_col:
                 df.rename(columns={date_col: 'Date'}, inplace=True)
                 df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                 df.set_index('Date', inplace=True) # Set index BACK for compatibility
            else: st.error(f"No Date column found for {ticker}."); return None
            df['Ticker'] = ticker; df.columns = df.columns.str.strip()
            return df
        except Exception as e: st.error(f"Error fetching {ticker}: {e}"); return None

    # <<< SỬA ĐỔI: calculate_basic_features_for_prediction >>>
    def _calculate_basic_features_for_prediction(self, df):
        """
        Calculates a SUBSET of features needed for prediction, mirroring
        the logic in data_collection's process_data, but simplified.
        This is used ONLY for preparing prediction input data.
        """
        ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns and not df.empty else 'Unknown'
        st.write(f"Calculating basic features for prediction input ({ticker})...")
        df_feat = df.copy()
        try:
            # Ensure basic columns are numeric and filled
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col in df_feat.columns: df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
                else: df_feat[col] = 0
            df_feat.fillna(method='ffill', inplace=True); df_feat.fillna(method='bfill', inplace=True); df_feat.fillna(0, inplace=True)

            # Calculate indicators using TA-Lib (preferred) or basic fallback
            try:
                import talib
                op = df_feat['Open'].values.astype(float); hi = df_feat['High'].values.astype(float); lo = df_feat['Low'].values.astype(float); cl = df_feat['Close'].values.astype(float); vo = df_feat['Volume'].values.astype(float)
                min_period = 50
                if len(df_feat) >= min_period:
                    df_feat['SMA_5'] = talib.SMA(cl, 5); df_feat['SMA_20'] = talib.SMA(cl, 20); df_feat['SMA_50'] = talib.SMA(cl, 50)
                    df_feat['EMA_5'] = talib.EMA(cl, 5); df_feat['EMA_20'] = talib.EMA(cl, 20)
                    df_feat['RSI'] = talib.RSI(cl, 14)
                    macd, macdsignal, macdhist = talib.MACD(cl, 12, 26, 9); df_feat['MACD'], df_feat['MACD_Signal'], df_feat['MACD_Hist'] = macd, macdsignal, macdhist
                    upper, middle, lower = talib.BBANDS(cl, 20, 2, 2, 0); df_feat['BB_Upper'], df_feat['BB_Middle'], df_feat['BB_Lower'] = upper, middle, lower
                    slowk, slowd = talib.STOCH(hi, lo, cl, 5, 3, 0, 3, 0); df_feat['SlowK'], df_feat['SlowD'] = slowk, slowd
                    df_feat['ADX'] = talib.ADX(hi, lo, cl, 14)
                    df_feat['Chaikin_AD'] = talib.AD(hi, lo, cl, vo)
                    df_feat['OBV'] = talib.OBV(cl, vo)
                    df_feat['ATR'] = talib.ATR(hi, lo, cl, 14)
                    df_feat['Williams_R'] = talib.WILLR(hi, lo, cl, 14)
                    df_feat['ROC'] = talib.ROC(cl, 10)
                    df_feat['CCI'] = talib.CCI(hi, lo, cl, 20)
                else: # Fallback if data too short for TA-Lib
                    st.warning(f"[{ticker}] Prediction data too short ({len(df_feat)}). Using basic indicator calcs.")
                    df_feat = self._calculate_basic_indicators(df_feat) # Use the fallback
            except ImportError: # Fallback if TA-Lib not installed
                st.warning(f"[{ticker}] TA-Lib not found. Using basic indicator calcs for prediction.")
                df_feat = self._calculate_basic_indicators(df_feat)
            except Exception as e_ind:
                st.error(f"[{ticker}] Error calculating indicators for prediction: {e_ind}")
                df_feat = self._calculate_basic_indicators(df_feat) # Fallback on error

            # Calculate Ratios, Lags, Volatility, Time features (matching data_collection)
            df_feat['Close_Open_Ratio'] = (df_feat['Close'] / df_feat['Open'].replace(0, 1e-6))
            df_feat['High_Low_Diff'] = df_feat['High'] - df_feat['Low']
            df_feat['Close_Prev_Ratio'] = (df_feat['Close'] / df_feat['Close'].shift(1).replace(0, 1e-6))
            lags = [1, 3, 5]
            for lag in lags:
                df_feat[f'Close_Lag_{lag}'] = df_feat['Close'].shift(lag)
                df_feat[f'Volume_Lag_{lag}'] = df_feat['Volume'].shift(lag)
                # NOTE: Cannot easily add lagged sentiment/trends/macro here without fetching them again
                # Model might expect these columns - handle missing features later
                for col in ['Compound', 'Interest', 'FEDFUNDS']: # Add placeholders if expected by model
                    if f'{col}_Lag_{lag}' not in df_feat.columns: df_feat[f'{col}_Lag_{lag}'] = 0
            df_feat['Volatility_20D'] = df_feat['Close'].rolling(window=20, min_periods=5).std()
            df_feat['Day_Of_Week'] = df_feat.index.dayofweek

            # Final Fill NaNs/Infs
            df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_feat = df_feat.fillna(method='ffill').fillna(method='bfill').fillna(0)

            return df_feat

        except Exception as e:
            st.error(f"Error calculating basic features for prediction ({ticker}): {e}")
            traceback.print_exc()
            return df.copy() # Return original df on error

    # --- GIỮ NGUYÊN: calculate_technical_indicators (chỉ dùng cho display) ---
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators FOR DISPLAY ONLY (simplified)."""
        # This function remains for creating the display charts, but prediction
        # uses _calculate_basic_features_for_prediction for consistency.
        if df is None or df.empty: return df.copy() if df is not None else None
        df_indicators = df.copy()
        if not isinstance(df_indicators.index, pd.DatetimeIndex):
             if 'Date' in df_indicators.columns: df_indicators = df_indicators.set_index('Date')
             else: return None
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df_indicators.columns: st.error(f"'{col}' missing."); return None
            try: df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce').fillna(method='ffill').fillna(method='bfill').fillna(0.0)
            except Exception as e: st.error(f"Error processing '{col}': {e}"); return None
        try:
            import talib
            cl = df_indicators['Close'].astype(float).values; hi = df_indicators['High'].astype(float).values; lo = df_indicators['Low'].astype(float).values; vo = df_indicators['Volume'].astype(float).values
            min_len = 50
            if len(cl) >= min_len:
                df_indicators['SMA_20'] = talib.SMA(cl, 20); df_indicators['SMA_50'] = talib.SMA(cl, 50)
                df_indicators['RSI'] = talib.RSI(cl, 14)
                macd, macdsignal, macdhist = talib.MACD(cl, 12, 26, 9); df_indicators['MACD'], df_indicators['MACD_Signal'], df_indicators['MACD_Hist'] = macd, macdsignal, macdhist
                upper, middle, lower = talib.BBANDS(cl, 20, 2, 2, 0); df_indicators['BB_Upper'], df_indicators['BB_Middle'], df_indicators['BB_Lower'] = upper, middle, lower
            else: # Basic fallback for display
                st.warning(f"Display data too short ({len(cl)}<{min_len}), using basic display indicators.")
                close_s = df_indicators['Close']
                df_indicators['SMA_20'] = close_s.rolling(20, min_periods=1).mean(); df_indicators['SMA_50'] = close_s.rolling(50, min_periods=1).mean()
                delta = close_s.diff(); gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean(); loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
                rs = gain / loss.replace(0, 1e-6); df_indicators['RSI'] = 100 - (100 / (1 + rs)); df_indicators['RSI'].fillna(50, inplace=True)
                ema12 = close_s.ewm(span=12, adjust=False).mean(); ema26 = close_s.ewm(span=26, adjust=False).mean()
                df_indicators['MACD'] = ema12 - ema26; df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean(); df_indicators['MACD_Hist'] = df_indicators['MACD'] - df_indicators['MACD_Signal']
                df_indicators['BB_Middle'] = df_indicators['SMA_20']; std = close_s.rolling(20, min_periods=1).std().fillna(0); df_indicators['BB_Upper'] = df_indicators['BB_Middle'] + (std * 2); df_indicators['BB_Lower'] = df_indicators['BB_Middle'] - (std * 2)
        except ImportError: st.warning("TA-Lib not found. Display indicators limited.") # Handle missing TA-Lib for display
        except Exception as e: st.error(f"Error calculating display indicators: {e}")
        # Final fill for display columns
        for col in ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower']:
            if col in df_indicators: df_indicators[col] = df_indicators[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)
            else: df_indicators[col] = 0.0 # Ensure columns exist
        return df_indicators

    # --- Giữ nguyên: create_price_chart, create_technical_indicators_chart ---
    def create_price_chart(self, df, ticker, prediction_df=None, is_classification=False):
        # (Logic vẽ biểu đồ giá, Bollinger Bands, SMA, và marker dự đoán giữ nguyên)
        # ... (code vẽ biểu đồ giá như cũ) ...
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], subplot_titles=(f"{ticker} Stock Price", "Volume"))
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price", increasing_line_color=PRIMARY_COLOR, decreasing_line_color='cyan'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=SECONDARY_COLOR), row=2, col=1)
        if 'SMA_20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5)), row=1, col=1)
        if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='rgba(30, 144, 255, 0.7)', width=1.5)), row=1, col=1)
        if all(c in df.columns for c in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(173, 216, 230, 0.7)', width=1), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], line=dict(color='rgba(173, 216, 230, 0.7)', width=1, dash='dash'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(173, 216, 230, 0.7)', width=1), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)', name='Bollinger Bands'), row=1, col=1)
        # Prediction marker logic (classification)
        if prediction_df is not None and not prediction_df.empty and is_classification and 'Predicted_Class' in prediction_df.columns and 'Probability' in prediction_df.columns:
            prediction_start_timestamp = df.index[-1]
            prediction_start_date_for_vline = prediction_start_timestamp.to_pydatetime()
            fig.add_vline(x=prediction_start_date_for_vline, line_width=1, line_dash="dash", line_color="grey") # Vline only
            pred_class = prediction_df['Predicted_Class'].iloc[0]; pred_prob = prediction_df['Probability'].iloc[0]
            pred_horizon_dates = prediction_df['Date']; last_close = df['Close'].iloc[-1]
            high_mean = df['High'].iloc[-5:].mean(); low_mean = df['Low'].iloc[-5:].mean(); price_range = high_mean - low_mean
            marker_offset = (price_range * 0.3) if price_range > 1e-6 else (last_close * 0.01)
            marker_symbol = 'triangle-up' if pred_class == 1 else 'triangle-down'; marker_color = 'lime' if pred_class == 1 else 'red'
            marker_name = f"Pred. Trend ({len(pred_horizon_dates)}d): {'Up' if pred_class == 1 else 'Down'}"; hover_text = f"Prob(Up): {pred_prob:.2f}"
            marker_plot_date = pred_horizon_dates.iloc[0]; marker_y_position = last_close + marker_offset if pred_class == 1 else last_close - marker_offset
            fig.add_trace(go.Scatter(x=[marker_plot_date], y=[marker_y_position], mode='markers', name=marker_name, marker=dict(symbol=marker_symbol, size=12, color=marker_color, line=dict(width=1, color='white')), hovertext=hover_text, hoverinfo='text+name'), row=1, col=1)
        fig.update_layout(title=f"{ticker} Stock Analysis & Trend Prediction", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, font=dict(color=TEXT_COLOR), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=700, xaxis_rangeslider_visible=False, margin=dict(l=50, r=50, t=85, b=50))
        fig.update_yaxes(title_text="Price ($)", row=1, col=1); fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig

    def create_technical_indicators_chart(self, df):
        """Create charts for technical indicators (RSI, MACD) for display."""
        # (Giữ nguyên logic vẽ biểu đồ chỉ báo)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("RSI (14)", "MACD"))
        if 'RSI' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color=PRIMARY_COLOR, width=1.5)), row=1, col=1); fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=1, col=1); fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=1, col=1)
        if all(c in df.columns for c in ['MACD', 'MACD_Signal', 'MACD_Hist']): fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='blue', width=1.5)), row=2, col=1); fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal", line=dict(color='red', width=1.5)), row=2, col=1); fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram", marker_color=SECONDARY_COLOR), row=2, col=1)
        fig.update_layout(title="Technical Indicators", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, font=dict(color=TEXT_COLOR), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500, margin=dict(l=50, r=50, t=85, b=50))
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1); fig.update_yaxes(title_text="MACD", row=2, col=1)
        return fig

    # --- Giữ nguyên: fetch_news, display_news_section ---
    def fetch_news(self, ticker, limit=5):
        """Fetch basic company info and news using yfinance."""
        # (Logic lấy info cơ bản và news từ yf.Ticker giữ nguyên)
        news_results = {'articles': [], 'info': {}}
        try:
             yf_ticker = ticker.replace('.', '-'); ticker_obj = yf.Ticker(yf_ticker); info = ticker_obj.info
             if info and info.get('symbol') == yf_ticker.upper():
                 news_results['info'] = {'name': info.get('longName', info.get('shortName', ticker)), 'sector': info.get('sector', 'N/A'), 'industry': info.get('industry', 'N/A'), 'summary': info.get('longBusinessSummary', 'N/A'), 'website': info.get('website', '#')}
             yf_news = ticker_obj.news
             if yf_news:
                 for item in yf_news[:limit]:
                     pub_ts = item.get('providerPublishTime')
                     pub_dt = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%dT%H:%M:%S") if pub_ts else "Unknown"
                     img_url = item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url')
                     news_results['articles'].append({'title': item.get('title', 'N/A'), 'url': item.get('link', '#'), 'source': item.get('publisher', 'N/A'), 'published_at': pub_dt, 'description': '', 'image_url': img_url})
        except Exception as e: st.warning(f"Error fetching yfinance info/news for {ticker}: {e}")
        return news_results

    def display_news_section(self, news_results):
        """Display company info and news articles."""
        # (Logic hiển thị info và news giữ nguyên)
        info = news_results.get('info', {})
        articles = news_results.get('articles', [])
        if info:
             st.subheader(f"ℹ️ About {info.get('name', 'N/A')}")
             st.markdown(f"**Sector:** {info.get('sector')} | **Industry:** {info.get('industry')}")
             with st.expander("Business Summary"): st.write(info.get('summary', 'N/A'))
             st.markdown(f"[Visit Website]({info.get('website')})"); st.markdown("---")
        st.subheader("📰 Latest News")
        if not articles: st.info("No recent news found."); return
        for article in articles:
            col1, col2 = st.columns([1, 4])
            with col1: st.image(article.get('image_url', '')) if article.get('image_url') else st.write("📰")
            with col2:
                st.markdown(f"**{article.get('title', 'N/A')}**")
                try: pub_date = pd.to_datetime(article.get('published_at'), errors='coerce').strftime("%Y-%m-%d %H:%M") if pd.notna(pd.to_datetime(article.get('published_at'), errors='coerce')) else "Unknown"
                except: pub_date = "Invalid Date"
                st.caption(f"{article.get('source', 'N/A')} | *{pub_date}*")
                st.write(article.get('description', '')); st.markdown(f"[Read more]({article.get('url', '#')})")
            st.markdown("---")

    # --- Giữ nguyên: run_data_collection ---
    # (Đã cập nhật để gọi DataCollector đúng cách)
    def run_data_collection(self, settings):
        """Run data collection pipeline."""
        # (Logic gọi DataCollector, xử lý callback giữ nguyên)
        # ... (code run_data_collection như cũ) ...
        progress_container = st.container()
        collected_stats = {}
        log_messages = ["### Data Collection Log"]
        with progress_container:
            st.subheader("📊 Collection Progress")
            status_text = st.empty(); progress_bar = st.progress(0)
            with st.expander("Show Detailed Logs", expanded=True): details_area = st.empty(); details_area.markdown("\n".join(log_messages), unsafe_allow_html=True)
        def progress_callback(progress_value): progress_bar.progress(max(0.0, min(1.0, progress_value)))
        def status_callback(message):
            ts = datetime.now().strftime('%H:%M:%S'); log_messages.append(f"- {ts}: {message}")
            status_text.info(message); details_area.markdown("\n".join(log_messages), unsafe_allow_html=True)
        processed_tickers, representative_features = None, None
        with st.spinner("Running data collection & processing pipeline..."):
            try:
                if not DATA_COLLECTION_AVAILABLE: st.error("DataCollector not available."); return None, None
                collector = DataCollector()
                start_str, end_str = settings['start_date'].strftime('%Y-%m-%d'), settings['end_date'].strftime('%Y-%m-%d')
                selected_companies = settings.get('selected_companies', [])
                if not selected_companies: status_text.error("No companies selected."); return None, None
                collector.companies = selected_companies; status_callback(f"Initialized for {len(collector.companies)} companies.")
                collected_stats["Companies Req."] = len(collector.companies)
                # Define steps based on settings
                total_steps = 1 # Stock
                if settings['use_reddit']: total_steps += 1
                if settings['use_google_trends']: total_steps += 1
                if settings['use_macro_data']: total_steps += 1
                total_steps += 1 # Process step
                current_step = 0

                # --- Step 1: Stock Data ---
                current_step += 1; status_callback(f"Step {current_step}/{total_steps}: Collecting stock data...")
                stock_data = collector.collect_stock_data(start_date=start_str, end_date=end_str, progress_callback=lambda c, t: progress_callback((current_step -1 + (c/t if t>0 else 0))/total_steps), status_callback=status_callback)
                if not stock_data: status_text.error("Failed to collect stock data. Aborting."); return None, None
                collected_stats["Stock Fetched"] = len(stock_data)

                # --- Step 2: Reddit Data ---
                if settings['use_reddit']:
                    current_step += 1; status_callback(f"Step {current_step}/{total_steps}: Collecting Reddit data...")
                    if settings['reddit_client_id'] and settings['reddit_client_secret']:
                        if collector.setup_reddit_api(settings['reddit_client_id'], settings['reddit_client_secret'], settings['reddit_user_agent']):
                             _ = collector.collect_reddit_sentiment(start_date=start_str, end_date=end_str, progress_callback=lambda p: progress_callback((current_step - 1 + p) / total_steps), status_callback=status_callback) # Returns None now
                             collected_stats["Reddit"] = "Attempted"
                        else: status_callback("Skipping Reddit (API setup failed).")
                    else: status_callback("Skipping Reddit (no credentials).")

                # --- Step 3: Google Trends ---
                if settings['use_google_trends']:
                    current_step += 1; status_callback(f"Step {current_step}/{total_steps}: Collecting Google Trends...")
                    _ = collector.collect_google_trends(start_date=start_str, end_date=end_str, progress_callback=lambda c, t: progress_callback((current_step - 1 + (c/t if t>0 else 0)) / total_steps), status_callback=status_callback) # Returns None now
                    collected_stats["Trends"] = "Attempted"

                # --- Step 4: Macro Data ---
                if settings['use_macro_data']:
                    current_step += 1; status_callback(f"Step {current_step}/{total_steps}: Collecting Macro data (FEDFUNDS)...")
                    _ = collector.collect_fred_data(start_date=start_str, end_date=end_str, series_id='FEDFUNDS', status_callback=status_callback) # Returns None now
                    collected_stats["Macro"] = "Attempted"

                # --- Step 5: Process Data ---
                current_step += 1; status_callback(f"Step {current_step}/{total_steps}: Processing collected data (reading raw files)...")
                # Pass only stock_data dict, process_data reads other raw files now
                processed_tickers, representative_features = collector.process_data(stock_data, progress_callback=lambda c, t: progress_callback((current_step - 1 + (c/t if t>0 else 0)) / total_steps), status_callback=status_callback)

                progress_bar.progress(1.0)
                if processed_tickers: status_text.success("Data collection & processing complete!")
                else: status_text.error("Processing failed for all tickers."); st.error("Check logs.")
            except Exception as e: status_text.error(f"Pipeline Error: {e}"); st.code(traceback.format_exc())
        # Display final stats
        st.subheader("Collection Stats"); num_stats = len(collected_stats); stats_cols = st.columns(min(num_stats, 4)) if num_stats > 0 else []
        for i, (label, value) in enumerate(collected_stats.items()): stats_cols[i % len(stats_cols)].metric(label, value)
        st.subheader("Overall Summary"); summary_cols = st.columns(3)
        num_proc = len(processed_tickers) if processed_tickers else 0; summary_cols[0].metric("Companies Processed", num_proc)
        num_recs = sum(len(d) for d in stock_data.values() if d is not None) if stock_data else 0; summary_cols[1].metric("Stock Records", num_recs)
        num_feats = len(representative_features) if representative_features else 0; summary_cols[2].metric("Features", num_feats)
        if representative_features: st.write(f"**Rep. Features:** `{', '.join(representative_features[:15])}...`")
        return processed_tickers, representative_features


    # --- Giữ nguyên: render_data_collection_page ---
    # (Đã hoạt động tốt với logic mới của run_data_collection)
    def render_data_collection_page(self):
        """Render the data collection page"""
        # (UI giữ nguyên, gọi run_data_collection đã được cập nhật)
        # ... (code render_data_collection_page như cũ) ...
        st.title("📊 Data Collection & Processing")
        settings_expander = st.expander("Collection Settings", expanded=True)
        status_container = st.container(); results_container = st.container()
        with settings_expander:
            st.subheader("Select Companies")
            co_map = {f"{r['ticker']} - {r['name']}": r for _, r in self.available_companies.iterrows()}
            co_opts = list(co_map.keys()); co_sel = st.radio("Selection", ["Top N", "Custom"], horizontal=True, key="dc_sel_method")
            sel_co_dicts = []
            if co_sel == "Top N":
                num_co = st.slider("# Companies", 1, len(co_opts), 10, key="dc_num_co"); top_n = co_opts[:num_co]
                sel_co_dicts = [co_map[d] for d in top_n]; st.write(f"Selected: {', '.join([d['ticker'] for d in sel_co_dicts])}")
            else:
                sel_opts = st.multiselect("Select", options=co_opts, default=[co_opts[0]] if co_opts else [], key="dc_multi_co")
                sel_co_dicts = [co_map[d] for d in sel_opts]
                custom_in = st.text_input("Add Tickers (comma-sep)", placeholder="e.g., MSFT, AAPL", key="dc_custom_ticker")
                if custom_in:
                    custom_tickers = [t.strip().upper() for t in custom_in.split(',') if t.strip()]
                    for t in custom_tickers: sel_co_dicts.append({'ticker': t, 'name': t}) # Add as dict
            # Ensure uniqueness
            seen = set(); unique_sel_co_dicts = []
            for d in sel_co_dicts:
                 if d['ticker'] not in seen: unique_sel_co_dicts.append(d); seen.add(d['ticker'])
            sel_co_dicts = unique_sel_co_dicts

            st.subheader("Date Range"); c1, c2 = st.columns(2)
            default_start = datetime.now() - timedelta(days=365*3) # Default 3 years
            start_date = c1.date_input("Start", default_start, min_value=datetime(2010,1,1), max_value=datetime.now()-timedelta(days=1), key="dc_start")
            end_date = c2.date_input("End", datetime.now(), min_value=start_date, max_value=datetime.now(), key="dc_end")

            st.subheader("Data Sources"); src1, src2 = st.columns(2)
            src1.checkbox("Stock Price", True, disabled=True, key="dc_src_stock")
            src1.checkbox("Indicators", True, disabled=True, key="dc_src_tech")
            src_reddit = src1.checkbox("Reddit Sentiment", False, key="dc_src_reddit")
            src_google = src2.checkbox("Google Trends", False, key="dc_src_google")
            src_macro = src2.checkbox("Macro (FEDFUNDS)", True, key="dc_src_macro")
            if src_reddit and not (st.session_state.get('reddit_client_id') and st.session_state.get('reddit_client_secret')): st.warning("Reddit credentials needed in Settings.")

            if st.button("🚀 Start Collection & Processing", type="primary", use_container_width=True, key="dc_start_button"):
                if not sel_co_dicts: st.error("Please select at least one company.")
                else:
                    status_container.empty(); results_container.empty()
                    with status_container:
                        try:
                            settings = {'start_date': start_date, 'end_date': end_date, 'use_reddit': src_reddit, 'reddit_client_id': st.session_state.get('reddit_client_id', ''), 'reddit_client_secret': st.session_state.get('reddit_client_secret', ''), 'reddit_user_agent': st.session_state.get('reddit_user_agent', 'StockApp/1.0'), 'use_google_trends': src_google, 'use_macro_data': src_macro, 'selected_companies': sel_co_dicts}
                            processed_tickers, rep_features = self.run_data_collection(settings)
                            with results_container:
                                st.subheader("Run Summary")
                                if processed_tickers is not None:
                                    if processed_tickers: st.success(f"Processed {len(processed_tickers)} tickers: {', '.join(processed_tickers)}")
                                    else: st.error("Failed to process any tickers.")
                                else: st.error("Critical failure during collection.")
                        except Exception as e: st.error(f"Error initiating collection: {e}"); st.code(traceback.format_exc())

        # --- Display Existing Processed Data ---
        if os.path.exists(PROCESSED_DATA_DIR) and any(f.endswith('_processed_data.csv') for f in os.listdir(PROCESSED_DATA_DIR)):
            with results_container:
                st.markdown("---"); st.subheader("Existing Processed Data")
                proc_files = sorted([f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')])
                st.info(f"Found {len(proc_files)} files in `{PROCESSED_DATA_DIR}`.")
                sel_preview = st.selectbox("Select File to Preview", options=[""] + proc_files, format_func=lambda x: x.split('_proc')[0] if x else "...", key="dc_preview_select")
                if sel_preview:
                    prev_path = os.path.join(PROCESSED_DATA_DIR, sel_preview); ticker_prev = sel_preview.split('_proc')[0]
                    st.write(f"Previewing: `{ticker_prev}`")
                    try:
                        df_prev = pd.read_csv(prev_path, parse_dates=['Date'], index_col='Date') # Set index for plot
                        pc1, pc2, pc3 = st.columns(3); pc1.metric("Rows", len(df_prev)); pc2.metric("Cols", len(df_prev.columns))
                        try: pc3.metric("Date Range", f"{df_prev.index.min():%Y-%m-%d} to {df_prev.index.max():%Y-%m-%d}")
                        except: pc3.metric("Date Range", "N/A")
                        st.dataframe(df_prev.head().round(3))
                        st.subheader("Quick Viz"); num_cols = df_prev.select_dtypes(include=np.number).columns.tolist()
                        def_plot = 'Close' if 'Close' in num_cols else (num_cols[0] if num_cols else None)
                        if def_plot:
                            plot_col = st.selectbox(f"Plot column for {ticker_prev}", num_cols, index=num_cols.index(def_plot) if def_plot in num_cols else 0, key=f"prev_plot_{ticker_prev}")
                            if plot_col:
                                try: fig_prev = px.line(df_prev, y=plot_col, title=f"{ticker_prev} - {plot_col}"); fig_prev.update_layout(template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR); st.plotly_chart(fig_prev, use_container_width=True)
                                except Exception as e: st.warning(f"Plot error: {e}")
                        else: st.info("No numeric columns.")
                    except Exception as e: st.error(f"Preview error: {e}")
        else: 
            with results_container: st.info("No processed data yet.")

    # --- Giữ nguyên: render_settings_page ---
    # (Đã ổn định)
    def render_settings_page(self):
        """Render the settings page"""
        # (UI và logic lưu/xóa giữ nguyên)
        # ... (code render_settings_page như cũ, bao gồm cả xóa data/model) ...
        st.title("⚙️ Settings")
        with st.expander("System Information", expanded=True):
            c1, c2 = st.columns(2)
            with c1: st.subheader("App Info"); st.write("**Version:** 3.0 (LGBM Enhanced)"); st.write(f"**Data Dir:** {DATA_DIR}"); st.write(f"**Model Dir:** {MODEL_DIR}")
            with c2: st.subheader("System"); st.write(f"**Python:** {platform.python_version()}"); st.write(f"**OS:** {platform.system()}"); st.write(f"**LightGBM:** {lgb.__version__ if lgb else 'N/A'}")

        with st.expander("App Settings", expanded=True):
             st.subheader("Defaults")
             def_opts = [f"{r['ticker']} - {r['name']}" for _, r in self.available_companies.iterrows()]
             def_idx = 0; current_def = st.session_state.get('default_ticker')
             if current_def:
                 try: def_idx = [opt.split(' - ')[0] for opt in def_opts].index(current_def)
                 except ValueError: pass
             def_sel = st.selectbox("Default Stock", options=def_opts, index=def_idx, key="set_def_ticker")
             if st.button("Save Default", type="primary", key="set_save_def"): st.session_state['default_ticker'] = def_sel.split(' - ')[0]; self._save_settings(); st.success("Default saved!")

        with st.expander("API Settings"):
            st.subheader("Reddit API (Optional)")
            st.markdown("Used for sentiment. Get from [Reddit Apps](https://www.reddit.com/prefs/apps).")
            r_id = st.text_input("Client ID", value=st.session_state.get('reddit_client_id', ''), type="password", key="set_r_id")
            r_sec = st.text_input("Client Secret", value=st.session_state.get('reddit_client_secret', ''), type="password", key="set_r_sec")
            r_ua = st.text_input("User Agent", value=st.session_state.get('reddit_user_agent', 'StockApp/1.0'), key="set_r_ua")
            if st.button("Save API", key="set_save_api"): st.session_state['reddit_client_id'] = r_id; st.session_state['reddit_client_secret'] = r_sec; st.session_state['reddit_user_agent'] = r_ua; self._save_settings(); st.success("API settings saved!")

        with st.expander("Data Management"):
            st.subheader("Cleanup")
            raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.csv', '.json'))] if os.path.exists(RAW_DATA_DIR) else []
            proc_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')] if os.path.exists(PROCESSED_DATA_DIR) else []
            top_files = [f for f in os.listdir(DATA_DIR) if f.startswith('top_') and f.endswith('_companies.csv')] if os.path.exists(DATA_DIR) else []
            all_data = raw_files + proc_files + top_files
            if all_data:
                st.write(f"Found {len(raw_files)} raw, {len(proc_files)} processed, {len(top_files)} company files.")
                if st.button("⚠️ Delete ALL Data", type="primary", key="set_del_all_data"):
                    count = 0; errors = []
                    for f in raw_files:
                        try: os.remove(os.path.join(RAW_DATA_DIR, f)); count += 1
                        except Exception as e: errors.append(f"Raw {f}: {e}")
                    for f in proc_files:
                        try: os.remove(os.path.join(PROCESSED_DATA_DIR, f)); count += 1
                        except Exception as e: errors.append(f"Proc {f}: {e}")
                    for f in top_files:
                        try: os.remove(os.path.join(DATA_DIR, f)); count += 1
                        except Exception as e: errors.append(f"Top {f}: {e}")
                    st.success(f"Deleted {count} data files!");
                    if errors: st.error(f"Errors: {errors}")
                    st.rerun()
            else: st.info("No data files found.")

        with st.expander("Model Management"):
            st.subheader("LGBM Models")
            mod_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('lgbm_') and f.endswith(('.joblib', '.json', '.png', '.csv'))] if os.path.exists(MODEL_DIR) else []
            if mod_files:
                st.write(f"Found {len(mod_files)} LGBM related files.")
                if st.button("⚠️ Delete ALL LGBM Artifacts", type="primary", key="set_del_models"):
                    count = 0; errors = []
                    for f in mod_files:
                        try: os.remove(os.path.join(MODEL_DIR, f)); count += 1
                        except Exception as e: errors.append(f"{f}: {e}")
                    st.success(f"Deleted {count} LGBM files!")
                    if errors: st.error(f"Errors: {errors}")
                    self.lgbm_loaded = False; self.lgbm_model = None; self.lgbm_scaler = None; self.lgbm_model_info = {} # Reset state
                    st.rerun()
            else: st.info("No trained LGBM models found.")

        with st.expander("About"): st.markdown("StockAI v3.0 - LGBM Enhanced Prediction. Hope Project 2025.")

    # <<< SỬA ĐỔI: render_prediction_page >>>
    def render_prediction_page(self):
        """Render the prediction page using the loaded LightGBM model."""
        st.title("🔮 Stock Trend Prediction (LightGBM)")

        # Attempt to load model if not already loaded
        if not self.lgbm_loaded:
            if not self.load_lgbm_model():
                st.error("No trained LightGBM model loaded. Please train on 'Train' tab.")
                return

        # Double-check essential components
        if not all([self.lgbm_model, self.lgbm_scaler, self.lgbm_features, self.lgbm_horizon is not None]):
             st.error("Model components missing. Try reloading/retraining.")
             if st.button("Attempt Reload"): self.load_lgbm_model(); st.rerun()
             return

        # Display Model Info
        with st.expander("Loaded Model Information", expanded=False):
            st.json(self.lgbm_model_info)
            metrics = self.lgbm_model_info.get('best_optuna_trial', {}).get('avg_metrics', {})
            st.metric("Forecast Horizon", f"{self.lgbm_horizon} Days")
            st.metric("Prediction Threshold", f"{self.lgbm_threshold:.4f}")
            if metrics: st.dataframe(pd.DataFrame(metrics.items(), columns=['Metric', 'Avg Value']).style.format({'Avg Value': '{:.4f}'}))

        # --- Prediction Input ---
        st.subheader("Make New Prediction")
        col1, col2 = st.columns([2, 1])
        with col1:
            pred_ticker_opts = [f"{r['ticker']} - {r['name']}" for _, r in self.available_companies.iterrows()]
            def_ticker = st.session_state.get('default_ticker', pred_ticker_opts[0].split(' - ')[0] if pred_ticker_opts else 'AAPL')
            try: def_idx = [opt.split(' - ')[0] for opt in pred_ticker_opts].index(def_ticker)
            except ValueError: def_idx = 0
            sel_ticker_opt = st.selectbox("Select Stock", pred_ticker_opts, index=def_idx, key="pred_ticker")
            ticker = sel_ticker_opt.split(' - ')[0]

            st.subheader("Fetch Historical Data")
            fetch_days = max(90, self.lgbm_horizon * 3 + 60) # Need enough for lags/indicators + buffer
            today = datetime.now().date()
            start_date = st.date_input("Fetch Start Date", value=today - timedelta(days=fetch_days), max_value=today - timedelta(days=10), key="pred_start", help=f"Need ~{fetch_days} days before today.")
            end_date = today # Predict based on data up to today

        with col2: # News/Info Section
            st.subheader(f"{ticker} Info & News")
            news_res = self.fetch_news(ticker); self.display_news_section(news_res)

        # --- Prediction Button ---
        if st.button(f"🚀 Predict Trend ({self.lgbm_horizon}-Day)", type="primary", use_container_width=True):
            status_cont = st.empty(); prog_bar = st.progress(0); results_cont = st.container()
            try:
                status_cont.info(f"Fetching data for {ticker}...")
                hist_df_raw = self.fetch_stock_data(ticker, start_date, end_date)
                if hist_df_raw is None or hist_df_raw.empty: status_cont.error("Failed to fetch data."); return
                if len(hist_df_raw) < 60: status_cont.error(f"Need >= 60 data points, got {len(hist_df_raw)}."); return
                prog_bar.progress(10, text="Data Fetched.")

                # Prepare features for MODEL INPUT using the dedicated function
                status_cont.info("Calculating features for prediction input...")
                input_data_with_features = self._calculate_basic_features_for_prediction(hist_df_raw.copy())
                if input_data_with_features is None or input_data_with_features.empty:
                     status_cont.error("Feature calculation for prediction failed."); return
                prog_bar.progress(30, text="Features Calculated.")

                # Ensure all required features are present, fill missing with 0
                missing_feats = [f for f in self.lgbm_features if f not in input_data_with_features.columns]
                if missing_feats:
                    st.warning(f"Input data missing {len(missing_feats)} model features: {missing_feats[:5]}... Filling with 0.")
                    for feat in missing_feats: input_data_with_features[feat] = 0.0

                # Select only required features in correct order and get last row
                try:
                    last_row_features_raw = input_data_with_features[self.lgbm_features].iloc[-1:]
                except KeyError as e: status_cont.error(f"Feature mismatch: {e}. Check model/data."); return
                except IndexError: status_cont.error("Cannot get last row of features."); return

                if last_row_features_raw.empty: status_cont.error("Failed to prepare feature row."); return

                # Final NaN/Inf check on the row to be scaled
                if last_row_features_raw.isna().any().any() or np.isinf(last_row_features_raw.values).any():
                    st.warning("NaN/Inf in final feature row. Filling with 0."); last_row_features_raw = last_row_features_raw.fillna(0).replace([np.inf, -np.inf], 0)
                prog_bar.progress(50, text="Features Prepared.")

                # Scale features
                status_cont.info("Scaling features...")
                try: X_pred_scaled = self.lgbm_scaler.transform(last_row_features_raw)
                except ValueError as ve: st.error(f"Scaling error: {ve}"); return
                except Exception as e_scale: st.error(f"Scaling error: {e_scale}"); return
                prog_bar.progress(70, text="Features Scaled.")

                # Make Prediction
                status_cont.info("Making prediction...")
                pred_proba_up = self.lgbm_model.predict_proba(X_pred_scaled)[0, 1]
                pred_class = int(pred_proba_up >= self.lgbm_threshold)
                prog_bar.progress(90, text="Prediction Made.")

                # Prepare results DataFrame
                last_hist_date = input_data_with_features.index[-1]
                forecast_dates = pd.date_range(start=last_hist_date + timedelta(days=1), periods=self.lgbm_horizon, freq='B')
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Predicted_Class': pred_class, 'Predicted_Trend': 'Up' if pred_class == 1 else 'Down',
                    'Probability (Up)': pred_proba_up, 'Confidence (%)': abs(pred_proba_up - 0.5) * 2 * 100
                }, index=range(self.lgbm_horizon)) # Ensure simple index for display


                # --- Display Results ---
                with results_cont:
                    st.subheader(f"Prediction Result ({self.lgbm_horizon}-Day Horizon)")
                    res_cols = st.columns(3)
                    res_cols[0].metric("Predicted Trend", forecast_df['Predicted_Trend'].iloc[0])
                    res_cols[1].metric("Probability (Up)", f"{forecast_df['Probability (Up)'].iloc[0]:.2%}")
                    res_cols[2].metric("Confidence", f"{forecast_df['Confidence (%)'].iloc[0]:.1f}%")
                    st.dataframe(forecast_df.style.format({'Probability (Up)': '{:.2%}', 'Confidence (%)': '{:.1f}%'}), hide_index=True, use_container_width=True)

                    # Calculate display indicators on historical data
                    hist_df_display = self.calculate_technical_indicators(hist_df_raw.copy())
                    st.subheader("Price Chart with Prediction")
                    fig_pred = self.create_price_chart(hist_df_display.iloc[-90:], ticker, forecast_df, is_classification=True)
                    st.plotly_chart(fig_pred, use_container_width=True)
                    st.subheader("Technical Indicators (Display Only)")
                    fig_tech = self.create_technical_indicators_chart(hist_df_display.iloc[-90:])
                    st.plotly_chart(fig_tech, use_container_width=True)
                    st.info("⚠️ Disclaimer: Predictions are indicative, not financial advice.")

                prog_bar.progress(100); status_cont.success(f"Prediction generated!")

            except Exception as e: status_cont.error(f"Prediction Error: {e}"); st.code(traceback.format_exc())

    # --- Giữ nguyên: render_home_page, show_about_section, run, _load_settings, _save_settings ---
    def render_home_page(self):
        """Render the home page"""
        st.title("📈 StockAI System (v3 - LGBM Enhanced)")
        st.markdown("Advanced Stock Trend Prediction. Use the sidebar to navigate.")
        self.show_about_section()

    def show_about_section(self):
        """Display about section content"""
        st.markdown("""
        Leveraging LightGBM and Optuna for N-day stock trend forecasting.
        Utilizes pre-processed features including technical indicators, sentiment (optional), trends (optional), and macro data.
        """)
        st.subheader("🚀 Key Features"); c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("### 📊 Data")
            st.markdown("- Collect Raw (OHLCV, etc)\n- Process & Feature Eng.\n- Save Raw/Processed")
            if st.button("Go to Data", key="home_data", use_container_width=True): st.session_state.app_mode = "Data Collection"; st.rerun()
        with c2:
            st.markdown("### 🧠 Train")
            st.markdown("- LGBM + Optuna\n- Pre-processed Features\n- Walk-Forward Val.")
            if st.button("Go to Train", key="home_train", use_container_width=True): st.session_state.app_mode = "Model Training"; st.rerun()
        with c3:
            st.markdown("### 🔮 Predict")
            st.markdown("- N-Day Trend Forecast\n- Probability Score\n- Interactive Charts")
            if st.button("Go to Predict", key="home_predict", use_container_width=True): st.session_state.app_mode = "Prediction"; st.rerun()
        st.subheader("📌 System Status"); sc1, sc2 = st.columns(2)
        proc_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')] if os.path.exists(PROCESSED_DATA_DIR) else []; sc1.metric("Data Status", f"✅ {len(proc_files)} processed" if proc_files else "❌ No data")
        lgbm_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('lgbm_model_info_') and f.endswith('d.json')] if os.path.exists(MODEL_DIR) else []; model_stat = "❌ No LGBM model"
        if lgbm_files: latest_info = max(lgbm_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f))); model_horizon = latest_info.split('_')[-1].replace('d.json', '?'); model_stat = f"✅ {len(lgbm_files)} LGBM ({model_horizon}d)"
        sc2.metric("Model Status", model_stat)
        st.info("⚠️ Disclaimer: Educational tool. Not financial advice.")


    def run(self):
        """Run the main Streamlit app loop"""
        if 'app_mode' not in st.session_state: st.session_state.app_mode = 'Home'
        if 'settings' not in st.session_state: st.session_state.settings = {}
        self._load_settings() # Load settings on startup
        self.render_sidebar()
        app_mode = st.session_state.get('app_mode', 'Home')
        page_render_map = {"Home": self.render_home_page, "Data Collection": self.render_data_collection_page, "Model Training": self.render_model_training_page, "Prediction": self.render_prediction_page, "Settings": self.render_settings_page}
        render_func = page_render_map.get(app_mode, self.render_home_page)
        render_func()

    def _load_settings(self):
        """Load settings from settings.json"""
        # (Logic giữ nguyên)
        settings_path = 'settings.json'
        if os.path.exists(settings_path):
            try:
                 with open(settings_path, 'r') as f: loaded = json.load(f)
                 for k, v in loaded.items(): st.session_state[k] = v # Load into session state
            except Exception as e: print(f"Warn: Failed to load settings: {e}") # Use print
        # Ensure defaults exist in session state
        for key, default in [('reddit_client_id',''), ('reddit_client_secret',''), ('reddit_user_agent','StockApp/1.0'), ('default_ticker',None)]:
             if key not in st.session_state: st.session_state[key] = default

    def _save_settings(self):
        """Save settings to settings.json"""
        # (Logic giữ nguyên)
        try:
            settings = {k: st.session_state.get(k) for k in ['default_ticker', 'reddit_client_id', 'reddit_client_secret', 'reddit_user_agent'] if st.session_state.get(k) is not None}
            with open('settings.json', 'w') as f: json.dump(settings, f, indent=4)
        except Exception as e: st.warning(f"Failed to save settings: {e}")


    # --- CLI Functions ---
    # <<< SỬA ĐỔI: run_data_collection_cli để phản ánh logic mới >>>
    def run_data_collection_cli(self):
        """Run data collection from command line."""
        print("--- CLI: Data Collection & Processing ---")
        num_cli = 5; start_cli = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d'); end_cli = datetime.now().strftime('%Y-%m-%d')
        def cli_p(c, t): print(f"\rProgress: {c}/{t}", end='')
        def cli_s(m): print(f"\nStatus: {m}")
        if not DATA_COLLECTION_AVAILABLE: print("E: DataCollector not available."); return False
        try:
            collector = DataCollector()
            cli_s(f"Fetching top {num_cli} companies...")
            companies = collector.get_top_companies(num_companies=num_cli, status_callback=cli_s)
            if not companies: print("E: Failed to get companies."); return False
            cli_s(f"Collecting stock data ({start_cli} to {end_cli})...")
            stock_data = collector.collect_stock_data(start_date=start_cli, end_date=end_cli, progress_callback=cli_p, status_callback=cli_s)
            if not stock_data: print("E: Failed to collect stock data."); return False
            cli_s("Collecting FRED data...") # Add other raw data collection if needed
            collector.collect_fred_data(start_date=start_cli, end_date=end_cli, status_callback=cli_s)
            cli_s("Processing data...")
            processed, features = collector.process_data(stock_data, progress_callback=cli_p, status_callback=cli_s)
            print("\nCLI Data pipeline complete.")
            if processed: print(f"Processed {len(processed)} tickers: {', '.join(processed)}")
            else: print("No tickers processed successfully.")
            return True
        except Exception as e: print(f"\nE: CLI data collection: {e}"); traceback.print_exc(); return False

    # <<< SỬA ĐỔI: run_model_training_cli để phản ánh logic mới >>>
    def run_model_training_cli(self):
        """Run LightGBM model training from command line."""
        print("--- CLI: Model Training (LGBM - Pre-existing Features) ---")
        cli_horizon = 5; cli_trials = 10; cli_splits = 3 # Smaller params for CLI demo
        if not MODEL_TRAINING_AVAILABLE or not run_optuna_optimization_lgbm: print("E: Training function unavailable."); return False
        if not os.path.exists(PROCESSED_DATA_DIR) or not any(f.endswith('_processed_data.csv') for f in os.listdir(PROCESSED_DATA_DIR)):
            print(f"E: No processed data in {PROCESSED_DATA_DIR}. Run data collection."); return False
        proc_files = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')]
        print(f"Found {len(proc_files)} processed files.")
        # Note: No longer need to determine base features here, model_training handles it.
        try:
            print(f"Starting Optuna ({cli_horizon}d, {cli_trials} trials, {cli_splits} splits)...")
            # Call training function WITHOUT feature_columns argument
            final_model, _, _, _, _ = run_optuna_optimization_lgbm(
                processed_files=proc_files, forecast_horizon=cli_horizon,
                n_trials=cli_trials, num_splits=cli_splits
            )
            if final_model: print("\nCLI Training finished. Model saved."); return True
            else: print("\nCLI Training failed."); return False
        except Exception as e: print(f"E: CLI training: {e}"); traceback.print_exc(); return False

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StockAI System v3')
    parser.add_argument('--mode', type=str, default='streamlit', choices=['streamlit', 'cli'], help='Mode: streamlit or cli')
    parser.add_argument('--cli-action', type=str, default='data_collection', choices=['data_collection', 'model_training'], help='Action for CLI mode')
    args = parser.parse_args()

    if args.mode == 'streamlit':
        # Create app instance only for Streamlit mode here
        app = StockPredictionApp()
        app.run()
    elif args.mode == 'cli':
        print(f"--- Running CLI Mode (Action: {args.cli_action}) ---")
        # Create app instance for CLI access to methods, but it won't run the Streamlit loop
        cli_app_instance = StockPredictionApp()
        if args.cli_action == 'data_collection':
            cli_app_instance.run_data_collection_cli()
        elif args.cli_action == 'model_training':
            cli_app_instance.run_model_training_cli()
        print("--- CLI Mode Finished ---")
