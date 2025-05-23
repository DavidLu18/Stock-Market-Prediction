# input_file_0.py (app.py)
import os
import sys
import platform
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import joblib # For loading scaler
from sklearn.preprocessing import StandardScaler # IMPORTED
import argparse
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time
import traceback
from typing import Optional, List, Dict, Tuple, Any, Callable

# Import PyTorch
import torch
import torch.nn as nn # For model structure definition if needed in app

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
    st.error(f"Failed to import DataCollector: {e}. Data collection functionality disabled.")
    DATA_COLLECTION_AVAILABLE = False
    DataCollector = None # type: ignore

try:
    from model_training import (
        run_optuna_optimization_resnls,
        engineer_advanced_features as mt_engineer_advanced_features,
        ResNLS # Class itself
    )
    MODEL_TRAINING_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"(app.py) INFO: PyTorch using device: {DEVICE}")

except ImportError as e:
    st.error(f"Failed to import from model_training: {e}. Training functionality disabled.")
    MODEL_TRAINING_AVAILABLE = False
    run_optuna_optimization_resnls = None # type: ignore
    mt_engineer_advanced_features = None # type: ignore
    ResNLS = None # type: ignore
    DEVICE = "cpu"


# --- Page Config (MUST be first st command) ---
st.set_page_config(page_title="StockAI System (ResNLS)", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- Directory Setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = APP_DIR
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    if not os.path.exists(dir_path):
        try: os.makedirs(dir_path); print(f"Created directory: {dir_path}")
        except OSError as e: st.error(f"Error creating directory {dir_path}: {e}")

# --- Theme Colors & CSS ---
PRIMARY_COLOR = "#00AEEF"
ACCENT_COLOR = "#F4A261"
BG_COLOR = "#121212"
TEXT_COLOR = "#E0E0E0"
TEXT_MUTED_COLOR = "#A0A0A0"
CARD_BG_COLOR = "#1E1E1E"
BORDER_COLOR = "#333333"

def load_css():
    st.markdown(f"""
    <style>
        .main {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
        h1, h2, h3, h4, h5, h6 {{ color: {PRIMARY_COLOR}; font-weight: 300; }}
        h1 {{ border-bottom: 2px solid {PRIMARY_COLOR}; padding-bottom: 10px; margin-bottom: 20px;}}
        h2 {{ border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 8px; margin-bottom: 15px;}}
        .stButton>button {{
            background-color: {PRIMARY_COLOR}; color: white;
            border-radius: 8px; border: 1px solid {PRIMARY_COLOR};
            padding: 10px 24px; transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2); font-weight: 500;
        }}
        .stButton>button:hover {{
            background-color: {ACCENT_COLOR}; border-color: {ACCENT_COLOR};
            color: {BG_COLOR}; transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        .stSelectbox>div>div, .stDateInput>div>div, .stTextInput>div>div, .stNumberInput>div>div {{
            background-color: {CARD_BG_COLOR} !important; color: {TEXT_COLOR} !important;
            border: 1px solid {BORDER_COLOR} !important; border-radius: 6px;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 5px; border-bottom: 2px solid {BORDER_COLOR}; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent; color: {TEXT_MUTED_COLOR};
            border-radius: 6px 6px 0px 0px; padding: 12px 22px;
            border: none; font-weight: 500; transition: all 0.2s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{ background-color: {CARD_BG_COLOR}; color: {PRIMARY_COLOR}; }}
        .stTabs [aria-selected="true"] {{
            background-color: {CARD_BG_COLOR}; color: {PRIMARY_COLOR};
            border-bottom: 3px solid {PRIMARY_COLOR};
        }}
        .sidebar .sidebar-content {{ background-color: {BG_COLOR}; border-right: 1px solid {BORDER_COLOR};}}
        .stProgress > div > div > div > div {{ background-color: {PRIMARY_COLOR}; }}
        .details-log {{
            background-color: {CARD_BG_COLOR}; border: 1px solid {BORDER_COLOR};
            border-radius: 5px; padding: 15px; height: 250px; overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace; font-size: 0.9em; color: {TEXT_MUTED_COLOR};
        }}
        .stMetricLabel {{ color: {TEXT_MUTED_COLOR}; font-size: 0.9rem; }}
        .stMetricValue {{ color: {TEXT_COLOR}; font-size: 1.8rem; font-weight: bold;}}
        .stExpander header {{ font-size: 1.1rem; font-weight: 500; color: {PRIMARY_COLOR}; }}
    </style>
    """, unsafe_allow_html=True)


class StockPredictionApp:
    def __init__(self):
        load_css()
        self.available_companies = self.load_available_companies()

        self.resnls_ensemble_models: List[Optional['ResNLS']] = []
        self.resnls_scaler: Optional['StandardScaler'] = None
        self.resnls_model_info: dict = {} # Changed from Dict to dict
        self.resnls_feature_columns: List[str] = []
        self.resnls_target_col: Optional[str] = None
        self.resnls_forecast_horizon: Optional[int] = None
        self.resnls_sequence_length: Optional[int] = None
        self.resnls_optimal_threshold: float = 0.5
        self.resnls_ensemble_loaded: bool = False

    def load_available_companies(self):
        top_10_companies_list = [
            {'ticker': 'AAPL', 'name': 'Apple Inc.'},
            {'ticker': 'MSFT', 'name': 'Microsoft Corp.'},
            {'ticker': 'GOOGL', 'name': 'Alphabet Inc. (A)'},
            {'ticker': 'GOOG', 'name': 'Alphabet Inc. (C)'},
            {'ticker': 'AMZN', 'name': 'Amazon.com, Inc.'},
            {'ticker': 'NVDA', 'name': 'NVIDIA Corp.'},
            {'ticker': 'META', 'name': 'Meta Platforms, Inc.'},
            {'ticker': 'TSLA', 'name': 'Tesla, Inc.'},
            {'ticker': 'LLY', 'name': 'Eli Lilly and Company'},
            {'ticker': 'V', 'name': 'Visa Inc.'}
        ]
        latest_file = None
        if os.path.exists(DATA_DIR):
            try:
                all_top = [f for f in os.listdir(DATA_DIR) if f.startswith('top_') and f.endswith('_companies.csv')]
                if all_top:
                    nums = []
                    for f_name in all_top:
                        try: parts = f_name.split('_'); num_str = parts[1]; nums.append(int(num_str))
                        except (IndexError, ValueError): continue
                    if nums: latest_file = os.path.join(DATA_DIR, f'top_{max(nums)}_companies.csv')
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
            print(f"Loaded {len(combined_df)} unique companies, prioritizing top 10.")
            return combined_df

        print(f"Using predefined list of {len(df_top_10)} top companies.")
        return df_top_10


    def load_resnls_ensemble_model(self):
        latest_info_file = None
        try:
            if os.path.exists(MODEL_DIR):
                candidate_info_files = [f for f in os.listdir(MODEL_DIR) if '_info.json' in f and f.startswith('resnls_ensemble_info_')]
                if candidate_info_files:
                    candidate_info_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)), reverse=True)
                    latest_info_file = candidate_info_files[0]
            else:
                st.warning(f"Model directory not found: {MODEL_DIR}")
                self.resnls_ensemble_loaded = False; return False

            if not latest_info_file:
                st.warning("No ResNLS Ensemble model info file found. Please train a model.")
                self.resnls_ensemble_loaded = False; return False

            info_path = os.path.join(MODEL_DIR, latest_info_file)
            with open(info_path, 'r') as f:
                self.resnls_model_info = json.load(f)

            self.resnls_forecast_horizon = self.resnls_model_info.get('forecast_horizon_days')
            ensemble_member_filenames = self.resnls_model_info.get('ensemble_member_filenames', [])
            scaler_filename_from_info = self.resnls_model_info.get('scaler_filename')
            self.resnls_feature_columns = self.resnls_model_info.get('feature_columns')
            self.resnls_target_col = self.resnls_model_info.get('target_variable')
            self.resnls_sequence_length = self.resnls_model_info.get('sequence_length')

            optimal_thresh_info = self.resnls_model_info.get('final_ensemble_evaluation', {}).get('metrics', {})
            self.resnls_optimal_threshold = optimal_thresh_info.get('optimal_threshold_on_test_set', 0.5)
            if self.resnls_optimal_threshold is None: self.resnls_optimal_threshold = 0.5


            if not all([self.resnls_forecast_horizon, ensemble_member_filenames, scaler_filename_from_info,
                        self.resnls_feature_columns, self.resnls_target_col, self.resnls_sequence_length]):
                st.error(f"Incomplete info in {latest_info_file}. Critical attributes missing.")
                self.resnls_ensemble_loaded = False; return False

            st.write(f"Loading ResNLS Ensemble artifacts for **{self.resnls_forecast_horizon}-Day Horizon** model (from `{latest_info_file}`)")

            scaler_path = os.path.join(MODEL_DIR, scaler_filename_from_info)
            if not os.path.exists(scaler_path):
                st.error(f"Scaler file not found: {scaler_path}"); self.resnls_ensemble_loaded = False; return False
            self.resnls_scaler = joblib.load(scaler_path)

            self.resnls_ensemble_models = []
            optuna_best_params = self.resnls_model_info.get('best_optuna_trial_summary', {}).get('params', {})
            if not optuna_best_params:
                st.error("Optuna best parameters not found in model info. Cannot reconstruct model structure."); self.resnls_ensemble_loaded = False; return False

            for member_fname in ensemble_member_filenames:
                model_path = os.path.join(MODEL_DIR, member_fname)
                if not os.path.exists(model_path):
                    st.error(f"Ensemble member file not found: {model_path}"); self.resnls_ensemble_loaded = False; return False

                try:
                    if ResNLS is None: raise ImportError("ResNLS class not available for model loading.")
                    model_instance = ResNLS(
                        num_features=len(self.resnls_feature_columns),
                        seq_len=self.resnls_sequence_length,
                        resnet_blocks=optuna_best_params.get("resnet_blocks", 2),
                        resnet_out_channels=optuna_best_params.get("resnet_out_channels", 64),
                        lstm_hidden_size=optuna_best_params.get("lstm_hidden_size", 128),
                        lstm_layers=optuna_best_params.get("lstm_layers", 2),
                        fc_hidden_size=optuna_best_params.get("fc_hidden_size", 64),
                        dropout_rate=optuna_best_params.get("dropout_rate", 0.3)
                    ).to(DEVICE)
                    model_instance.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    model_instance.eval()
                    self.resnls_ensemble_models.append(model_instance)
                except Exception as e_load_member:
                    st.error(f"Error loading ensemble member {member_fname}: {e_load_member}"); self.resnls_ensemble_loaded = False; return False

            if len(self.resnls_ensemble_models) != len(ensemble_member_filenames):
                 st.error(f"Mismatch in loaded ensemble members. Expected {len(ensemble_member_filenames)}, got {len(self.resnls_ensemble_models)}"); self.resnls_ensemble_loaded = False; return False


            st.success(f"ResNLS Ensemble Model ({self.resnls_forecast_horizon}d Horizon, {len(self.resnls_ensemble_models)} members) loaded successfully. Using prediction threshold: {self.resnls_optimal_threshold:.4f}")
            self.resnls_ensemble_loaded = True
            return True

        except Exception as e:
            st.error(f"Error loading ResNLS Ensemble model: {e}")
            traceback.print_exc()
            self.resnls_ensemble_loaded = False; return False


    def render_model_training_page(self):
        st.header("🧠 Model Training (ResNLS Ensemble - PyTorch)")

        if not MODEL_TRAINING_AVAILABLE or not run_optuna_optimization_resnls:
            st.error("Model training (ResNLS) unavailable. Check `model_training.py` import and PyTorch setup.")
            return

        processed_files = []
        if os.path.exists(PROCESSED_DATA_DIR):
            processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')]

        if not processed_files:
            st.warning("No processed data found in 'data/processed/'. Collect/process data first on the 'Data' tab.")
            return

        st.info("This page trains a ResNLS Ensemble model using PyTorch, advanced features, and Optuna hyperparameter optimization, based on the selected processed data files. This process can be very time-consuming, especially with many Optuna trials or ensemble members.")

        with st.expander("⚙️ Training Configuration (ResNLS)", expanded=True):
            col1_train, col2_train, col3_train = st.columns(3)
            with col1_train:
                forecast_horizon_train = st.slider("Forecast Horizon (days)", 1, 30, 5, key="train_resnls_horizon", help="Predict trend over the next N trading days.")
                target_threshold_train = st.slider("Target Definition Threshold (%)", 0.1, 5.0, 1.0, step=0.1, key="train_resnls_target_thresh", help="Percentage increase required for a positive target label.") / 100.0
            with col2_train:
                n_trials_optuna_train = st.slider("Optuna Trials", 10, 200, 25, key="train_resnls_optuna_trials", help="Number of hyperparameter sets Optuna will test. More trials can yield better models but take longer.")
                num_cv_splits_optuna_train = st.slider("Optuna CV Splits", 2, 5, 3, key="train_resnls_cv_splits", help="Number of walk-forward validation folds per Optuna trial.")
            with col3_train:
                num_ensemble_models_train = st.slider("Ensemble Members", 1, 10, 3, key="train_resnls_ensemble_members", help="Number of ResNLS models to train in the final ensemble.")
                final_model_epochs_train = st.slider("Final Model Epochs", 20, 150, 50, key="train_resnls_final_epochs", help="Max epochs for training each final ensemble member.")

            st.markdown("##### Select Processed Data Files for Training")
            st.caption("Select `_processed_data.csv` files. More diverse data (multiple tickers, longer periods) generally leads to more robust models.")

            default_selection_train = []
            top_10_tickers_train = [] # Initialize
            if hasattr(self, 'available_companies') and isinstance(self.available_companies, pd.DataFrame) and not self.available_companies.empty:
                company_dicts_for_default = self.available_companies.head(10).to_dict('records')
                top_10_tickers_train = [c['ticker'].upper() for c in company_dicts_for_default]


            if processed_files:
                for f_name_train in sorted(processed_files):
                    ticker_in_fname_train = f_name_train.split('_processed_data.csv')[0].upper()
                    if ticker_in_fname_train in top_10_tickers_train: # top_10_tickers_train is list of UPPERCASE strings
                        default_selection_train.append(f_name_train)
                if not default_selection_train and processed_files:
                    default_selection_train = sorted(processed_files)[:min(len(processed_files), 5)]

            selected_files_train = st.multiselect("Select Files", options=sorted(processed_files),
                                            format_func=lambda x: x.split('_processed_data.csv')[0],
                                            default=default_selection_train,
                                            key="train_resnls_files")
            can_train_ui = bool(selected_files_train)
            if not selected_files_train:
                st.warning("Please select at least one processed data file.")

            st.markdown(f"**Training will use device: `{DEVICE}`** (PyTorch). Ensure CUDA is available and PyTorch is built with GPU support for faster training if desired.")


        if st.button(f"🚀 Start ResNLS Ensemble Training ({forecast_horizon_train}-Day Horizon, {target_threshold_train*100:.1f}% Thr)", type="primary", use_container_width=True, disabled=not can_train_ui):
            progress_container_train = st.container()
            with progress_container_train:
                st.subheader("🏋️‍♂️ Training Progress (ResNLS Ensemble)")
                overall_progress_bar_train = st.progress(0, text="Initializing ResNLS Training...")
                status_area_train = st.empty()
                metrics_area_train = st.container()

            def training_status_callback_streamlit(message, is_error=False, indent_level=None):
                if is_error: status_area_train.error(f"{'  '* (indent_level or 0)}{message}")
                else: status_area_train.info(f"{'  '* (indent_level or 0)}{message}")

            def training_progress_callback_streamlit(progress_value, text_message):
                overall_progress_bar_train.progress(max(0.0, min(1.0, progress_value)), text=text_message)

            try:
                status_area_train.info("Starting ResNLS Ensemble training with Optuna... This will take a significant amount of time.")
                processed_file_paths_train = [os.path.join(PROCESSED_DATA_DIR, f) for f in selected_files_train]

                ensemble_model_paths_trained, scaler_trained, best_optuna_params_trained, \
                avg_optuna_metrics_trained, target_column_name_trained, \
                feature_names_selected_trained, sequence_length_trained = run_optuna_optimization_resnls(
                    processed_files=processed_file_paths_train,
                    forecast_horizon=forecast_horizon_train,
                    n_trials_optuna=n_trials_optuna_train,
                    num_cv_splits_optuna=num_cv_splits_optuna_train,
                    target_threshold=target_threshold_train,
                    num_ensemble_models=num_ensemble_models_train,
                    final_model_epochs=final_model_epochs_train,
                    status_callback=training_status_callback_streamlit,
                    progress_callback=training_progress_callback_streamlit
                )
                overall_progress_bar_train.progress(1.0, text="Training Pipeline Complete!")

                if ensemble_model_paths_trained and scaler_trained and best_optuna_params_trained:
                    status_area_train.success(f"🏆 ResNLS Ensemble Training complete! {len(ensemble_model_paths_trained)} members saved for {forecast_horizon_train}-Day model.")

                    with metrics_area_train:
                        st.subheader("📊 Best Optuna Trial - Average Validation Metrics (from CV)")
                        if avg_optuna_metrics_trained:
                            m_cols_train = st.columns(4)
                            m_cols_train[0].metric("Avg Accuracy", f"{avg_optuna_metrics_trained.get('accuracy', 0):.4f}")
                            m_cols_train[1].metric("Avg F1 (Positive)", f"{avg_optuna_metrics_trained.get('f1_positive', 0):.4f}")
                            m_cols_train[2].metric("Avg AUC", f"{avg_optuna_metrics_trained.get('roc_auc', 0):.4f}")
                            m_cols_train[3].metric("Avg Recall (Positive)", f"{avg_optuna_metrics_trained.get('recall_positive', 0):.4f}")
                        else:
                            st.info("Average Optuna validation metrics not available from training function.")

                        col_params_train, col_metrics_detail_train = st.columns(2)
                        with col_params_train:
                            st.markdown("##### Optimal Hyperparameters (Optuna Best Trial)")
                            st.json(best_optuna_params_trained, expanded=False)
                        with col_metrics_detail_train:
                            st.markdown("##### Detailed Avg Optuna Validation Metrics")
                            if avg_optuna_metrics_trained:
                                metrics_disp_train = {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k,v in avg_optuna_metrics_trained.items()}
                                st.dataframe(pd.DataFrame(metrics_disp_train.items(), columns=['Metric', 'Avg Value']), use_container_width=True, hide_index=True)

                    st.info("Reloading newly trained ResNLS Ensemble model for the application...")
                    self.load_resnls_ensemble_model()

                    with st.expander("📄 Saved Model Ensemble Summary (from Info JSON)", expanded=True):
                         try:
                            resnls_info_files = []
                            if os.path.exists(MODEL_DIR):
                                resnls_info_files = sorted(
                                    [f for f in os.listdir(MODEL_DIR) if f.startswith('resnls_ensemble_info_') and f.endswith('.json')],
                                    key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)),
                                    reverse=True
                                )
                            if resnls_info_files:
                                latest_model_info_path = os.path.join(MODEL_DIR, resnls_info_files[0])
                                with open(latest_model_info_path, 'r') as f_disp: model_info_disp = json.load(f_disp)
                                st.json(model_info_disp, expanded=False)

                                final_eval_metrics_disp = model_info_disp.get('final_ensemble_evaluation', {}).get('metrics', {})
                                eval_set_desc_disp = model_info_disp.get('final_ensemble_evaluation', {}).get('eval_set_description', 'N/A')
                                if final_eval_metrics_disp:
                                    st.markdown(f"##### Ensemble Metrics on Final Hold-Out Test Set")
                                    st.caption(f"Test Set: {eval_set_desc_disp}")
                                    st.caption(f"Threshold used for binary metrics on test set: {final_eval_metrics_disp.get('optimal_threshold_on_test_set', 0.5):.4f}")

                                    test_m_cols = st.columns(4)
                                    test_m_cols[0].metric("Accuracy", f"{final_eval_metrics_disp.get('accuracy',0):.4f}")
                                    test_m_cols[1].metric("F1 (Pos)", f"{final_eval_metrics_disp.get('f1_positive',0):.4f}")
                                    test_m_cols[2].metric("AUC", f"{final_eval_metrics_disp.get('roc_auc',0):.4f}")
                                    test_m_cols[3].metric("Recall (Pos)", f"{final_eval_metrics_disp.get('recall_positive',0):.4f}")
                            else:
                                st.warning("Could not find the specific ResNLS info JSON file for display. Model was saved, but its summary might not appear here.")
                         except Exception as e_info_disp: st.warning(f"Could not display detailed model info from JSON: {e_info_disp}")
                         st.markdown("--- \n### Next Steps\n- Go to the **🔮 Predict** tab to use the newly trained ResNLS ensemble model.")
                else:
                    status_area_train.error("ResNLS Ensemble training failed or did not produce a suitable model. Check console logs for details from `model_training.py`.")
            except Exception as e_train_app:
                if 'overall_progress_bar_train' in locals() : overall_progress_bar_train.empty()
                status_area_train.error(f"❌ Training Error in App (ResNLS): {e_train_app}"); st.code(traceback.format_exc())


    def render_sidebar(self):
        with st.sidebar:
            st.markdown(f"<h1 style='text-align: center; color: {PRIMARY_COLOR};'>📈 StockAI (ResNLS)</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 0.9em; color: {TEXT_MUTED_COLOR};'>v4.0.0 - ResNLS PyTorch</p>", unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("📌 Navigation")
            selected_mode = st.session_state.get('app_mode', 'Home')
            nav_buttons = {
                "Home": "🏠 Home", "Data Collection": "📊 Data",
                "Model Training": "🧠 Train", "Prediction": "🔮 Predict", "Settings": "⚙️ Settings"
            }
            for mode, label in nav_buttons.items():
                if st.button(label, key=f"{mode}_btn_resnls", use_container_width=True, type="secondary" if selected_mode != mode else "primary"):
                    if selected_mode != mode:
                        st.session_state.app_mode = mode
                        st.rerun()
            st.markdown("---")
            st.subheader("💻 System Status")

            model_status_text = "⚠️ No Trained ResNLS Model"; model_status_color = "error"
            resnls_model_info_files = []
            if os.path.exists(MODEL_DIR):
                resnls_model_info_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('resnls_ensemble_info_') and f.endswith('.json')]

            if resnls_model_info_files:
                resnls_model_info_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)), reverse=True)
                latest_model_info_file = resnls_model_info_files[0]
                try:
                    if self.resnls_ensemble_loaded and self.resnls_forecast_horizon is not None:
                        target_thresh_display = self.resnls_model_info.get('target_threshold_percent', 'N/A')
                        if isinstance(target_thresh_display, (float, int)): target_thresh_display = f"{target_thresh_display:.1f}%"
                        num_members = len(self.resnls_ensemble_models) if self.resnls_ensemble_models else self.resnls_model_info.get('ensemble_member_filenames', [])
                        if isinstance(num_members, list): num_members = len(num_members)
                        model_status_text = f"✅ Active ({self.resnls_forecast_horizon}d, {target_thresh_display}, {num_members} members)"
                    else:
                        parts = latest_model_info_file.split('_')
                        horizon_disp = "N/A"; thresh_disp = "N/A"
                        for p_idx, part_val in enumerate(parts):
                            if part_val.endswith('d') and part_val[:-1].isdigit(): horizon_disp = part_val[:-1]; break
                        for p_idx, part_val in enumerate(parts):
                             if part_val.startswith('thr') and part_val.endswith('pct'): thresh_disp = part_val[3:-3]; break
                        model_status_text = f"✅ Model Ready ({horizon_disp}d, {thresh_disp}%)"
                    model_status_color = "success"
                except Exception as e_parse_sidebar:
                    print(f"Sidebar model status parse error: {e_parse_sidebar}")
                    model_status_text = f"✅ {len(resnls_model_info_files)} ResNLS Models Available"
                    model_status_color = "success"

            st.markdown(f"""<div style="padding: 10px; background-color: {CARD_BG_COLOR}; border-radius: 6px; margin-bottom: 10px;"><span style="font-size: 0.9em; color: {TEXT_MUTED_COLOR};">Model Status</span><br><span style="font-weight: bold; color: {'#28a745' if model_status_color == 'success' else '#dc3545'};">{model_status_text}</span></div>""", unsafe_allow_html=True)

            processed_files_count = 0
            if os.path.exists(PROCESSED_DATA_DIR):
                processed_files_count = len([f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')])
            data_status_text = "⚠️ No Processed Data"; data_status_color = "error"
            if processed_files_count > 0: data_status_text = f"✅ {processed_files_count} Processed Files"; data_status_color = "success"
            st.markdown(f"""<div style="padding: 10px; background-color: {CARD_BG_COLOR}; border-radius: 6px;"><span style="font-size: 0.9em; color: {TEXT_MUTED_COLOR};">Data Status</span><br><span style="font-weight: bold; color: {'#28a745' if data_status_color == 'success' else '#dc3545'};">{data_status_text}</span></div>""", unsafe_allow_html=True)
            st.markdown("---")
            st.caption(f"<p style='text-align: center; color: {TEXT_MUTED_COLOR}'>StockAI ResNLS v4.0.0<br>Hope Project 2025</p>", unsafe_allow_html=True)


    def fetch_stock_data(self, ticker, start_date, end_date):
        try:
            yf_ticker = ticker.replace('.', '-');
            end_date_yf_str = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')

            df = None
            for attempt in range(3):
                try:
                    ticker_obj = yf.Ticker(yf_ticker);
                    df_hist = ticker_obj.history(start=start_date_str, end=end_date_yf_str, interval="1d", auto_adjust=False, actions=True)
                    if not df_hist.empty:
                        df = df_hist;
                        df = df[df.index <= pd.to_datetime(end_date).tz_localize(df.index.tzinfo if df.index.tz else None)]
                        break
                except requests.exceptions.ConnectionError as e_conn:
                    if attempt < 2: st.warning(f"Connection error fetching {ticker} (attempt {attempt+1}/3): {e_conn}. Retrying in 2s..."); time.sleep(2)
                    else: raise
                except Exception as e_fetch_inner:
                    st.error(f"yfinance error fetching {ticker} (attempt {attempt+1}): {e_fetch_inner}"); return None

            if df is None or df.empty :
                st.error(f"No data returned for {ticker} after retries ({start_date_str} to {end_date})."); return None

            df.reset_index(inplace=True)
            date_col_found = None
            for potential_date_col in ['Date', 'Datetime', 'Datetime (UTC)']:
                if potential_date_col in df.columns: date_col_found = potential_date_col; break

            if date_col_found:
                 df.rename(columns={date_col_found: 'Date'}, inplace=True)
                 df['Date'] = pd.to_datetime(df['Date'])
                 if df['Date'].dt.tz is not None: df['Date'] = df['Date'].dt.tz_localize(None)
                 df.set_index('Date', inplace=True)
            else:
                st.error(f"Date column not found in data for {ticker}. Columns: {df.columns.tolist()}"); return None

            df['Ticker'] = ticker; df.columns = df.columns.str.strip().str.replace(' ', '_')
            essential_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
            rename_map = {'Adj_Close': 'Adj Close'}
            df.rename(columns=rename_map, inplace=True)
            essential_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

            for ecol in essential_cols:
                if ecol not in df.columns:
                    df[ecol] = np.nan
                    st.warning(f"Column '{ecol}' was missing for {ticker} and added as NaN.")
            return df
        except Exception as e:
            st.error(f"Critical error fetching stock data for {ticker}: {e}"); traceback.print_exc(); return None

    def _prepare_data_for_prediction(self, df_historical_raw_orig: pd.DataFrame) -> Optional[pd.DataFrame]:
        ticker = df_historical_raw_orig['Ticker'].iloc[0] if 'Ticker' in df_historical_raw_orig.columns and not df_historical_raw_orig.empty else 'Unknown'
        st.write(f"--- Preparing features for ResNLS prediction ({ticker}) ---")

        df_historical_raw = df_historical_raw_orig.copy()

        if not isinstance(df_historical_raw.index, pd.DatetimeIndex):
            if 'Date' in df_historical_raw.columns:
                try:
                    df_historical_raw['Date'] = pd.to_datetime(df_historical_raw['Date'])
                    if df_historical_raw['Date'].dt.tz is not None: df_historical_raw['Date'] = df_historical_raw['Date'].dt.tz_localize(None)
                    df_historical_raw = df_historical_raw.set_index('Date')
                except Exception as e_idx:
                    st.error(f"Prediction feature calc: Error setting DatetimeIndex for {ticker}: {e_idx}. Cannot proceed.")
                    return None
            else:
                st.error(f"Prediction feature calc: DataFrame for {ticker} missing 'Date' column for index. Cannot proceed.")
                return None
        df_historical_raw = df_historical_raw.sort_index()

        df_current_features = df_historical_raw.copy()
        if DATA_COLLECTION_AVAILABLE and DataCollector is not None:
            dc_instance_for_pred = DataCollector()

            ohlcv_cols = [dc_instance_for_pred.COL_OPEN, dc_instance_for_pred.COL_HIGH,
                          dc_instance_for_pred.COL_LOW, dc_instance_for_pred.COL_CLOSE,
                          dc_instance_for_pred.COL_VOLUME, dc_instance_for_pred.COL_ADJ_CLOSE]
            for col_name in ohlcv_cols:
                if col_name not in df_current_features.columns:
                    df_current_features[col_name] = np.nan
                df_current_features[col_name] = pd.to_numeric(df_current_features[col_name], errors='coerce')
            df_current_features[ohlcv_cols] = df_current_features[ohlcv_cols].ffill().bfill()

            df_with_tas = dc_instance_for_pred._calculate_basic_technical_indicators(
                df_current_features.copy(), ticker=ticker
            )
            if df_with_tas is None or df_with_tas.empty:
                st.error(f"TA calculation failed during prediction preparation for {ticker}.")
                return None
            df_current_features = df_with_tas

            df_with_base_features = dc_instance_for_pred._engineer_base_features(
                df_current_features.copy(), ticker=ticker
            )
            if df_with_base_features is None or df_with_base_features.empty:
                st.error(f"Base feature engineering failed during prediction preparation for {ticker}.")
                return None
            df_current_features = df_with_base_features
        else:
            st.error("DataCollector module not available. Cannot prepare features accurately for prediction.")
            return None

        df_pred_feat = df_current_features.copy()
        if mt_engineer_advanced_features:
            df_pred_engineered = mt_engineer_advanced_features(df_pred_feat.copy())
        else:
            st.error("`engineer_advanced_features` from model_training not available. Cannot create advanced features for prediction.")
            return None

        if df_pred_engineered is None or df_pred_engineered.empty:
            st.error(f"Data for {ticker} became empty or None after advanced feature engineering. Cannot proceed."); return None

        if not self.resnls_scaler or not self.resnls_feature_columns:
            st.error("Scaler or feature list for ResNLS model not loaded. Cannot proceed.")
            return None

        for col_expected in self.resnls_feature_columns:
            if col_expected not in df_pred_engineered.columns:
                df_pred_engineered[col_expected] = 0.0
                st.caption(f"Feature '{col_expected}' (expected by model) not in current data, filled with 0.0 for prediction.")

        df_final_features = df_pred_engineered[self.resnls_feature_columns].copy()

        df_final_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_final_features = df_final_features.ffill().bfill().fillna(0)
        return df_final_features


    def calculate_technical_indicators(self, df):
        if df is None or df.empty: return df.copy() if df is not None else None
        df_indicators = df.copy()

        if not isinstance(df_indicators.index, pd.DatetimeIndex):
             if 'Date' in df_indicators.columns:
                 try:
                    df_indicators['Date'] = pd.to_datetime(df_indicators['Date'])
                    if df_indicators['Date'].dt.tz is not None: df_indicators['Date'] = df_indicators['Date'].dt.tz_localize(None)
                    df_indicators = df_indicators.set_index('Date')
                 except Exception as e_idx_ti: st.error(f"Display indicators: Error setting DatetimeIndex: {e_idx_ti}"); return None
             else: st.error("Display indicators: DataFrame has no 'Date' column for index."); return None
        df_indicators = df_indicators.sort_index()

        COL_OPEN_DISP, COL_HIGH_DISP, COL_LOW_DISP, COL_CLOSE_DISP, COL_VOLUME_DISP = 'Open', 'High', 'Low', 'Close', 'Volume'
        required_cols_disp = [COL_OPEN_DISP, COL_HIGH_DISP, COL_LOW_DISP, COL_CLOSE_DISP, COL_VOLUME_DISP]

        for col_disp in required_cols_disp:
            if col_disp not in df_indicators.columns:
                st.error(f"Display indicators: Column '{col_disp}' missing."); return None
            try:
                df_indicators[col_disp] = pd.to_numeric(df_indicators[col_disp], errors='coerce')
                df_indicators[col_disp] = df_indicators[col_disp].ffill().bfill()
                if df_indicators[col_disp].isnull().any(): df_indicators[col_disp] = df_indicators[col_disp].fillna(0.0)
            except Exception as e_disp_numeric: st.error(f"Display indicators: Error processing column '{col_disp}': {e_disp_numeric}"); return None
        if df_indicators[COL_CLOSE_DISP].isnull().all():
            st.error(f"Display indicators: '{COL_CLOSE_DISP}' column is all NaN. Cannot calculate display TAs."); return None

        cl_disp = df_indicators[COL_CLOSE_DISP].astype(float).values
        hi_disp = df_indicators[COL_HIGH_DISP].astype(float).values
        lo_disp = df_indicators[COL_LOW_DISP].astype(float).values

        if TALIB_AVAILABLE and len(cl_disp) >= 20:
            try:
                df_indicators['SMA_20'] = talib.SMA(cl_disp, 20); df_indicators['SMA_50'] = talib.SMA(cl_disp, 50)
                df_indicators['RSI_14'] = talib.RSI(cl_disp, 14)
                macd_disp, macdsignal_disp, macdhist_disp = talib.MACD(cl_disp, 12, 26, 9)
                df_indicators['MACD'], df_indicators['MACD_Signal'], df_indicators['MACD_Hist'] = macd_disp, macdsignal_disp, macdhist_disp
                upper_disp, middle_disp, lower_disp = talib.BBANDS(cl_disp, 20, 2, 2, 0)
                df_indicators['BB_Upper'], df_indicators['BB_Middle'], df_indicators['BB_Lower'] = upper_disp, middle_disp, lower_disp
            except Exception as e_talib_disp: st.error(f"Error calculating display TAs with TA-Lib: {e_talib_disp}")

        if not all(c_disp in df_indicators.columns for c_disp in ['SMA_20', 'RSI_14', 'MACD', 'BB_Middle']):
            st.warning("Using basic pandas calculations for display TAs (TA-Lib missing/errored or data insufficient).")
            close_s_disp = df_indicators[COL_CLOSE_DISP].astype(float)
            df_indicators['SMA_20'] = close_s_disp.rolling(20, min_periods=1).mean()
            df_indicators['SMA_50'] = close_s_disp.rolling(50, min_periods=1).mean()
            delta_disp = close_s_disp.diff(); gain_disp = delta_disp.where(delta_disp > 0, 0.0).rolling(14, min_periods=1).mean()
            loss_disp = (-delta_disp.where(delta_disp < 0, 0.0)).rolling(14, min_periods=1).mean()
            rs_disp = gain_disp / loss_disp.replace(0, np.nan); rs_disp.fillna(method='ffill', inplace=True)
            df_indicators['RSI_14'] = 100.0 - (100.0 / (1.0 + rs_disp)); df_indicators['RSI_14'].fillna(50.0, inplace=True); df_indicators['RSI_14'] = np.clip(df_indicators['RSI_14'], 0, 100)
            ema12_disp = close_s_disp.ewm(span=12, adjust=False, min_periods=1).mean(); ema26_disp = close_s_disp.ewm(span=26, adjust=False, min_periods=1).mean()
            df_indicators['MACD'] = ema12_disp - ema26_disp; df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            df_indicators['MACD_Hist'] = df_indicators['MACD'] - df_indicators['MACD_Signal']
            df_indicators['BB_Middle'] = df_indicators['SMA_20'] ; std_dev_disp = close_s_disp.rolling(20, min_periods=1).std().fillna(0)
            df_indicators['BB_Upper'] = df_indicators['BB_Middle'] + (std_dev_disp * 2); df_indicators['BB_Lower'] = df_indicators['BB_Middle'] - (std_dev_disp * 2)

        display_indicator_cols_final = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower']
        for col_final_disp in display_indicator_cols_final:
            if col_final_disp in df_indicators:
                df_indicators[col_final_disp] = df_indicators[col_final_disp].interpolate(method='linear').ffill().bfill()
                df_indicators[col_final_disp].fillna(50.0 if col_final_disp == 'RSI_14' else 0.0, inplace=True)
            else: df_indicators[col_final_disp] = 50.0 if col_final_disp == 'RSI_14' else 0.0
        return df_indicators

    def create_price_chart(self, df, ticker, prediction_df=None, is_classification=False):
        COL_OPEN_CHART, COL_HIGH_CHART, COL_LOW_CHART, COL_CLOSE_CHART, COL_VOLUME_CHART = 'Open', 'High', 'Low', 'Close', 'Volume'
        if df is None or df.empty:
            fig = go.Figure(); fig.update_layout(title=f"{ticker} - No Data Available", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
            return fig
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25], subplot_titles=(f"{ticker} Price Analysis", "Volume"))
        if not all(c in df.columns for c in [COL_OPEN_CHART, COL_HIGH_CHART, COL_LOW_CHART, COL_CLOSE_CHART]):
            st.error(f"Candlestick chart for {ticker} requires Open, High, Low, Close columns.")
            return go.Figure().update_layout(title=f"{ticker} - Missing OHLC Data", template="plotly_dark")
        fig.add_trace(go.Candlestick(x=df.index, open=df[COL_OPEN_CHART], high=df[COL_HIGH_CHART], low=df[COL_LOW_CHART], close=df[COL_CLOSE_CHART], name="Price", increasing_line_color=PRIMARY_COLOR, decreasing_line_color='#00BFFF'), row=1, col=1)
        if COL_VOLUME_CHART in df.columns: fig.add_trace(go.Bar(x=df.index, y=df[COL_VOLUME_CHART], name="Volume", marker_color=ACCENT_COLOR, opacity=0.7), row=2, col=1)
        if 'SMA_20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5)), row=1, col=1)
        if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='rgba(30, 144, 255, 0.8)', width=1.5)), row=1, col=1)
        if all(c in df.columns for c in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(173, 216, 230, 0.5)', width=1), showlegend=False, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(173, 216, 230, 0.5)', width=1), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)', name='Bollinger Bands', hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], line=dict(color='rgba(173, 216, 230, 0.6)', width=1, dash='dot'), name='BB Middle'), row=1, col=1)
        if prediction_df is not None and not prediction_df.empty and is_classification and 'Predicted_Class' in prediction_df.columns and 'Probability (Up)' in prediction_df.columns:
            forecast_horizon_days_chart = self.resnls_forecast_horizon if hasattr(self, 'resnls_forecast_horizon') and self.resnls_forecast_horizon else 5
            pred_start_date = prediction_df['Date'].min()
            pred_end_date = prediction_df['Date'].max()
            fig.add_vrect(x0=pred_start_date, x1=pred_end_date + pd.Timedelta(days=0.9),
                          fillcolor=f"rgba({int(ACCENT_COLOR[1:3], 16)}, {int(ACCENT_COLOR[3:5], 16)}, {int(ACCENT_COLOR[5:7], 16)}, 0.15)",
                          layer="below", line=dict(color=ACCENT_COLOR, width=1.5, dash="dash"),
                          annotation_text=f"<b>{forecast_horizon_days_chart}D Forecast</b>", annotation_position="top left",
                          annotation_font=dict(size=12, color=TEXT_MUTED_COLOR), row=1, col=1)
            marker_y_position = df[COL_CLOSE_CHART].iloc[-1]
            pred_class = prediction_df['Predicted_Class'].iloc[0]; pred_prob_up = prediction_df['Probability (Up)'].iloc[0]
            marker_symbol = 'triangle-up' if pred_class == 1 else 'triangle-down'; marker_color = '#2ECC71' if pred_class == 1 else '#E74C3C'
            trend_text = 'Up' if pred_class == 1 else 'Down'
            confidence_text = prediction_df['Confidence (%)'].iloc[0] if 'Confidence (%)' in prediction_df.columns else 'N/A'
            prob_up_display = f"{pred_prob_up:.2%}" if isinstance(pred_prob_up, (float, int)) else "N/A"
            conf_display = f"{confidence_text:.1f}%" if isinstance(confidence_text, (float, int)) else "N/A"
            hover_text_chart = f"<b>Predicted Trend: {trend_text}</b><br>Prob(Up): {prob_up_display}<br>Confidence: {conf_display}"
            marker_plot_date = pred_start_date + pd.Timedelta(days=int(forecast_horizon_days_chart/2))
            if marker_plot_date > pred_end_date : marker_plot_date = pred_end_date
            if marker_plot_date < pred_start_date : marker_plot_date = pred_start_date
            fig.add_trace(go.Scatter(x=[marker_plot_date], y=[marker_y_position], mode='markers+text', name=f"Forecast: {trend_text}",
                marker=dict(symbol=marker_symbol, size=18, color=marker_color, line=dict(width=1.5, color='white')),
                text=[f"<b>{trend_text}</b><br><span style='font-size:0.8em;'>{prob_up_display}</span>"],
                textposition="middle right" if pred_class == 1 else "middle left", textfont=dict(color=TEXT_COLOR, size=12),
                hovertext=hover_text_chart, hoverinfo='text', showlegend=True), row=1, col=1)
        fig.update_layout(title=None, template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, font=dict(color=TEXT_COLOR, family="Arial, sans-serif"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor=CARD_BG_COLOR, bordercolor=BORDER_COLOR,borderwidth=1, font=dict(size=10)), height=650, xaxis_rangeslider_visible=False, margin=dict(l=40, r=40, t=50, b=40))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=BORDER_COLOR, zeroline=False);
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=BORDER_COLOR, zeroline=False, row=1, col=1, title_text="Price ($)");
        fig.update_yaxes(showgrid=False, row=2, col=1, title_text="Volume")
        return fig

    def create_technical_indicators_chart(self, df):
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
                    st.markdown(f"""<div style="border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 15px; margin-bottom: 15px; background-color: {CARD_BG_COLOR};"><h6><a href="{article_disp.get('url', '#')}" target="_blank" style="color: {PRIMARY_COLOR}; text-decoration: none;">{article_disp.get('title', 'N/A')}</a></h6><small style="color: {TEXT_MUTED_COLOR};">{article_disp.get('source', 'N/A')} | <i>{article_disp.get('published_at')}</i></small></div>""", unsafe_allow_html=True)

    def run_data_collection(self, settings_dc):
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

        processed_tickers_res = None; representative_features_res = None
        with st.spinner("⚙️ Running data collection & processing pipeline... This may take some time."):
            try:
                if not DATA_COLLECTION_AVAILABLE or not DataCollector : status_callback_app_dc("DataCollector module not available.", True); return None, None
                collector = DataCollector();
                start_str = settings_dc['start_date'].strftime('%Y-%m-%d'); end_str = settings_dc['end_date'].strftime('%Y-%m-%d')
                selected_companies_dicts = settings_dc.get('selected_companies', [])
                if not selected_companies_dicts: status_callback_app_dc("No companies selected.", True); return None, None
                if settings_dc.get('use_reddit') and settings_dc.get('reddit_client_id') and settings_dc.get('reddit_client_secret'):
                    ua = settings_dc.get('reddit_user_agent', 'StockAIStreamlitApp/1.0 by User')
                    collector.setup_reddit_api(settings_dc['reddit_client_id'], settings_dc['reddit_client_secret'], ua, status_callback=status_callback_app_dc)
                status_callback_app_dc(f"Initialized for {len(selected_companies_dicts)} companies.")
                processed_tickers_res, representative_features_res = collector.run_full_pipeline(
                    companies_to_process=selected_companies_dicts, start_date_str=start_str, end_date_str=end_str,
                    use_market_indices=True, use_fred_data=settings_dc.get('use_macro_data', True),
                    use_reddit_sentiment=settings_dc.get('use_reddit', False) and collector.reddit_client is not None,
                    use_google_trends=settings_dc.get('use_google_trends', True),
                    progress_callback=progress_callback_app_dc, status_callback=status_callback_app_dc
                )
                progress_callback_app_dc(1.0, "All tasks complete.")
                if processed_tickers_res: status_callback_app_dc(f"Pipeline completed for {len(processed_tickers_res)} tickers!")
                else: status_callback_app_dc("Processing failed or no tickers processed.", True)
            except Exception as e_pipeline: status_callback_app_dc(f"Critical pipeline error in app.py: {e_pipeline}", True); st.code(traceback.format_exc())
            finally:
                if 'dc_log_messages' in st.session_state: del st.session_state['dc_log_messages']

        st.subheader("📊 Collection Summary"); stats_display = {}
        if 'selected_companies_dicts' in locals() and selected_companies_dicts : stats_display["Companies Requested"] = len(selected_companies_dicts)
        if processed_tickers_res: stats_display["Companies Processed"] = len(processed_tickers_res)
        if representative_features_res: stats_display["Features Generated"] = len(representative_features_res)
        if stats_display:
            num_stats = len(stats_display); stats_cols = st.columns(min(num_stats, 3)) if num_stats > 0 else []
            for i, (lbl, val) in enumerate(stats_display.items()): stats_cols[i % len(stats_cols)].metric(lbl, str(val))
        else: st.info("No collection statistics to display.")
        if representative_features_res:
            with st.expander("Representative Features List (from last processed file)", expanded=False): st.code(f"{', '.join(representative_features_res)}")
        return processed_tickers_res, representative_features_res

    def render_data_collection_page(self):
        st.header("📊 Data Collection & Processing"); tab1, tab2 = st.tabs(["⚙️ Settings & Run", "📂 Existing Data"])
        with tab1:
            st.subheader("Setup Pipeline"); col_co, col_date = st.columns(2)
            with col_co:
                st.markdown("##### Select Companies")
                company_map = {f"{r['ticker']} - {r['name']}": r for _, r in self.available_companies.iterrows()}
                company_options = list(company_map.keys())
                sel_method = st.radio("Selection Method:", ["Top N", "Custom"], horizontal=True, key="dc_sel_method_resnls")
                selected_company_dicts_ui = []
                if sel_method == "Top N":
                    num_to_sel = st.slider("Number of Companies", 1, len(company_options), min(10, len(company_options)), key="dc_num_co_resnls")
                    selected_company_dicts_ui = [company_map[d] for d in company_options[:num_to_sel]]
                    st.caption(f"Selected: {', '.join([d['ticker'] for d in selected_company_dicts_ui[:5]])}{'...' if len(selected_company_dicts_ui) > 5 else ''}")
                else:
                    default_custom = [opt for opt in company_options if any(top_ticker['ticker'] in opt for top_ticker in self.available_companies[:3].to_dict('records'))]
                    if not default_custom and company_options: default_custom = [company_options[0]]
                    selected_multi = st.multiselect("Select Companies", options=company_options, default=default_custom, key="dc_multi_co_resnls")
                    selected_company_dicts_ui = [company_map[d] for d in selected_multi]
                    custom_tickers_input = st.text_input("Add Tickers (comma-separated)", placeholder="e.g., BRK.A, JPM", key="dc_custom_ticker_resnls")
                    if custom_tickers_input:
                        custom_list = [t.strip().upper() for t in custom_tickers_input.split(',') if t.strip()]
                        for ticker_str in custom_list:
                            if not any(d_item['ticker'] == ticker_str for d_item in selected_company_dicts_ui): selected_company_dicts_ui.append({'ticker': ticker_str, 'name': ticker_str})
                seen_tickers_ui = set(); final_selected_companies_ui = []
                for d_item_ui in selected_company_dicts_ui:
                     if d_item_ui['ticker'] not in seen_tickers_ui: final_selected_companies_ui.append(d_item_ui); seen_tickers_ui.add(d_item_ui['ticker'])
            with col_date:
                st.markdown("##### Date Range")
                default_start = datetime.now() - timedelta(days=365*5 + 90)
                start_date_in = st.date_input("Start Date", default_start, min_value=datetime(2000,1,1), max_value=datetime.now()-timedelta(days=180), key="dc_start_resnls")
                end_date_in = st.date_input("End Date", datetime.now().date(), min_value=start_date_in + timedelta(days=365), max_value=datetime.now().date(), key="dc_end_resnls")
            st.markdown("##### Data Sources"); ds_cols = st.columns(4)
            ds_cols[0].checkbox("Stock Price & Volume", True, disabled=True, key="dc_src_stock_resnls")
            ds_cols[1].checkbox("Technical Indicators", True, disabled=True, key="dc_src_tech_resnls")
            src_reddit = ds_cols[2].checkbox("Reddit Sentiment", False, key="dc_src_reddit_resnls")
            if src_reddit and not (st.session_state.get('reddit_client_id') and st.session_state.get('reddit_client_secret')):
                st.warning("Reddit API credentials not set in Settings. Reddit data will be skipped.", icon="⚠️")
            src_google = ds_cols[3].checkbox("Google Trends", True, key="dc_src_google_resnls")
            src_macro = st.checkbox("Macro Data (FEDFUNDS, etc.)", True, key="dc_src_macro_resnls")
            if st.button("🚀 Start Collection & Processing", type="primary", use_container_width=True, key="dc_start_button_resnls"):
                if not final_selected_companies_ui: st.error("Please select at least one company.")
                elif end_date_in <= start_date_in : st.error("End Date must be after Start Date and range should be sufficient (e.g., >1 year).")
                else:
                    collection_settings = {'start_date': start_date_in, 'end_date': end_date_in, 'use_reddit': src_reddit,
                        'reddit_client_id': st.session_state.get('reddit_client_id', ''), 'reddit_client_secret': st.session_state.get('reddit_client_secret', ''),
                        'reddit_user_agent': st.session_state.get('reddit_user_agent', 'StockAIStreamlitApp/1.0'),
                        'use_google_trends': src_google, 'use_macro_data': src_macro, 'selected_companies': final_selected_companies_ui}
                    self.run_data_collection(collection_settings)
        with tab2:
            st.subheader("Preview Existing Data")
            processed_files_list = []
            if os.path.exists(PROCESSED_DATA_DIR):
                processed_files_list = sorted([f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')])
            if processed_files_list:
                st.info(f"Found {len(processed_files_list)} processed data files.")
                selected_file_preview = st.selectbox("Select file:", [""] + processed_files_list, format_func=lambda x: x.split('_processed_data.csv')[0] if x else "--- Select ---", key="dc_preview_sel_resnls")
                if selected_file_preview:
                    file_path_preview = os.path.join(PROCESSED_DATA_DIR, selected_file_preview)
                    ticker_preview = selected_file_preview.split('_processed_data.csv')[0]
                    st.markdown(f"#### Preview: `{ticker_preview}`")
                    try:
                        df_preview = pd.read_csv(file_path_preview, parse_dates=['Date'])
                        df_disp = df_preview.set_index('Date') if 'Date' in df_preview.columns and pd.api.types.is_datetime64_any_dtype(df_preview['Date']) else df_preview
                        c1,c2,c3 = st.columns(3); c1.metric("Rows",f"{len(df_disp):,}"); c2.metric("Cols",len(df_disp.columns))
                        dr_str = "N/A";
                        if isinstance(df_disp.index,pd.DatetimeIndex) and not df_disp.empty: dr_str=f"{df_disp.index.min():%Y-%m-%d} to {df_disp.index.max():%Y-%m-%d}"
                        c3.metric("Date Range",dr_str)
                        st.dataframe(df_disp.head().round(3))
                        st.markdown("##### Quick Plot")
                        num_cols = df_disp.select_dtypes(include=np.number).columns.tolist()
                        def_plot_col = 'Close' if 'Close' in num_cols else (num_cols[0] if num_cols else None)
                        if def_plot_col:
                            plot_col_sel = st.selectbox(f"Plot col for {ticker_preview}", num_cols, index=num_cols.index(def_plot_col) if def_plot_col in num_cols else 0, key=f"prev_plot_{ticker_preview.replace('.','_')}")
                            if plot_col_sel:
                                try:
                                    fig_prev = None
                                    if isinstance(df_disp.index, pd.DatetimeIndex): fig_prev = px.line(df_disp, y=plot_col_sel, title=f"{ticker_preview} - {plot_col_sel}")
                                    elif 'Date' in df_preview.columns and pd.api.types.is_datetime64_any_dtype(df_preview['Date']): fig_prev = px.line(df_preview, x='Date', y=plot_col_sel, title=f"{ticker_preview} - {plot_col_sel}")
                                    if fig_prev: fig_prev.update_layout(template="plotly_dark",plot_bgcolor=BG_COLOR,paper_bgcolor=BG_COLOR,font_color=TEXT_COLOR); st.plotly_chart(fig_prev, use_container_width=True)
                                    else: st.warning("Could not generate plot (Date index issue).")
                                except Exception as plot_err: st.warning(f"Plot error: {plot_err}")
                        else: st.info("No numeric columns to plot.")
                    except Exception as e_prev: st.error(f"Error previewing {selected_file_preview}: {e_prev}")
            else: st.info("No processed data found. Run collection on 'Settings & Run' tab.")

    def render_settings_page(self):
        st.header("⚙️ Settings & Management")
        with st.expander("ℹ️ System Info", expanded=False):
            c1,c2=st.columns(2);
            c1.subheader("App Details"); c1.markdown(f"**Version:** 4.0.0 (ResNLS)"); c1.markdown(f"**Data Dir:** `{DATA_DIR}`"); c1.markdown(f"**Model Dir:** `{MODEL_DIR}`")
            c2.subheader("Environment"); c2.markdown(f"**Python:** {platform.python_version()}"); c2.markdown(f"**OS:** {platform.system()} {platform.release()}");
            c2.markdown(f"**PyTorch:** {torch.__version__ if ResNLS else 'N/A'} ({DEVICE})"); c2.markdown(f"**TA-Lib:** {'✅ Yes' if TALIB_AVAILABLE else '❌ No'}")
        with st.expander("⚙️ App Preferences", expanded=True):
             st.subheader("Default Stock for Prediction Page"); st.caption("Set the stock that appears by default on the 'Prediction' page.")
             def_ticker_opts = [f"{r['ticker']} - {r['name']}" for _, r in self.available_companies.iterrows()]
             curr_def_ticker = st.session_state.get('default_ticker', None)
             def_idx = 0
             if curr_def_ticker and def_ticker_opts:
                 try: def_idx = [opt.split(' - ')[0] for opt in def_ticker_opts].index(curr_def_ticker)
                 except ValueError: def_idx = 0
             sel_def_opt = st.selectbox("Default Stock", options=def_ticker_opts, index=def_idx, key="settings_def_ticker_sel_resnls")
             if st.button("Save Default Stock", type="primary", key="settings_save_def_ticker_btn_resnls"):
                 if sel_def_opt: st.session_state['default_ticker'] = sel_def_opt.split(' - ')[0]; self._save_settings(); st.success(f"Default stock saved: {st.session_state['default_ticker']}")
        with st.expander("🔑 API Credentials (Optional)", expanded=False):
            st.subheader("Reddit API"); st.markdown("Required for 'Reddit Sentiment' in Data Collection. Obtain from [Reddit Apps](https://www.reddit.com/prefs/apps).")
            rid = st.text_input("Client ID", value=st.session_state.get('reddit_client_id', ''), type="password", key="settings_rid_resnls")
            rsecret = st.text_input("Client Secret", value=st.session_state.get('reddit_client_secret', ''), type="password", key="settings_rsecret_resnls")
            rua = st.text_input("User Agent", value=st.session_state.get('reddit_user_agent', 'StockAIStreamlitApp/1.0'), key="settings_rua_resnls")
            if st.button("Save API Credentials", key="settings_save_api_btn_resnls"):
                st.session_state['reddit_client_id'] = rid; st.session_state['reddit_client_secret'] = rsecret; st.session_state['reddit_user_agent'] = rua
                self._save_settings(); st.success("Reddit API credentials saved!")
        with st.expander("🧹 Data Management", expanded=False):
            st.subheader("Clear Cached Data"); st.warning("⚠️ This action is irreversible.", icon="❗")
            raw_count = 0; proc_count = 0; co_list_count = 0
            if os.path.exists(RAW_DATA_DIR):
                for sub_dir in os.listdir(RAW_DATA_DIR):
                    if os.path.isdir(os.path.join(RAW_DATA_DIR, sub_dir)): raw_count += len([item for item in os.listdir(os.path.join(RAW_DATA_DIR, sub_dir)) if item.endswith(('.csv','.json'))])
            if os.path.exists(PROCESSED_DATA_DIR): proc_count = len([f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')])
            if os.path.exists(DATA_DIR): co_list_count = len([f for f in os.listdir(DATA_DIR) if f.startswith('top_') and f.endswith('_companies.csv')])
            st.markdown(f"- Raw Data: `{raw_count}` files"); st.markdown(f"- Processed Data: `{proc_count}` files"); st.markdown(f"- Company Lists: `{co_list_count}` files")
            if raw_count > 0 or proc_count > 0 or co_list_count > 0:
                if st.button("🗑️ Delete ALL Data Files", type="primary", key="settings_del_data_btn_resnls"):
                    del_count = 0; err_msgs = []
                    for d_path, is_subdir_root in [(RAW_DATA_DIR, True), (PROCESSED_DATA_DIR, False), (DATA_DIR, False)]:
                        if os.path.exists(d_path):
                            for item_name in os.listdir(d_path):
                                item_path = os.path.join(d_path, item_name)
                                if is_subdir_root and os.path.isdir(item_path):
                                    for sub_item_name in os.listdir(item_path):
                                        try: os.remove(os.path.join(item_path, sub_item_name)); del_count +=1
                                        except Exception as e: err_msgs.append(f"Err del {os.path.join(item_name,sub_item_name)}: {e}")
                                elif not is_subdir_root and os.path.isfile(item_path) and (item_path.endswith('.csv') or item_path.endswith('.json')):
                                    if d_path == DATA_DIR and not (item_name.startswith('top_') and item_name.endswith('_companies.csv')): continue
                                    try: os.remove(item_path); del_count +=1
                                    except Exception as e: err_msgs.append(f"Err del {item_name}: {e}")
                    st.success(f"Deleted {del_count} data files!");
                    if err_msgs: st.warning(f"Some files not deleted: {err_msgs}")
                    st.rerun()
            else: st.info("No data files to delete.")
        with st.expander("🤖 Model Management", expanded=False):
            st.subheader("Clear Models & Artifacts"); st.warning("⚠️ This action is irreversible.", icon="❗")
            model_files_count = 0
            if os.path.exists(MODEL_DIR): model_files_count = len([f for f in os.listdir(MODEL_DIR) if f.startswith(('resnls_','.joblib','.json'))])
            st.markdown(f"- Model Artifacts: `{model_files_count}` files in `{os.path.basename(MODEL_DIR)}`.")
            if model_files_count > 0:
                if st.button("🗑️ Delete ALL Model Artifacts", type="primary", key="settings_del_models_btn_resnls"):
                    del_count_mod = 0; err_msgs_mod = []
                    if os.path.exists(MODEL_DIR):
                        for f_name_mod in os.listdir(MODEL_DIR):
                            if f_name_mod.startswith(('resnls_', 'lgbm_')) and (f_name_mod.endswith(('.pt', '.joblib', '.json'))):
                                try: os.remove(os.path.join(MODEL_DIR, f_name_mod)); del_count_mod += 1
                                except Exception as e_del_mod: err_msgs_mod.append(f"Model artifact '{f_name_mod}': {e_del_mod}")
                    st.success(f"Deleted {del_count_mod} model artifacts!");
                    if err_msgs_mod: st.warning(f"Some artifacts not deleted: {err_msgs_mod}")
                    self.resnls_ensemble_models = []; self.resnls_scaler = None; self.resnls_model_info = {}
                    self.resnls_feature_columns = []; self.resnls_target_col = None; self.resnls_forecast_horizon = None
                    self.resnls_sequence_length = None; self.resnls_optimal_threshold = 0.5; self.resnls_ensemble_loaded = False
                    st.rerun()
            else: st.info("No model artifacts to delete.")
        with st.expander("📖 About StockAI (ResNLS)", expanded=False):
            st.markdown("""**StockAI v4.0.0 - ResNLS Ensemble**
            This application uses a PyTorch-based ResNet+LSTM (ResNLS) ensemble model for stock trend prediction.
            It features automated data gathering, feature engineering, Optuna hyperparameter tuning, and interactive prediction.
            *Disclaimer:* Educational tool. Not financial advice. Markets are volatile. DYOR.
            *Hope Project 2025*""")

    def render_prediction_page(self):
        st.header("🔮 Stock Trend Prediction (ResNLS Ensemble - PyTorch)")

        if not self.resnls_ensemble_loaded or not self.resnls_ensemble_models or not self.resnls_scaler:
            st.info("ResNLS Ensemble model not loaded. Attempting to load the latest trained model...");
            if not self.load_resnls_ensemble_model():
                st.error("Failed to load ResNLS model. Please train a model on the '🧠 Train' tab or check the 'models' directory.");
                return

        try:
            with st.expander("ℹ️ Loaded ResNLS Ensemble Model Info", expanded=False):
                if self.resnls_model_info:
                    st.json(self.resnls_model_info, expanded=False)
                    optuna_metrics = self.resnls_model_info.get('best_optuna_trial_summary', {}).get('avg_validation_metrics_cv', {})
                    final_eval_metrics = self.resnls_model_info.get('final_ensemble_evaluation', {}).get('metrics', {})

                    pred_info_cols = st.columns(4)
                    pred_info_cols[0].metric("Horizon", f"{self.resnls_forecast_horizon} Days" if self.resnls_forecast_horizon else "N/A")
                    pred_info_cols[1].metric("Seq. Length", f"{self.resnls_sequence_length}" if self.resnls_sequence_length else "N/A")
                    pred_info_cols[2].metric("Features Used", len(self.resnls_feature_columns) if self.resnls_feature_columns else "N/A")
                    pred_info_cols[3].metric("Ensemble Size", len(self.resnls_ensemble_models) if self.resnls_ensemble_models else "N/A")

                    if optuna_metrics:
                        st.markdown("##### Avg Optuna CV Metrics (Best Trial, @0.5 Thresh)")
                        m_cols = st.columns(4)
                        m_cols[0].metric("Accuracy", f"{optuna_metrics.get('accuracy', 0):.4f}")
                        m_cols[1].metric("F1 (Pos)", f"{optuna_metrics.get('f1_positive', 0):.4f}")
                        m_cols[2].metric("AUC", f"{optuna_metrics.get('roc_auc', 0):.4f}")
                        m_cols[3].metric("Recall (Pos)", f"{optuna_metrics.get('recall_positive', 0):.4f}")

                    if final_eval_metrics and 'accuracy' in final_eval_metrics:
                        eval_desc = self.resnls_model_info.get('final_ensemble_evaluation', {}).get('eval_set_description', '')
                        opt_thresh_test = final_eval_metrics.get('optimal_threshold_on_test_set', self.resnls_optimal_threshold)
                        st.markdown(f"##### Ensemble Metrics on Final Test Set (Optimal Threshold: {opt_thresh_test:.4f})")
                        if eval_desc: st.caption(f"Test Set: {eval_desc}")
                        pm_cols = st.columns(4)
                        pm_cols[0].metric("Accuracy", f"{final_eval_metrics.get('accuracy',0):.4f}")
                        pm_cols[1].metric("F1 (Pos)", f"{final_eval_metrics.get('f1_positive',0):.4f}")
                        pm_cols[2].metric("AUC", f"{final_eval_metrics.get('roc_auc',0):.4f}")
                        pm_cols[3].metric("Recall (Pos)", f"{final_eval_metrics.get('recall_positive',0):.4f}")
                else: st.warning("No detailed ResNLS model information available.")

            st.subheader("📈 Make New Prediction"); input_col, news_col = st.columns([2, 1.5])
            with input_col:
                st.markdown("##### Select Stock & Data Period for Prediction")
                pred_ticker_options = [f"{r['ticker']} - {r['name']}" for _, r in self.available_companies.iterrows()]
                default_ticker_sym = st.session_state.get('default_ticker', pred_ticker_options[0].split(' - ')[0] if pred_ticker_options else 'AAPL')
                default_pred_idx = 0
                if default_ticker_sym and pred_ticker_options:
                    try: default_pred_idx = [opt.split(' - ')[0] for opt in pred_ticker_options].index(default_ticker_sym)
                    except ValueError: default_pred_idx = 0

                selected_ticker_opt = st.selectbox("Stock:", pred_ticker_options, index=default_pred_idx, key="pred_page_ticker_sel_resnls")
                ticker_to_predict = selected_ticker_opt.split(' - ')[0]

                required_hist_days = (self.resnls_sequence_length or 60) + 100
                today_date = datetime.now().date()
                default_fetch_start = today_date - timedelta(days=int(required_hist_days * 1.5))

                fetch_start_date_ui = st.date_input("Fetch History From:", value=default_fetch_start,
                    max_value=today_date - timedelta(days=(self.resnls_sequence_length or 60) + 10),
                    min_value=today_date - timedelta(days=365*10), key="pred_page_start_date_resnls",
                    help=f"At least ~{required_hist_days} days of history ending before today are recommended for feature calculation and forming one sequence.")
                fetch_end_date_ui = today_date

            if st.button(f"🚀 Predict {ticker_to_predict} Trend ({self.resnls_forecast_horizon}-Day, ResNLS)", type="primary", use_container_width=True):
                status_container = st.empty(); progress_bar = st.progress(0, text="Initializing prediction..."); results_container = st.container()
                try:
                    status_container.info(f"Fetching data for {ticker_to_predict} ({fetch_start_date_ui} to {fetch_end_date_ui})...");
                    hist_df_raw = self.fetch_stock_data(ticker_to_predict, fetch_start_date_ui, fetch_end_date_ui)

                    if hist_df_raw is None or hist_df_raw.empty:
                        status_container.error(f"Failed to fetch data for {ticker_to_predict}."); progress_bar.empty(); return

                    min_data_points_for_one_sequence = (self.resnls_sequence_length or 60) + 50
                    if len(hist_df_raw) < min_data_points_for_one_sequence:
                        status_container.error(f"Data too short: {len(hist_df_raw)} points, need at least ~{min_data_points_for_one_sequence} for features & one sequence. Adjust 'Fetch History From' date.");
                        progress_bar.empty(); return
                    progress_bar.progress(0.1, text="Data fetched.")

                    status_container.info("Preparing features for ResNLS prediction...");
                    df_features_for_scaling = self._prepare_data_for_prediction(hist_df_raw.copy())

                    if df_features_for_scaling is None or df_features_for_scaling.empty:
                        status_container.error(f"Feature preparation failed for {ticker_to_predict}."); progress_bar.empty(); return
                    progress_bar.progress(0.3, text="Features prepared.")

                    scaled_features_np = self.resnls_scaler.transform(df_features_for_scaling)
                    scaled_features_df = pd.DataFrame(scaled_features_np, columns=self.resnls_feature_columns, index=df_features_for_scaling.index)
                    progress_bar.progress(0.4, text="Features scaled.")

                    if len(scaled_features_df) < self.resnls_sequence_length:
                        status_container.error(f"Not enough data ({len(scaled_features_df)} rows) after feature prep to form a sequence of length {self.resnls_sequence_length}. Adjust date range.");
                        progress_bar.empty(); return

                    last_sequence_np = scaled_features_df.iloc[-self.resnls_sequence_length:].values
                    last_sequence_tensor = torch.tensor(last_sequence_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    progress_bar.progress(0.5, text="Input sequence created.")

                    status_container.info(f"Making prediction with {len(self.resnls_ensemble_models)} ResNLS ensemble members...");
                    ensemble_probas = []
                    with torch.no_grad():
                        for model_member in self.resnls_ensemble_models:
                            model_member.eval()
                            output = model_member(last_sequence_tensor)
                            proba = torch.sigmoid(output).item()
                            ensemble_probas.append(proba)

                    predicted_probability_up = np.mean(ensemble_probas)
                    predicted_class_label = int(predicted_probability_up >= self.resnls_optimal_threshold)
                    progress_bar.progress(0.8, text="Prediction generated.")

                    last_historical_date = scaled_features_df.index[-1]
                    forecast_period_dates = pd.bdate_range(start=last_historical_date + pd.Timedelta(days=1), periods=self.resnls_forecast_horizon)

                    prediction_results_df = pd.DataFrame({
                        'Date': forecast_period_dates,
                        'Predicted_Class': predicted_class_label,
                        'Predicted_Trend': 'Up' if predicted_class_label == 1 else 'Down',
                        'Probability (Up)': predicted_probability_up,
                        'Confidence (%)': abs(predicted_probability_up - 0.5) * 2 * 100
                    })

                    with results_container:
                        st.subheader(f"🎯 Prediction: Next {self.resnls_forecast_horizon} Trading Days for {ticker_to_predict}");
                        res_cols = st.columns(3)
                        trend_delta_txt = "Potential Uptrend" if prediction_results_df['Predicted_Trend'].iloc[0] == "Up" else "Potential Downtrend"
                        res_cols[0].metric("Predicted Trend", prediction_results_df['Predicted_Trend'].iloc[0], delta=trend_delta_txt, delta_color="normal" if prediction_results_df['Predicted_Trend'].iloc[0] == "Up" else "inverse")
                        res_cols[1].metric("Prob(Uptrend)", f"{prediction_results_df['Probability (Up)'].iloc[0]:.2%}")
                        res_cols[2].metric("Confidence", f"{prediction_results_df['Confidence (%)'].iloc[0]:.1f}%")
                        st.dataframe(prediction_results_df.style.format({'Date': '{:%Y-%m-%d}', 'Probability (Up)': '{:.2%}', 'Confidence (%)': '{:.1f}%'}), hide_index=True, use_container_width=True)

                        chart_hist_df = self.calculate_technical_indicators(hist_df_raw.iloc[-120:].copy())
                        if chart_hist_df is not None and not chart_hist_df.empty:
                            st.subheader(f"Chart for {ticker_to_predict}");
                            price_chart = self.create_price_chart(chart_hist_df, ticker_to_predict, prediction_results_df, is_classification=True)
                            st.plotly_chart(price_chart, use_container_width=True)
                            st.subheader(f"Technical Indicators for {ticker_to_predict}");
                            tech_chart = self.create_technical_indicators_chart(chart_hist_df)
                            st.plotly_chart(tech_chart, use_container_width=True)
                        else: st.warning("Could not generate charts as display indicator calculation failed.")
                        st.info("⚠️ Disclaimer: AI predictions are for informational and educational purposes only. Not financial advice.")

                    progress_bar.progress(1.0, text="Results Displayed!");
                    status_container.success(f"Prediction for {ticker_to_predict} using ResNLS Ensemble successful!")

                except Exception as e_pred_loop:
                    status_container.error(f"ResNLS Prediction process error: {e_pred_loop}"); st.code(traceback.format_exc());
                finally:
                    if 'progress_bar' in locals() and progress_bar: progress_bar.empty()

            with news_col:
                st.markdown(f"##### 📰 Info & News: {ticker_to_predict}");
                with st.spinner(f"Fetching news for {ticker_to_predict}..."):
                    news_data = self.fetch_news(ticker_to_predict, limit=5)
                self.display_news_section(news_data, st.container())

        except Exception as e_render_pred:
            st.error(f"An critical error occurred while rendering the ResNLS prediction page: {e_render_pred}")
            st.code(traceback.format_exc())
            if st.button("Return to Home Page"): st.session_state.app_mode = 'Home'; st.rerun()

    def render_home_page(self):
        st.title(f"📈 Welcome to StockAI (ResNLS Ensemble)"); st.markdown(f"#### Advanced Stock Trend Prediction (v4.0.0)"); st.markdown("---");
        st.markdown("Navigate using the sidebar to collect data, train ResNLS models, or make predictions.");

        with st.expander("About StockAI v4.0.0 (ResNLS)", expanded=False):
            st.markdown("""
            StockAI (ResNLS Edition) leverages a sophisticated ensemble of ResNet+LSTM models built with PyTorch
            to analyze historical stock data and attempt to predict future price trends. It features:
            - **Data Collection**: Gathers data from yfinance, FRED, Google Trends, and optionally Reddit.
            - **Feature Engineering**: Creates a rich set of technical indicators and market features.
            - **Model Training**: Employs PyTorch for ResNLS model definition, Optuna for hyperparameter tuning,
              walk-forward validation concepts, and ensemble methods for robustness.
            - **Prediction**: Offers trend forecasts with associated probabilities.

            **Remember**: Financial markets are complex. Use this tool as part of a broader research strategy.
            This is an educational tool and not financial advice.
            """)

        st.subheader("📌 Quick Access"); home_cols = st.columns(3)
        if home_cols[0].button("📊 Data Collection", use_container_width=True, key="home_data_btn_resnls"): st.session_state.app_mode = "Data Collection"; st.rerun()
        if home_cols[1].button("🧠 Model Training", use_container_width=True, key="home_train_btn_resnls"): st.session_state.app_mode = "Model Training"; st.rerun()
        if home_cols[2].button("🔮 Prediction", use_container_width=True, key="home_predict_btn_resnls"): st.session_state.app_mode = "Prediction"; st.rerun()

        st.subheader("⚙️ System Status Summary"); status_cols = st.columns(2)
        proc_files_count_home = 0
        if os.path.exists(PROCESSED_DATA_DIR): proc_files_count_home = len([f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')])
        data_msg_home = f"✅ {proc_files_count_home} processed files" if proc_files_count_home > 0 else "⚠️ No processed data"
        status_cols[0].metric("Processed Data Status", data_msg_home)

        model_msg_home = "⚠️ No ResNLS model loaded/found"
        if self.resnls_ensemble_loaded and self.resnls_ensemble_models:
            thresh_disp = self.resnls_model_info.get('target_threshold_percent', 'N/A')
            if isinstance(thresh_disp, (float,int)): thresh_disp = f"{thresh_disp:.1f}%"
            num_m = len(self.resnls_ensemble_models)
            model_msg_home = f"✅ ResNLS Loaded ({self.resnls_forecast_horizon}d, Thr: {thresh_disp}, {num_m} members)"
        elif os.path.exists(MODEL_DIR) and any(f.startswith('resnls_ensemble_info_') and f.endswith('.json') for f in os.listdir(MODEL_DIR)):
            resnls_infos = [f for f in os.listdir(MODEL_DIR) if f.startswith('resnls_ensemble_info_') and f.endswith('.json')]
            if resnls_infos:
                 resnls_infos.sort(key=lambda f_sort: os.path.getmtime(os.path.join(MODEL_DIR, f_sort)), reverse=True)
                 latest_m_file = resnls_infos[0]
                 m_parts = latest_m_file.split('_');
                 m_hor = next((p for p in m_parts if p.endswith('d') and p[:-1].isdigit()), None)
                 m_thr = next((p_thr for p_thr in m_parts if p_thr.startswith('thr') and p_thr.endswith('pct')), None)
                 hor_disp = m_hor.replace('d', '') if m_hor else "N/A"
                 thr_disp = m_thr.replace('thr','').replace('pct','') if m_thr else "N/A"
                 model_msg_home = f"ℹ️ ResNLS available ({hor_disp}d, Thr: {thr_disp}%) - Load on Predict page."
        status_cols[1].metric("Trained Model Status", model_msg_home)
        st.markdown("---"); st.info("⚠️ Disclaimer: This tool is for educational and informational purposes only. It does not constitute financial advice.")

    def run(self):
        if 'app_mode' not in st.session_state: st.session_state.app_mode = 'Home'
        self._load_settings()
        self.render_sidebar()
        app_mode = st.session_state.get('app_mode', 'Home')
        page_render_map = {"Home": self.render_home_page, "Data Collection": self.render_data_collection_page,
                           "Model Training": self.render_model_training_page, "Prediction": self.render_prediction_page,
                           "Settings": self.render_settings_page}
        render_func = page_render_map.get(app_mode, self.render_home_page)
        render_func()

    def _load_settings(self):
        settings_path = os.path.join(ROOT_DIR,'app_settings.json')
        defaults = {'reddit_client_id': '', 'reddit_client_secret': '', 'reddit_user_agent': 'StockAIStreamlitApp/1.0', 'default_ticker': 'AAPL' }
        if os.path.exists(settings_path):
            try:
                 with open(settings_path, 'r') as f: loaded = json.load(f)
                 for k, dv in defaults.items(): st.session_state[k] = loaded.get(k, dv)
            except Exception as e: print(f"Error loading settings: {e}. Using defaults."); [st.session_state.setdefault(k,dv) for k,dv in defaults.items()]
        else: [st.session_state.setdefault(k,dv) for k,dv in defaults.items()]

    def _save_settings(self):
        settings_path = os.path.join(ROOT_DIR,'app_settings.json')
        to_save = {k: st.session_state.get(k) for k in ['default_ticker', 'reddit_client_id', 'reddit_client_secret', 'reddit_user_agent']}
        to_save_clean = {k: v for k, v in to_save.items() if v is not None}
        try:
            with open(settings_path, 'w') as f: json.dump(to_save_clean, f, indent=4)
        except Exception as e: st.warning(f"Failed to save app settings: {e}")

    def run_data_collection_cli(self):
        print("--- CLI: Data Collection & Processing (ResNLS context) ---")
        st.warning("CLI data collection for ResNLS not fully implemented in this version snippet.")
        if DATA_COLLECTION_AVAILABLE and DataCollector:
            print("DataCollector is available. Basic CLI run might be possible if DataCollector itself is robust.")
        return False

    def run_model_training_cli(self):
        print("--- CLI: Model Training (ResNLS Ensemble - PyTorch) ---")
        st.warning("CLI model training for ResNLS not fully implemented in this version snippet.")
        if MODEL_TRAINING_AVAILABLE and run_optuna_optimization_resnls:
            print("run_optuna_optimization_resnls is available. A CLI wrapper could be built.")
        return False

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StockAI System v4.0.0 (ResNLS)')
    parser.add_argument('--mode', type=str, default='streamlit', choices=['streamlit', 'cli'], help="Operation mode: 'streamlit' (default) or 'cli'.")
    parser.add_argument('--cli-action', type=str, default='all', choices=['data_collection', 'model_training', 'all'], help="Action in CLI: 'data_collection', 'model_training', or 'all'.")
    args = parser.parse_args()

    if args.mode == 'streamlit':
        app_instance = StockPredictionApp()
        app_instance.run()
    elif args.mode == 'cli':
        print(f"--- StockAI (ResNLS): CLI Mode (Action: {args.cli_action}) ---")
        cli_app_instance = StockPredictionApp()
        if args.cli_action == 'data_collection':
            cli_app_instance.run_data_collection_cli()
        elif args.cli_action == 'model_training':
            cli_app_instance.run_model_training_cli()
        elif args.cli_action == 'all':
            print("\n>>> Running CLI: Data Collection Phase <<<");
            data_success = cli_app_instance.run_data_collection_cli()
            if data_success:
                print("\n>>> Running CLI: Model Training Phase (ResNLS) <<<");
                cli_app_instance.run_model_training_cli()
            else: print("\nSkipping model training due to data collection failure in CLI 'all' mode.")
        print("--- StockAI (ResNLS): CLI Mode Finished ---")