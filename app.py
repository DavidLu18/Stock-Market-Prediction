"""
Stock Price Prediction System - Web Application
This module implements a Streamlit-based web interface for the stock price prediction system
with a black and dark pink theme.
"""

import os
import sys
import platform
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px  # Added this import
from plotly.subplots import make_subplots
import torch
import json
import joblib
import argparse
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time
from PIL import Image
import matplotlib.pyplot as plt
import io
import traceback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from scipy import stats # For confidence interval calculation

# Set Streamlit page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Stock Price Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define root and data directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import using relative paths (assuming app.py is run from within StockPrediction or its parent)
# Direct top-level imports (assuming sibling modules)
from data_collection import DataCollector
# Import the grid search function and ModelTrainer (still needed for predict)
from model_training import RecurrentAttentionModel, ModelTrainer, run_optuna_optimization # Import Optuna function


# Directory setup using constants from data_collection if possible, otherwise define here
# Define paths relative to the script location
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# Ensure directories exist (moved creation here for clarity)
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            # Use st.error only if Streamlit context is available
            if 'streamlit' in sys.modules:
                st.error(f"Error creating directory {dir_path}: {e}")
            else:
                print(f"Error creating directory {dir_path}: {e}")


# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom theme colors
PRIMARY_COLOR = "#ff1493"  # Dark pink
BG_COLOR = "#0e1117"       # Black
TEXT_COLOR = "#ffffff"     # White
SECONDARY_COLOR = "#ff69b4"  # Lighter pink

# Custom CSS to style the app
def load_css():
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff1493;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff69b4;
        color: white;
    }
    .stSelectbox>div>div {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #ff1493;
    }
    .stDateInput>div>div {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #ff1493;
    }
    .css-145kmo2 { /* Assuming this targets the main content area border */
        border: 2px solid #ff1493;
    }
    .css-l6i7oy { /* Assuming this targets some background element */
        background-color: #1e1e1e;
    }
    .sidebar .sidebar-content {
        background-color: #0e1117;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ff1493;
    }
    .stProgress > div > div > div > div {
        background-color: #ff1493;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        color: white;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff1493;
        color: white;
    }
    /* Style the details area for logs */
    .details-log {
        background-color: #1e1e1e;
        border: 1px solid #444;
        border-radius: 5px;
        padding: 10px;
        height: 200px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.9em;
        color: #ccc;
    }
    </style>
    """, unsafe_allow_html=True)

class StockPredictionApp:
    def __init__(self):
        self.setup_page()
        # Load the full list initially for selection purposes
        self.available_companies = self.load_available_companies()
        self.model_trainer = None # Will hold the ModelTrainer instance after loading
        self.model_loaded = False

    def setup_page(self):
        """Setup the page configuration and theme"""
        load_css()

    def load_available_companies(self, default_num=50):
        """Load the list of available companies from the latest top_N file or default."""
        # Find the latest 'top_X_companies.csv' file
        latest_file = None
        latest_num = 0
        if os.path.exists(DATA_DIR):
            try:
                all_top_files = [f for f in os.listdir(DATA_DIR) if f.startswith('top_') and f.endswith('_companies.csv')]
                if all_top_files:
                    # Extract numbers and find the max
                    nums = []
                    for f in all_top_files:
                        try: nums.append(int(f.split('_')[1]))
                        except: pass
                    if nums:
                        latest_num = max(nums)
                        latest_file = os.path.join(DATA_DIR, f'top_{latest_num}_companies.csv')
            except Exception as e:
                st.warning(f"Error finding latest companies file: {e}")

        companies_path_to_load = latest_file

        if companies_path_to_load and os.path.exists(companies_path_to_load):
            try:
                df = pd.read_csv(companies_path_to_load)
                # Standardize columns: expect 'ticker', 'name' from DataCollector now
                if 'ticker' not in df.columns and 'Symbol' in df.columns:
                     df.rename(columns={'Symbol': 'ticker'}, inplace=True)
                if 'name' not in df.columns and 'Name' in df.columns:
                     df.rename(columns={'Name': 'name'}, inplace=True)

                if 'ticker' not in df.columns or 'name' not in df.columns:
                     st.warning(f"CSV {os.path.basename(companies_path_to_load)} missing 'ticker' or 'name' column. Attempting fallback.")
                     raise ValueError("Missing required columns in CSV")
                st.info(f"Loaded {len(df)} companies from {os.path.basename(companies_path_to_load)}")
                # Return DataFrame with 'ticker' and 'name'
                return df[['ticker', 'name']]
            except Exception as read_err:
                 st.error(f"Error reading {os.path.basename(companies_path_to_load)}: {read_err}")

        # Fallback default list if loading fails or no file exists (Expanded List)
        st.warning("Using default company list (approx. 50).")
        default_companies = [
            {'ticker': 'AAPL', 'name': 'Apple Inc.'}, {'ticker': 'MSFT', 'name': 'Microsoft Corporation'},
            {'ticker': 'GOOGL', 'name': 'Alphabet Inc. (Class A)'}, {'ticker': 'GOOG', 'name': 'Alphabet Inc. (Class C)'},
            {'ticker': 'AMZN', 'name': 'Amazon.com, Inc.'}, {'ticker': 'NVDA', 'name': 'NVIDIA Corporation'},
            {'ticker': 'META', 'name': 'Meta Platforms, Inc.'}, {'ticker': 'TSLA', 'name': 'Tesla, Inc.'},
            {'ticker': 'LLY', 'name': 'Eli Lilly and Company'}, {'ticker': 'V', 'name': 'Visa Inc.'},
            {'ticker': 'JPM', 'name': 'JPMorgan Chase & Co.'}, {'ticker': 'WMT', 'name': 'Walmart Inc.'},
            {'ticker': 'XOM', 'name': 'Exxon Mobil Corporation'}, {'ticker': 'UNH', 'name': 'UnitedHealth Group Incorporated'},
            {'ticker': 'MA', 'name': 'Mastercard Incorporated'}, {'ticker': 'JNJ', 'name': 'Johnson & Johnson'},
            {'ticker': 'PG', 'name': 'Procter & Gamble Company'}, {'ticker': 'AVGO', 'name': 'Broadcom Inc.'},
            {'ticker': 'HD', 'name': 'The Home Depot, Inc.'}, {'ticker': 'ORCL', 'name': 'Oracle Corporation'},
            {'ticker': 'MRK', 'name': 'Merck & Co., Inc.'}, {'ticker': 'CVX', 'name': 'Chevron Corporation'},
            {'ticker': 'ABBV', 'name': 'AbbVie Inc.'}, {'ticker': 'KO', 'name': 'The Coca-Cola Company'},
            {'ticker': 'PEP', 'name': 'PepsiCo, Inc.'}, {'ticker': 'COST', 'name': 'Costco Wholesale Corporation'},
            {'ticker': 'ADBE', 'name': 'Adobe Inc.'}, {'ticker': 'BAC', 'name': 'Bank of America Corporation'},
            {'ticker': 'CRM', 'name': 'Salesforce, Inc.'}, {'ticker': 'MCD', 'name': "McDonald's Corporation"},
            {'ticker': 'CSCO', 'name': 'Cisco Systems, Inc.'}, {'ticker': 'TMO', 'name': 'Thermo Fisher Scientific Inc.'},
            {'ticker': 'ACN', 'name': 'Accenture plc'}, {'ticker': 'ABT', 'name': 'Abbott Laboratories'},
            {'ticker': 'NFLX', 'name': 'Netflix, Inc.'}, {'ticker': 'LIN', 'name': 'Linde plc'},
            {'ticker': 'AMD', 'name': 'Advanced Micro Devices, Inc.'}, {'ticker': 'DIS', 'name': 'The Walt Disney Company'},
            {'ticker': 'PFE', 'name': 'Pfizer Inc.'}, {'ticker': 'WFC', 'name': 'Wells Fargo & Company'},
            {'ticker': 'CMCSA', 'name': 'Comcast Corporation'}, {'ticker': 'TXN', 'name': 'Texas Instruments Incorporated'},
            {'ticker': 'VZ', 'name': 'Verizon Communications Inc.'}, {'ticker': 'DHR', 'name': 'Danaher Corporation'},
            {'ticker': 'INTC', 'name': 'Intel Corporation'}, {'ticker': 'UPS', 'name': 'United Parcel Service, Inc.'},
            {'ticker': 'PM', 'name': 'Philip Morris International Inc.'}, {'ticker': 'NEE', 'name': 'NextEra Energy, Inc.'},
            {'ticker': 'MS', 'name': 'Morgan Stanley'}, {'ticker': 'RTX', 'name': 'RTX Corporation'}
        ]
        return pd.DataFrame(default_companies, columns=['ticker', 'name'])

    def load_model(self):
        """Load the trained model and scaler"""
        try:
            model_info_path = os.path.join(MODEL_DIR, 'model_info.json')
            model_path = os.path.join(MODEL_DIR, 'stock_prediction_model.pth')
            scalers_path = os.path.join(MODEL_DIR, 'scalers.pkl')

            if not os.path.exists(model_info_path) or not os.path.exists(model_path) or not os.path.exists(scalers_path):
                st.warning("No trained model found or artifacts missing. Please train a model first.")
                self.model_loaded = False
                return False

            # Load model info
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)

            feature_columns = model_info['feature_columns']
            lookback_window = model_info['lookback_window']

            # Load scaler dictionary (should contain scalers per ticker or a combined one)
            scalers_dict = joblib.load(scalers_path)
            if not isinstance(scalers_dict, dict):
                 st.error("Error loading scalers: scalers.pkl is not a dictionary.")
                 self.model_loaded = False
                 return False

            # Create model trainer instance and assign loaded components
            # Note: ModelTrainer now expects scalers dict, not single scaler
            trainer = ModelTrainer(lookback_window=lookback_window)
            trainer.feature_columns = feature_columns
            trainer.scalers = scalers_dict # Assign the loaded dictionary

            # Create model architecture based on saved info
            input_dim = len(feature_columns)
            # Get parameters from 'best_params' if available (Optuna), else from top level
            params_source = model_info.get('best_params', model_info)
            hidden_dim = params_source.get('hidden_dim', 128)
            num_layers = params_source.get('num_layers', 2)
            dropout_prob = params_source.get('dropout_prob', 0.3) # Use correct key
            bidirectional = params_source.get('bidirectional', False)
            rnn_type = model_info.get('rnn_type', 'lstm') # rnn_type might be top-level
            output_dim = 1

            # Instantiate RecurrentAttentionModel with loaded parameters
            model = RecurrentAttentionModel( # Use the correct class name
                input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                output_dim=output_dim, lookback_window=lookback_window,
                rnn_type=rnn_type, # Pass loaded rnn_type
                dropout_prob=dropout_prob, bidirectional=bidirectional
            ).to(device)
            # Load model weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # Set model in trainer
            trainer.model = model

            self.model_trainer = trainer # Store the trainer instance
            self.model_loaded = True
            st.success("Model loaded successfully.")
            return True

        except Exception as e:
            st.error(f"Error loading model: {e}")
            traceback.print_exc()
            self.model_loaded = False
            return False

    def render_model_training_page(self):
        """Render the model training page"""
        st.title("🧠 Model Training")

        # Look for processed data files in the PROCESSED_DATA_DIR
        processed_files = []
        if os.path.exists(PROCESSED_DATA_DIR):
             processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')]

        if not processed_files:
            st.warning("No processed data files found in 'data/processed/'. Please collect and process data first.")
            st.info("Go to the 'Data Collection' tab to gather and process stock data.")
            return

        col1, col2 = st.columns([2,1]) # Adjust column width
        with col1:
            # Allow selecting multiple files for training
            selected_files = st.multiselect("Select Processed Data File(s) for Training",
                                            options=processed_files,
                                            format_func=lambda x: x.split('_processed_data.csv')[0], # Show ticker name
                                            default=processed_files if processed_files else None) # Default to all

            if not selected_files:
                st.warning("Please select at least one processed data file.")
                return

            target_options = ['Close_Next', 'Price_Change', 'Trend']
            target_var = st.radio("Select Target Variable", target_options, index=0, help="Target variable for the model to predict.")
            is_classification = target_var == 'Trend'
            model_type = "Classification" if is_classification else "Regression"
            st.info(f"Model Type: {model_type}")
            st.subheader("Training Settings") # Renamed subheader
            lookback_window = st.slider("Lookback Window (days)", 5, 60, 30, help="Number of previous days' data to use for predicting the next day.")
            epochs = st.slider("Training Epochs (per trial)", 10, 200, 50, help="Number of times the model sees the training data in each Optuna trial.") # Adjusted label/default
            num_splits = st.slider("Walk-Forward Splits (per trial)", 2, 10, 3, help="Number of validation folds within each Optuna trial.") # Adjusted default
            n_trials = st.slider("Optuna Optimization Trials", 5, 100, 20, help="Number of hyperparameter combinations Optuna will test.") # Added n_trials

        with col2:
            st.subheader("Hardware") # Renamed subheader
            use_gpu = st.checkbox("Use GPU", torch.cuda.is_available(), disabled=not torch.cuda.is_available())
            if torch.cuda.is_available(): st.success(f"GPU: {torch.cuda.get_device_name(0)}")
            elif use_gpu: st.warning("GPU not available. Using CPU.")
            # Removed Optuna-controlled hyperparameter widgets (hidden_dim, num_layers, dropout, bidirectional, rnn_type, batch_size, learning_rate)
            # RNN Type selection is now handled within Optuna search space in model_training.py
            st.subheader("Feature Selection")
            # Load and combine data from selected files to determine available features
            combined_df_for_features = pd.DataFrame()
            try:
                dfs_to_combine = []
                for file in selected_files:
                    file_path = os.path.join(PROCESSED_DATA_DIR, file)
                    try:
                        df_single = pd.read_csv(file_path, parse_dates=['Date']) # Ensure Date is parsed
                        dfs_to_combine.append(df_single)
                    except Exception as e:
                        st.error(f"Error loading {file}: {e}")
                if dfs_to_combine:
                    combined_df_for_features = pd.concat(dfs_to_combine, ignore_index=True)

                if not combined_df_for_features.empty:
                    # Determine features from the combined dataframe
                    all_features = [col for col in combined_df_for_features.columns if col not in ['Date', 'Ticker', 'Company', 'Close_Next', 'Price_Change', 'Trend']]
                    price_features = [col for col in all_features if any(p in col for p in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])]
                    technical_features = [col for col in all_features if col not in price_features and not col.startswith(('Compound', 'Positive', 'Neutral', 'Negative', 'Count', 'Interest'))] # Removed Reddit/Google prefixes
                    sentiment_features = [col for col in all_features if col in ['Compound', 'Positive', 'Neutral', 'Negative', 'Count']]
                    trends_features = [col for col in all_features if col == 'Interest']

                    st.write(f"Total features: {len(all_features)}, Price: {len(price_features)}, Technical: {len(technical_features)}, Sentiment: {len(sentiment_features)}, Trends: {len(trends_features)}")
                    use_price = st.checkbox("Use Price Features", True)
                    use_tech = st.checkbox("Use Technical Indicators", True)
                    use_sent = st.checkbox("Use Sentiment Features", True if sentiment_features else False, disabled=not sentiment_features)
                    use_trends = st.checkbox("Use Google Trends Features", True if trends_features else False, disabled=not trends_features)

                    selected_features = []
                    if use_price: selected_features.extend(price_features)
                    if use_tech: selected_features.extend(technical_features)
                    if use_sent: selected_features.extend(sentiment_features)
                    if use_trends: selected_features.extend(trends_features)

                    selected_features = sorted(list(set(selected_features))) # Ensure unique and sorted
                    st.write(f"Selected features ({len(selected_features)}):")
                    st.write(f"`{', '.join(selected_features)}`")
                    with st.expander("Preview Combined Training Data (Head)"): st.dataframe(combined_df_for_features.head().round(3))
                else:
                    st.error("No data loaded for feature selection.")
                    selected_features = []
            except Exception as e:
                st.error(f"Error loading data for feature selection: {e}")
                selected_features = []

        can_train = len(selected_features) > 0 and not combined_df_for_features.empty
        if not can_train: st.warning("No features selected or data loaded.")

        if st.button("Train Model with Walk-Forward Validation", type="primary", use_container_width=True, disabled=not can_train):
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_area = st.empty()
                metrics_area = st.empty()
                # Placeholder for multiple fold loss plots if needed, or just show last one
                plot_area = st.container() # Use container to hold plots

            all_fold_metrics = []
            last_fold_model_state = None
            last_fold_scaler = None
            final_model_to_save = None
            final_scaler_to_save = None

            try:
                status_area.info("Loading and combining data...")
                training_device = torch.device('cuda') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
                device_info = f"Using {training_device}" + (f": {torch.cuda.get_device_name(0)}" if training_device.type == 'cuda' else "")
                status_area.info(device_info)

                # Get full paths for processed files
                processed_file_paths = [os.path.join(PROCESSED_DATA_DIR, f) for f in selected_files]

                # Call the Optuna optimization function
                status_area.info(f"Starting Optuna Optimization ({n_trials} trials)...")
                # run_optuna_optimization handles data loading, splitting, training, evaluation, and saving
                best_model, best_scaler_obj, best_params, best_avg_metrics = run_optuna_optimization(
                    processed_files=processed_file_paths,
                    feature_columns=selected_features,
                    n_trials=n_trials, # Pass number of trials
                    epochs=epochs,     # Pass epochs per trial
                    num_splits=num_splits, # Pass splits per trial
                    lookback_window=lookback_window
                )

                progress_bar.progress(1.0) # Mark as complete after grid search returns

                # --- Process results from grid search ---
                if best_model and best_params and best_avg_metrics:
                    status_area.success("Training complete! Best model found and saved.")

                    # Display the best parameters found by Optuna
                    metrics_area.subheader("Best Parameters Found by Optuna")
                    metrics_area.json(best_params)

                    # Display the average metrics achieved with these parameters
                    metrics_area.subheader("Average Walk-Forward Validation Metrics")
                    avg_metrics_df = pd.DataFrame(best_avg_metrics.items(), columns=['Metric', 'Average Value'])
                    # Ensure 'confusion_matrix' is handled if present
                    avg_metrics_df = avg_metrics_df[avg_metrics_df['Metric'] != 'confusion_matrix']
                    metrics_area.dataframe(avg_metrics_df.style.format({'Average Value': '{:.4f}'}), use_container_width=True)

                    # Display key average metrics
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Avg Accuracy", f"{best_avg_metrics.get('accuracy', 0):.4f}")
                    metric_cols[1].metric("Avg F1 (Positive)", f"{best_avg_metrics.get('f1_positive', 0):.4f}")
                    metric_cols[2].metric("Avg ROC AUC", f"{best_avg_metrics.get('roc_auc', 0):.4f}")
                    metric_cols[3].metric("Avg Recall (Positive)", f"{best_avg_metrics.get('recall_positive', 0):.4f}")

                    # Update the app's model state if needed for immediate prediction
                    self.model_trainer = ModelTrainer(lookback_window=lookback_window) # Re-init trainer
                    self.model_trainer.model = best_model
                    self.model_trainer.scalers = {'final_scaler': best_scaler_obj} # Store the single best scaler
                    self.model_trainer.feature_columns = selected_features
                    self.model_loaded = True
                    with st.expander("Saved Model Summary", expanded=True):
                         try:
                             with open(os.path.join(MODEL_DIR, 'model_info.json'), 'r') as f: model_info = json.load(f)
                             st.json(model_info)
                         except Exception as e: st.warning(f"Could not reload model_info.json: {e}")
                         st.info(device_info)
                         st.markdown("### Next Steps\n- Go to **Prediction** tab...")

                else:
                    status_area.error("Training process failed or did not find a suitable model.")
                    # Optionally display partial results if available

                # --- Remove the old walk-forward loop and direct training calls ---
                # (The following code block from the original file is now replaced by the call
                # to run_grid_search_training and its result handling above)
                # The old training loop code that was previously commented out here has been removed.
                # Training is now handled by the call to run_grid_search_training above.

            except Exception as e:
                progress_bar.empty()
                status_area.error(f"❌ Error during training process: {e}") # Updated error message context
                st.error("Stack trace:"); st.code(traceback.format_exc())

    def render_sidebar(self):
        """Render the sidebar with navigation and status"""
        with st.sidebar:
            st.markdown("<h1 style='text-align: center; color: #FF69B4;'>📈 StockAI</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Advanced Stock Prediction</p>", unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("📌 Navigation")
            selected_mode = st.session_state.get('app_mode', 'Home')
            nav_buttons = {
                "Home": "🏠 Home", "Data Collection": "📊 Data Collection",
                "Model Training": "🧠 Model Training", "Prediction": "🔮 Prediction",
                "Settings": "⚙️ Settings"
            }
            for mode, label in nav_buttons.items():
                button_type = "primary" if selected_mode == mode else "secondary"
                if st.button(label, key=f"{mode}_btn", use_container_width=True, type=button_type):
                    if selected_mode != mode:
                        st.session_state.app_mode = mode
                        st.rerun()
            st.markdown("---")
            st.subheader("💻 System Status")
            gpu_status = f"✅ GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "⚠️ Using CPU (slower)"
            st.write(gpu_status)
            model_status = "✅ Model loaded" if os.path.exists(os.path.join(MODEL_DIR, 'stock_prediction_model.pth')) else "⚠️ No model trained"
            st.write(model_status)
            # Check for processed data files
            processed_data_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')] if os.path.exists(PROCESSED_DATA_DIR) else []
            data_status = f"✅ {len(processed_data_files)} processed files" if processed_data_files else "⚠️ No processed data"
            st.write(data_status)
            st.markdown("---")
            st.caption("StockAI Prediction System v1.1") # Version bump
            st.caption("2025 Hope Project")

    def fetch_stock_data(self, ticker, start_date, end_date):
        """Fetch stock data using yfinance"""
        try:
            # Sanitize ticker for yfinance (replace dot with dash)
            yf_ticker = ticker.replace('.', '-')
            df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
            if df.empty: st.error(f"No data found for {ticker}."); return None
            # Add original ticker column if needed elsewhere
            df['Ticker'] = ticker
            return df
        except Exception as e: st.error(f"Error fetching stock data for {ticker}: {e}"); return None

    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for visualization.
        Returns the DataFrame with indicators added, or None if a critical error occurs.
        """
        if df is None or df.empty:
            st.warning("No data provided for indicator calculation.")
            return df.copy() if df is not None else None
        if len(df) < 50:
            st.warning(f"Need >50 data points for all indicators, got {len(df)}.")

        df_indicators = df.copy()
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        processed_data = {} # Store 1D float numpy arrays

        # --- Rigorous Column Handling & Preparation ---
        for col in required_cols:
            if col not in df_indicators.columns:
                st.error(f"Critical column '{col}' missing. Cannot calculate indicators.")
                return None # Signal critical failure

            col_data = df_indicators[col]
            final_array = None

            try:
                if isinstance(col_data, pd.Series):
                    # Attempt direct conversion to numeric, coercing errors
                    numeric_series = pd.to_numeric(col_data, errors='coerce')
                    # Fill NaNs introduced by coercion or already present
                    filled_series = numeric_series.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
                    final_array = filled_series.astype(float).values
                elif isinstance(col_data, pd.DataFrame):
                    st.warning(f"Column '{col}' is a DataFrame. Attempting to extract first column.")
                    if not col_data.empty:
                        # Select first column, convert to numeric Series, fill NaNs
                        series_from_df = pd.to_numeric(col_data.iloc[:, 0], errors='coerce')
                        filled_series = series_from_df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
                        final_array = filled_series.astype(float).values
                    else:
                        st.warning(f"Column '{col}' is an empty DataFrame. Filling with zeros.")
                        final_array = np.zeros(len(df_indicators), dtype=float)
                elif isinstance(col_data, np.ndarray):
                     st.warning(f"Column '{col}' is already a NumPy array. Ensuring 1D float.")
                     final_array = col_data.flatten().astype(float)
                     # Basic NaN/inf handling for arrays
                     final_array[np.isnan(final_array)] = 0.0
                     final_array[np.isinf(final_array)] = 0.0
                else:
                    # Attempt conversion for other types, might fail
                    st.warning(f"Column '{col}' has unexpected type {type(col_data)}. Attempting conversion.")
                    try:
                        converted_data = np.array(col_data, dtype=float)
                        final_array = converted_data.flatten()
                        final_array[np.isnan(final_array)] = 0.0
                        final_array[np.isinf(final_array)] = 0.0
                    except Exception as conv_err:
                        st.error(f"Could not convert column '{col}' of type {type(col_data)} to float array: {conv_err}")
                        final_array = np.zeros(len(df_indicators), dtype=float) # Fallback to zeros

                # Ensure the result is a 1D array
                if final_array is not None:
                    if final_array.ndim > 1:
                        st.warning(f"Data for '{col}' has shape {final_array.shape}. Squeezing to 1D.")
                        final_array = np.squeeze(final_array)
                    if final_array.ndim != 1:
                         st.error(f"Could not ensure 1D array for '{col}'. Final shape: {final_array.shape}. Aborting.")
                         return None # Signal critical failure
                    processed_data[col] = final_array
                    df_indicators[col] = final_array # Update DataFrame column as well
                else:
                    # This case should ideally be handled by fallbacks above, but as a safeguard:
                    st.error(f"Processing failed unexpectedly for column '{col}'. Aborting.")
                    return None # Signal critical failure

            except Exception as e:
                st.error(f"Error processing critical column '{col}': {e}")
                traceback.print_exc()
                return None # Signal critical failure
        # --- End Column Handling ---

        # --- Indicator Calculation ---
        try:
            import talib
            # Use the pre-processed numpy arrays (already ensured to be 1D float)
            open_p, high, low, close, volume = (processed_data['Open'], processed_data['High'],
                                                processed_data['Low'], processed_data['Close'],
                                                processed_data['Volume'])

            # Calculate indicators (TA-Lib expects float64)
            df_indicators['SMA_5'] = talib.SMA(close, 5)
            df_indicators['SMA_20'] = talib.SMA(close, 20)
            df_indicators['SMA_50'] = talib.SMA(close, 50)
            df_indicators['EMA_5'] = talib.EMA(close, 5)
            df_indicators['EMA_20'] = talib.EMA(close, 20)
            df_indicators['RSI'] = talib.RSI(close, 14)
            macd, macdsignal, macdhist = talib.MACD(close, 12, 26, 9)
            df_indicators['MACD'], df_indicators['MACD_Signal'], df_indicators['MACD_Hist'] = macd, macdsignal, macdhist
            upper, middle, lower = talib.BBANDS(close, 20, 2, 2, 0)
            df_indicators['BB_Upper'], df_indicators['BB_Middle'], df_indicators['BB_Lower'] = upper, middle, lower
            df_indicators['SlowK'], df_indicators['SlowD'] = talib.STOCH(high, low, close, 5, 3, 0, 3, 0)
            df_indicators['ADX'] = talib.ADX(high, low, close, 14)
            df_indicators['Chaikin_AD'] = talib.AD(high, low, close, volume)
            df_indicators['OBV'] = talib.OBV(close, volume)
            df_indicators['ATR'] = talib.ATR(high, low, close, 14)
            df_indicators['Williams_R'] = talib.WILLR(high, low, close, 14)
            df_indicators['ROC'] = talib.ROC(close, 10)
            df_indicators['CCI'] = talib.CCI(high, low, close, 14)

        except ImportError:
            st.warning("TA-Lib not available, using basic indicators.")
            # Basic calculations (use the processed numpy arrays converted back to Series for convenience)
            close_series = pd.Series(processed_data['Close'], index=df_indicators.index)
            open_series = pd.Series(processed_data['Open'], index=df_indicators.index)
            high_series = pd.Series(processed_data['High'], index=df_indicators.index)
            low_series = pd.Series(processed_data['Low'], index=df_indicators.index)
            volume_series = pd.Series(processed_data['Volume'], index=df_indicators.index)

            df_indicators['SMA_5'] = close_series.rolling(5, min_periods=1).mean()
            df_indicators['SMA_20'] = close_series.rolling(20, min_periods=1).mean()
            df_indicators['SMA_50'] = close_series.rolling(50, min_periods=1).mean()
            df_indicators['EMA_5'] = close_series.ewm(span=5, adjust=False).mean()
            df_indicators['EMA_20'] = close_series.ewm(span=20, adjust=False).mean()
            delta = close_series.diff(); gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean(); loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan); df_indicators['RSI'] = 100 - (100 / (1 + rs))
            ema12 = close_series.ewm(span=12, adjust=False).mean(); ema26 = close_series.ewm(span=26, adjust=False).mean()
            df_indicators['MACD'] = ema12 - ema26; df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
            df_indicators['MACD_Hist'] = df_indicators['MACD'] - df_indicators['MACD_Signal']
            df_indicators['BB_Middle'] = close_series.rolling(20, min_periods=1).mean(); std = close_series.rolling(20, min_periods=1).std()
            df_indicators['BB_Upper'] = df_indicators['BB_Middle'] + (std * 2); df_indicators['BB_Lower'] = df_indicators['BB_Middle'] - (std * 2)
            # Add basic versions of other indicators if needed (placeholders for now)
            basic_indicator_cols = ['SlowK', 'SlowD', 'ADX', 'Chaikin_AD', 'OBV', 'ATR', 'Williams_R', 'ROC', 'CCI']
            for col in basic_indicator_cols: df_indicators[col] = 0.0 # Assign float 0.0

        except Exception as ta_err:
             st.error(f"Error during TA-Lib calculation: {ta_err}")
             # Fallback to basic if TA-Lib fails unexpectedly
             st.warning("Falling back to basic indicators due to TA-Lib error.")
             close_series = pd.Series(processed_data['Close'], index=df_indicators.index)
             df_indicators['SMA_5'] = close_series.rolling(5, min_periods=1).mean()
             df_indicators['SMA_20'] = close_series.rolling(20, min_periods=1).mean()
             # ... (add other basic calcs as above) ...
             basic_indicator_cols = ['SMA_50', 'EMA_5', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SlowK', 'SlowD', 'ADX', 'Chaikin_AD', 'OBV', 'ATR', 'Williams_R', 'ROC', 'CCI']
             for col in basic_indicator_cols:
                 if col not in df_indicators.columns: df_indicators[col] = 0.0


        # Ratio features (handle division by zero robustly)
        df_indicators['Close_Open_Ratio'] = (df_indicators['Close'] / df_indicators['Open'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        df_indicators['High_Low_Diff'] = df_indicators['High'] - df_indicators['Low']
        df_indicators['Close_Prev_Ratio'] = (df_indicators['Close'] / df_indicators['Close'].shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


        # Final Fill NaNs robustly AFTER all calculations
        indicator_cols = [
            'SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal',
            'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SlowK', 'SlowD', 'ADX',
            'Chaikin_AD', 'OBV', 'ATR', 'Williams_R', 'ROC', 'CCI',
            'Close_Open_Ratio', 'High_Low_Diff', 'Close_Prev_Ratio' # Include ratio features
        ]
        for col in indicator_cols:
            if col in df_indicators.columns:
                # Ensure column is numeric before filling
                df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')
                df_indicators[col] = df_indicators[col].fillna(method='ffill').fillna(method='bfill').fillna(0.0) # Use float 0.0
            else:
                 # If an indicator failed completely, add it with zeros
                 st.warning(f"Indicator column '{col}' was missing after calculation. Adding with zeros.")
                 df_indicators[col] = 0.0

        # --- Flatten MultiIndex Columns if necessary ---
        if isinstance(df_indicators.columns, pd.MultiIndex):
            st.warning("Detected MultiIndex columns after indicator calculation. Flattening.")
            # More robust flattening: handle potential duplicate names
            flat_cols = []
            seen_cols = {}
            for col_tuple in df_indicators.columns.values:
                # Join tuple elements, filter out empty strings
                flat_name = '_'.join(filter(None, col_tuple))
                if flat_name in seen_cols:
                     seen_cols[flat_name] += 1
                     flat_name = f"{flat_name}_{seen_cols[flat_name]}"
                else:
                     seen_cols[flat_name] = 0
                flat_cols.append(flat_name)
            df_indicators.columns = flat_cols

        return df_indicators

    def create_price_chart(self, df, ticker, prediction_df=None, is_classification=False):
        """Create an interactive price chart with Plotly, optionally adding prediction markers."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], subplot_titles=(f"{ticker} Stock Price", "Volume"))
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price", increasing_line_color=PRIMARY_COLOR, decreasing_line_color='cyan'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=SECONDARY_COLOR), row=2, col=1)
        if 'SMA_20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5)), row=1, col=1)
        if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='rgba(30, 144, 255, 0.7)', width=1.5)), row=1, col=1)
        if all(c in df.columns for c in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", line=dict(color='rgba(173, 216, 230, 0.7)', width=1), showlegend=False), row=1, col=1) # Hide legend for clarity
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name="BB Middle", line=dict(color='rgba(173, 216, 230, 0.7)', width=1, dash='dash'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", line=dict(color='rgba(173, 216, 230, 0.7)', width=1), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)', showlegend=True), row=1, col=1) # Show legend only for lower band

        # Add Predictions (Regression or Classification Markers)
        if prediction_df is not None and not prediction_df.empty:
            prediction_start_date = df.index[-1]
            fig.add_vline(x=prediction_start_date, line_width=1, line_dash="dash", line_color="grey", annotation_text="Prediction Start", annotation_position="bottom right")

            if is_classification:
                # Add markers for classification predictions
                up_preds = prediction_df[prediction_df['Predicted_Trend'] == 1]
                down_preds = prediction_df[prediction_df['Predicted_Trend'] == 0]
                # Get last close price to position markers slightly above/below
                last_close = df['Close'].iloc[-1]
                marker_offset = (df['High'].iloc[-1] - df['Low'].iloc[-1]) * 0.5 # Adjust offset based on recent volatility

                fig.add_trace(go.Scatter(
                    x=up_preds['Date'], y=[last_close + marker_offset] * len(up_preds), # Position above last close
                    mode='markers', name='Pred. Up',
                    marker=dict(symbol='triangle-up', size=10, color='lime'),
                    hovertext=up_preds['Probability (Up)'].apply(lambda p: f'Prob(Up): {p:.2f}'),
                    hoverinfo='text'
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=down_preds['Date'], y=[last_close - marker_offset] * len(down_preds), # Position below last close
                    mode='markers', name='Pred. Down',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    hovertext=down_preds['Probability (Up)'].apply(lambda p: f'Prob(Up): {p:.2f}'),
                    hoverinfo='text'
                ), row=1, col=1)
            else: # Regression
                # Plot predicted line and confidence interval
                fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted'], mode='lines+markers', name="Prediction", line=dict(color='yellow', width=2, dash='dash'), marker=dict(size=6, color='gold')), row=1, col=1)
                if 'Lower_Bound' in prediction_df.columns and 'Upper_Bound' in prediction_df.columns:
                    fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Upper_Bound'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Lower_Bound'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.2)', name='Confidence Interval'), row=1, col=1)

        fig.update_layout(title=f"{ticker} Stock Analysis", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, font=dict(color=TEXT_COLOR), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=700, xaxis_rangeslider_visible=False, margin=dict(l=50, r=50, t=85, b=50))
        fig.update_yaxes(title_text="Price ($)", row=1, col=1); fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig

    def create_technical_indicators_chart(self, df):
        """Create charts for technical indicators"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("RSI (14)", "MACD"))
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color=PRIMARY_COLOR, width=1.5)), row=1, col=1)
            fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=1, col=1)
        if all(c in df.columns for c in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='blue', width=1.5)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal", line=dict(color='red', width=1.5)), row=2, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram", marker_color=SECONDARY_COLOR), row=2, col=1)
        fig.update_layout(title="Technical Indicators", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, font=dict(color=TEXT_COLOR), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500, margin=dict(l=50, r=50, t=85, b=50)) # Reduced height
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1); fig.update_yaxes(title_text="MACD", row=2, col=1)
        return fig

    def fetch_news(self, ticker, api_key, limit=5):
        """Fetch latest news articles using MarketAux API"""
        if not api_key: return []
        try:
            url = f"https://api.marketaux.com/v1/news/all?symbols={ticker}&filter_entities=true&language=en&api_token={api_key}&limit={limit}"
            response = requests.get(url)
            if response.status_code == 200: return response.json().get('data', [])
            else: st.warning(f"Error fetching news: {response.status_code}"); return []
        except Exception as e: st.warning(f"Error fetching news: {e}"); return []

    def display_news_section(self, news_articles):
        """Display news articles in the app"""
        if not news_articles: st.info("No news articles available or API key not provided."); return
        st.subheader("📰 Latest News")
        for article in news_articles:
            col1, col2 = st.columns([1, 3])
            with col1:
                img_url = article.get('image_url')
                if img_url:
                    try: st.image(img_url, use_container_width=True) # Replaced use_column_width
                    except: st.write("📰") # Fallback icon
                else: st.write("📰")
            with col2:
                pub_date = datetime.strptime(article['published_at'][:19], "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M")
                st.markdown(f"**{article['title']}**")
                st.markdown(f"*{pub_date}*")
                st.write(article.get('description', 'No description available.'))
                st.markdown(f"[Read more]({article['url']})")
            st.markdown("---")

    def run_data_collection(self, settings):
        """Run data collection process using the provided settings."""
        progress_container = st.container()
        collected_stats = {}
        log_messages = ["### Data Collection Log"] # Start log with a header

        with progress_container:
            st.subheader("📊 Collection Progress")
            status_text = st.empty()
            progress_bar = st.progress(0)
            # Use an expander for detailed logs, default to expanded
            log_expander = st.expander("Show Detailed Logs", expanded=True)
            with log_expander:
                details_area = st.empty()
                details_area.markdown("\n".join(log_messages), unsafe_allow_html=True) # Initial empty log

        # Define callbacks for DataCollector
        def progress_callback(progress_value): # Takes a float value between 0.0 and 1.0
            # Clamp the value between 0.0 and 1.0 to prevent errors
            clamped_progress = max(0.0, min(1.0, progress_value))
            progress_bar.progress(clamped_progress)
        def status_callback(message):
            timestamp = datetime.now().strftime('%H:%M:%S')
            log_entry = f"- {timestamp}: {message}"
            log_messages.append(log_entry)
            status_text.info(message) # Show current status prominently
            # Update the details area with all logs so far
            details_area.markdown("\n".join(log_messages), unsafe_allow_html=True)

        with st.spinner("Running data collection pipeline..."):
            processed_tickers = None # Initialize to None
            representative_features = None
            stock_data = {} # Initialize stock_data
            try:
                # Import moved back to top level
                collector = DataCollector()
                # Get settings from the dictionary
                start_date_str = settings['start_date'].strftime('%Y-%m-%d')
                end_date_str = settings['end_date'].strftime('%Y-%m-%d')
                selected_companies_list = settings.get('selected_companies', []) # List of {'ticker': T, 'name': N}

                if not selected_companies_list:
                    status_text.error("No companies selected for collection.")
                    return None, None

                # --- Set the companies list in the collector ---
                collector.companies = selected_companies_list
                status_callback(f"Initialized collector for {len(collector.companies)} companies.")
                collected_stats["Companies Requested"] = len(collector.companies)

                # Define total steps based on settings
                total_steps = 1 # Stock data is always collected
                if settings['use_reddit']: total_steps += 1
                if settings['use_google_trends']: total_steps += 1
                total_steps += 1 # Add step for processing
                current_step = 0

                # --- Step 1: Collect Stock Data ---
                current_step += 1
                status_callback(f"Step {current_step}/{total_steps}: Collecting stock data...")
                stock_data = collector.collect_stock_data(
                    start_date=start_date_str, end_date=end_date_str,
                    progress_callback=lambda c, t: progress_callback( (current_step - 1 + (c / t if t > 0 else 0)) / total_steps ), # Calculate final 0.0-1.0 value
                    status_callback=status_callback
                )
                if not stock_data:
                     status_text.error("Failed to collect stock data for any ticker. Aborting.")
                     return None, None
                collected_stats["Stock Datasets Fetched"] = len(stock_data)

                # --- Step 2: Collect Reddit Data ---
                reddit_data = {}
                if settings['use_reddit']:
                    current_step += 1
                    status_callback(f"Step {current_step}/{total_steps}: Collecting Reddit data...")
                    if settings['reddit_client_id'] and settings['reddit_client_secret']:
                        if collector.setup_reddit_api(settings['reddit_client_id'], settings['reddit_client_secret'], settings['reddit_user_agent']):
                            reddit_data = collector.collect_reddit_sentiment(
                                start_date=start_date_str, end_date=end_date_str,
                                progress_callback=lambda overall_progress_fraction: progress_callback(overall_progress_fraction), # Pass the single calculated fraction
                                status_callback=status_callback
                            )
                            collected_stats["Reddit Tickers Found"] = len(reddit_data)
                        else:
                            status_callback("Skipping Reddit (API setup failed).")
                    else:
                        status_callback("Skipping Reddit (no credentials).")
                elif settings['use_reddit']: # Should not happen if checkbox logic is correct, but good fallback
                    status_callback("Skipping Reddit (disabled in settings but flag was true?).")

                # --- Step 3: Collect Google Trends ---
                google_trends = {}
                if settings['use_google_trends']:
                    current_step += 1
                    status_callback(f"Step {current_step}/{total_steps}: Collecting Google Trends...")
                    try:
                        google_trends = collector.collect_google_trends(
                            start_date=start_date_str, end_date=end_date_str,
                            progress_callback=lambda c, t: progress_callback( (current_step - 1 + (c / t if t > 0 else 0)) / total_steps ), # Calculate final 0.0-1.0 value
                            status_callback=status_callback
                        )
                        # Check if google_trends is None (could happen on critical error) and default to {}
                        if google_trends is None:
                             google_trends = {}
                             status_callback("Warning: Google Trends collection returned None. Proceeding without Trends data.")
                        trends_found_count = sum(1 for df in google_trends.values() if df is not None and not df.empty)
                        collected_stats["Trends Tickers Found"] = trends_found_count
                        status_callback(f"Google Trends step completed. Found data for {trends_found_count} tickers.")
                    except Exception as trends_err:
                         status_callback(f"⚠️ Warning: Google Trends collection failed: {trends_err}. Proceeding without Trends data.")
                         google_trends = {} # Ensure it's an empty dict on error
                         collected_stats["Trends Tickers Found"] = 0 # Record failure
                elif settings['use_google_trends']: # This case might be redundant now but kept for safety
                    status_callback("Skipping Google Trends (disabled in settings but flag was true?).")

                # --- Step 3.5: Collect Macro Data (Optional) ---
                if settings['use_macro_data']:
                    # Adjust step count dynamically if macro is included
                    # Find the current step number based on previous steps completed
                    steps_done = 1 # Stock data
                    if settings['use_reddit']: steps_done += 1
                    if settings['use_google_trends']: steps_done += 1
                    current_step = steps_done + 1 # This is the macro step
                    total_steps = current_step + 1 # +1 for the final processing step

                    status_callback(f"Step {current_step}/{total_steps}: Collecting Macro data (FEDFUNDS)...")
                    # Call the new macro collection method
                    macro_df = collector.collect_fred_data(
                        start_date=start_date_str, end_date=end_date_str,
                        series_id='FEDFUNDS', # Hardcoded for now, could be made configurable
                        status_callback=status_callback
                        # No progress callback needed here as it's usually fast
                    )
                    if macro_df is not None and not macro_df.empty:
                         collected_stats["Macro Data Points"] = len(macro_df)
                    else:
                         status_callback("Macro data collection failed or returned empty.")
                         collected_stats["Macro Data Points"] = 0
                elif settings['use_macro_data']: # If checkbox was true but step skipped
                     status_callback("Skipping Macro Data collection (disabled in settings but flag was true?).")


                # --- Step 4 (or 5): Process Data ---
                # Recalculate current/total steps for the processing step message
                steps_done = 1 # Stock data
                if settings['use_reddit']: steps_done += 1
                if settings['use_google_trends']: steps_done += 1
                if settings['use_macro_data']: steps_done += 1
                current_step = steps_done + 1
                total_steps = current_step # Processing is the last step
                current_step += 1
                status_callback(f"Step {current_step}/{total_steps}: Processing collected data (reading raw files)...")
                # process_data now reads raw files directly, doesn't need dicts passed
                processed_tickers, representative_features = collector.process_data(
                    stock_data, # Only pass stock_data dict
                    progress_callback=lambda c, t: progress_callback( (current_step - 1 + (c / t if t > 0 else 0)) / total_steps ), # Calculate final 0.0-1.0 value
                    status_callback=status_callback
                )

                progress_bar.progress(1.0) # Ensure 100% at the end
                if processed_tickers:
                    status_text.success("Data collection and processing complete!")
                else:
                    # Keep the error message visible after spinner stops
                    status_text.error("Data collection ran, but failed to process any tickers successfully. Check details below.")
                    # Ensure the log expander is open if processing failed
                    details_area.markdown("\n".join(log_messages), unsafe_allow_html=True) # Update logs one last time
                    # Force expander open (this might require JS or more complex state management,
                    # but showing the error prominently is key)
                    st.error("Processing failed for all tickers. See logs above for details.")


            except Exception as e:
                 status_text.error(f"Critical error during data collection pipeline: {e}")
                 st.error("Stack trace:"); st.code(traceback.format_exc())
                 return None, None # Indicate failure

        # Display final stats outside the spinner
        st.subheader("Collection Statistics")
        if collected_stats:
             # Use max 4 columns for stats display
             num_stats = len(collected_stats)
             stats_cols = st.columns(min(num_stats, 4))
             col_idx = 0
             for label, value in collected_stats.items():
                 with stats_cols[col_idx % min(num_stats, 4)]:
                     st.metric(label, value)
                 col_idx += 1

        # Display overall summary based on processed tickers
        st.subheader("Overall Summary")
        summary_cols = st.columns(3)
        num_processed = len(processed_tickers) if processed_tickers else 0
        with summary_cols[0]: st.metric("Companies Processed", num_processed)
        total_records = sum(len(d) for d in stock_data.values() if d is not None) if stock_data else 0
        with summary_cols[1]: st.metric("Stock Records Fetched", total_records)
        num_features = len(representative_features) if representative_features else 0
        with summary_cols[2]: st.metric("Features", num_features)

        if representative_features:
            st.subheader("Representative Features")
            safe_features = [str(f) for f in representative_features if f is not None]
            st.write(f"(From last processed ticker: {processed_tickers[-1] if processed_tickers else 'N/A'}): `{', '.join(safe_features)}`" if safe_features else "None determined.")
        elif num_processed > 0: # If processed but no features (shouldn't happen ideally)
             st.warning("Tickers processed, but no representative features determined.")
        # No message needed if num_processed is 0, error shown above

        # Return list of tickers and features
        return processed_tickers, representative_features

    def render_data_collection_page(self):
        """Render the data collection page"""
        st.title("📊 Data Collection")
        settings_expander = st.expander("Collection Settings", expanded=True)
        status_container = st.container() # Container for status updates and logs
        results_container = st.container() # Container for results summary and preview

        with settings_expander:
            st.subheader("Select Companies")
            # Use the loaded available companies DataFrame
            company_options_map = {f"{r['ticker']} - {r['name']}": {'ticker': r['ticker'], 'name': r['name']}
                                   for _, r in self.available_companies.iterrows()}
            company_display_options = list(company_options_map.keys())

            company_selection = st.radio("Selection Method", ["Top Companies", "Custom"], horizontal=True, key="data_sel_method")
            selected_companies_dicts = [] # Will store list of {'ticker': T, 'name': N}

            if company_selection == "Top Companies":
                num_companies = st.slider("Number of Top Companies", 1, len(company_display_options), 10, key="data_num_comp")
                # Get the display strings for the top N
                top_n_display = company_display_options[:num_companies]
                # Convert back to list of dicts
                selected_companies_dicts = [company_options_map[display_str] for display_str in top_n_display]
                st.write(f"Selected: {', '.join([d['ticker'] for d in selected_companies_dicts])}")
            else: # Custom selection
                # Default to first company if available
                default_selection = [company_display_options[0]] if company_display_options else []
                selected_options_display = st.multiselect("Select Companies", options=company_display_options, default=default_selection, key="data_multi_comp")
                # Convert selected display strings back to dicts
                selected_companies_dicts = [company_options_map[display_str] for display_str in selected_options_display]

                custom_ticker_input = st.text_input("Add Custom Ticker(s) (comma-separated)", placeholder="e.g., MSFT, AAPL", key="data_custom_ticker")
                if custom_ticker_input:
                    custom_tickers = [t.strip().upper() for t in custom_ticker_input.split(',') if t.strip()]
                    # For custom tickers, we don't have the name, use ticker as name
                    for ticker in custom_tickers:
                         # Avoid adding duplicates if already selected
                         if not any(d['ticker'] == ticker for d in selected_companies_dicts):
                              selected_companies_dicts.append({'ticker': ticker, 'name': ticker})

            # Ensure uniqueness (based on ticker)
            seen_tickers = set()
            unique_selected_companies = []
            for company_dict in selected_companies_dicts:
                if company_dict['ticker'] not in seen_tickers:
                    unique_selected_companies.append(company_dict)
                    seen_tickers.add(company_dict['ticker'])
            selected_companies_dicts = unique_selected_companies

            st.subheader("Date Range")
            col1, col2 = st.columns(2)
            with col1: start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*2), min_value=datetime(2010, 1, 1), max_value=datetime.now() - timedelta(days=1), key="data_start_date") # Default 2 years
            with col2: end_date = st.date_input("End Date", datetime.now(), min_value=start_date, max_value=datetime.now(), key="data_end_date")

            st.subheader("Data Sources")
            collect_stock = st.checkbox("Stock Price Data (OHLCV)", True, disabled=True, key="data_src_stock") # Always collect stock data
            collect_tech = st.checkbox("Technical Indicators", True, disabled=True, key="data_src_tech") # Calculated during processing
            collect_reddit = st.checkbox("Reddit Sentiment", False, key="data_src_reddit")
            collect_google = st.checkbox("Google Trends", False, key="data_src_google")
            collect_macro = st.checkbox("Macro Data (FEDFUNDS)", False, key="data_src_macro") # Added Macro checkbox

            if collect_reddit:
                st.warning("Reddit API credentials required in Settings.")
                if not (st.session_state.get('reddit_client_id') and st.session_state.get('reddit_client_secret')):
                    st.error("Reddit API credentials not found in Settings. Reddit collection will be skipped.")
                # Reddit settings (limit, subreddits) could be added here if needed

            if collect_google: st.info("Google Trends uses pytrends library.") # Settings could be added

            if st.button("Start Data Collection", type="primary", use_container_width=True, key="data_start_button"):
                if not selected_companies_dicts: st.error("Please select at least one company.")
                else:
                    # Clear previous status/results before starting
                    status_container.empty()
                    results_container.empty()
                    with status_container: # Show status updates within this container
                        try:
                            settings = {
                                'start_date': start_date, 'end_date': end_date,
                                'use_reddit': collect_reddit,
                                'reddit_client_id': st.session_state.get('reddit_client_id', ''),
                                'reddit_client_secret': st.session_state.get('reddit_client_secret', ''),
                                'reddit_user_agent': st.session_state.get('reddit_user_agent', 'StockPredictionApp/1.0'),
                                'use_google_trends': collect_google,
                                'use_macro_data': collect_macro, # Pass macro setting
                                'use_macro_data': collect_macro, # Pass macro setting
                                'selected_companies': selected_companies_dicts, # Pass list of dicts
                            }
                            # run_data_collection now returns (processed_tickers, representative_features)
                            processed_tickers, representative_features = self.run_data_collection(settings)

                            # Display results summary after completion (outside the spinner/status area)
                            with results_container:
                                st.subheader("Collection Run Summary") # New header for clarity
                                if processed_tickers is not None: # Check if collection ran, even if it processed 0 tickers
                                     if processed_tickers: # Check if the list is not empty
                                         st.success(f"Data collected and processed successfully for {len(processed_tickers)} tickers.")
                                         # Optionally list processed tickers
                                         st.write(f"Processed: {', '.join(processed_tickers)}")
                                     else:
                                         # Collection ran but processed 0 tickers
                                         st.error("Data collection finished, but failed to process any tickers successfully. Review the detailed status messages above.")
                                else:
                                     # Collection failed to even start or return properly
                                     st.error("Data collection process failed critically. Check logs.")

                        except Exception as e:
                            # Catch errors from the button click action itself
                            st.error(f"Error initiating data collection: {e}")
                            st.error("Stack trace:"); st.code(traceback.format_exc())

        # Display summary of existing processed data files (remains the same)
        if os.path.exists(PROCESSED_DATA_DIR) and any(f.endswith('_processed_data.csv') for f in os.listdir(PROCESSED_DATA_DIR)):
            with results_container: # Show existing files summary below the run summary
                st.markdown("---") # Separator
                st.subheader("Existing Processed Data Files")
                processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')]
                st.info(f"Found {len(processed_files)} processed data files in `{PROCESSED_DATA_DIR}`.")
                selected_file_preview = st.selectbox("Select Processed File to Preview",
                                                     options=[""] + processed_files, # Add empty option
                                                     format_func=lambda x: x.split('_processed_data.csv')[0] if x else "Select...",
                                                     key="preview_select") # Unique key
                if selected_file_preview:
                    preview_path = os.path.join(PROCESSED_DATA_DIR, selected_file_preview)
                    ticker_preview = selected_file_preview.split('_processed_data.csv')[0]
                    st.write(f"Previewing: `{ticker_preview}`")
                    try:
                        df_preview = pd.read_csv(preview_path)
                        # --- Robust Date Column Handling ---
                        date_col_found = False
                        potential_date_cols = ['Date', 'Datetime', 'timestamp', 'date']
                        for col in potential_date_cols:
                            if col in df_preview.columns:
                                try:
                                    df_preview[col] = pd.to_datetime(df_preview[col])
                                    df_preview.rename(columns={col: 'Date'}, inplace=True)
                                    date_col_found = True
                                    break
                                except Exception: continue
                        if not date_col_found:
                            try:
                                df_temp = pd.read_csv(preview_path, index_col=0, parse_dates=True)
                                if isinstance(df_temp.index, pd.DatetimeIndex):
                                    df_preview = df_temp.reset_index()
                                    index_name = df_preview.columns[0]
                                    df_preview.rename(columns={index_name: 'Date'}, inplace=True)
                                    date_col_found = True
                                else:
                                     if df_temp.index.name: df_preview = df_temp.reset_index()
                            except Exception: pass

                        # --- Display Data ---
                        if date_col_found:
                            df_preview['Date'] = pd.to_datetime(df_preview['Date'])
                            potential_numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SlowK', 'SlowD', 'ADX', 'Chaikin_AD', 'OBV', 'ATR', 'Williams_R', 'ROC', 'CCI', 'Compound', 'Positive', 'Neutral', 'Negative', 'Count', 'Interest']
                            df_preview_cols_lower = {c.lower(): c for c in df_preview.columns}
                            for potential_col in potential_numeric_cols:
                                original_col_name = df_preview_cols_lower.get(potential_col.lower())
                                if original_col_name:
                                    try: df_preview[original_col_name] = pd.to_numeric(df_preview[original_col_name], errors='coerce')
                                    except Exception as conv_err: st.warning(f"Could not convert '{original_col_name}' to numeric: {conv_err}")

                            cols = st.columns(3)
                            cols[0].metric("Rows", len(df_preview))
                            cols[1].metric("Columns", len(df_preview.columns))
                            try: cols[2].metric("Date Range", f"{df_preview['Date'].min():%Y-%m-%d} to {df_preview['Date'].max():%Y-%m-%d}")
                            except Exception: cols[2].metric("Date Range", "N/A")

                            st.dataframe(df_preview.head().round(3)) # Show head

                            st.subheader("Quick Visualization")
                            numeric_cols_preview = df_preview.select_dtypes(include=np.number).columns.tolist()
                            close_col_actual = df_preview_cols_lower.get('close')
                            default_plot_col = close_col_actual if close_col_actual in numeric_cols_preview else (numeric_cols_preview[0] if numeric_cols_preview else None)

                            if default_plot_col:
                                plot_column = st.selectbox(f"Select column to plot for {ticker_preview}", numeric_cols_preview, index=numeric_cols_preview.index(default_plot_col) if default_plot_col in numeric_cols_preview else 0, key=f"preview_plot_select_{ticker_preview}")
                                if plot_column:
                                    try:
                                        plot_title = f"{ticker_preview} - {plot_column}"
                                        fig_preview = px.line(df_preview, x='Date', y=plot_column, title=plot_title)
                                        fig_preview.update_layout(template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
                                        st.plotly_chart(fig_preview, use_container_width=True) # Already correct, no change needed here but included for context
                                    except Exception as plot_err: st.warning(f"Could not plot {plot_column} for {ticker_preview}: {plot_err}")
                            else: st.info("No numeric columns found to plot.")
                        else:
                            st.warning(f"Could not identify a 'Date' column in '{selected_file_preview}'. Displaying raw data.")
                            st.dataframe(df_preview.head())

                    except Exception as e:
                        st.error(f"Error displaying preview for {selected_file_preview}: {str(e)}")
                        st.code(traceback.format_exc())
        else:
             with results_container: st.info("No processed data files found yet. Use settings above to collect data.")


    def render_settings_page(self):
        """Render the settings page"""
        st.title("⚙️ Settings")
        with st.expander("System Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Application Info"); st.write("**Version:** 1.1.0"); st.write("**Last Updated:** 2025-03-31") # Updated version/date
                st.write(f"**Data Directory:** {DATA_DIR}"); st.write(f"**Model Directory:** {MODEL_DIR}")
            with col2:
                st.subheader("System Info"); cuda_available = torch.cuda.is_available()
                st.write(f"**Device:** {'CUDA' if cuda_available else 'CPU'}")
                if cuda_available: st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
                st.write(f"**PyTorch:** {torch.__version__}"); st.write(f"**Python:** {platform.python_version()}")
                st.write(f"**OS:** {platform.system()} {platform.release()}")

        with st.expander("Application Settings", expanded=True):
            st.subheader("Default Settings")
            # Use available_companies which has 'ticker' and 'name'
            default_ticker_options = [f"{r['ticker']} - {r['name']}" for _, r in self.available_companies.iterrows()]
            default_ticker_index = 0
            if 'default_ticker' in st.session_state:
                 try: default_ticker_index = [opt.split(' - ')[0] for opt in default_ticker_options].index(st.session_state['default_ticker'])
                 except ValueError: pass # Keep index 0 if not found
            default_ticker_selection = st.selectbox("Default Stock", options=default_ticker_options, index=default_ticker_index, key="settings_default_ticker")
            default_lookback = st.slider("Default Lookback Window", 10, 120, st.session_state.get('default_lookback', 60), 5, key="settings_lookback")
            default_forecast = st.slider("Default Forecast Days", 1, 30, st.session_state.get('default_forecast', 7), 1, key="settings_forecast")
            if st.button("Save Default Settings", type="primary", key="settings_save_defaults"):
                st.session_state['default_ticker'] = default_ticker_selection.split(' - ')[0] # Save only the ticker symbol
                st.session_state['default_lookback'] = default_lookback
                st.session_state['default_forecast'] = default_forecast
                self._save_settings() # Save immediately
                st.success("Default settings saved!")

        with st.expander("API Settings"):
            st.subheader("Reddit API (Optional)")
            st.markdown("Used for sentiment analysis during data collection. Get credentials from [Reddit Apps](https://www.reddit.com/prefs/apps).")
            reddit_client_id = st.text_input("Client ID", value=st.session_state.get('reddit_client_id', ''), type="password", key="settings_reddit_id")
            reddit_client_secret = st.text_input("Client Secret", value=st.session_state.get('reddit_client_secret', ''), type="password", key="settings_reddit_secret")
            reddit_user_agent = st.text_input("User Agent", value=st.session_state.get('reddit_user_agent', 'StockPredictionApp/1.0'), key="settings_reddit_agent")
            if st.button("Save API Settings", key="settings_save_api"):
                st.session_state['reddit_client_id'] = reddit_client_id
                st.session_state['reddit_client_secret'] = reddit_client_secret
                st.session_state['reddit_user_agent'] = reddit_user_agent
                self._save_settings() # Save immediately
                st.success("API settings saved!")

        with st.expander("Data Management"):
            st.subheader("Data Cleanup")
            # Check both raw and processed directories
            raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')] if os.path.exists(RAW_DATA_DIR) else []
            processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')] if os.path.exists(PROCESSED_DATA_DIR) else []
            # Check for combined files in DATA_DIR
            combined_files = [f for f in os.listdir(DATA_DIR) if f in ['reddit_sentiment.csv', 'google_trends_data.csv']] if os.path.exists(DATA_DIR) else []
            # Check for top companies files
            top_company_files = [f for f in os.listdir(DATA_DIR) if f.startswith('top_') and f.endswith('_companies.csv')] if os.path.exists(DATA_DIR) else []

            all_data_files = raw_files + processed_files + combined_files + top_company_files

            if all_data_files:
                st.write(f"Found {len(raw_files)} raw, {len(processed_files)} processed, {len(combined_files)} combined, and {len(top_company_files)} company list files.")
                if st.button("⚠️ Delete All Data Files", type="primary", key="settings_delete_all_data"):
                    deleted_count = 0
                    error_list = []
                    # Delete raw files
                    for file in raw_files:
                        try: os.remove(os.path.join(RAW_DATA_DIR, file)); deleted_count += 1
                        except Exception as e: error_list.append(f"Error deleting raw file {file}: {e}")
                    # Delete processed files
                    for file in processed_files:
                        try: os.remove(os.path.join(PROCESSED_DATA_DIR, file)); deleted_count += 1
                        except Exception as e: error_list.append(f"Error deleting processed file {file}: {e}")
                    # Delete combined files
                    for file in combined_files:
                         path_to_del = os.path.join(DATA_DIR, file)
                         try: os.remove(path_to_del); deleted_count += 1
                         except Exception as e: error_list.append(f"Error deleting combined file {file}: {e}")
                    # Delete top company files
                    for file in top_company_files:
                         path_to_del = os.path.join(DATA_DIR, file)
                         try: os.remove(path_to_del); deleted_count += 1
                         except Exception as e: error_list.append(f"Error deleting company file {file}: {e}")

                    st.success(f"Deleted {deleted_count} data files!")
                    if error_list:
                         st.error("Some errors occurred during deletion:")
                         for err in error_list: st.error(f"- {err}")
                    st.rerun()

                # Option to delete selected processed files
                if processed_files:
                    selected_processed_del = st.multiselect("Select Processed Files to Delete", options=processed_files, key="settings_select_proc_del")
                    if selected_processed_del and st.button("Delete Selected Processed Files", key="settings_delete_proc_sel"):
                        for file in selected_processed_del:
                            try: os.remove(os.path.join(PROCESSED_DATA_DIR, file)); st.success(f"Deleted {file}")
                            except Exception as e: st.error(f"Error deleting {file}: {e}")
                        st.rerun()
            else: st.info("No data files found in raw, processed, or data directories.")


        with st.expander("Model Management"):
            st.subheader("Trained Models")
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(('.pth', '.pkl', '.json'))] if os.path.exists(MODEL_DIR) else []
            if model_files:
                st.write(f"Found {len(model_files)} model-related file(s).")
                if st.button("⚠️ Delete All Models & Scalers", type="primary", key="settings_delete_models"):
                    deleted_count = 0
                    error_list = []
                    for file in model_files: # Delete .pth, .pkl, .json
                         try: os.remove(os.path.join(MODEL_DIR, file)); deleted_count += 1
                         except Exception as e: error_list.append(f"Error deleting {file}: {e}")
                    st.success(f"Deleted {deleted_count} model-related files!")
                    if error_list:
                         st.error("Some errors occurred during deletion:")
                         for err in error_list: st.error(f"- {err}")
                    # Reset model loaded state
                    self.model_loaded = False
                    self.model_trainer = None
                    st.rerun()
            else: st.info("No trained models found.")

        with st.expander("About"):
            st.markdown("""
            ## Stock Price Prediction System v1.1
            Developed by Hope Project Team - March 2025
            Uses LSTM networks via PyTorch, data from yfinance, and Streamlit for the UI.
            Optional sentiment analysis via Reddit API (PRAW).
            Optional Google Trends data (pytrends).
            Technical indicators via TA-Lib (optional, basic fallback provided).
            """)

    def render_prediction_page(self):
        """Render the prediction page"""
        st.title("🔮 Stock Price Prediction")

        if not self.model_loaded:
            if not self.load_model(): # Attempt to load model if not already loaded
                st.error("No trained model found. Please train a model first on the 'Model Training' tab.")
                return

        # Ensure model_trainer is initialized
        if self.model_trainer is None or self.model_trainer.model is None or not self.model_trainer.scalers:
             st.error("Model trainer not initialized correctly. Please try reloading the model or retraining.")
             return

        model_info = {}
        try:
            with open(os.path.join(MODEL_DIR, 'model_info.json'), 'r') as f: model_info = json.load(f)
            with st.expander("Model Information", expanded=False):
                st.json(model_info)
                target_var = model_info.get('target_variable', 'Unknown')
                is_classification = model_info.get('is_classification', False)
                model_type = "Classification (Trend)" if is_classification else f"Regression ({target_var})"
                st.info(f"Model Type: {model_type}")
                metrics = model_info.get('metrics', {})
                if metrics: st.subheader("Model Performance Metrics"); st.dataframe(pd.DataFrame(metrics.items(), columns=['Metric', 'Value']))
        except Exception as e: st.warning(f"Could not load model info: {e}")

        st.subheader("Make New Predictions")
        col1, col2 = st.columns(2)
        with col1:
            # Use available_companies DataFrame
            pred_ticker_options = [f"{r['ticker']} - {r['name']}" for _, r in self.available_companies.iterrows()]
            default_ticker_symbol = st.session_state.get('default_ticker', self.available_companies['ticker'].iloc[0] if not self.available_companies.empty else 'AAPL')
            # Find the index for the default ticker
            try:
                 default_pred_index = [opt.split(' - ')[0] for opt in pred_ticker_options].index(default_ticker_symbol)
            except ValueError:
                 default_pred_index = 0 # Fallback to first option

            selected_ticker_option = st.selectbox("Select Stock", pred_ticker_options, index=default_pred_index, key="pred_ticker_select")
            ticker_symbol = selected_ticker_option.split(' - ')[0]

            today = datetime.now().date()
            default_lookback_days = st.session_state.get('default_lookback', 60) * 2 # Fetch more history than lookback
            start_date_default = today - timedelta(days=max(365, default_lookback_days)) # At least 1 year or double lookback
            st.subheader("Historical Data Range")
            start_date = st.date_input("Start Date", value=start_date_default, max_value=today - timedelta(days=1), key="pred_start_date")
            end_date = st.date_input("End Date", value=today, max_value=today, key="pred_end_date")

            st.subheader("Prediction Settings")
            forecast_days = st.slider("Forecast Horizon (Days)", 1, 30, st.session_state.get('default_forecast', 7), key="pred_forecast_days")
            confidence_level = st.slider("Confidence Level (%)", 50, 99, 95, help="For regression model confidence intervals", key="pred_conf_level")

        with col2:
            st.subheader(f"Current {ticker_symbol} Info")
            try:
                # Use sanitized ticker for yfinance download
                yf_ticker_symbol = ticker_symbol.replace('.', '-')
                current_data = yf.download(yf_ticker_symbol, start=today - timedelta(days=5), end=today + timedelta(days=1), progress=False)
                # --- Robust check for current data ---
                if not current_data.empty and 'Close' in current_data.columns and len(current_data.index) > 0:
                    last_date = current_data.index[-1]
                    last_price_val = current_data['Close'].iloc[-1] # Use iloc for position-based access
                    delta_text = "N/A"
                    if len(current_data) > 1 and 'Close' in current_data.columns and len(current_data['Close']) > 1:
                        # Ensure prev_price is obtained correctly as a scalar
                        prev_price_val = current_data['Close'].iloc[-2]
                        # Check if prev_price is a valid number before calculating delta
                        # Use float conversion attempt for robustness
                        try:
                            # Attempt to get scalar value using .item() if it's a Series/array like object
                            if isinstance(prev_price_val, (pd.Series, np.ndarray)) and prev_price_val.size == 1:
                                prev_price_float = prev_price_val.item()
                            else:
                                prev_price_float = float(prev_price_val) # Fallback to float conversion

                            if pd.notna(prev_price_float) and prev_price_float != 0:
                                delta = (float(last_price_val) - prev_price_float) / prev_price_float * 100
                                delta_text = f"{delta:.2f}%"
                            else:
                                delta_text = "N/A" # Handle case where previous price is invalid or zero
                        except (TypeError, ValueError, AttributeError): # Catch potential .item() error too
                             delta_text = "N/A" # Handle case where prev_price is not convertible to float

                    # Explicitly convert last_price to float before formatting in st.metric
                    try:
                        # Attempt to get scalar value using .item() if it's a Series/array like object
                        if isinstance(last_price_val, (pd.Series, np.ndarray)) and last_price_val.size == 1:
                            last_price_float = last_price_val.item()
                        else:
                            last_price_float = float(last_price_val) # Fallback to float conversion

                        # --- Add check ---
                        if isinstance(last_price_float, (int, float)):
                            st.metric(f"Last Close ({last_date:%Y-%m-%d})", f"${last_price_float:.2f}", delta=delta_text)
                        else:
                            st.metric(f"Last Close ({last_date:%Y-%m-%d})", "N/A (Invalid Price)", delta=delta_text)
                            st.warning(f"Could not format last price. Type: {type(last_price_float)}, Value: {last_price_float}")

                    except (TypeError, ValueError, AttributeError):
                         st.metric(f"Last Close ({last_date:%Y-%m-%d})", "N/A (Error)", delta=delta_text)
                         st.warning("Could not format last price due to conversion error.")


                    # Simple recent history plot (check for required columns)
                    plot_cols = ['Open', 'High', 'Low', 'Close']
                    if all(col in current_data.columns for col in plot_cols):
                        fig_curr = go.Figure(data=[go.Candlestick(x=current_data.index, open=current_data['Open'], high=current_data['High'], low=current_data['Low'], close=current_data['Close'])])
                        fig_curr.update_layout(title="Recent Days", xaxis_rangeslider_visible=False, height=250, margin=dict(l=0,r=0,t=30,b=0), template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
                        st.plotly_chart(fig_curr, use_container_width=True)
                    else:
                        st.warning("Could not display recent history plot (missing OHLC columns).")
                else:
                    st.warning("No recent data found or 'Close' column missing.")
            except Exception as e:
                st.error(f"Error fetching/displaying current info: {str(e)}")
                st.code(traceback.format_exc()) # Print full traceback for fetch error

        if st.button("Generate Prediction", type="primary", use_container_width=True, key="pred_generate_button"):
            status_container = st.empty(); progress_bar = st.progress(0); results_container = st.container()
            try:
                status_container.info(f"Fetching historical data for {ticker_symbol}...")
                hist_df = self.fetch_stock_data(ticker_symbol, start_date, end_date) # Use original ticker here
                if hist_df is None or hist_df.empty: status_container.error("Failed to fetch data."); return
                progress_bar.progress(20)

                status_container.info("Calculating technical indicators...")
                # Pass status_callback to see indicator calculation errors
                hist_df_indicators = self.calculate_technical_indicators(hist_df.copy())
                if hist_df_indicators is None: # Check if indicator calculation failed critically
                     status_container.error("Technical indicator calculation failed. Cannot proceed.")
                     return
                progress_bar.progress(40)

                # --- Add explicit flattening ---
                if isinstance(hist_df_indicators.columns, pd.MultiIndex):
                    status_container.info("Flattening MultiIndex columns after indicator calculation...")
                    # Use the same robust flattening logic from calculate_technical_indicators
                    flat_cols = []
                    seen_cols = {}
                    for col_tuple in hist_df_indicators.columns.values:
                        flat_name = '_'.join(filter(None, [str(c) for c in col_tuple])).strip('_') # Ensure strings
                        if flat_name in seen_cols:
                             seen_cols[flat_name] += 1
                             flat_name = f"{flat_name}_{seen_cols[flat_name]}"
                        else:
                             seen_cols[flat_name] = 0
                        flat_cols.append(flat_name)
                    hist_df_indicators.columns = flat_cols
                    status_container.info(f"Columns flattened: {hist_df_indicators.columns.tolist()[:5]}...") # Log first few flattened cols

                status_container.info("Preparing data for model...")
                # Ensure columns are simple strings AFTER calculating indicators (redundant check now, but safe)

                # Ensure all required features are present
                required_features = self.model_trainer.feature_columns
                missing_after_indicators = [f for f in required_features if f not in hist_df_indicators.columns]
                if missing_after_indicators:
                     # Attempt to add missing features with 0, common for sentiment/trends if not collected
                     st.warning(f"Missing features after indicator calculation: {missing_after_indicators}. Filling with 0.")
                     for feat in missing_after_indicators: hist_df_indicators[feat] = 0.0 # Fill with float 0.0
                     # Recheck after filling - if still missing, it's a critical error
                     missing_after_fill = [f for f in required_features if f not in hist_df_indicators.columns]
                     if missing_after_fill:
                          status_container.error(f"Still missing critical features after fill attempt: {missing_after_fill}. Cannot predict.")
                          return

                # Select only the required features *as defined during model training*
                # This ensures consistency even if the scaler expects more/different features
                model_required_features = model_info.get('feature_columns')
                if not model_required_features:
                     status_container.error("Feature columns not found in model_info.json. Cannot proceed.")
                     return

                # Ensure all *model-required* features are present in the historical data (after indicator calculation and filling)
                missing_model_features = [f for f in model_required_features if f not in hist_df_indicators.columns]
                if missing_model_features:
                     # This case should ideally be handled by the earlier filling step, but as a safeguard:
                     st.warning(f"Model required features missing after indicator calculation/fill: {missing_model_features}. Filling with 0 again.")
                     for feat in missing_model_features: hist_df_indicators[feat] = 0.0

                try:
                    # Select only the features the *model* was trained on, in the correct order
                    input_data_for_model = hist_df_indicators[model_required_features].copy()
                    st.info(f"Selected {len(model_required_features)} features for model input.")
                except KeyError as e:
                     status_container.error(f"Error selecting model required features: {e}. Columns available: {hist_df_indicators.columns.tolist()}")
                     return
                except Exception as e_sel:
                     status_container.error(f"Unexpected error selecting model features: {e_sel}")
                     return


                # --- Determine Scaler Key ---
                scaler_key = 'combined' # Default assumption
                available_scaler_keys = list(self.model_trainer.scalers.keys())

                if len(available_scaler_keys) == 1:
                    # If only one scaler exists, use its key
                    scaler_key = available_scaler_keys[0]
                    status_container.info(f"Using single available scaler key: '{scaler_key}'")
                elif ticker_symbol in available_scaler_keys:
                    # If a scaler exists for the specific ticker, prefer that
                    scaler_key = ticker_symbol
                    status_container.info(f"Using ticker-specific scaler key: '{scaler_key}'")
                elif 'combined' in available_scaler_keys:
                    # If 'combined' exists and ticker-specific doesn't, use 'combined'
                    scaler_key = 'combined'
                    status_container.info(f"Using 'combined' scaler key.")
                elif 'default' in available_scaler_keys:
                    # Fallback to 'default' if it exists and others don't match
                    scaler_key = 'default'
                    status_container.info(f"Using 'default' scaler key.")
                else:
                    # No suitable key found
                    status_container.error(f"Could not determine a suitable scaler key. Available keys: {available_scaler_keys}. Ticker: {ticker_symbol}. Cannot proceed.")
                    return

                # Final check if the determined key actually exists (should be redundant but safe)
                if scaler_key not in self.model_trainer.scalers:
                     status_container.error(f"Determined scaler key '{scaler_key}' not found in loaded scalers dictionary. Cannot proceed.")
                     return

                scaler = self.model_trainer.scalers[scaler_key]
                status_container.info(f"Scaling features using scaler: '{scaler_key}'")

                # --- Feature Count Check & Alignment (REMOVED - Scaling now uses input_data_for_model) ---
                # The alignment logic previously here compared input_data (potentially 38 features from hist_df_indicators)
                # with the scaler's expected features. This caused issues if the scaler expected more features
                # than the model was trained on.
                # We now scale ONLY the features the model expects (input_data_for_model).

                # --- Scaling ---
                try:
                    # Scale only the features required by the model
                    scaled_features = scaler.transform(input_data_for_model)
                except ValueError as ve:
                     # Check if the error is due to feature mismatch between input_data_for_model and scaler
                     if 'X has' in str(ve) and 'features, but StandardScaler is expecting' in str(ve):
                          status_container.error(f"Feature count mismatch during scaling: Input data has {input_data_for_model.shape[1]} features (expected by model), but scaler '{scaler_key}' was fitted on {scaler.n_features_in_} features. Model and scaler are incompatible. Please retrain the model.")
                          st.error(f"Model Features ({len(model_required_features)}): {model_required_features}")
                          st.error(f"Scaler Features ({scaler.n_features_in_}): {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'Not available'}")
                     else:
                          status_container.error(f"Error during scaling: {ve}")
                          st.error(f"Input data shape: {input_data_for_model.shape}, Columns: {input_data_for_model.columns.tolist()}")
                     return
                except Exception as scale_err:
                     status_container.error(f"Unexpected error during scaling: {scale_err}")
                     return

                # Create DataFrame with scaled features, using the model's required feature names
                scaled_df = pd.DataFrame(scaled_features, index=input_data_for_model.index, columns=model_required_features)

                # Get the last sequence needed for the first prediction
                # Ensure we use the scaled_df which now has the correct number of columns (matching the model)
                last_sequence_scaled = scaled_df.iloc[-self.model_trainer.lookback_window:].values
                if len(last_sequence_scaled) < self.model_trainer.lookback_window:
                    status_container.error(f"Not enough historical data ({len(last_sequence_scaled)} points) for lookback window ({self.model_trainer.lookback_window}).")
                    return

                progress_bar.progress(60)
                status_container.info("Making predictions...")

                # --- Prediction Logic ---
                predictions_output = [] # Store raw model outputs (logits for classification, scaled values for regression)
                current_sequence_np = last_sequence_scaled.copy() # Start with the last known sequence
                is_classification = model_info.get('is_classification', False) # Get model type

                self.model_trainer.model.eval()
                with torch.no_grad():
                    for i in range(forecast_days):
                        # Prepare current sequence as tensor
                        current_sequence_tensor = torch.tensor(current_sequence_np, dtype=torch.float32).unsqueeze(0).to(device)

                        # Ensure tensor shape matches model input
                        expected_features = len(model_required_features)
                        if current_sequence_tensor.shape[-1] != expected_features:
                            status_container.error(f"Tensor shape mismatch! Expected {expected_features}, got {current_sequence_tensor.shape[-1]}. Cannot predict.")
                            return # Stop prediction

                        # Predict the next step (raw output: logit or scaled value)
                        pred_output_tensor = self.model_trainer.model(current_sequence_tensor)
                        pred_output_value = pred_output_tensor.item()
                        predictions_output.append(pred_output_value)

                        # --- Iterative Update (Only for Regression) ---
                        if not is_classification:
                            # Construct the *next* input feature vector using the predicted scaled value
                            last_actual_features_scaled = current_sequence_np[-1, :]
                            target_col_name = model_info.get('target_variable', 'Close_Next') # Target used during training

                            try:
                                # Find index in the *model's* required_features list
                                target_index = model_required_features.index(target_col_name)
                            except ValueError:
                                # Try finding 'Close' if target was 'Close_Next' or similar
                                try: target_index = model_required_features.index('Close')
                                except ValueError:
                                    status_container.error(f"Cannot find target '{target_col_name}' or 'Close' in features {model_required_features} for iterative regression prediction.")
                                    return # Stop prediction

                            # Ensure next_feature_vector_scaled has the correct shape
                            next_feature_vector_scaled = last_actual_features_scaled.copy()
                            if next_feature_vector_scaled.shape[0] != len(model_required_features):
                                status_container.error(f"Shape mismatch constructing next feature vector. Expected {len(model_required_features)}, got {next_feature_vector_scaled.shape[0]}.")
                                return # Stop prediction

                            # Update the value at the correct index with the predicted scaled value
                            next_feature_vector_scaled[target_index] = pred_output_value

                            # Append the new vector and remove the oldest one
                            current_sequence_np = np.vstack((current_sequence_np[1:], next_feature_vector_scaled))
                        else:
                            # For classification, we don't iterate using the output probability/logit.
                            # We just predict each day based on the *last known actual data*.
                            # This means the loop runs, but `current_sequence_np` doesn't change after the first iteration.
                            # Alternatively, break after the first prediction if only single-step classification is desired.
                            # Let's predict for all forecast_days using the same initial sequence.
                            pass # Keep using the initial `last_sequence_scaled` for subsequent classification predictions

                    # --- End Prediction Loop ---


                    progress_bar.progress(80)
                    status_container.info("Processing predictions and creating visualization...")

                    # Process raw predictions based on model type
                    target_col_name = model_info.get('target_variable', 'Close_Next') # Target name used during training
                    predictions_orig = [] # For regression
                    probabilities_up = [] # For classification

                    # Create forecast dataframe
                    last_hist_date = hist_df_indicators.index[-1]
                    # Ensure forecast_dates matches the number of predictions we actually have
                    forecast_dates = pd.date_range(start=last_hist_date + timedelta(days=1), periods=len(predictions_output), freq='B') # Use Business Day freq
                    forecast_data = {'Date': forecast_dates}

                    if is_classification:
                        # Convert logits to probabilities
                        probabilities_up = (1 / (1 + np.exp(-np.array(predictions_output)))).tolist()
                        forecast_data['Probability (Up)'] = probabilities_up
                        forecast_data['Predicted_Trend'] = (np.array(probabilities_up) > 0.5).astype(int)
                        forecast_data['Trend_Label'] = ['Up' if p > 0.5 else 'Down' for p in probabilities_up]
                        forecast_data['Confidence (%)'] = abs(np.array(probabilities_up) - 0.5) * 2 * 100
                    else: # Regression
                        # Inverse transform scaled predictions
                        predictions_orig = [self.model_trainer.inverse_transform(p, ticker=scaler_key, column_name=target_col_name) for p in predictions_output]
                        forecast_data['Predicted'] = predictions_orig

                        # Confidence Interval Calculation (Refined)
                        try:
                            mse = model_info.get('metrics', {}).get('mse') # Get MSE from training
                            if mse is not None and mse > 0:
                                # Calculate margin in scaled units
                                z_score = stats.norm.ppf(1 - (1 - confidence_level / 100) / 2)
                                scaled_margin = z_score * np.sqrt(mse)

                                # Calculate scaled bounds
                                upper_scaled = np.array(predictions_output) + scaled_margin
                                lower_scaled = np.array(predictions_output) - scaled_margin

                                # Inverse transform bounds
                                forecast_data['Upper_Bound'] = [self.model_trainer.inverse_transform(p, ticker=scaler_key, column_name=target_col_name) for p in upper_scaled]
                                forecast_data['Lower_Bound'] = [self.model_trainer.inverse_transform(p, ticker=scaler_key, column_name=target_col_name) for p in lower_scaled]
                            else:
                                st.warning("MSE not found or is zero in model info. Cannot calculate confidence interval.")
                                forecast_data['Lower_Bound'] = predictions_orig # No interval
                                forecast_data['Upper_Bound'] = predictions_orig
                        except Exception as ci_err:
                            st.warning(f"Could not calculate confidence interval: {ci_err}")
                            forecast_data['Lower_Bound'] = predictions_orig
                            forecast_data['Upper_Bound'] = predictions_orig

                    forecast_df = pd.DataFrame(forecast_data)

                    # Visualization
                    # Display Results
                    with results_container:
                        st.subheader("Prediction Results")
                        # Use st.dataframe for interactivity
                        st.dataframe(forecast_df.style.format({
                            'Probability (Up)': '{:.2%}',
                            'Confidence (%)': '{:.1f}%',
                            'Predicted': '${:.2f}',
                            'Lower_Bound': '${:.2f}',
                            'Upper_Bound': '${:.2f}'
                        }), use_container_width=True) # Replaced use_column_width
                        # Create plot showing history and predictions
                        if not isinstance(hist_df_indicators.index, pd.DatetimeIndex):
                            if 'Date' in hist_df_indicators.columns:
                                hist_df_indicators.set_index('Date', inplace=True)
                            else:
                                st.warning("Cannot set Date index for plotting.")

                        # Pass is_classification to chart function
                        fig_pred = self.create_price_chart(hist_df_indicators.iloc[-60:], ticker_symbol, forecast_df, is_classification)
                        st.plotly_chart(fig_pred, use_container_width=True)

                        # Add probability chart for classification
                        if is_classification:
                            st.subheader("Predicted Trend Probability (Up)")
                            fig_prob = px.line(forecast_df, x='Date', y='Probability (Up)', range_y=[0,1], markers=True)
                            fig_prob.add_hline(y=0.5, line_dash="dash", line_color="grey")
                            fig_prob.update_layout(template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, height=300)
                            st.plotly_chart(fig_prob, use_container_width=True)

                        st.info("⚠️ Disclaimer: Predictions are based on historical data and model assumptions. Not financial advice.")

                    progress_bar.progress(100)
                    status_container.success(f"Prediction generated for {ticker_symbol}!")

            except Exception as e:
                status_container.error(f"Error during prediction: {e}")
                st.error("Stack trace:"); st.code(traceback.format_exc())

    def render_home_page(self):
        """Render the home page"""
        st.title("📈 Stock Price Prediction System")
        st.markdown("Welcome! Use the sidebar to navigate.")
        # Use the about section content here for the home page
        self.show_about_section() # Call the about section rendering

    def show_about_section(self):
        """Display about section content (can be called by home page)"""
        # This content was previously duplicated in render_home_page, now centralized
        st.markdown("""
        An advanced tool for analyzing stock market data and making predictions using deep learning.
        Combines technical indicators, market data, and optional sentiment analysis.
        """)
        st.subheader("🚀 Key Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 Data Collection"); st.markdown("- OHLCV\n- Reddit Sentiment\n- Google Trends\n- Indicators")
            if st.button("Go to Data Collection", key="about_data", use_container_width=True): st.session_state.app_mode = "Data Collection"; st.rerun()
        with col2:
            st.markdown("### 🧠 Model Training"); st.markdown("- LSTM Network\n- Custom Hyperparams\n- Feature Selection\n- Metrics")
            if st.button("Go to Model Training", key="about_train", use_container_width=True): st.session_state.app_mode = "Model Training"; st.rerun()
        with col3:
            st.markdown("### 🔮 Prediction"); st.markdown("- Forecasting\n- Trend Prediction\n- Confidence Intervals\n- Visualizations")
            if st.button("Go to Prediction", key="about_predict", use_container_width=True): st.session_state.app_mode = "Prediction"; st.rerun()

        st.subheader("📌 System Status")
        status_cols = st.columns(3)
        # Check processed files for data status
        processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')] if os.path.exists(PROCESSED_DATA_DIR) else []
        status_cols[0].metric("Data Status", f"✅ {len(processed_files)} processed files" if processed_files else "❌ No processed data")
        model_path = os.path.join(MODEL_DIR, 'stock_prediction_model.pth')
        model_info_path = os.path.join(MODEL_DIR, 'model_info.json')
        model_status = "❌ No model"
        if os.path.exists(model_path) and os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r') as f: model_info = json.load(f)
                model_status = f"✅ Trained {model_info.get('training_date', '')}"
            except: model_status = "✅ Model files exist"
        status_cols[1].metric("Model Status", model_status)
        gpu_status = f"✅ {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
        status_cols[2].metric("Hardware", gpu_status)

        st.subheader("💡 Getting Started")
        st.markdown("1. **Collect Data**\n2. **Train Model**\n3. **Make Predictions**")
        st.info("⚠️ Disclaimer: For educational purposes only. Not financial advice.")

    def run(self):
        """Run the main Streamlit app"""
        if 'app_mode' not in st.session_state: st.session_state.app_mode = 'Home'
        if 'settings' not in st.session_state: st.session_state.settings = {} # Init settings
        self._load_settings() # Load saved settings

        self.render_sidebar()
        app_mode = st.session_state.get('app_mode', 'Home')

        page_render_map = {
            "Home": self.render_home_page,
            "Data Collection": self.render_data_collection_page,
            "Model Training": self.render_model_training_page,
            "Prediction": self.render_prediction_page,
            "Settings": self.render_settings_page
        }
        render_func = page_render_map.get(app_mode, self.render_home_page)
        render_func()

    def _load_settings(self):
         """Load settings from settings.json if it exists"""
         settings_path = 'settings.json'
         if os.path.exists(settings_path):
             try:
                 with open(settings_path, 'r') as f:
                     loaded_settings = json.load(f)
                 # Update session state, preserving existing keys if not in file
                 for key, value in loaded_settings.items():
                     # Load into session state directly
                     st.session_state[key] = value
                 # Ensure required API keys are present in session_state even if empty/not in file
                 for key in ['reddit_client_id', 'reddit_client_secret', 'reddit_user_agent']:
                      if key not in st.session_state:
                           st.session_state[key] = loaded_settings.get(key, '') # Use loaded value or empty string
             except Exception as e:
                 st.warning(f"Could not load settings from {settings_path}: {e}")
         # Ensure default API keys are initialized if not loaded/present
         if 'reddit_client_id' not in st.session_state: st.session_state['reddit_client_id'] = ''
         if 'reddit_client_secret' not in st.session_state: st.session_state['reddit_client_secret'] = ''
         if 'reddit_user_agent' not in st.session_state: st.session_state['reddit_user_agent'] = 'StockPredictionApp/1.0'


    def _save_settings(self):
        """Save settings to settings.json"""
        try:
            # Gather relevant settings from session_state
            settings_to_save = {
                'default_ticker': st.session_state.get('default_ticker'),
                'default_lookback': st.session_state.get('default_lookback'),
                'default_forecast': st.session_state.get('default_forecast'),
                'reddit_client_id': st.session_state.get('reddit_client_id'),
                'reddit_client_secret': st.session_state.get('reddit_client_secret'),
                'reddit_user_agent': st.session_state.get('reddit_user_agent'),
                # Add other settings as needed
            }
            # Remove None values before saving
            settings_to_save = {k: v for k, v in settings_to_save.items() if v is not None}

            with open('settings.json', 'w') as f:
                json.dump(settings_to_save, f, indent=4)
        except Exception as e:
            st.warning(f"Failed to save settings: {e}")

    def run_data_collection_cli(self):
        """Run data collection from command line"""
        print("Starting CLI data collection...")
        num_cli_companies = 3 # Example: fetch top 3 for CLI
        start_cli_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        end_cli_date = datetime.now().strftime('%Y-%m-%d')

        collector = DataCollector()

        def cli_progress(current, total): print(f"\rProgress: {current}/{total}", end='')
        def cli_status(message): print(f"\nStatus: {message}")

        try:
            # Fetch top N companies list first
            cli_status(f"Fetching top {num_cli_companies} companies list...")
            # get_top_companies now returns list of dicts and sets self.companies
            top_companies_list = collector.get_top_companies(num_companies=num_cli_companies, status_callback=cli_status)
            if not top_companies_list:
                 print("\nError: Failed to fetch companies list. Aborting.")
                 return False
            # collector.companies is now set internally by get_top_companies

            # Collect stock data (saved individually now)
            cli_status(f"Collecting stock data from {start_cli_date} to {end_cli_date}...")
            # collect_stock_data uses collector.companies internally
            stock_data_dict = collector.collect_stock_data(start_date=start_cli_date, end_date=end_cli_date, progress_callback=cli_progress, status_callback=cli_status)
            if not stock_data_dict:
                 print("\nError: Failed to collect stock data. Aborting.")
                 return False

            # Process data (Reddit/Trends disabled for basic CLI)
            cli_status("Processing data...")
            # process_data now saves files individually and returns list of processed tickers
            # Corrected call signature: removed extra {} arguments
            processed_tickers, features = collector.process_data(stock_data_dict, progress_callback=cli_progress, status_callback=cli_status)

            print("\nCLI Data collection complete.")
            if processed_tickers:
                print(f"Processed {len(processed_tickers)} tickers: {', '.join(processed_tickers)}")
                print(f"Features from last ticker ({processed_tickers[-1]}): {features}")
            else:
                print("No tickers were processed successfully.")
            return True
        except Exception as e:
            print(f"\nError during CLI data collection: {e}")
            traceback.print_exc()
            return False

    def run_model_training_cli(self):
        """Run model training from command line using data from processed folder"""
        print("Starting CLI model training...")
        # Check if processed data directory exists and has files
        if not os.path.exists(PROCESSED_DATA_DIR) or not any(f.endswith('_processed_data.csv') for f in os.listdir(PROCESSED_DATA_DIR)):
            print(f"Error: No processed data files found in {PROCESSED_DATA_DIR}. Run data collection first."); return False

        # Select files to use (e.g., use all available processed files for CLI training)
        processed_files = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')]
        print(f"Found {len(processed_files)} processed data files for training.")

        settings = { # Default CLI settings
            'target_variable': 'Close_Next', 'lookback_window': 30, 'test_split': 0.2,
            'epochs': 10, 'batch_size': 64, 'learning_rate': 0.001,
            'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.2,
            'rnn_type': 'lstm', # Add rnn_type for CLI, default to lstm
            'bidirectional': False,
            'is_classification': False,
        }
        settings['is_classification'] = settings['target_variable'] == 'Trend'

        # Import ModelTrainer here
        try:
            from model_training import ModelTrainer
        except ImportError:
             print("Error: Failed to import ModelTrainer for CLI training.")
             return False
        # Define a simple param grid for CLI (can be expanded)
        cli_param_grid = {
            'rnn_type': [settings.get('rnn_type', 'lstm')],
            'hidden_dim': [settings.get('hidden_dim', 64)],
            'num_layers': [settings.get('num_layers', 1)],
            'dropout_prob': [settings.get('dropout', 0.2)],
            'learning_rate': [settings.get('learning_rate', 0.001)],
            'bidirectional': [settings.get('bidirectional', False)],
            'batch_size': [settings.get('batch_size', 64)]
        }

        # Determine feature columns (attempt loading from first file)
        feature_columns_cli = None
        try:
            temp_df = pd.read_csv(processed_files[0])
            # Use Price_Increase as target column name
            feature_columns_cli = [col for col in temp_df.columns if col not in ['Date', 'Ticker', 'Company', 'Close_Next', 'Price_Change', 'Price_Increase']]
            print(f"Using {len(feature_columns_cli)} features from {os.path.basename(processed_files[0])}.")
        except Exception as e:
            print(f"Error reading features from {processed_files[0]}: {e}")
            # Use a default fallback if reading fails (ensure this list is maintained)
            feature_columns_cli = [ # Fallback list
                'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Compound', 'Positive', 'Neutral', 'Negative', 'Count', 'Interest', 'FEDFUNDS',
                'SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SlowK', 'SlowD', 'ADX',
                'Chaikin_AD', 'OBV', 'ATR', 'Williams_R', 'ROC', 'CCI', 'Close_Open_Ratio', 'High_Low_Diff', 'Close_Prev_Ratio', 'Close_Lag_1', 'Volume_Lag_1',
                'Compound_Lag_1', 'Interest_Lag_1', 'FEDFUNDS_Lag_1', 'Close_Lag_3', 'Volume_Lag_3', 'Compound_Lag_3', 'Interest_Lag_3', 'FEDFUNDS_Lag_3',
                'Close_Lag_5', 'Volume_Lag_5', 'Compound_Lag_5', 'Interest_Lag_5', 'FEDFUNDS_Lag_5', 'Volatility_20D', 'Day_Of_Week' ]
            try: # Filter fallback by actual columns in the first file
                temp_df_cols = pd.read_csv(processed_files[0], nrows=1).columns
                feature_columns_cli = [f for f in feature_columns_cli if f in temp_df_cols]
                print(f"Warning: Using filtered default feature set ({len(feature_columns_cli)}).")
            except Exception: print(f"Warning: Using unfiltered default feature set ({len(feature_columns_cli)}).")

        if not feature_columns_cli:
            print("Error: Could not determine feature columns for CLI training.")
            return False

        # Call the main training function from model_training.py
        try:
            print(f"Starting CLI training run with {len(processed_files)} files...")
            # Use run_optuna_optimization for CLI as well, with a small number of trials
            best_model, best_scaler, best_params, best_metrics = run_optuna_optimization(
                processed_files=processed_files,
                feature_columns=feature_columns_cli,
                param_grid=cli_param_grid, # Pass the simple grid defined above for CLI
                n_trials=5, # Run a small number of trials for CLI example
                epochs=settings['epochs'],
                num_splits=3, # Use a fixed number of splits for CLI example, e.g., 3
                lookback_window=settings['lookback_window']
            )

            if best_model:
                print("\nCLI Training finished successfully.")
                # Saving is handled within run_optuna_optimization
                return True
            else:
                print("\nCLI Training failed or found no suitable model.")
                return False

        except Exception as e:
            print(f"Error during CLI model training execution: {e}")
            traceback.print_exc()
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Price Prediction System')
    parser.add_argument('--mode', type=str, default='streamlit', choices=['streamlit', 'cli'], help='Mode: streamlit (default) or cli')
    parser.add_argument('--cli-action', type=str, default='data_collection', choices=['data_collection', 'model_training'], help='Action for CLI mode')
    args = parser.parse_args()

    app = StockPredictionApp()
    if args.mode == 'streamlit':
        app.run()
    elif args.mode == 'cli':
        if args.cli_action == 'data_collection':
            app.run_data_collection_cli()
        elif args.cli_action == 'model_training':
            app.run_model_training_cli()
