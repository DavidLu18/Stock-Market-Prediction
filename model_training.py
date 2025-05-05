import os
import pandas as pd
import numpy as np
import joblib
import json
import traceback
import time
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# Import LightGBM and check GPU support
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    try:
        lgb.LGBMClassifier(device='gpu')
        LGBM_GPU_SUPPORT = True
        print("LightGBM GPU support detected.")
    except Exception as e:
        print(f"Warning: LightGBM GPU support check failed: {e}. Will attempt CPU training.")
        LGBM_GPU_SUPPORT = False
except ImportError:
    LIGHTGBM_AVAILABLE = False; LGBM_GPU_SUPPORT = False
    print("ERROR: lightgbm library not found."); exit()

# Directory setup (Assume running relative to app.py or specific environment)
# If running standalone, adjust BASE_DIR
try: BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes model_training is in a subdir
except NameError: BASE_DIR = os.getcwd() # Fallback for interactive sessions
DATA_DIR_INPUT = '/kaggle/input/newway01/data' # <<< Point to WHERE PROCESSED DATA IS <<<
MODEL_DIR_OUTPUT = '/kaggle/working/' # <<< Point to WHERE MODELS SHOULD BE SAVED <<<

# Use environment variables if defined (e.g., for Kaggle)
PROCESSED_DATA_DIR = '/kaggle/input/newway01/data/processed'
MODEL_DIR = os.environ.get('MODEL_OUTPUT_PATH', MODEL_DIR_OUTPUT)

# Create MODEL_DIR if it doesn't exist
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(PROCESSED_DATA_DIR):
    raise FileNotFoundError(f"Input data directory not found: {PROCESSED_DATA_DIR}")
else:
    print(f"Using processed data from: {PROCESSED_DATA_DIR}")
    print(f"Saving models to: {MODEL_DIR}")


# --- Feature Engineering Function (DISABLED) ---
# >>> This function is NOT called when using pre-processed features <<<
def engineer_features(df):
    print("SKIPPING internal feature engineering. Using features from input CSV.")
    return df

# --- Helper Functions (Keep create_future_target, prepare_data, get_splits) ---

def create_future_target(df, target_col_base='Close', horizon=5):
    """Creates the N-day future return and binary target variable."""
    # (Giữ nguyên logic)
    print(f"  Creating {horizon}-day future target based on '{target_col_base}'...")
    if target_col_base not in df.columns: return df, None
    if not pd.api.types.is_numeric_dtype(df[target_col_base]): return df, None
    df_copy = df.copy(); target_name = f'Price_Increase_{horizon}D'
    try:
        df_copy[f'{target_col_base}_Future_{horizon}D'] = df_copy[target_col_base].shift(-horizon)
        current_price = df_copy[target_col_base]
        future_price = df_copy[f'{target_col_base}_Future_{horizon}D']
        df_copy[f'Future_Return_{horizon}D'] = np.where(
            current_price.isna() | (current_price == 0) | future_price.isna(), np.nan,
            (future_price - current_price) / current_price)
        df_copy[target_name] = np.where(df_copy[f'Future_Return_{horizon}D'].isna(), np.nan,
                                       (df_copy[f'Future_Return_{horizon}D'] > 0).astype(int))
        df_out = pd.merge(df, df_copy[[target_name]], left_index=True, right_index=True, how='left')
        original_len = len(df_out)
        df_out.dropna(subset=[target_name], inplace=True)
        rows_dropped = original_len - len(df_out)
        print(f"  Target '{target_name}' created. Dropped {rows_dropped} rows.")
        df_out[target_name] = df_out[target_name].astype(int)
        return df_out, target_name
    except Exception as e: print(f"  Error creating target: {e}"); return df, None

def prepare_data_tree_nday(df, feature_columns, target_column_nday):
    """Prepares data for tree models using pre-existing features."""
    # (Giữ nguyên logic - làm sạch NaN/inf, chọn cột)
    if target_column_nday not in df.columns: print(f"E: Target '{target_column_nday}' not found."); return None, None
    df_clean = df.copy()
    feature_columns_present = [f for f in feature_columns if f in df_clean.columns]
    missing = list(set(feature_columns) - set(feature_columns_present))
    if missing: print(f"W: {len(missing)} features missing: {missing[:5]}... Using {len(feature_columns_present)}.")
    if not feature_columns_present: print("E: No usable features."); return None, None
    cols_to_keep = feature_columns_present + [target_column_nday]
    try: df_clean = df_clean[cols_to_keep]
    except KeyError as e: print(f"E: KeyError selecting columns: {e}. Available: {df_clean.columns.tolist()}"); return None, None
    inf_before = np.isinf(df_clean[feature_columns_present].values).sum()
    if inf_before > 0: df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feature_columns_present:
        if not pd.api.types.is_numeric_dtype(df_clean[col]): df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    nan_after_convert = df_clean[feature_columns_present].isna().sum().sum()
    if nan_after_convert > 0 or inf_before > 0:
        df_clean = df_clean.ffill().bfill()
        df_clean[feature_columns_present] = df_clean[feature_columns_present].fillna(0)
    if df_clean[target_column_nday].isna().any():
        nan_target = df_clean[target_column_nday].isna().sum()
        print(f"W: {nan_target} NaN in target '{target_column_nday}'. Dropping rows.")
        df_clean.dropna(subset=[target_column_nday], inplace=True)
    if df_clean.empty: print("E: DataFrame empty after cleaning."); return None, None
    X = df_clean[feature_columns_present]
    y = df_clean[target_column_nday].astype(int)
    if X.isna().any().any() or np.isinf(X.values).any(): print("E: NaN/Inf still in features! Final fillna(0)."); X = X.fillna(0).replace([np.inf, -np.inf], 0)
    return X, y

def get_walk_forward_splits_rf(df, num_splits, min_train_perc=0.3, test_period_days=None, gap_days=1):
    """Generates walk-forward splits."""
    # (Giữ nguyên logic - kiểm tra ngày tháng bất thường đã được thêm)
    if not isinstance(df.index, pd.DatetimeIndex):
        try: df.index = pd.to_datetime(df.index); df = df.sort_index(); print("Converted index to DatetimeIndex.")
        except Exception as e: print(f"W: Index conversion failed: {e}. Proceeding..."); df = df.sort_index()
    total_len = len(df); min_required = int(num_splits * 2 * max(10, gap_days)) + 100
    if total_len < min_required: raise ValueError(f"Data size ({total_len}) too small for {num_splits} splits/gap {gap_days}.")
    initial_train_size = max(50, int(total_len * min_train_perc))
    if test_period_days is None: rem = total_len - initial_train_size - (num_splits * gap_days); val_size = max(10, rem // num_splits) if rem > 0 else 10
    else: val_size = test_period_days
    max_initial_train = total_len - num_splits * (val_size + gap_days)
    initial_train_size = min(initial_train_size, max(50, max_initial_train))
    if initial_train_size < 50: raise ValueError(f"Initial train size ({initial_train_size}) too small.")
    print(f"\nGenerating {num_splits} walk-forward splits:"); print(f"  Total data: {total_len}, Initial train: ~{initial_train_size}, Gap: {gap_days}d")
    train_end_idx = initial_train_size; split_count = 0
    for i in range(num_splits):
        current_split_num = i + 1; train_start_idx = 0
        validation_start_idx = train_end_idx + gap_days
        if test_period_days: validation_end_idx = validation_start_idx + test_period_days
        else: rem_val = total_len - validation_start_idx; step_size = max(10, rem_val // (num_splits - i)) if rem_val > 0 and (num_splits - i)>0 else 10; validation_end_idx = validation_start_idx + step_size
        validation_end_idx = min(validation_end_idx, total_len)
        if validation_start_idx >= total_len or validation_end_idx <= validation_start_idx: print(f"W: Not enough data for val split {current_split_num}. Stopping."); break
        train_df = df.iloc[train_start_idx : train_end_idx]; validation_df = df.iloc[validation_start_idx : validation_end_idx]
        if train_df.empty or validation_df.empty: print(f"W: Empty train/val set for split {current_split_num}. Stopping."); break
        split_count += 1
        print(f"\n  Split {split_count}/{num_splits}:")
        try:
            print(f"    Train: {train_df.index.min().date()} - {train_df.index.max().date()} ({len(train_df)} pts)")
            print(f"    Val:   {validation_df.index.min().date()} - {validation_df.index.max().date()} ({len(validation_df)} pts)")
            if validation_df.index.max() > pd.Timestamp.now() + pd.Timedelta(days=1): print(f"    <<<< CRITICAL WARNING >>>> Val data has future dates ({validation_df.index.max().date()})!")
        except AttributeError: print(f"    Train/Val Index not datetime: {train_df.index.min()} - {validation_df.index.max()}")
        yield train_df, validation_df
        train_end_idx = validation_end_idx - gap_days; train_end_idx = max(initial_train_size, train_end_idx)
    if split_count < num_splits: print(f"\nW: Only generated {split_count}/{num_splits} splits.")


# --- LGBM Training/Evaluation Function ---
def train_evaluate_lgbm_fold(train_df, validation_df, feature_columns, target_column_nday, params, use_gpu):
    """Trains and evaluates a LightGBM model on a single fold."""
    # (Giữ nguyên logic)
    if not LIGHTGBM_AVAILABLE: return None, {'error': 'LGBM not available'}, None
    model, scaler, metrics = None, None, {}; fold_start_time = time.time()
    try:
        X_train_raw, y_train = prepare_data_tree_nday(train_df, feature_columns, target_column_nday)
        X_val_raw, y_val = prepare_data_tree_nday(validation_df, feature_columns, target_column_nday)
        if X_train_raw is None or X_val_raw is None or y_train is None or y_val is None or X_train_raw.empty or X_val_raw.empty: return None, {'error': 'Data prep failed'}, None
        if len(y_train.unique()) < 2: return None, {'error': f'Single class ({y_train.unique()}) in train'}, None
        if len(y_val) == 0: return None, {'error': 'Empty validation target'}, None
        if len(X_train_raw) < 20 or len(X_val_raw) < 10: return None, {'error': f'Samples train({len(X_train_raw)}<20)/val({len(X_val_raw)}<10)'}, None
        scaler = StandardScaler(); X_train = scaler.fit_transform(X_train_raw); X_val = scaler.transform(X_val_raw)
        lgbm_class_weight_param = params.pop('class_weight', None); scale_pos_weight_param = params.pop('scale_pos_weight', None)
        effective_scale_pos_weight = None; lgbm_class_weight = None
        if lgbm_class_weight_param == 'balanced': lgbm_class_weight = 'balanced'
        elif scale_pos_weight_param is not None: effective_scale_pos_weight = scale_pos_weight_param
        else: n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum(); effective_scale_pos_weight = n_neg / n_pos if n_pos > 0 and n_neg > 0 else None
        lgbm_extra_params = {'device': 'gpu'} if use_gpu and LGBM_GPU_SUPPORT else {'device': 'cpu'}
        model = lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42, n_jobs=-1, class_weight=lgbm_class_weight, scale_pos_weight=effective_scale_pos_weight, **params, **lgbm_extra_params)
        eval_set = [(X_val, y_val)]; callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(X_train, y_train, eval_set=eval_set, eval_metric='auc', callbacks=callbacks)
        y_pred_proba = model.predict_proba(X_val)[:, 1]; y_pred_class = (y_pred_proba >= 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred_class); precision = precision_score(y_val, y_pred_class, pos_label=1, zero_division=0); recall = recall_score(y_val, y_pred_class, pos_label=1, zero_division=0); f1 = f1_score(y_val, y_pred_class, pos_label=1, zero_division=0); roc_auc = np.nan
        if len(np.unique(y_val)) > 1:
            try: roc_auc = roc_auc_score(y_val, y_pred_proba)
            except ValueError as e: print(f"W: ROC AUC calc failed: {e}"); roc_auc = np.nan
        else: roc_auc = 0.0
        try: cm = confusion_matrix(y_val, y_pred_class)
        except ValueError: cm = np.array([[0,0],[0,0]]); # Handle single class y_val
        metrics = {'accuracy': accuracy, 'roc_auc': roc_auc if pd.notna(roc_auc) else 0.0, 'precision_positive': precision, 'recall_positive': recall, 'f1_positive': f1, 'confusion_matrix': cm.tolist(), 'num_val_samples': len(y_val), 'val_positive_ratio': y_val.mean() if len(y_val) > 0 else 0.0, 'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None else params.get('n_estimators', -1)}
        if lgbm_class_weight_param == 'balanced': params['class_weight'] = 'balanced' # Restore params
        if scale_pos_weight_param is not None: params['scale_pos_weight'] = scale_pos_weight_param
        return model, metrics, scaler
    except Exception as e: print(f"Error in fold: {e}"); #traceback.print_exc()
    return None, {'error': str(e)}, scaler # Return scaler even on error


# --- Optuna Objective Function ---
def objective_lgbm(trial, df_full_with_target, feature_columns, target_column_nday, num_splits, use_gpu):
    """Optuna objective function."""
    # (Giữ nguyên logic với không gian siêu tham số đã cải tiến)
    if not LIGHTGBM_AVAILABLE: raise ImportError("LGBM not found.")
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000, step=100), 'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.08, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300), 'max_depth': trial.suggest_int('max_depth', 5, 35), 'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
        'weighting_strategy': trial.suggest_categorical('weighting_strategy', ['none', 'scale_pos_weight_calc']),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.05), 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
    }
    weighting_strategy = params.pop('weighting_strategy'); params['class_weight'] = None; params['scale_pos_weight'] = None
    fold_metrics_list = []; fold_num = 0; total_fold_time = 0
    split_generator = get_walk_forward_splits_rf(df_full_with_target, num_splits, gap_days=3)
    for train_fold_df, validation_fold_df in split_generator:
        fold_num += 1; fold_start_time = time.time(); fold_params = params.copy()
        if weighting_strategy == 'scale_pos_weight_calc': fold_params['scale_pos_weight'] = None
        model, metrics, _ = train_evaluate_lgbm_fold(train_fold_df, validation_fold_df, feature_columns, target_column_nday, fold_params, use_gpu)
        fold_duration = time.time() - fold_start_time; total_fold_time += fold_duration
        if model and metrics and 'accuracy' in metrics and pd.notna(metrics['accuracy']) and 'error' not in metrics:
            fold_metrics_list.append(metrics); acc_fold = metrics.get('accuracy', 0.0); f1_fold = metrics.get('f1_positive', 0.0)
            objective_value_fold = (0.4 * acc_fold) + (0.6 * f1_fold); trial.report(-objective_value_fold, fold_num)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        else: error_msg = metrics.get('error', 'Unknown'); trial.report(float('-inf'), fold_num); # Report bad score
        if fold_num <= 1 and ('error' in metrics or not model): raise optuna.exceptions.TrialPruned(f"Failed fold {fold_num}: {error_msg}") # Prune early
    if not fold_metrics_list: return float('inf')
    avg_metrics = {}; metric_keys = ['accuracy', 'f1_positive', 'precision_positive', 'recall_positive', 'roc_auc']
    for key in metric_keys: valid_values = [m[key] for m in fold_metrics_list if key in m and pd.notna(m[key])]; avg_metrics[key] = np.mean(valid_values) if valid_values else 0.0
    avg_metrics['num_successful_folds'] = len(fold_metrics_list); avg_metrics['avg_val_positive_ratio'] = np.mean([m.get('val_positive_ratio', 0.0) for m in fold_metrics_list])
    valid_iters = [m['best_iteration'] for m in fold_metrics_list if m.get('best_iteration', -1) > 0]; avg_metrics['avg_best_iteration'] = np.mean(valid_iters) if valid_iters else -1.0
    trial.set_user_attr('avg_metrics', avg_metrics)
    final_acc = avg_metrics.get('accuracy', 0.0); final_f1 = avg_metrics.get('f1_positive', 0.0); final_weighted_score = (0.4 * final_acc) + (0.6 * final_f1)
    return -final_weighted_score

# --- Optuna Optimization Runner ---
# <<< SỬA ĐỔI: Bỏ tham số feature_columns, lấy từ PRE_EXISTING_FEATURES_LIST >>>
def run_optuna_optimization_lgbm(processed_files, forecast_horizon, n_trials, num_splits, use_gpu=True):
    """Runs Optuna optimization using PRE-EXISTING features from processed files."""
    global df_full_final
    if not LIGHTGBM_AVAILABLE: print("E: LightGBM not installed."); return None, None, None, None, None

    # --- LIST OF EXPECTED PRE-EXISTING FEATURES ---
    # <<< QUAN TRỌNG: Danh sách này phải khớp với các cột trong file *_processed_data.csv >>>
    PRE_EXISTING_FEATURES_LIST = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Compound', 'Positive', 'Neutral', 'Negative', 'Count',
        'Interest', 'FEDFUNDS', 'SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'RSI', 'MACD',
        'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SlowK', 'SlowD', 'ADX',
        'Chaikin_AD', 'OBV', 'ATR', 'Williams_R', 'ROC', 'CCI', 'Close_Open_Ratio', 'High_Low_Diff',
        'Close_Prev_Ratio', 'Close_Lag_1', 'Volume_Lag_1', 'Compound_Lag_1', 'Interest_Lag_1', 'FEDFUNDS_Lag_1',
        'Close_Lag_3', 'Volume_Lag_3', 'Compound_Lag_3', 'Interest_Lag_3', 'FEDFUNDS_Lag_3',
        'Close_Lag_5', 'Volume_Lag_5', 'Compound_Lag_5', 'Interest_Lag_5', 'FEDFUNDS_Lag_5',
        'Volatility_20D', 'Day_Of_Week'
        # <<< Bỏ các cột target gốc hoặc cột không phải feature ở đây >>>
    ]
    print(f"\n--- Starting Optuna (LGBM {forecast_horizon}D Target - Pre-existing Features) ---")
    print(f"Settings: Trials={n_trials}, Folds={num_splits}, GPU={use_gpu}")
    start_time = time.time()

    # --- 1. Load and Combine Data ---
    print("Loading and combining processed data...")
    all_dfs = []; loaded_files_count = 0
    target_tickers = ['AAPL', 'AMZN', 'GOOG', 'GOOGL', 'LLY', 'META', 'MSFT', 'NVDA', 'TSLA', 'V'] # Define target tickers

    for file_path in processed_files:
        filename = os.path.basename(file_path)
        # <<< Sửa đổi: Lấy ticker an toàn hơn >>>
        try: ticker_in_filename = filename.split('_processed_data.csv')[0].upper()
        except IndexError: continue # Skip if filename format is wrong
        if ticker_in_filename in target_tickers: # Chỉ load các ticker mục tiêu
            try:
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                if not isinstance(df.index, pd.DatetimeIndex): # Fallback parsing
                    df = pd.read_csv(file_path, parse_dates=['Date']).set_index('Date')
                if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                    df = df.sort_index()
                    # Bỏ các cột target gốc nếu có
                    cols_to_drop = ['Close_Next', 'Price_Change', 'Price_Increase']
                    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
                    if 'Ticker' not in df.columns: df['Ticker'] = ticker_in_filename # Add ticker if missing
                    all_dfs.append(df)
                    loaded_files_count += 1
                # else: print(f"W: Skipped {filename} due to index/empty issue.") # Bớt log
            except Exception as e: print(f"E: Loading {filename}: {e}")
        # else: print(f"Skipping {filename} (not in target list)") # Bớt log

    if not all_dfs: print(f"E: No valid data loaded for target tickers. Searched in {PROCESSED_DATA_DIR}"); return None, None, None, None, None
    print(f"\nLoaded data for {loaded_files_count} target tickers.")
    df_full = pd.concat(all_dfs, axis=0).sort_index()

    # Clean combined data (Remove rows where essential columns are all NaN)
    print(f"Combined shape before cleaning: {df_full.shape}")
    essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    initial_len = len(df_full); df_full.dropna(subset=essential_cols, how='all', inplace=True); rows_dropped = initial_len - len(df_full)
    if rows_dropped > 0: print(f"Dropped {rows_dropped} potentially invalid rows.")
    print(f"Combined shape after cleaning: {df_full.shape}")
    if df_full.empty: print("E: Combined DataFrame empty after cleaning."); return None, None, None, None, None
    try: # Verify date range
        min_date, max_date = df_full.index.min(), df_full.index.max()
        print(f"Verified Data Index Range: {min_date.date()} to {max_date.date()}")
        if max_date > pd.Timestamp.now() + pd.Timedelta(days=7): print(f"<<<< CRITICAL WARNING >>>> Max date {max_date.date()} is in the future!")
    except Exception as date_e: print(f"W: Could not verify date range: {date_e}")

    # --- 2. Feature Engineering --> SKIPPED (Using pre-processed) ---
    print("\nUsing pre-existing features from loaded files.")

    # --- 3. Create N-Day Target Variable ---
    print(f"\nCreating {forecast_horizon}-day target variable...")
    df_full_final, target_column_nday = create_future_target(df_full, horizon=forecast_horizon, target_col_base='Close')
    del df_full # Free memory

    if target_column_nday is None or df_full_final is None or df_full_final.empty: print("E: Failed to create target or DataFrame empty."); return None, None, None, None, None
    print(f"Target '{target_column_nday}' created. Final data shape: {df_full_final.shape}")
    print(f"Target distribution:\n{df_full_final[target_column_nday].value_counts(normalize=True)}")

    # --- 4. Final Feature Selection (Based on PRE_EXISTING_FEATURES_LIST) ---
    print("\nDetermining final feature set from expected list...")
    exclude_cols = ['Date', 'Ticker', 'Company', target_column_nday] + [col for col in df_full_final.columns if 'Future_' in col]
    available_cols = df_full_final.columns.tolist()
    # <<< SỬA ĐỔI: Dùng PRE_EXISTING_FEATURES_LIST làm cơ sở >>>
    final_feature_columns = [col for col in PRE_EXISTING_FEATURES_LIST if col in available_cols and col not in exclude_cols]
    missing_expected = list(set(PRE_EXISTING_FEATURES_LIST) - set(final_feature_columns) - set(exclude_cols))
    if missing_expected: print(f"  W: {len(missing_expected)} expected features missing/excluded: {missing_expected[:5]}...")

    print(f"Checking data types for {len(final_feature_columns)} potential features...")
    cols_to_drop_type = []
    for col in final_feature_columns:
        if pd.api.types.is_numeric_dtype(df_full_final[col]):
            if not (df_full_final[col].notna().sum() > 0 and np.isfinite(df_full_final[col]).sum() > 0): cols_to_drop_type.append(col) # Remove all NaN/Inf cols
        else: # Try converting non-numeric
            try: df_full_final[col] = pd.to_numeric(df_full_final[col], errors='raise')
            except: cols_to_drop_type.append(col) # Remove if conversion fails

    if cols_to_drop_type:
        final_feature_columns = [col for col in final_feature_columns if col not in cols_to_drop_type]
        print(f"  Removed {len(cols_to_drop_type)} non-numeric/invalid columns.")

    print(f"Final feature set: {len(final_feature_columns)} numeric features.")
    if not final_feature_columns: print("E: No numeric features remaining."); return None, None, None, None, None

    # --- 5. Prepare & Run Optuna ---
    # (Giữ nguyên logic setup và run Optuna)
    print("\nSetting up Optuna study...")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2, interval_steps=1)
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True, constant_liar=True)
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler=sampler, study_name=f'lgbm_{forecast_horizon}d_preproc_v3_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}')
    print(f"Starting Optuna optimization ({n_trials} trials)...")
    try:
        study.optimize(lambda trial: objective_lgbm(trial, df_full_final.copy(), final_feature_columns, target_column_nday, num_splits, use_gpu),
                       n_trials=n_trials, gc_after_trial=True, show_progress_bar=True, n_jobs=1)
    except KeyboardInterrupt: print("\nOptuna interrupted.")
    except Exception as e: print(f"\nError during Optuna: {e}"); traceback.print_exc()
    optimization_duration = time.time() - start_time
    print(f"\n--- Optuna Complete ({optimization_duration:.2f}s) ---")

    # --- 7. Process Results ---
    # (Giữ nguyên logic xử lý kết quả Optuna)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials: print("E: No Optuna trials completed."); return None, None, None, None, None
    try:
        best_trial = study.best_trial; best_value = -best_trial.value if best_trial.value is not None else None
        best_params_raw = best_trial.params; best_avg_metrics = best_trial.user_attrs.get('avg_metrics', {})
        print(f"\nBest Trial: #{best_trial.number}"); print(f"Best Score: {best_value:.4f}" if best_value else "N/A")
        best_params_cleaned = best_params_raw.copy(); best_params_cleaned.pop('weighting_strategy', None)
        print("\nBest Hyperparameters:"); print(json.dumps(best_params_cleaned, indent=2))
        print("\nBest Avg Metrics (Optuna):"); print(json.dumps(best_avg_metrics, indent=2, default=default_serializer))
        if 'avg_best_iteration' in best_avg_metrics and best_avg_metrics['avg_best_iteration'] > 0: print(f"Avg Best Iteration: {best_avg_metrics['avg_best_iteration']:.1f}")
    except Exception as e: print(f"E: Processing Optuna results: {e}"); return None, None, None, None, None

    # --- 8. Final Model Training ---
    # (Giữ nguyên logic huấn luyện model cuối cùng)
    print("\n--- Training Final Model on Full Data ---")
    final_model, final_scaler = None, None; train_duration = None
    try:
        X_final_raw, y_final = prepare_data_tree_nday(df_full_final, final_feature_columns, target_column_nday)
        if X_final_raw is None: print("E: Final data prep failed."); return None, None, best_params_cleaned, best_avg_metrics, target_column_nday
        print(f"Scaling final data ({X_final_raw.shape})..."); final_scaler = StandardScaler(); X_final_scaled = final_scaler.fit_transform(X_final_raw)
        final_params = best_params_cleaned.copy(); final_ws = best_params_raw.get('weighting_strategy', 'none')
        final_spw = None
        if final_ws == 'scale_pos_weight_calc': n_neg, n_pos = (y_final == 0).sum(), (y_final == 1).sum(); final_spw = n_neg / n_pos if n_pos > 0 else None; print(f"  Final scale_pos_weight: {final_spw:.4f}")
        final_extra = {'device': 'gpu'} if use_gpu and LGBM_GPU_SUPPORT else {'device': 'cpu'}; print(f"  Final device: {final_extra['device']}")
        final_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1, class_weight=None, scale_pos_weight=final_spw, **final_params, **final_extra)
        print(f"Training final model ({len(X_final_scaled)} samples)..."); t_start = time.time(); final_model.fit(X_final_scaled, y_final); train_duration = time.time() - t_start
        print(f"Final training complete ({train_duration:.2f}s).")
    except Exception as e: print(f"E: Final training: {e}"); traceback.print_exc(); return None, final_scaler, best_params_cleaned, best_avg_metrics, target_column_nday

    # --- 9. Save Artifacts ---
    # (Giữ nguyên logic lưu trữ)
    try:
        print("\nSaving artifacts...")
        model_suffix = '_gpu.joblib' if use_gpu and LGBM_GPU_SUPPORT else '_cpu.joblib'
        model_filename = f'lgbm_model_{forecast_horizon}d{model_suffix}'
        scaler_filename = f'lgbm_scaler_{forecast_horizon}d.joblib'; info_filename = f'lgbm_model_info_{forecast_horizon}d.json'
        model_path = os.path.join(MODEL_DIR, model_filename); scaler_path = os.path.join(MODEL_DIR, scaler_filename); info_path = os.path.join(MODEL_DIR, info_filename)
        joblib.dump(final_model, model_path); joblib.dump(final_scaler, scaler_path)
        print(f"  Model saved: {model_path}"); print(f"  Scaler saved: {scaler_path}")
        try: skv = joblib.__version__.split('.')[0] + '.' + joblib.__version__.split('.')[1]
        except: skv = joblib.__version__
        model_info = {
            'model_type': f'LGBMClassifier_{forecast_horizon}D_PreProc_v3', 'model_filename': model_filename, 'scaler_filename': scaler_filename,
            'feature_columns': final_feature_columns, 'target_variable': target_column_nday, 'forecast_horizon_days': forecast_horizon,
            'training_data_source': PROCESSED_DATA_DIR, 'loaded_tickers': target_tickers, 'final_training_samples': len(X_final_scaled) if 'X_final_scaled' in locals() else None,
            'training_timestamp': datetime.datetime.now().isoformat(), 'total_optimization_duration_seconds': optimization_duration, 'final_model_train_duration_seconds': train_duration,
            'optimization_details': { 'method': 'Optuna (TPE, MedianPruner)', 'n_trials_completed': len(completed_trials), 'n_trials_requested': n_trials, 'num_splits_walk_forward': num_splits, 'walk_forward_gap_days': 3, 'use_gpu': use_gpu and LGBM_GPU_SUPPORT, 'objective': 'Minimize -(0.4*Acc + 0.6*F1)'},
            'best_optuna_trial': {'number': best_trial.number, 'score': best_value, 'params': best_params_cleaned, 'raw_params': best_params_raw, 'avg_metrics': best_avg_metrics},
            'library_versions': {'lightgbm': lgb.__version__ if lgb else None, 'optuna': optuna.__version__, 'sklearn': skv, 'pandas': pd.__version__, 'numpy': np.__version__}
        }
        with open(info_path, 'w') as f: json.dump(model_info, f, indent=4, default=default_serializer)
        print(f"  Model info saved: {info_path}")
    except Exception as e: print(f"E: Saving artifacts: {e}"); traceback.print_exc()

    return final_model, final_scaler, best_params_cleaned, best_avg_metrics, target_column_nday


# --- JSON Serializer Helper ---
def default_serializer(obj):
    # (Giữ nguyên logic)
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)): return float(obj) if not pd.isna(obj) else None
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, (datetime.datetime, datetime.date)): return obj.isoformat()
    elif isinstance(obj, np.number): return obj.item() if not pd.isna(obj) else None
    elif isinstance(obj, pd.Timestamp): return obj.isoformat()
    elif pd.isna(obj): return None
    try: return repr(obj)
    except: raise TypeError(f"Type {type(obj)} not serializable")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n======================================================================")
    print(" Starting Stock Prediction Model Training (LGBM - N-Day Target) ")
    print("          >>> Using Pre-Processed Features Only - v3 <<<              ")
    print("======================================================================")
    sns.set_style("darkgrid")
    # --- Configuration ---
    N_TRIALS_LGBM = 200      # <<< Tăng số trials >>>
    NUM_SPLITS_LGBM = 5
    FORECAST_HORIZON = 5
    USE_GPU_TRAINING = True if LGBM_GPU_SUPPORT else False

    # --- Load Processed File List ---
    if not os.path.exists(PROCESSED_DATA_DIR): print(f"E: Input dir not found: {PROCESSED_DATA_DIR}"); exit()
    processed_files = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv') and os.path.isfile(os.path.join(PROCESSED_DATA_DIR, f))]
    if not processed_files: print(f"E: No '*_processed_data.csv' found in {PROCESSED_DATA_DIR}"); exit()
    print(f"\nFound {len(processed_files)} processed files.")

    # --- Run Optuna Optimization ---
    # <<< SỬA ĐỔI: Không cần truyền feature_columns nữa >>>
    final_model, final_scaler, best_params, best_metrics, target_column_main = run_optuna_optimization_lgbm(
        processed_files=processed_files,
        forecast_horizon=FORECAST_HORIZON,
        n_trials=N_TRIALS_LGBM,
        num_splits=NUM_SPLITS_LGBM,
        use_gpu=USE_GPU_TRAINING,
    )

    # --- Post-Training Summary ---
    print("\n=========================================================")
    print(" Training Process Summary")
    print("=========================================================")
    if final_model and final_scaler and best_params and best_metrics and target_column_main:
        print(f"\n--- LGBM Optuna & Training Successful ({FORECAST_HORIZON}-Day) ---")
        print("\nBest Hyperparameters:"); print(json.dumps(best_params, indent=2))
        print("\nBest Avg Metrics (Optuna):"); formatted_metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in best_metrics.items()}; print(json.dumps(formatted_metrics, indent=2, default=default_serializer))
        target_accuracy_val = best_metrics.get('accuracy', 0) * 100; target_f1_val = best_metrics.get('f1_positive', 0) * 100
        print(f"\n---> Avg Val Accuracy: {target_accuracy_val:.2f}%"); print(f"---> Avg Val F1 Score: {target_f1_val:.2f}%")
        if target_accuracy_val >= 80: print("  ✅ Goal >80% accuracy MET during Optuna!")
        else: print("  ❌ Goal >80% accuracy NOT MET during Optuna.")

        # Verify saved files
        print("\nVerifying artifacts...")
        model_suffix_verify = '_gpu' if USE_GPU_TRAINING and LGBM_GPU_SUPPORT else '_cpu'
        model_path = os.path.join(MODEL_DIR, f'lgbm_model_{FORECAST_HORIZON}d{model_suffix_verify}.joblib')
        scaler_path = os.path.join(MODEL_DIR, f'lgbm_scaler_{FORECAST_HORIZON}d.joblib')
        info_path = os.path.join(MODEL_DIR, f'lgbm_model_info_{FORECAST_HORIZON}d.json')
        thresh_path = os.path.join(MODEL_DIR, f'lgbm_threshold_info_{FORECAST_HORIZON}d.json')
        artifacts_ok = True
        for p, name in [(model_path, "Model"), (scaler_path, "Scaler"), (info_path, "Info")]:
            if os.path.exists(p): print(f"  ✅ {name} found: {os.path.basename(p)}")
            else: print(f"  ❌ {name} NOT found: {os.path.basename(p)}"); artifacts_ok = False
        if artifacts_ok: print("Essential artifacts saved.")
        else: print("W: Some essential artifacts missing.")

        # --- Feature Importance ---
        # (Giữ nguyên logic)
        if os.path.exists(info_path) and final_model:
            try:
                print("\n--- Feature Importance (Final Model) ---")
                with open(info_path, 'r') as f: model_info_imp = json.load(f)
                final_features = model_info_imp.get('feature_columns')
                if final_features and hasattr(final_model, 'feature_importances_') and len(final_features) == len(final_model.feature_importances_):
                    imp_df = pd.DataFrame({'f': final_features, 'i': final_model.feature_importances_}).sort_values('i', ascending=False).reset_index(drop=True)
                    print("\nTop 30:"); print(imp_df.head(30).to_string())
                    imp_f = f'lgbm_feature_importance_{FORECAST_HORIZON}d.csv'; imp_p = os.path.join(MODEL_DIR, imp_f)
                    try: imp_df.to_csv(imp_p, index=False); print(f"\nImportance saved: {imp_f}")
                    except Exception as e: print(f"E: Saving importance: {e}")
                    try: plt.figure(figsize=(10, 12)); sns.barplot(x="i", y="f", data=imp_df.head(30), palette="viridis", orient='h'); plt.xlabel("Importance"); plt.ylabel("Feature"); plt.title(f"Top 30 Features ({FORECAST_HORIZON}D)"); plt.tight_layout(); plot_f = f'lgbm_feature_importance_{FORECAST_HORIZON}d.png'; plt.savefig(os.path.join(MODEL_DIR, plot_f)); print(f"Plot saved: {plot_f}"); plt.close()
                    except Exception as e: print(f"E: Plotting importance: {e}")
                else: print("W: Could not generate importance (missing info/model data/mismatch).")
            except Exception as e: print(f"E: Importance analysis: {e}")
        else: print("\nSkipping Importance (info/model missing).")

        # --- Threshold Tuning ---
        # (Giữ nguyên logic, bao gồm cả cảnh báo)
        if os.path.exists(info_path) and final_model and final_scaler and 'df_full_final' in globals() and df_full_final is not None:
            try:
                print(f"\n--- Threshold Tuning Example (Last Fold Proxy) ---")
                print("    <<< WARNING: Thresholds from one fold may not generalize well! >>>")
                with open(info_path, 'r') as f: mi_thresh = json.load(f)
                features_thresh = mi_thresh.get('feature_columns')
                if features_thresh:
                    split_gen = get_walk_forward_splits_rf(df_full_final, NUM_SPLITS_LGBM, gap_days=3)
                    _, last_val_fold = None, None; sc = 0
                    try:
                        for _, va in split_gen: last_val_fold = va; sc += 1
                    except ValueError as e: last_val_fold = None; print(f"W: Split gen failed: {e}")
                    if last_val_fold is not None and not last_val_fold.empty and sc == NUM_SPLITS_LGBM:
                        print(f"Using last fold (Split {sc}, {len(last_val_fold)} samples).")
                        X_tune_raw, y_tune = prepare_data_tree_nday(last_val_fold, features_thresh, target_column_main)
                        if X_tune_raw is not None and not X_tune_raw.empty and y_tune is not None:
                            X_tune_s = final_scaler.transform(X_tune_raw); y_prob_tune = final_model.predict_proba(X_tune_s)[:, 1]
                            if np.all(y_prob_tune < 0.01) or np.all(y_prob_tune > 0.99): print("    W: Probabilities skewed. Tuning unreliable.")
                            prec, rec, thr_pr = precision_recall_curve(y_tune, y_prob_tune); f1s = np.divide(2*prec*rec, prec+rec+1e-9, out=np.zeros_like(prec), where=(prec+rec)>1e-9)[:-1]; thr_f1 = thr_pr
                            best_thr_f1, best_f1 = 0.5, 0.0
                            if len(f1s) > 0: idx = np.argmax(f1s); best_thr_f1 = thr_f1[idx]; best_f1 = f1s[idx]
                            best_acc, best_thr_acc = 0.0, 0.5
                            for thr in np.arange(0.05, 0.96, 0.01):
                                acc = accuracy_score(y_tune, (y_prob_tune >= thr).astype(int))
                                if acc > best_acc: best_acc, best_thr_acc = acc, thr
                                elif acc == best_acc and abs(thr-0.5) < abs(best_thr_acc-0.5): best_thr_acc = thr
                            print("\nThreshold Results (Last Fold):")
                            print(f"  Val Size: {len(y_tune)}, Pos Ratio: {y_tune.mean():.2f}")
                            acc_def = accuracy_score(y_tune, (y_prob_tune >= 0.5).astype(int)); f1_def = f1_score(y_tune, (y_prob_tune >= 0.5).astype(int), zero_division=0)
                            f1_at_ba = f1_score(y_tune, (y_prob_tune >= best_thr_acc).astype(int), zero_division=0); acc_at_bf1 = accuracy_score(y_tune, (y_prob_tune >= best_thr_f1).astype(int))
                            print(f"  Default (0.50): Acc={acc_def:.4f}, F1={f1_def:.4f}")
                            if best_acc >= 0.99 or best_f1 >= 0.99: print("    <<<< WARNING >>>> Near-perfect scores on tuning fold! Likely UNRELIABLE!")
                            print(f"  Best Thr (Acc): {best_thr_acc:.2f} -> Acc={best_acc:.4f}, F1={f1_at_ba:.4f}")
                            print(f"  Best Thr (F1):  {best_thr_f1:.2f} -> Acc={acc_at_bf1:.4f}, F1={best_f1:.4f}")
                            thresh_info = {'tuning_method': 'Last Fold Proxy', 'size': len(y_tune), 'pos_ratio': y_tune.mean(), 'default_threshold': 0.5, 'default_accuracy': acc_def, 'default_f1': f1_def, 'best_threshold_accuracy': best_thr_acc, 'best_accuracy_on_val': best_acc, 'f1_at_best_accuracy': f1_at_ba, 'best_threshold_f1': best_thr_f1, 'accuracy_at_best_f1': acc_at_bf1, 'best_f1_on_val': best_f1}
                            thresh_f = f'lgbm_threshold_info_{FORECAST_HORIZON}d.json'; thresh_p = os.path.join(MODEL_DIR, thresh_f)
                            try: 
                                with open(thresh_p, 'w') as f: json.dump(thresh_info, f, indent=4, default=default_serializer); print(f"\nThreshold info saved: {thresh_f}")
                            except Exception as e: print(f"E: Saving threshold info: {e}")
                        else: print("W: Could not prepare data for tuning.")
                    else: print(f"W: Could not get last fold ({sc}/{NUM_SPLITS_LGBM}).")
                else: print("W: Missing features list for tuning.")
            except Exception as e: print(f"E: Threshold tuning: {e}"); traceback.print_exc()
        else: print("\nSkipping Threshold Tuning (info/model/data missing).")

        # --- Example Prediction ---
        # (Giữ nguyên logic)
        if os.path.exists(info_path) and final_model and final_scaler and processed_files:
            try:
                print(f"\n--- Example Prediction ({FORECAST_HORIZON}-Day) ---")
                with open(info_path, 'r') as f: mi_pred = json.load(f)
                features_pred = mi_pred.get('feature_columns'); 
                if not features_pred: raise ValueError("Missing features.")
                tuned_thr = 0.5; chosen_thr_key = 'best_threshold_f1'
                thresh_p_load = os.path.join(MODEL_DIR, f'lgbm_threshold_info_{FORECAST_HORIZON}d.json')
                if os.path.exists(thresh_p_load):
                    try:
                        with open(thresh_p_load, 'r') as f: thr_load = json.load(f)
                        loaded_thr = thr_load.get(chosen_thr_key)
                        if loaded_thr and 0.05 < loaded_thr < 0.95: tuned_thr = loaded_thr; print(f"Using tuned threshold ({chosen_thr_key}): {tuned_thr:.4f}")
                        else: print(f"W: Invalid threshold '{chosen_thr_key}'. Using 0.5.")
                    except Exception as e: print(f"W: Loading threshold failed ({e}). Using 0.5.")
                else: print(f"Threshold info not found. Using 0.5.")
                sample_f = processed_files[-1]; print(f"\nLoading sample: {os.path.basename(sample_f)}")
                try:
                    sample_df = pd.read_csv(sample_f, index_col='Date', parse_dates=True); sample_df = sample_df.sort_index()
                    essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] # Define locally for prediction scope
                    sample_df.dropna(subset=essential_cols, how='all', inplace=True) # Clean sample
                    if not sample_df.empty:
                        print(f"Prep last row (Sample shape: {sample_df.shape})...")
                        missing_f = [f for f in features_pred if f not in sample_df.columns]
                        if missing_f: print(f"  W: Sample missing {len(missing_f)} features. Adding 0."); sample_df = sample_df.reindex(columns=list(sample_df.columns) + missing_f, fill_value=0)
                        try:
                            last_row_raw = sample_df[features_pred].iloc[-1:]
                        except Exception as e:
                            print(f"E: Selecting features: {e}"); last_row_raw = pd.DataFrame()
                        if not last_row_raw.empty:
                            if last_row_raw.isna().any().any() or np.isinf(last_row_raw.values).any():
                                print(" W: NaN/Inf in last row. Filling 0."); last_row_raw = last_row_raw.fillna(0).replace([np.inf, -np.inf], 0)
                            X_pred_s = final_scaler.transform(last_row_raw); pred_prob = final_model.predict_proba(X_pred_s)[0, 1]; pred_class = int(pred_prob >= tuned_thr)
                            last_date = sample_df.index[-1].strftime('%Y-%m-%d'); ticker_p = sample_df.get('Ticker', os.path.basename(sample_f).split('_')[0]).iloc[-1]
                            print("\n--- Prediction Result ---"); print(f"  Ticker: {ticker_p}, Data up to: {last_date}"); print(f"  Horizon: Next {FORECAST_HORIZON} days"); print("-"*25)
                            print(f"  Prob (Increase): {pred_prob:.4f}"); print(f"  Threshold: {tuned_thr:.4f} ({chosen_thr_key})"); print(f"  Trend: {pred_class} ({'UP' if pred_class==1 else 'DOWN/STAY'})"); print("-"*25)
                            conf = abs(pred_prob - 0.5)*2; conf_lvl = "High" if conf > 0.7 else ("Medium" if conf > 0.4 else "Low"); print(f"  Confidence: {conf_lvl} ({conf:.2f})")
                        else:
                            print("E: Could not extract last row features.")
                    else:
                        print("E: Sample empty after cleaning.")
                except Exception as e:
                    print(f"E: Predicting sample: {e}"); traceback.print_exc(limit=2)
            except Exception as e:
                print(f"E: Setting up prediction: {e}")
        else:
            print("\nSkipping Example Prediction (missing components).")
    else:
        print("\n--- LGBM Training Failed ---"); print("Check logs for errors.")
    print("\n========================================================="); print(" Script Finished"); print("=========================================================")