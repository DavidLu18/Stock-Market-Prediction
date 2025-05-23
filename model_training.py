import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import optuna
# Not using LightGBMPruningCallback directly for PyTorch, Optuna has its own pruning for PyTorch trials
# from optuna.integration import LightGBMPruningCallback

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Callable

# --- Directory Setup ---
SCRIPT_DIR_MT = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR_MT = SCRIPT_DIR_MT
DATA_DIR_MT = os.path.join(ROOT_DIR_MT, 'data')
PROCESSED_DATA_DIR_MT = os.path.join(DATA_DIR_MT, 'processed')
MODEL_DIR_MT = os.path.join(ROOT_DIR_MT, 'models')

for dir_path in [MODEL_DIR_MT]:
    if not os.path.exists(dir_path):
        try: os.makedirs(dir_path); print(f"(ModelTraining) Created directory: {dir_path}")
        except OSError as e: print(f"(ModelTraining) Error creating directory {dir_path}: {e}")

# --- Constants ---
COL_DATE = 'Date'
COL_TICKER = 'Ticker'
COL_OPEN = 'Open'
COL_HIGH = 'High'
COL_LOW = 'Low'
COL_CLOSE = 'Close'
COL_ADJ_CLOSE = 'Adj Close'
COL_VOLUME = 'Volume'
COL_TARGET = 'Target_Trend_Up'
COL_LOG_RETURN = 'Log_Return'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Feature Engineering (v4 - kept from original, can be enhanced for DL) ---
def engineer_advanced_features(df_input: pd.DataFrame, status_callback: Optional[Callable[[str, bool, int], None]] = None) -> pd.DataFrame:
    if status_callback: status_callback("Starting advanced feature engineering (v4 style for ResNLS)...", False, 0)
    df = df_input.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if COL_DATE in df.columns:
            df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')
            df = df.set_index(COL_DATE)
        else:
            if status_callback: status_callback("DataFrame needs DatetimeIndex or 'Date' column for advanced features.", True, 0); return df_input
    df = df.sort_index()

    # Ensure basic price columns are numeric
    for col in [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_ADJ_CLOSE, COL_VOLUME]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    price_col_for_log_return = COL_ADJ_CLOSE if COL_ADJ_CLOSE in df.columns and df[COL_ADJ_CLOSE].isnull().sum() < len(df) else COL_CLOSE
    if price_col_for_log_return in df.columns:
        df[COL_LOG_RETURN] = np.log(df[price_col_for_log_return].replace(0, np.nan) / df[price_col_for_log_return].shift(1).replace(0, np.nan))
        if status_callback: status_callback(f"Calculated '{COL_LOG_RETURN}' using '{price_col_for_log_return}'.", False, 1)
    else:
        if status_callback: status_callback(f"Price columns for '{COL_LOG_RETURN}' missing or all NaN.", True, 1)
        df[COL_LOG_RETURN] = 0.0 # Add placeholder if calculation fails

    if COL_LOG_RETURN in df.columns:
        for window in [30, 60, 90, 120, 180]:
            col_name = f'Volatility_{window}D'
            df[col_name] = df[COL_LOG_RETURN].rolling(window=window, min_periods=int(window*0.6)).std(ddof=0) * np.sqrt(window) # Annualized for window
            if status_callback: status_callback(f"Engineered {col_name}", False, 1)

    # Interaction features (example)
    common_vol_col = 'Volatility_60D'
    if 'RSI_14' in df.columns and common_vol_col in df.columns: df['RSI_x_Volatility'] = df['RSI_14'] * df[common_vol_col]
    if 'MACD' in df.columns and common_vol_col in df.columns: df['MACD_x_Volatility'] = df['MACD'] * df[common_vol_col]
    if 'MFI_14' in df.columns and common_vol_col in df.columns: df['MFI_x_Volatility'] = df['MFI_14'] * df[common_vol_col]

    indicators_to_lag = {
        'RSI_14': [1, 3, 5, 10], 'MACD': [1, 3, 5, 10], 'MACD_Hist': [1, 3, 5],
        'SlowK': [1, 3, 5], 'ADX_14': [1,3,5], COL_LOG_RETURN: [1,2,3,5,10] # Added Log_Return lags directly here
    }
    for indicator, lags in indicators_to_lag.items():
        if indicator in df.columns:
            for lag in lags: df[f'{indicator}_Lag_{lag}'] = df[indicator].shift(lag)
            if status_callback: status_callback(f"Lags for {indicator}", False, 1)

    if isinstance(df.index, pd.DatetimeIndex):
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 6.0)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 6.0)
        df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12.0)
        df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12.0)
        df['DayOfYear_Sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
        df['DayOfYear_Cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)
        if status_callback: status_callback("Cyclical time features", False, 1)

    if 'VIX' in df.columns:
        df['VIX_Roll_Mean_5'] = df['VIX'].rolling(window=5, min_periods=1).mean()
        df['VIX_Roll_Mean_20'] = df['VIX'].rolling(window=20, min_periods=1).mean()
        df['VIX_vs_Mean_20'] = (df['VIX'] / df['VIX_Roll_Mean_20'].replace(0,np.nan)) - 1
        if status_callback: status_callback("VIX-based features", False, 1)

    if 'Market_Return' in df.columns:
         df['Market_Return_Lag_1'] = df['Market_Return'].shift(1)
         df['Market_Return_Roll_Sum_3D'] = df['Market_Return'].rolling(window=3, min_periods=1).sum()
         df['Market_Return_Roll_Sum_5D'] = df['Market_Return'].rolling(window=5, min_periods=1).sum()
         if status_callback: status_callback("Market_Return derived features", False, 1)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    # Smart fill: ffill for most, but for things like returns or volatility, 0 might be better if no prior data
    df[numeric_cols] = df[numeric_cols].ffill().bfill() # Generic fill first
    for col in numeric_cols: # Specific fills for some common features
        if 'Return' in col or 'Volatility' in col or 'MACD_Hist' in col:
             df[col].fillna(0, inplace=True)
    df.fillna(0, inplace=True) # Catch-all for any remaining NaNs

    if status_callback: status_callback("Advanced feature engineering complete.", False, 0)
    return df

# --- Target Variable Definition ---
def create_target_variable(df: pd.DataFrame, forecast_horizon: int, threshold: float,
                           status_callback: Optional[Callable[[str, bool, int], None]] = None) -> pd.DataFrame:
    if status_callback: status_callback(f"Creating target: {forecast_horizon}d horizon, {threshold*100:.2f}% threshold...", False, 0)
    df_target = df.copy()
    price_col_for_target = COL_ADJ_CLOSE if COL_ADJ_CLOSE in df_target.columns and not df_target[COL_ADJ_CLOSE].isnull().all() else COL_CLOSE

    if price_col_for_target not in df_target.columns or df_target[price_col_for_target].isnull().all():
        if status_callback: status_callback(f"'{price_col_for_target}' missing or all NaN. Cannot create target.", True, 1)
        df_target[COL_TARGET] = 0 # Assign a default target if price is unusable
        return df_target

    future_price = df_target[price_col_for_target].shift(-forecast_horizon)
    # Ensure current price is not zero to avoid division by zero
    current_price_safe = df_target[price_col_for_target].replace(0, np.nan)
    perc_change = (future_price - current_price_safe) / current_price_safe

    df_target[COL_TARGET] = np.where(perc_change >= threshold, 1, 0).astype(int)

    # Remove rows where target cannot be calculated (typically last `forecast_horizon` rows)
    df_target.dropna(subset=[COL_TARGET], inplace=True)
    if status_callback: status_callback("Target variable created and NaN rows dropped.", False, 0)
    return df_target

# --- PyTorch ResNLS Model ---
class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_rate=0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ResNLS(nn.Module):
    def __init__(self, num_features, seq_len, resnet_blocks=2, resnet_out_channels=64,
                 lstm_hidden_size=128, lstm_layers=2, fc_hidden_size=64, dropout_rate=0.3):
        super().__init__()
        self.seq_len = seq_len

        layers = []
        current_channels = num_features
        for _ in range(resnet_blocks):
            layers.append(ResNetBlock1D(current_channels, resnet_out_channels, kernel_size=3, dropout_rate=dropout_rate))
            current_channels = resnet_out_channels
        self.resnet = nn.Sequential(*layers)

        self.lstm = nn.LSTM(input_size=resnet_out_channels, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0)

        self.fc_block = nn.Sequential(
            nn.Linear(lstm_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_size, 1) # Output for binary classification
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len) for Conv1D
        x = self.resnet(x)      # (batch_size, resnet_out_channels, seq_len_after_resnet)

        x = x.permute(0, 2, 1)  # (batch_size, seq_len_after_resnet, resnet_out_channels) for LSTM

        lstm_out, _ = self.lstm(x) # lstm_out: (batch_size, seq_len, lstm_hidden_size)

        # Use the output of the last LSTM time step
        last_lstm_out = lstm_out[:, -1, :] # (batch_size, lstm_hidden_size)

        out = self.fc_block(last_lstm_out) # (batch_size, 1)
        return out

# --- PyTorch Dataset & Data Preparation ---
def create_sequences(data: pd.DataFrame, target: pd.Series, seq_length: int):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length): # Target already aligned (no future data needed for y)
        X_seq.append(data.iloc[i:i + seq_length].values)
        y_seq.append(target.iloc[i + seq_length -1]) # Target for the END of the sequence
    return np.array(X_seq), np.array(y_seq)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Training and Evaluation Loop ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.sigmoid(outputs).round().cpu().detach().numpy()
        all_preds.extend(preds.flatten())
        all_targets.extend(y_batch.cpu().detach().numpy().flatten())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    return avg_loss, acc, f1

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds_proba, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().detach().numpy()
            all_preds_proba.extend(probs.flatten())
            all_targets.extend(y_batch.cpu().detach().numpy().flatten())

    avg_loss = total_loss / len(dataloader)

    if len(np.unique(all_targets)) < 2: # Handle case where validation set has only one class
        roc_auc = 0.5
        try:
            all_preds_binary = (np.array(all_preds_proba) >= 0.5).astype(int)
            acc = accuracy_score(all_targets, all_preds_binary)
            f1 = f1_score(all_targets, all_preds_binary, zero_division=0)
            precision = precision_score(all_targets, all_preds_binary, zero_division=0)
            recall = recall_score(all_targets, all_preds_binary, zero_division=0)
        except:
            acc, f1, precision, recall = 0.0, 0.0, 0.0, 0.0
    else:
        all_preds_binary = (np.array(all_preds_proba) >= 0.5).astype(int)
        acc = accuracy_score(all_targets, all_preds_binary)
        f1 = f1_score(all_targets, all_preds_binary, zero_division=0)
        roc_auc = roc_auc_score(all_targets, all_preds_proba)
        precision = precision_score(all_targets, all_preds_binary, zero_division=0)
        recall = recall_score(all_targets, all_preds_binary, zero_division=0)

    return avg_loss, acc, f1, roc_auc, precision, recall, all_preds_proba, all_targets


# --- Optuna Objective Function for ResNLS ---
def objective_resnls(
    trial: optuna.Trial, X_train_val_df: pd.DataFrame, y_train_val_series: pd.Series,
    feature_names: List[str], num_splits_walk_forward: int,
    status_callback: Optional[Callable[[str, bool, int], None]] = None
) -> float:
    cfg = {
        "seq_len": trial.suggest_int("seq_len", 30, 120, step=10),
        "resnet_blocks": trial.suggest_int("resnet_blocks", 1, 3),
        "resnet_out_channels": trial.suggest_categorical("resnet_out_channels", [32, 64, 128]),
        "lstm_hidden_size": trial.suggest_categorical("lstm_hidden_size", [64, 128, 256]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "fc_hidden_size": trial.suggest_categorical("fc_hidden_size", [32, 64, 128]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("epochs", 20, 100, step=10), # Max epochs for Optuna trial
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    early_stopping_patience = 10 # For Optuna trials

    def _obj_status(msg, err=False, indent=2): status_callback(msg, err, indent) if status_callback else None
    _obj_status(f"Optuna Trial {trial.number}: Starting {num_splits_walk_forward} walk-forward splits...")

    tscv = TimeSeriesSplit(n_splits=num_splits_walk_forward)
    fold_metrics = {'f1_positive': [], 'roc_auc': [], 'accuracy': [], 'precision_positive': [], 'recall_positive': []}

    X_data_trial = X_train_val_df[feature_names].copy()
    y_data_trial = y_train_val_series.copy()

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data_trial)):
        _obj_status(f"  Fold {fold+1}/{num_splits_walk_forward}")
        X_train_fold_df, X_val_fold_df = X_data_trial.iloc[train_idx], X_data_trial.iloc[val_idx]
        y_train_fold_s, y_val_fold_s = y_data_trial.iloc[train_idx], y_data_trial.iloc[val_idx]

        X_train_seq, y_train_seq = create_sequences(X_train_fold_df, y_train_fold_s, cfg["seq_len"])
        X_val_seq, y_val_seq = create_sequences(X_val_fold_df, y_val_fold_s, cfg["seq_len"])

        if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(np.unique(y_val_seq)) < 2:
            _obj_status(f"  Fold {fold+1}: Skip (insufficient data or single class in val).", True, 3)
            continue

        train_dataset = StockDataset(X_train_seq, y_train_seq)
        val_dataset = StockDataset(X_val_seq, y_val_seq)
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        model = ResNLS(
            num_features=len(feature_names), seq_len=cfg["seq_len"],
            resnet_blocks=cfg["resnet_blocks"], resnet_out_channels=cfg["resnet_out_channels"],
            lstm_hidden_size=cfg["lstm_hidden_size"], lstm_layers=cfg["lstm_layers"],
            fc_hidden_size=cfg["fc_hidden_size"], dropout_rate=cfg["dropout_rate"]
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        criterion = nn.BCEWithLogitsLoss()

        best_val_f1 = 0
        epochs_no_improve = 0
        for epoch in range(cfg["epochs"]):
            train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_acc, val_f1, val_auc, val_prec, val_recall, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                _obj_status(f"    Fold {fold+1}: Early stopping at epoch {epoch+1}.", False, 4)
                break

        fold_metrics['f1_positive'].append(val_f1)
        fold_metrics['roc_auc'].append(val_auc)
        fold_metrics['accuracy'].append(val_acc)
        fold_metrics['precision_positive'].append(val_prec)
        fold_metrics['recall_positive'].append(val_recall)

        trial.report(np.mean(fold_metrics['roc_auc']) if fold_metrics['roc_auc'] else 0.0, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if not fold_metrics['f1_positive']:
        _obj_status(f"Trial {trial.number}: All folds skipped or failed. Return 0.0 F1.", True)
        return 0.0

    avg_f1_positive = np.mean(fold_metrics['f1_positive'])
    avg_auc = np.mean(fold_metrics['roc_auc'])
    _obj_status(f"Trial {trial.number}: Avg F1(pos)={avg_f1_positive:.4f}, Avg AUC={avg_auc:.4f}")

    trial.set_user_attr("average_metrics", {k: np.mean(v) if v else 0.0 for k,v in fold_metrics.items()})
    return avg_f1_positive

# --- Main Training Orchestration for ResNLS Ensemble ---
def run_optuna_optimization_resnls(
    processed_files: List[str], forecast_horizon: int, n_trials_optuna: int, num_cv_splits_optuna: int,
    target_threshold: float = 0.01, train_val_split_ratio: float = 0.8, eval_set_ratio: float = 0.1,
    num_ensemble_models: int = 5, final_model_epochs: int = 50, final_model_patience: int = 10,
    status_callback: Optional[Callable[[str, bool, Optional[int]], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[Optional[List[str]], Optional[StandardScaler], Optional[Dict], Optional[Dict], Optional[str], Optional[List[str]], Optional[int]]:

    def _status(msg, err=False, indent=0): status_callback(msg,err,indent) if status_callback else print(f"{'  '*indent}[{'ERR' if err else 'INF'}] {msg}")
    def _progress(val, msg): progress_callback(val,msg) if progress_callback else print(f"Prog: {val*100:.1f}% - {msg}")

    _status(f"Starting ResNLS Ensemble model training pipeline... Device: {DEVICE}"); _progress(0.0, "Initializing...")

    _status("Loading & combining data..."); all_dfs=[]
    for fp_idx, fp in enumerate(processed_files):
        try:
            df = pd.read_csv(fp);
            df[COL_DATE]=pd.to_datetime(df[COL_DATE], errors='coerce')
            all_dfs.append(df)
            _status(f"Loaded {os.path.basename(fp)} ({df.shape})",False,1)
        except Exception as e: _status(f"Err load {fp}: {e}",True,1); continue
        _progress(0.01 + 0.09 * (fp_idx+1)/len(processed_files), f"Loading data files...")

    if not all_dfs: _status("No data loaded. Aborting training.",True,0); return (None,)*7
    combo_df = pd.concat(all_dfs,ignore_index=True)
    if COL_DATE in combo_df: combo_df=combo_df.set_index(COL_DATE).sort_index()
    elif isinstance(combo_df.index,pd.DatetimeIndex): combo_df=combo_df.sort_index()
    else: _status("Date index missing after combining. Abort.",True,0); return (None,)*7
    _status(f"Combined data shape: {combo_df.shape}",False,1); _progress(0.1, "Data combined.")

    _status("Advanced feature engineering..."); df_eng=engineer_advanced_features(combo_df.copy(),_status); _progress(0.15,"Adv feats done.")
    _status(f"Target variable creation: {forecast_horizon}d horizon, {target_threshold*100:.2f}% thresh...");
    df_tgt=create_target_variable(df_eng.copy(),forecast_horizon,target_threshold,_status); _progress(0.20,"Target var done.")

    potential_features = [c for c in df_tgt.columns if c not in [COL_TARGET, COL_TICKER] and df_tgt[c].dtype in [np.int64, np.float64, np.int32, np.float32]]
    X = df_tgt[potential_features].copy()
    y = df_tgt[COL_TARGET].copy()

    valid_idx = y.dropna().index.intersection(X.dropna(how='any').index)
    X = X.loc[valid_idx]; y = y.loc[valid_idx]

    if X.empty or len(X) < 500:
        _status(f"Insufficient data after cleaning for X ({X.shape}). Aborting.",True,0); return (None,)*7
    _status(f"X/y shape after cleaning: {X.shape}, {y.shape}",False,1)
    target_dist = y.value_counts(normalize=True).to_dict()
    _status(f"Target distribution: {target_dist}",False,1)
    if len(np.unique(y)) < 2: _status("Target variable has only one class. Aborting.",True,0); return (None,)*7
    _progress(0.25, "X/y prepared.")

    split_idx_test = int(len(X) * (1 - eval_set_ratio))
    X_optuna_tv, X_test_final = X.iloc[:split_idx_test], X.iloc[split_idx_test:]
    y_optuna_tv, y_test_final = y.iloc[:split_idx_test], y.iloc[split_idx_test:]

    _status(f"Data for Optuna (Train/Val): {X_optuna_tv.shape}, Final Hold-Out Test: {X_test_final.shape}",False,1)
    eval_desc = ""
    if X_test_final.empty or y_test_final.empty or len(np.unique(y_test_final)) < 2:
        _status("Final hold-out test set is empty or single-class. Optuna will use all data. Post-hoc eval on this set will be skipped.",True,1)
        X_optuna_tv, y_optuna_tv = X.copy(), y.copy()
        X_test_final, y_test_final = pd.DataFrame(), pd.Series()
        eval_desc = "Skipped (final test set issue or too small)"
    else:
        eval_desc = f"Final {eval_set_ratio*100:.0f}% of data ({X_test_final.index.min():%Y-%m-%d} to {X_test_final.index.max():%Y-%m-%d})"
    _progress(0.30, "Data split (Optuna Train/Val, Final Test).")

    _status("Scaling features...");
    scaler = StandardScaler()
    X_optuna_tv_scaled_np = scaler.fit_transform(X_optuna_tv[potential_features])
    X_optuna_tv_scaled_df = pd.DataFrame(X_optuna_tv_scaled_np, columns=potential_features, index=X_optuna_tv.index)

    X_test_final_scaled_df = pd.DataFrame()
    if not X_test_final.empty:
        X_test_final_scaled_np = scaler.transform(X_test_final[potential_features])
        X_test_final_scaled_df = pd.DataFrame(X_test_final_scaled_np, columns=potential_features, index=X_test_final.index)
    _progress(0.35, "Features scaled.")

    selected_feature_names = potential_features
    _status(f"Using {len(selected_feature_names)} features for ResNLS model.", False, 1)

    _status(f"Optuna hyperparameter optimization ({n_trials_optuna} trials)...",False,0)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

    def optuna_status_wrapper(msg, err=False, indent_level=None):
        _status(msg, err, indent_level if indent_level is not None else 0)

    try:
        study.optimize(
            lambda t: objective_resnls(t, X_optuna_tv_scaled_df, y_optuna_tv, selected_feature_names, num_cv_splits_optuna, optuna_status_wrapper),
            n_trials=n_trials_optuna,
            callbacks=[lambda s,ft: _progress(0.35 + ((ft.number+1)/n_trials_optuna * 0.35),f"Optuna trial {ft.number+1}/{n_trials_optuna}")]
        )
    except optuna.exceptions.OptunaError as e_opt:
        _status(f"Optuna optimization failed: {e_opt}. Aborting.",True,0); return (None,)*7

    best_params_optuna = study.best_params
    avg_metrics_optuna = study.best_trial.user_attrs.get("average_metrics", {})
    best_sequence_length = best_params_optuna.get("seq_len", 60)

    _status(f"Optuna finished. Best trial #{study.best_trial.number} with F1(pos): {study.best_value:.4f}",False,1)
    _status(f"  Best Params from Optuna: {best_params_optuna}",False,2)
    _status(f"  Avg Validation Metrics (from best trial's CV): {avg_metrics_optuna}",False,2)
    _progress(0.70, "Optuna optimization complete.")

    _status(f"Training final ensemble of {num_ensemble_models} ResNLS models...",False,0)

    X_final_train_seq, y_final_train_seq = create_sequences(X_optuna_tv_scaled_df[selected_feature_names], y_optuna_tv, best_sequence_length)
    if len(X_final_train_seq) == 0:
         _status(f"Not enough data to form sequences for final model training with seq_len {best_sequence_length}. Aborting.", True, 0); return (None,)*7

    final_train_dataset = StockDataset(X_final_train_seq, y_final_train_seq)
    final_train_loader = DataLoader(final_train_dataset, batch_size=best_params_optuna["batch_size"], shuffle=True, num_workers=0)

    ensemble_model_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i_model in range(num_ensemble_models):
        _status(f"  Training ensemble member {i_model+1}/{num_ensemble_models}...", False, 1)

        model = ResNLS(
            num_features=len(selected_feature_names), seq_len=best_sequence_length,
            resnet_blocks=best_params_optuna["resnet_blocks"], resnet_out_channels=best_params_optuna["resnet_out_channels"],
            lstm_hidden_size=best_params_optuna["lstm_hidden_size"], lstm_layers=best_params_optuna["lstm_layers"],
            fc_hidden_size=best_params_optuna["fc_hidden_size"], dropout_rate=best_params_optuna["dropout_rate"]
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=best_params_optuna["lr"], weight_decay=best_params_optuna["weight_decay"])
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(final_model_epochs):
            model.train()
            for X_batch, y_batch in final_train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                 _status(f"    Member {i_model+1}, Epoch {epoch+1}/{final_model_epochs}, Loss: {loss.item():.4f}", False, 2)
            _progress(0.70 + (i_model/num_ensemble_models * 0.15) + ((epoch+1)/(final_model_epochs*num_ensemble_models) * 0.15) , f"Training Ensemble Member {i_model+1}")


        model_filename = f"resnls_ensemble_member_{i_model}_{forecast_horizon}d_thr{target_threshold*100:.1f}pct_{ts}.pt"
        model_path = os.path.join(MODEL_DIR_MT, model_filename)
        torch.save(model.state_dict(), model_path)
        ensemble_model_paths.append(model_filename)
        _status(f"  Saved member {i_model+1} to {model_filename}", False, 1)
    _progress(0.85, "Ensemble training complete.")

    final_test_metrics = {}
    optimal_threshold_ensemble = 0.5

    if not X_test_final_scaled_df.empty and not y_test_final.empty and len(np.unique(y_test_final)) >= 2:
        _status(f"Evaluating ensemble on final hold-out test set ({eval_desc})...", False, 0)
        X_final_test_seq, y_final_test_seq = create_sequences(X_test_final_scaled_df[selected_feature_names], y_test_final, best_sequence_length)

        if len(X_final_test_seq) > 0:
            final_test_dataset = StockDataset(X_final_test_seq, y_final_test_seq)
            final_test_loader = DataLoader(final_test_dataset, batch_size=best_params_optuna["batch_size"], shuffle=False)

            ensemble_predictions_proba = np.zeros((len(X_final_test_seq), num_ensemble_models))

            for i_model, model_fname in enumerate(ensemble_model_paths):
                model = ResNLS(
                    num_features=len(selected_feature_names), seq_len=best_sequence_length,
                    resnet_blocks=best_params_optuna["resnet_blocks"], resnet_out_channels=best_params_optuna["resnet_out_channels"],
                    lstm_hidden_size=best_params_optuna["lstm_hidden_size"], lstm_layers=best_params_optuna["lstm_layers"],
                    fc_hidden_size=best_params_optuna["fc_hidden_size"], dropout_rate=best_params_optuna["dropout_rate"]
                ).to(DEVICE)
                model.load_state_dict(torch.load(os.path.join(MODEL_DIR_MT, model_fname), map_location=DEVICE))
                model.eval()

                member_preds_proba = []
                with torch.no_grad():
                    for X_batch, _ in final_test_loader:
                        X_batch = X_batch.to(DEVICE)
                        outputs = model(X_batch)
                        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                        member_preds_proba.extend(probs)
                ensemble_predictions_proba[:, i_model] = np.array(member_preds_proba)[:len(ensemble_predictions_proba)]


            avg_ensemble_proba = np.mean(ensemble_predictions_proba, axis=1)

            prec, rec, thresh_prc = precision_recall_curve(y_final_test_seq, avg_ensemble_proba, pos_label=1)
            f1s_prc = np.array([])
            denom = prec[1:] + rec[1:]
            if len(prec) > 1 and len(rec) > 1: f1s_prc = np.where(denom > 1e-9, (2 * prec[1:] * rec[1:]) / denom, 0.0)

            if len(f1s_prc) > 0 and len(thresh_prc) == len(f1s_prc):
                best_f1_idx = np.argmax(f1s_prc)
                optimal_threshold_ensemble = thresh_prc[best_f1_idx]
            else:
                _status("Could not determine optimal threshold robustly from PR curve for ensemble on test set. Using 0.5.", True, 1)
                optimal_threshold_ensemble = 0.5

            ensemble_preds_binary = (avg_ensemble_proba >= optimal_threshold_ensemble).astype(int)

            final_test_metrics = {
                'accuracy': accuracy_score(y_final_test_seq, ensemble_preds_binary),
                'f1_positive': f1_score(y_final_test_seq, ensemble_preds_binary, pos_label=1, zero_division=0),
                'precision_positive': precision_score(y_final_test_seq, ensemble_preds_binary, pos_label=1, zero_division=0),
                'recall_positive': recall_score(y_final_test_seq, ensemble_preds_binary, pos_label=1, zero_division=0),
                'roc_auc': roc_auc_score(y_final_test_seq, avg_ensemble_proba),
                'confusion_matrix': confusion_matrix(y_final_test_seq, ensemble_preds_binary).tolist(),
                'optimal_threshold_on_test_set': float(optimal_threshold_ensemble)
            }
            _status(f"Ensemble performance on Final Test Set (Thresh: {optimal_threshold_ensemble:.4f}): {final_test_metrics}",False,1)
        else:
             _status("Not enough data to form sequences for final test set. Skipping evaluation.", True, 0)
             eval_desc += " (Skipped evaluation due to insufficient seq data)"
    else:
        _status("Final hold-out test set was empty or single-class. Skipping ensemble evaluation on it.",False,1)
    _progress(0.95, "Final evaluation done.")

    _status("Saving scaler and model information file...",False,0)
    scaler_filename = f"resnls_scaler_{forecast_horizon}d_thr{target_threshold*100:.1f}pct_{ts}.joblib"
    scaler_path = os.path.join(MODEL_DIR_MT, scaler_filename)
    joblib.dump(scaler, scaler_path)

    info_filename = f"resnls_ensemble_info_{forecast_horizon}d_thr{target_threshold*100:.1f}pct_{ts}.json"
    info_path = os.path.join(MODEL_DIR_MT, info_filename)

    model_info = {
        'model_type': 'ResNLS_Ensemble',
        'ensemble_member_filenames': ensemble_model_paths,
        'scaler_filename': scaler_filename,
        'training_timestamp_utc': datetime.utcnow().isoformat(),
        'forecast_horizon_days': forecast_horizon,
        'target_threshold_percent': target_threshold * 100,
        'target_variable': COL_TARGET,
        'sequence_length': best_sequence_length,
        'feature_columns': selected_feature_names,
        'optuna_trials': n_trials_optuna,
        'optuna_cv_splits': num_cv_splits_optuna,
        'best_optuna_trial_summary': {
            'trial_number': study.best_trial.number,
            'value_optimized (f1_pos_avg)': study.best_value,
            'params': best_params_optuna,
            'avg_validation_metrics_cv': avg_metrics_optuna
        },
        'final_ensemble_evaluation': {
            'eval_set_description': eval_desc,
            'metrics': final_test_metrics
        },
        'data_summary': {
            'processed_files_used': [os.path.basename(f) for f in processed_files],
            'combined_data_shape_initial': combo_df.shape,
            'data_shape_for_model_X_before_sequencing': X.shape,
            'target_class_distribution_overall': target_dist,
            'optuna_train_val_set_period': f"{X_optuna_tv.index.min():%Y-%m-%d} to {X_optuna_tv.index.max():%Y-%m-%d}" if not X_optuna_tv.empty else "N/A",
        }
    }
    try:
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4, default=lambda o: str(o) if isinstance(o,(datetime,pd.Timestamp,pd.Series)) else o)
        _status(f"Ensemble info saved to {info_filename}",False,1)
    except Exception as e:
        _status(f"Error saving model info JSON: {e}", True, 1)
        return (None,)*7

    _progress(1.0,"Pipeline complete."); _status("ResNLS Ensemble training finished successfully.",False,0)
    return ensemble_model_paths, scaler, best_params_optuna, avg_metrics_optuna, COL_TARGET, selected_feature_names, best_sequence_length


# --- CLI Test Section ---
if __name__ == "__main__":
    print(f"--- Running ModelTraining (ResNLS Ensemble) Standalone Test ---")
    print(f"Using device: {DEVICE}")

    def cli_status_callback(message: str, is_error: bool = False, indent_level: Optional[int] = 0):
        actual_indent = indent_level if indent_level is not None else 0
        prefix = "  " * actual_indent; level = "ERROR" if is_error else "INFO"
        print(f"{prefix}[{level}] {message}")

    def cli_progress_callback(progress_value: float, text_message: Optional[str] = None):
        bar_length = 30; filled_length = int(bar_length * progress_value)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\rProgress: |{bar}| {progress_value*100:.1f}% ({text_message or ""})      ', end='')
        if progress_value >= 1.0: print()

    test_processed_files = []
    if os.path.exists(PROCESSED_DATA_DIR_MT):
        all_processed = [os.path.join(PROCESSED_DATA_DIR_MT, f) for f in os.listdir(PROCESSED_DATA_DIR_MT) if f.endswith('_processed_data.csv')]
        preferred_tickers_test = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META', 'LLY', 'V']
        for ticker_test in preferred_tickers_test:
            found_file = next((f for f in all_processed if ticker_test.upper() in os.path.basename(f).upper()), None)
            if found_file and len(test_processed_files) < 5 :
                if found_file not in test_processed_files: test_processed_files.append(found_file)
        if not test_processed_files and all_processed:
             test_processed_files = all_processed[:min(len(all_processed), 2)]

    if not test_processed_files:
        cli_status_callback("No processed data files found for testing. Please run data collection first.", True)
    else:
        cli_status_callback(f"Using files for test: {', '.join([os.path.basename(f) for f in test_processed_files])}")

        test_forecast_horizon = 5
        test_optuna_trials = 10
        test_optuna_cv_splits = 2
        test_target_threshold = 0.01
        test_num_ensemble = 2
        test_final_epochs = 15

        cli_status_callback(f"Starting test: Horizon={test_forecast_horizon}d, Optuna Trials={test_optuna_trials}, CV Splits={test_optuna_cv_splits}, Target Thr={test_target_threshold*100:.1f}%, Ensemble Size={test_num_ensemble}")

        model_paths, scaler_obj, optuna_params, optuna_metrics, target_col_name, feat_names, seq_len = run_optuna_optimization_resnls(
            processed_files=test_processed_files,
            forecast_horizon=test_forecast_horizon,
            n_trials_optuna=test_optuna_trials,
            num_cv_splits_optuna=test_optuna_cv_splits,
            target_threshold=test_target_threshold,
            num_ensemble_models=test_num_ensemble,
            final_model_epochs=test_final_epochs,
            status_callback=cli_status_callback,
            progress_callback=cli_progress_callback
        )

        if model_paths and scaler_obj:
            cli_status_callback("\n--- ResNLS Ensemble Test Training Successful ---")
            cli_status_callback(f"  Ensemble model paths: {model_paths}")
            cli_status_callback(f"  Scaler saved: {scaler_obj is not None}")
            lr_val = optuna_params.get('lr', 'N/A')
            lr_str = f"{lr_val:.5f}" if isinstance(lr_val, float) else lr_val
            cli_status_callback(f"  Best Optuna Params (example): LR={lr_str}, SeqLen={optuna_params.get('seq_len','N/A')}, LSTM Hidden={optuna_params.get('lstm_hidden_size','N/A')}")

            f1_val = optuna_metrics.get('f1_positive',0) if optuna_metrics else 0
            auc_val = optuna_metrics.get('roc_auc',0) if optuna_metrics else 0
            cli_status_callback(f"  Avg Optuna CV F1(pos): {f1_val:.4f}, AUC: {auc_val:.4f}")
            cli_status_callback(f"  Target Column: {target_col_name}, Selected Sequence Length: {seq_len}")
            cli_status_callback(f"  Number of Features Used: {len(feat_names) if feat_names else 'N/A'}")
            if feat_names: cli_status_callback(f"  Features sample: {feat_names[:3]}...")

            latest_info_file = None
            if os.path.exists(MODEL_DIR_MT):
                info_files = sorted(
                    [f for f in os.listdir(MODEL_DIR_MT) if f.endswith('_info.json') and f.startswith('resnls_ensemble_info_')],
                    key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR_MT, f)),
                    reverse=True
                )
                if info_files: latest_info_file = info_files[0]

            if latest_info_file:
                cli_status_callback(f"  Latest model info file: {latest_info_file}")
                try:
                    with open(os.path.join(MODEL_DIR_MT, latest_info_file), 'r') as f_inf:
                        loaded_info = json.load(f_inf)
                    final_eval_metrics = loaded_info.get('final_ensemble_evaluation',{}).get('metrics',{})
                    test_f1 = final_eval_metrics.get('f1_positive', 'N/A')
                    test_acc = final_eval_metrics.get('accuracy', 'N/A')
                    test_auc = final_eval_metrics.get('roc_auc', 'N/A')
                    cli_status_callback(f"  Metrics on Final Test Set (from info file): F1(pos)={test_f1:.4f}, Acc={test_acc:.4f}, AUC={test_auc:.4f}"
                                        if isinstance(test_f1,float) else f"  Metrics on Final Test Set: F1(pos)={test_f1}, Acc={test_acc}, AUC={test_auc}")
                except Exception as e:
                    cli_status_callback(f"  Error verifying info file: {e}",True)
        else:
            cli_status_callback("\n--- ResNLS Ensemble Test Training Failed ---", True)

    print("\n--- ModelTraining (ResNLS Ensemble) Standalone Test Complete ---")