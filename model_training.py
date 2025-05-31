import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import joblib
import traceback
from typing import List, Dict, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

# XGBoost và các thư viện ML
import xgboost as xgb # XGBoost is directly imported and assumed to be available
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

# Import additional advanced ML libraries
try:
    import talib
    TALIB_AVAILABLE = True
    print("TA-Lib library available for technical indicators")
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Some technical indicators disabled.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("LightGBM library available for enhanced gradient boosting")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Some advanced features disabled.")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("CatBoost library available for categorical features handling")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Some advanced features disabled.")

# Thêm imblearn để xử lý class imbalance
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours # Required for SMOTETomek
    IMBLEARN_AVAILABLE = True
    print("imblearn library available for class balancing")
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imblearn not available. Class balancing features disabled.")

# Additional feature engineering libraries
try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some advanced features disabled.")

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

for dir_path in [MODEL_DIR]:
    if not os.path.exists(dir_path):
        try: os.makedirs(dir_path); print(f"Created directory: {dir_path}")
        except OSError as e: print(f"Error creating directory {dir_path}: {e}")

class StockVolatilityPredictor:
    def __init__(self, forecast_horizon: int = 5):
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.target_column = 'Target_Volatility'
        self.model_info = {}
        self.prediction_type = 'volatility'

    def _status_update(self, message: str, status_callback: Optional[Callable[[str, bool], None]] = None, is_error: bool = False):
        if status_callback: status_callback(message, is_error)
        else: print(f"[{'ERROR' if is_error else 'INFO'}] {message}")

    def _progress_update(self, progress: float, message: str, progress_callback: Optional[Callable[[float, str], None]] = None):
        if progress_callback: progress_callback(progress, message)
        else: print(f"Progress: {progress*100:.1f}% - {message}")

    def load_and_combine_data(self, processed_files: List[str], status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        # This method remains the same
        self._status_update("Đang tải và kết hợp dữ liệu...", status_callback)
        all_data = []
        for file_path in processed_files:
            try:
                if not os.path.exists(file_path):
                    self._status_update(f"File không tồn tại: {file_path}", status_callback, True); continue
                df = pd.read_csv(file_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                ticker = os.path.basename(file_path).split('_processed_data.csv')[0]
                df['Ticker'] = ticker
                all_data.append(df)
            except Exception as e:
                self._status_update(f"Lỗi khi tải {file_path}: {e}", status_callback, True); continue
        if not all_data: raise ValueError("Không có dữ liệu nào được tải thành công")
        combined_df = pd.concat(all_data, ignore_index=False, sort=False)
        combined_df.sort_index(inplace=True)
        self._status_update(f"Đã kết hợp {len(combined_df)} rows từ {len(all_data)} files", status_callback)
        return combined_df

    def engineer_advanced_features(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        self._status_update("Đang tạo đặc trưng nâng cao (cho biến động - v2)...", status_callback)
        df_featured = df.copy()
        if 'Close' not in df_featured.columns:
            self._status_update("Không có cột 'Close'. Không thể tạo features.", status_callback, True); return df_featured

        if 'VIX' in df_featured.columns: 
            df_featured['VIX_Value'] = df_featured['VIX'] 
            df_featured['VIX_Pct_Change_1D'] = df_featured['VIX'].pct_change()
            df_featured['VIX_MA_5D'] = df_featured['VIX'].rolling(5).mean() 
            df_featured['VIX_MA_20D'] = df_featured['VIX'].rolling(20).mean() 
            df_featured['VIX_Ratio_To_MA20'] = df_featured['VIX'] / df_featured['VIX_MA_20D'].replace(0, np.nan)
            df_featured['VIX_Std_10D'] = df_featured['VIX'].rolling(10).std()
            self._status_update("✅ VIX features created/updated.", status_callback)
        else:
            self._status_update("⚠️  Không tìm thấy cột 'VIX'. Bỏ qua features liên quan đến VIX.", status_callback)
            if 'VIX_Value' not in df_featured.columns: df_featured['VIX_Value'] = 0.0


        returns = df_featured['Close'].pct_change()
        squared_returns = returns**2

        vol_periods = [5, 10, 20, 30, 60, 90, 120] 
        for period in vol_periods:
            df_featured[f'Hist_Vol_{period}D'] = returns.rolling(period).std() * np.sqrt(252)
            df_featured[f'Lag_Hist_Vol_{period}D_1'] = df_featured[f'Hist_Vol_{period}D'].shift(1)
            df_featured[f'Lag_Hist_Vol_{period}D_5'] = df_featured[f'Hist_Vol_{period}D'].shift(5)
            df_featured[f'GARCH_Proxy_Vol_{period}D'] = np.sqrt(squared_returns.ewm(span=period, adjust=False).mean() * 252)
            if f'Hist_Vol_{period}D' in df_featured:
                vov_window = max(5, period // 4) 
                df_featured[f'VoV_{period}D'] = df_featured[f'Hist_Vol_{period}D'].rolling(vov_window).std()

        if TALIB_AVAILABLE and all(c in df_featured for c in ['High', 'Low', 'Close']):
            try:
                for atr_period in [10, 14, 21]:
                    df_featured[f'ATR_{atr_period}'] = talib.ATR(df_featured['High'].values, df_featured['Low'].values, df_featured['Close'].values, timeperiod=atr_period)
                    df_featured[f'ATR_Norm_{atr_period}'] = df_featured[f'ATR_{atr_period}'] / df_featured['Close'].replace(0, np.nan)
                if 'High' in df_featured and 'Low' in df_featured:
                    hl_diff = df_featured['High'] - df_featured['Low']
                    ema_hl_10 = talib.EMA(hl_diff.values, timeperiod=10)
                    ema_hl_20 = talib.EMA(hl_diff.values, timeperiod=20)
                    safe_ema_hl_20 = ema_hl_20.copy()
                    safe_ema_hl_20[safe_ema_hl_20 == 0] = np.nan 
                    df_featured['CHAIKIN_VOL_PROXY'] = ((ema_hl_10 - ema_hl_20) / safe_ema_hl_20 * 100).fillna(0)
            except Exception as e_talib_atr: self._status_update(f"Lỗi tính ATR/Chaikin Proxy: {e_talib_atr}", status_callback)
        
        if 'High' in df_featured and 'Low' in df_featured:
            log_hl_ratio_sq = (np.log(df_featured['High'] / df_featured['Low'].replace(0, np.nan)))**2
            for period in [5, 10, 20]:
                df_featured[f'Parkinson_Vol_{period}D'] = np.sqrt(log_hl_ratio_sq.rolling(period).mean() * (252 / (4 * np.log(2) * period)))

        if 'Hist_Vol_20D' in df_featured and 'Hist_Vol_60D' in df_featured:
            df_featured['Vol_Ratio_S_L'] = df_featured['Hist_Vol_20D'] / df_featured['Hist_Vol_60D'].replace(0, np.nan)
        if 'Hist_Vol_5D' in df_featured and 'Hist_Vol_20D' in df_featured:
            df_featured['Vol_Ratio_VS_S'] = df_featured['Hist_Vol_5D'] / df_featured['Hist_Vol_20D'].replace(0, np.nan)

        if 'High' in df_featured and 'Low' in df_featured and 'Close' in df_featured:
            df_featured['Price_Range_Daily_Norm'] = (df_featured['High'] - df_featured['Low']) / df_featured['Close'].replace(0,np.nan)
            for p_range in [5, 10, 20]:
                df_featured[f'Price_Range_Avg_Norm_{p_range}D'] = df_featured['Price_Range_Daily_Norm'].rolling(p_range).mean()
                df_featured[f'Price_Range_Std_Norm_{p_range}D'] = df_featured['Price_Range_Daily_Norm'].rolling(p_range).std()

        if 'VIX_Value' in df_featured.columns and 'Hist_Vol_20D' in df_featured.columns:
            df_featured['StockVol_x_VIX'] = df_featured['Hist_Vol_20D'] * df_featured['VIX_Value']
            df_featured['StockVol_div_VIX'] = df_featured['Hist_Vol_20D'] / df_featured['VIX_Value'].replace(0, np.nan)
        elif 'Hist_Vol_20D' in df_featured.columns: 
             df_featured['StockVol_x_VIX'] = 0.0 
             df_featured['StockVol_div_VIX'] = 0.0
        else: 
             df_featured['StockVol_x_VIX'] = 0.0
             df_featured['StockVol_div_VIX'] = 0.0

        if all(col in df_featured.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
            bb_range = (df_featured['BB_Upper'] - df_featured['BB_Lower']).replace(0, np.nan)
            df_featured['BB_Width_Norm'] = bb_range / df_featured['Close'].replace(0, np.nan)
            if 'BB_Width_Norm' in df_featured.columns:
                 df_featured['BB_Squeeze_Index'] = df_featured['BB_Width_Norm'] / df_featured['BB_Width_Norm'].rolling(20).mean().replace(0,np.nan)

        if 'Volume' in df_featured.columns and 'Price_Range_Daily_Norm' in df_featured.columns:
            df_featured['Volume_x_PriceRange'] = df_featured['Volume'] * df_featured['Price_Range_Daily_Norm']
            df_featured['Volume_MA_Ratio_20'] = df_featured['Volume'] / df_featured['Volume'].rolling(20).mean().replace(0,np.nan)

        if 'RSI_14' in df_featured.columns:
            df_featured['RSI_Extreme'] = ((df_featured['RSI_14'] < 20) | (df_featured['RSI_14'] > 80)).astype(int)
        if 'MACD_Hist' in df_featured.columns: 
             df_featured['MACD_Hist_Abs_Mean_5D'] = df_featured['MACD_Hist'].abs().rolling(5).mean()

        try:
            if isinstance(df_featured.index, pd.DatetimeIndex):
                df_featured['Day_of_Week_Sin'] = np.sin(2 * np.pi * df_featured.index.dayofweek / 6)
                df_featured['Day_of_Week_Cos'] = np.cos(2 * np.pi * df_featured.index.dayofweek / 6)
                df_featured['Month_Sin'] = np.sin(2 * np.pi * df_featured.index.month/12)
                df_featured['Month_Cos'] = np.cos(2 * np.pi * df_featured.index.month/12)
            else: raise AttributeError("Index not DatetimeIndex")
        except AttributeError: 
             self._status_update("Index không phải DatetimeIndex, bỏ qua time features.", status_callback, True)
             for tc in ['Day_of_Week_Sin', 'Day_of_Week_Cos', 'Month_Sin', 'Month_Cos']:
                 if tc not in df_featured.columns: df_featured[tc] = 0.0

        df_featured = df_featured.replace([np.inf, -np.inf], np.nan)
        for col in df_featured.columns:
            if col in df_featured.columns and df_featured[col].dtype in ['float64', 'float32', 'float16']:
                df_featured[col] = df_featured[col].ffill().bfill().fillna(0)
        
        final_feature_count = len(df_featured.columns) - len(df.columns)
        self._status_update(f"Đã tạo/cập nhật {final_feature_count} đặc trưng nâng cao cho biến động (v2)", status_callback)
        return df_featured

    def engineer_ultra_advanced_features(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        self._status_update("Đang tạo features siêu nâng cao (cho biến động - v2)...", status_callback)
        df_featured = self.engineer_advanced_features(df, status_callback) 

        if 'Close' in df_featured.columns:
            returns = df_featured['Close'].pct_change()
            for window in [20, 60]: 
                df_featured[f'Returns_Skew_{window}D'] = returns.rolling(window).skew()
                df_featured[f'Returns_Kurt_{window}D'] = returns.rolling(window).kurt()

        if TALIB_AVAILABLE and all(c in df_featured for c in ['High', 'Low', 'Close']):
            try: 
                df_featured['ADX_14'] = talib.ADX(df_featured['High'], df_featured['Low'], df_featured['Close'], timeperiod=14)
                df_featured['Is_Trending_ADX'] = (df_featured['ADX_14'] > 25).astype(int) 
            except Exception as e_adx: self._status_update(f"Lỗi tính ADX: {e_adx}", status_callback)

        if 'Close' in df_featured.columns:
            close_prices = df_featured['Close']
            kernels = {
                'local_peak_3d': [-1, 2, -1], 'local_trough_3d': [1, -2, 1],
                'accel_up_3d': [1, -2, 1], 'accel_down_3d': [-1, 2, -1],
            }
            for name, kernel_arr in kernels.items():
                kernel = np.array(kernel_arr)
                df_featured[f'Pattern_{name}'] = close_prices.rolling(window=len(kernel), center=True).apply(lambda x: np.dot(x, kernel) if len(x)==len(kernel) else 0, raw=True).fillna(0)
            
            if TALIB_AVAILABLE:
                try:
                    df_featured['TEMA_20'] = talib.TEMA(close_prices.values, timeperiod=20)
                    df_featured['Price_vs_TEMA20'] = (df_featured['Close'] - df_featured['TEMA_20']) / df_featured['TEMA_20'].replace(0,np.nan)
                except Exception as e_tema: self._status_update(f"Lỗi tính TEMA_20: {e_tema}", status_callback)

        df_featured = df_featured.replace([np.inf, -np.inf], np.nan)
        for col in df_featured.columns:
            if col in df_featured.columns and df_featured[col].dtype in ['float64', 'float32', 'float16']:
                df_featured[col] = df_featured[col].ffill().bfill().fillna(0)
        self._status_update(f"🚀 Đã tạo thêm features siêu nâng cao (cho biến động - v2) - Tổng cộng {len(df_featured.columns)} features.", status_callback)
        return df_featured

    def create_target_variable(self, df: pd.DataFrame, volatility_threshold: float = 0.02, status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        self._status_update(f"Đang tạo biến mục tiêu BIẾN ĐỘNG với ngưỡng {volatility_threshold:.4f}...", status_callback)
        df_target = df.copy()
        if 'Close' not in df_target.columns:
            self._status_update("Thiếu cột 'Close', không thể tính toán target biến động.", status_callback, True)
            df_target['Target_Volatility'] = np.nan; df_target['Future_Volatility_Raw'] = np.nan
            return df_target.dropna(subset=['Target_Volatility'])

        daily_returns_full_series = df_target['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
        rolling_std_dev = daily_returns_full_series.rolling(window=self.forecast_horizon, min_periods=max(1, int(self.forecast_horizon * 0.8))).std()
        df_target['Future_Volatility_Raw'] = rolling_std_dev.shift(-self.forecast_horizon) 
        
        df_target['Target_Volatility'] = (df_target['Future_Volatility_Raw'] > volatility_threshold).astype(int)
        self.target_column = 'Target_Volatility'
        df_target = df_target.dropna(subset=['Target_Volatility', 'Future_Volatility_Raw'])

        if not df_target.empty:
            target_distribution = df_target['Target_Volatility'].value_counts(normalize=True) * 100
            self._status_update(f"Phân phối target (Biến Động): Cao(1): {target_distribution.get(1, 0):.2f}%, Thấp(0): {target_distribution.get(0, 0):.2f}%", status_callback)
            if len(target_distribution) < 2 and len(df_target) > 10: 
                self._status_update("CẢNH BÁO: Chỉ có một lớp trong biến mục tiêu. Mô hình có thể không huấn luyện tốt.", status_callback, True)
        else:
            self._status_update("CẢNH BÁO: Không có dữ liệu target nào được tạo. Kiểm tra forecast_horizon và dữ liệu đầu vào.", status_callback, True)
        return df_target

    def prepare_features(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[pd.DataFrame, List[str]]:
        self._status_update("Đang chuẩn bị đặc trưng (cho biến động)...", status_callback)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['Target_Volatility', 'Future_Volatility_Raw', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Ticker_Code', 'Ticker'] 
        feature_columns = [col for col in numeric_columns if col not in exclude_columns and col.lower() != 'adj_close' and 'ticker' not in col.lower()] 

        for col in feature_columns.copy(): 
            if df[col].isna().sum() / len(df) > 0.6: 
                feature_columns.remove(col)
                self._status_update(f"Loại bỏ {col} do >60% NaN", status_callback)
        X = df[feature_columns].copy()
        X = X.ffill().bfill().fillna(0) 
        X = X.replace([np.inf, -np.inf], 0) 
        self._status_update(f"Đã chuẩn bị {len(feature_columns)} đặc trưng cho biến động", status_callback)
        return X, feature_columns

    def advanced_feature_selection(self, X: pd.DataFrame, y: pd.Series, status_callback: Optional[Callable[[str, bool], None]] = None) -> List[str]:
        self._status_update("Đang thực hiện Advanced Feature Selection (cho biến động - v2)...", status_callback)
        if X.empty or y.empty or len(X) != len(y):
             self._status_update("Dữ liệu đầu vào cho feature selection không hợp lệ.", status_callback, True)
             return X.columns.tolist()[:min(50, X.shape[1] if X.shape[1]>0 else 50 )] 

        feature_scores = {col: [] for col in X.columns}
        try: 
            rf_selector = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced') 
            rf_selector.fit(X, y)
            rf_importance = pd.Series(rf_selector.feature_importances_, index=X.columns)
            max_score = rf_importance.max(); max_score = 1.0 if max_score == 0 else max_score
            for feature, imp in rf_importance.items(): feature_scores[feature].append(imp / max_score)
        except Exception as e: self._status_update(f"⚠️ RF feature selection thất bại: {e}", status_callback)

        try: 
            mi_scores_arr = mutual_info_classif(X, y, random_state=42)
            mi_scores = pd.Series(mi_scores_arr, index=X.columns)
            max_score = mi_scores.max(); max_score = 1.0 if max_score <= 0 else max_score
            for feature, mi in mi_scores.items(): feature_scores[feature].append(mi / max_score)
        except Exception as e: self._status_update(f"⚠️ MI feature selection thất bại: {e}", status_callback)
        
        try: # Add XGBoost importance
            xgb_sel = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
            xgb_sel.fit(X,y)
            xgb_imp = pd.Series(xgb_sel.feature_importances_, index=X.columns)
            max_s = xgb_imp.max(); max_s = 1.0 if max_s == 0 else max_s
            for f, imp_val in xgb_imp.items(): feature_scores[f].append(imp_val / max_s)
            self._status_update("✅ XGBoost feature importance (cho feature selection) tính toán xong.", status_callback)
        except Exception as e_xgb_fs: 
            self._status_update(f"⚠️ XGBoost Feature Selection thất bại: {e_xgb_fs}", status_callback)

        final_scores = {feature: np.mean(scores) if scores else 0.0 for feature, scores in feature_scores.items()}
        sorted_features_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        num_to_select = 0
        cumulative_importance = 0.0
        importance_threshold = 0.90 
        total_score_sum = sum(s for _,s in sorted_features_scores if s > 0)
        
        selected_features = []
        if total_score_sum > 0:
            for feature, score in sorted_features_scores:
                if score <= 0: break 
                selected_features.append(feature)
                cumulative_importance += (score / total_score_sum)
                num_to_select += 1
                if cumulative_importance >= importance_threshold and num_to_select >= 30 : 
                    break
                if num_to_select >= 100: 
                    break
        
        if not selected_features: 
            selected_features = [f[0] for f in sorted_features_scores[:min(50, len(sorted_features_scores))]]
        if not selected_features and not X.empty: 
            selected_features = X.columns.tolist()[:20]

        self._status_update(f"🎯 Đã chọn {len(selected_features)} features cho dự đoán biến động (v2)", status_callback)
        if selected_features:
             self._status_update(f"Top 10 features (biến động - v2): {[f for f in selected_features[:10]]}", status_callback)
        return selected_features

    def advanced_data_preprocessing(self, X: pd.DataFrame, y: pd.Series, status_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        self._status_update("Đang thực hiện Advanced Data Preprocessing (cho biến động)...", status_callback)
        X_processed, y_processed = X.copy(), y.copy()

        try:
            from sklearn.feature_selection import VarianceThreshold
            variance_selector = VarianceThreshold(threshold=1e-4) 
            X_temp = variance_selector.fit_transform(X_processed)
            retained_cols = X_processed.columns[variance_selector.get_support()]
            X_processed = pd.DataFrame(X_temp, columns=retained_cols, index=X_processed.index)
        except Exception as e: self._status_update(f"⚠️ Lọc variance thất bại: {e}", status_callback)

        try:
            correlation_matrix = X_processed.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            high_corr_features_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)] 
            if high_corr_features_to_drop:
                X_processed = X_processed.drop(columns=high_corr_features_to_drop)
        except Exception as e: self._status_update(f"⚠️ Lọc tương quan thất bại: {e}", status_callback)
        
        if not X_processed.empty:
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.01, random_state=42, n_estimators=100) 
                outlier_labels = iso_forest.fit_predict(X_processed)
                outlier_mask = outlier_labels == 1
                original_len = len(X_processed)
                X_processed = X_processed[outlier_mask]
                y_processed = y_processed[outlier_mask] 
                X_processed = X_processed.reset_index(drop=True)
                y_processed = y_processed.reset_index(drop=True)
            except Exception as e: self._status_update(f"⚠️ Loại bỏ outlier thất bại: {e}. Dữ liệu có thể chứa outliers.", status_callback)
        
        self._status_update(f"🎯 Preprocessing (biến động) hoàn thành: {X_processed.shape[0]} samples, {X_processed.shape[1]} features", status_callback)
        return X_processed, y_processed

    def smart_class_balancing(self, X: pd.DataFrame, y: pd.Series, strategy: str = 'adaptive',
                             status_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        self._status_update("Đang thực hiện Smart Class Balancing (cho biến động)...", status_callback)
        if not IMBLEARN_AVAILABLE:
            self._status_update("⚠️ imblearn không khả dụng, bỏ qua class balancing", status_callback); return X, y
        
        class_counts = y.value_counts()
        if len(class_counts) < 2 or X.empty or y.empty:
            self._status_update("Không đủ lớp hoặc dữ liệu trống để cân bằng. Bỏ qua.", status_callback); return X, y
        
        minority_ratio = min(class_counts) / max(class_counts) if max(class_counts) > 0 else 1.0
        self._status_update(f"Original distribution: {dict(class_counts)} (Ratio: {minority_ratio:.3f})", status_callback)

        if strategy == 'adaptive':
            if minority_ratio > 0.8: strategy = 'none' 
            elif minority_ratio > 0.5: strategy = 'tomek' 
            elif minority_ratio > 0.25: strategy = 'smote_tomek' 
            else: strategy = 'smote' 

        self._status_update(f"Selected balancing strategy: {strategy}", status_callback)
        if strategy == 'none': return X, y

        try:
            original_columns = X.columns.tolist()
            min_class_count = min(class_counts)
            k_neighbors_val = min(5, min_class_count - 1 if min_class_count > 1 else 1)

            if k_neighbors_val < 1 and strategy in ['smote', 'adasyn', 'borderline_smote', 'smote_tomek', 'smote_enn']:
                 self._status_update(f"Không đủ sample ({min_class_count}) cho k-neighbors={k_neighbors_val} trong {strategy}. Bỏ qua balancing.", status_callback); return X, y

            if strategy == 'smote': balancer = SMOTE(random_state=42, k_neighbors=k_neighbors_val)
            elif strategy == 'adasyn': balancer = ADASYN(random_state=42, n_neighbors=k_neighbors_val)
            elif strategy == 'borderline_smote': balancer = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors_val, kind='borderline-1')
            elif strategy == 'smote_tomek': balancer = SMOTETomek(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k_neighbors_val), tomek=TomekLinks(sampling_strategy='majority'))
            elif strategy == 'smote_enn': balancer = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k_neighbors_val), enn=EditedNearestNeighbours(n_neighbors=3, kind_sel='mode'))
            elif strategy == 'tomek': balancer = TomekLinks(sampling_strategy='majority')
            else: self._status_update(f"⚠️ Unknown strategy: {strategy}, using SMOTE", status_callback); balancer = SMOTE(random_state=42, k_neighbors=k_neighbors_val)
            
            X_balanced, y_balanced = balancer.fit_resample(X, y)
            X_balanced = pd.DataFrame(X_balanced, columns=original_columns, index=range(len(X_balanced)))
            y_balanced = pd.Series(y_balanced, name=y.name, index=range(len(y_balanced)))
            
            new_counts = y_balanced.value_counts()
            self._status_update(f"✅ Balanced distribution: {dict(new_counts)}", status_callback)
            return X_balanced, y_balanced
        except Exception as e:
            self._status_update(f"⚠️ Class balancing với {strategy} thất bại: {e}. Using original data.", status_callback); return X, y


    def train_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, status_callback: Optional[Callable[[str, bool], None]] = None) -> xgb.XGBClassifier:
        self._status_update("Đang thực hiện Advanced Hyperparameter Tuning cho XGBoost (Biến Động)...", status_callback)
        try: import optuna; OPTUNA_AVAILABLE = True
        except ImportError: OPTUNA_AVAILABLE = False
        
        scale_pos_weight_val = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
        self._status_update(f"Calculated scale_pos_weight for training: {scale_pos_weight_val:.2f}", status_callback)

        if OPTUNA_AVAILABLE:
            self._status_update("✅ Sử dụng Optuna cho Bayesian hyperparameter optimization (Biến Động)", status_callback)
            def objective(trial): 
                params = {
                    'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 
                    'random_state': 42, 'n_jobs': -1, 'verbosity': 0, 'early_stopping_rounds': 30, 
                    'max_depth': trial.suggest_int('max_depth', 3, 8), 
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 15), 
                    'subsample': trial.suggest_float('subsample', 0.5, 0.9), 
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9), 
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True), 
                    'n_estimators': trial.suggest_int('n_estimators', 400, 2500), 
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.5), 
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0), 
                    'gamma': trial.suggest_float('gamma', 0, 1.5),
                    'scale_pos_weight': scale_pos_weight_val, 
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            
            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=7, n_min_trials=5)) 
            study.optimize(objective, n_trials=50, show_progress_bar=False, n_jobs=1) 
            best_params = study.best_params
            self._status_update(f"Best trial score (AUC) from Optuna: {study.best_value:.4f}", status_callback)
            
            final_model_params = {**best_params, 'objective': 'binary:logistic', 'eval_metric': 'auc', 
                                  'tree_method': 'hist', 'random_state': 42, 'verbosity': 0, 'n_jobs': -1, 
                                  'early_stopping_rounds': 30, 'scale_pos_weight': scale_pos_weight_val}
            # n_estimators from best_params is the one Optuna decided was best for the given search space for n_estimators
            # If early stopping happened, the *actual* number of trees used might be less than this n_estimators.
            # We want to train the final model with the n_estimators Optuna chose, and let early stopping refine it if needed.
            
            best_model = xgb.XGBClassifier(**final_model_params)
            best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        else: 
            self._status_update("⚠️ Optuna không khả dụng, sử dụng GridSearchCV (Biến Động)", status_callback)
            param_grid = {'max_depth': [4, 7], 'learning_rate': [0.01, 0.03], 'n_estimators': [700, 1200], 'subsample': [0.7, 0.85], 'colsample_bytree': [0.7, 0.85], 'scale_pos_weight': [scale_pos_weight_val]}
            xgb_base = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', tree_method='hist', random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=30)
            grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='roc_auc', n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            best_model = grid_search.best_estimator_
        
        val_pred_proba = best_model.predict_proba(X_val)[:,1]
        val_auc = roc_auc_score(y_val, val_pred_proba)
        val_pred_class = (val_pred_proba > 0.5).astype(int) 
        val_accuracy = accuracy_score(y_val, val_pred_class) 
        val_f1 = f1_score(y_val, val_pred_class, pos_label=1, zero_division=0)

        self._status_update(f"Final Model Validation (Volatility) - AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}, F1 (HighVol): {val_f1:.4f}", status_callback)
        return best_model

    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, status_callback: Optional[Callable[[str, bool], None]] = None):
        self._status_update("Đang tạo ensemble model (cho biến động)...", status_callback)
        models_to_ensemble = []
        try:
            xgb_m = self.train_xgboost_model(X_train, y_train, X_val, y_val, status_callback)
            models_to_ensemble.append(('xgb', xgb_m))
        except Exception as e: self._status_update(f"⚠️ XGBoost (biến động) thất bại khi tạo ensemble: {e}", status_callback)
        
        if LIGHTGBM_AVAILABLE:
            try:
                lgbm = lgb.LGBMClassifier(random_state=42, n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1, class_weight='balanced')
                lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
                models_to_ensemble.append(('lgbm', lgbm))
            except Exception as e: self._status_update(f"⚠️ LightGBM (biến động) thất bại khi tạo ensemble: {e}", status_callback)

        if not models_to_ensemble:
            self._status_update("❌ Không có model nào huấn luyện thành công cho ensemble. Dùng XGBoost mặc định.", status_callback, True)
            fallback = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0); fallback.fit(X_train, y_train); return fallback
        if len(models_to_ensemble) == 1: return models_to_ensemble[0][1]
        
        try:
            ensemble_model = VotingClassifier(estimators=models_to_ensemble, voting='soft', n_jobs=-1) 
            ensemble_model.fit(X_train, y_train)
            return ensemble_model
        except Exception as e_ens: 
            self._status_update(f"Tạo Voting Ensemble (biến động) thất bại: {e_ens}. Trả về model đầu tiên.", status_callback)
            return models_to_ensemble[0][1]

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, status_callback: Optional[Callable[[str, bool], None]] = None) -> Dict[str, float]:
        # This method remains largely the same
        self._status_update("Đang đánh giá mô hình (dự đoán biến động)...", status_callback)
        y_pred = model.predict(X_test)
        try: y_pred_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError: 
             y_pred_proba = (model.decision_function(X_test) if hasattr(model, 'decision_function') else y_pred.astype(float))
             if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1: y_pred_proba = y_pred_proba[:,1]
             if not ((y_pred_proba >= 0) & (y_pred_proba <= 1)).all() and len(y_pred_proba)>0 :
                 min_val, max_val = y_pred_proba.min(), y_pred_proba.max()
                 y_pred_proba = (y_pred_proba - min_val) / (max_val - min_val) if max_val > min_val else np.full_like(y_pred_proba, 0.5)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_high_vol': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'recall_high_vol': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            'f1_score_high_vol': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            'precision_low_vol': precision_score(y_test, y_pred, pos_label=0, zero_division=0),
            'recall_low_vol': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
            'f1_score_low_vol': f1_score(y_test, y_pred, pos_label=0, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 and not np.all(y_pred_proba == y_pred_proba[0]) else 0.5
        }
        accuracy_pct = metrics['accuracy'] * 100
        target_reached_vol = "🎉 ĐẠT MỤC TIÊU (Biến Động)!" if accuracy_pct >= 85 else "⚠️ Chưa đạt 85% (Biến Động)"
        self._status_update(f"=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (BIẾN ĐỘNG) ===", status_callback)
        self._status_update(f"  Accuracy: {metrics['accuracy']:.4f} ({accuracy_pct:.2f}%) - {target_reached_vol}", status_callback)
        self._status_update(f"  Precision (Biến Động Cao): {metrics['precision_high_vol']:.4f}", status_callback)
        self._status_update(f"  Recall (Biến Động Cao): {metrics['recall_high_vol']:.4f}", status_callback)
        self._status_update(f"  F1-Score (Biến Động Cao): {metrics['f1_score_high_vol']:.4f}", status_callback)
        self._status_update(f"  ROC-AUC: {metrics['roc_auc']:.4f}", status_callback)
        return metrics

    def save_model(self, model, scaler: StandardScaler, feature_columns: List[str], metrics: Dict[str, float], target_threshold: float, status_callback: Optional[Callable[[str, bool], None]] = None) -> str:
        # This method remains largely the same
        self._status_update("Đang lưu mô hình (dự đoán biến động)...", status_callback)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        accuracy_pct = int(metrics['accuracy'] * 100)
        model_type = "Ensemble_Volatility" if hasattr(model, 'estimators_') or hasattr(model, 'estimators') or hasattr(model, 'models_data') else "XGBoost_Volatility"
        if isinstance(model, xgb.XGBClassifier): model_type = "XGBoost_Volatility"

        model_filename = f"{model_type.lower()}_predictor_{self.forecast_horizon}d_{accuracy_pct}pct_{timestamp}.joblib"
        scaler_filename = f"scaler_volatility_{self.forecast_horizon}d_{timestamp}.joblib"
        info_filename = f"model_info_volatility_{self.forecast_horizon}d_{timestamp}.json"
        model_path = os.path.join(MODEL_DIR, model_filename); scaler_path = os.path.join(MODEL_DIR, scaler_filename); info_path = os.path.join(MODEL_DIR, info_filename)
        joblib.dump(model, model_path); joblib.dump(scaler, scaler_path)
        model_info = {
            'model_type': model_type, 'model_purpose': 'volatility',
            'forecast_horizon_days': self.forecast_horizon, 'target_threshold': target_threshold,
            'target_column': self.target_column, 'feature_columns': feature_columns, 'metrics': metrics,
            'model_filename': model_filename, 'scaler_filename': scaler_filename,
            'training_timestamp': timestamp, 'feature_count': len(feature_columns),
            'accuracy_target_reached': metrics['accuracy'] >= 0.85
        }
        if hasattr(model, 'get_params'):
            try: model_info['model_params'] = {k:str(v) for k,v in model.get_params(deep=False).items()} 
            except: model_info['model_params'] = "Could not serialize params"

        with open(info_path, 'w') as f: json.dump(model_info, f, indent=4, default=str)
        self._status_update(f"Đã lưu {model_type} model: {model_filename}", status_callback)
        return model_path

    def optimize_target_threshold(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> float:
        # This method remains largely the same
        self._status_update("Đang tối ưu ngưỡng BIẾN ĐỘNG...", status_callback)
        if 'Close' not in df.columns: return 0.02
        
        daily_returns_full = df['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
        rolling_std = daily_returns_full.rolling(window=self.forecast_horizon, min_periods=max(1, int(self.forecast_horizon * 0.8))).std()
        calculated_future_volatility = rolling_std.shift(-self.forecast_horizon).dropna()

        if calculated_future_volatility.empty or len(calculated_future_volatility) < 20: return 0.02
        
        percentiles_to_test = [20, 30, 40, 50, 60, 70, 80] 
        volatility_threshold_candidates = np.percentile(calculated_future_volatility, percentiles_to_test)
        volatility_threshold_candidates = sorted(list(set(v for v in volatility_threshold_candidates if v > 1e-5))) 

        best_threshold_val = 0.02; best_balance_metric = -1
        for threshold_val_opt in volatility_threshold_candidates:
            target_opt = (calculated_future_volatility > threshold_val_opt).astype(int)
            counts = target_opt.value_counts()
            if len(counts) < 2: continue
            r0, r1 = counts.get(0,0)/len(target_opt), counts.get(1,0)/len(target_opt)
            metric = (1.0 - abs(r0-0.5)) * (min(r0,r1)/0.5) if min(r0,r1) > 0.05 else 0.0 
            if metric > best_balance_metric: best_balance_metric, best_threshold_val = metric, threshold_val_opt
        self._status_update(f"🎯 Ngưỡng biến động tối ưu: {best_threshold_val:.4f} (Metric cân bằng: {best_balance_metric:.3f})", status_callback)
        return best_threshold_val

    def engineer_deep_learning_inspired_features(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        # This method remains largely the same (simplified version)
        self._status_update("Tạo Deep Learning Inspired Features (phiên bản đơn giản hóa)...", status_callback)
        df_deep = df.copy()
        if 'Close' in df_deep.columns:
            close_prices = df_deep['Close']
            for window in [10, 30]: 
                returns = close_prices.pct_change().fillna(0)
                volatility = returns.rolling(window).std().fillna(0)
                exp_vol = np.exp(-volatility) 
                sum_exp_vol_rolling = exp_vol.rolling(window).sum().replace(0, 1e-9)
                att_weights = exp_vol / sum_exp_vol_rolling
                df_deep[f'Attention_Price_InvVol_{window}'] = (close_prices * att_weights).rolling(window).sum()
            self._status_update("✅ Attention-like (InvVol) features created", status_callback)
        return df_deep.fillna(0) 

    def run_full_training_pipeline(self, processed_files: List[str], target_threshold: float = 0.02,
                                 test_size: float = 0.2, status_callback: Optional[Callable[[str, bool], None]] = None,
                                 progress_callback: Optional[Callable[[float, str], None]] = None,
                                 prediction_type: str = 'volatility'
                                 ) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[List[str]]]:
        # This method's overall flow remains the same
        self.prediction_type = prediction_type 
        self.target_column = 'Target_Volatility'

        try:
            self._progress_update(0.05, "Tải dữ liệu...", progress_callback)
            df = self.load_and_combine_data(processed_files, status_callback)

            self._progress_update(0.15, "Tạo đặc trưng (cho biến động)...", progress_callback)
            df = self.engineer_ultra_advanced_features(df, status_callback) 

            self._progress_update(0.20, "Tối ưu ngưỡng biến động...", progress_callback)
            optimized_threshold = self.optimize_target_threshold(df, status_callback)
            if abs(optimized_threshold - target_threshold) > 1e-5 :
                target_threshold = optimized_threshold
                self._status_update(f"Ngưỡng biến động được cập nhật: {target_threshold:.4f}", status_callback)

            self._progress_update(0.25, "Tạo biến mục tiêu (biến động)...", progress_callback)
            df = self.create_target_variable(df, target_threshold, status_callback)
            if df.empty or self.target_column not in df.columns or df[self.target_column].isna().all():
                raise ValueError("Không có dữ liệu target hợp lệ sau khi tạo biến mục tiêu.")

            self._progress_update(0.35, "Chuẩn bị đặc trưng...", progress_callback)
            X, feature_cols_initial = self.prepare_features(df, status_callback)
            y = df[self.target_column]
            
            initial_rows = len(X)
            good_rows_mask = (X.isna().sum(axis=1) / (X.shape[1] if X.shape[1] > 0 else 1)) <= 0.5 
            X, y = X[good_rows_mask].reset_index(drop=True), y[good_rows_mask].reset_index(drop=True)
            self._status_update(f"Sau khi loại bỏ hàng nhiều NaN: {len(X)}/{initial_rows} rows còn lại", status_callback)
            if X.empty: raise ValueError("Dữ liệu trống sau khi loại bỏ hàng NaN.")


            self._progress_update(0.40, "Advanced data preprocessing...", progress_callback)
            X_processed, y_processed = self.advanced_data_preprocessing(X, y, status_callback)
            if X_processed.empty: raise ValueError("Dữ liệu trống sau advanced_data_preprocessing.")


            self._progress_update(0.42, "Smart Class Balancing...", progress_callback)
            X_processed, y_processed = self.smart_class_balancing(X_processed, y_processed, 'adaptive', status_callback)
            if X_processed.empty: raise ValueError("Dữ liệu trống sau smart_class_balancing.")


            self._progress_update(0.45, "Advanced feature selection...", progress_callback)
            selected_features = self.advanced_feature_selection(X_processed, y_processed, status_callback)
            X_selected = X_processed[selected_features]
            if X_selected.empty: raise ValueError("Dữ liệu trống sau feature selection.")


            self._progress_update(0.55, "Chia dữ liệu...", progress_callback)
            min_samples_needed_for_stratify = 2 
            stratify_y = y_processed if not y_processed.empty and y_processed.value_counts().min() >= min_samples_needed_for_stratify else None
            if stratify_y is None and not y_processed.empty and y_processed.value_counts().min() > 0:
                 self._status_update("Cảnh báo: Không đủ mẫu ở lớp thiểu số để thực hiện stratified split cho train/test.", status_callback)

            X_train, X_test, y_train, y_test = train_test_split(X_selected, y_processed, test_size=test_size, random_state=42, stratify=stratify_y)

            stratify_y_train = y_train if not y_train.empty and y_train.value_counts().min() >= min_samples_needed_for_stratify else None
            if stratify_y_train is None and not y_train.empty and y_train.value_counts().min() > 0:
                 self._status_update("Cảnh báo: Không đủ mẫu ở lớp thiểu số (trong tập train) để thực hiện stratified split cho train/validation.", status_callback)

            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_y_train) 
            self._status_update(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}", status_callback)
            if X_train.empty or X_val.empty or X_test.empty: raise ValueError("Một trong các tập dữ liệu train/val/test bị trống.")

            self._progress_update(0.65, "Chuẩn hóa dữ liệu...", progress_callback)
            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_val_scaled = scaler.transform(X_val); X_test_scaled = scaler.transform(X_test)
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=selected_features, index=X_val.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)

            self._progress_update(0.75, "Huấn luyện mô hình XGBoost...", progress_callback)
            model = self.train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val, status_callback)
            if model is None: raise ValueError("Huấn luyện mô hình thất bại.")


            self._progress_update(0.90, "Đánh giá mô hình...", progress_callback)
            metrics = self.evaluate_model(model, X_test_scaled, y_test, status_callback)

            self._progress_update(0.95, "Lưu mô hình...", progress_callback)
            model_path = self.save_model(model, scaler, selected_features, metrics, target_threshold, status_callback)

            self.model = model; self.scaler = scaler; self.feature_columns = selected_features
            self._progress_update(1.0, "🎉 Hoàn thành pipeline!", progress_callback)
            accuracy_pct = metrics['accuracy'] * 100
            if accuracy_pct >= 0.85: self._status_update(f"🏆 SUCCESS! Đã đạt mục tiêu accuracy (Biến Động): {accuracy_pct:.2f}% >= 85%", status_callback)
            else: self._status_update(f"📈 Accuracy (Biến Động): {accuracy_pct:.2f}% - Gần đạt mục tiêu 85%", status_callback)
            return model_path, metrics, selected_features
        except ValueError as ve: 
             self._status_update(f"❌ Lỗi dữ liệu trong pipeline: {ve}", status_callback, True); traceback.print_exc(); return None, None, []
        except Exception as e:
            self._status_update(f"❌ Lỗi trong quá trình huấn luyện (biến động): {e}", status_callback, True); traceback.print_exc(); raise e

def train_stock_prediction_model(processed_files: List[str], forecast_horizon: int,
                                 target_threshold: float, test_size: float = 0.2,
                                 status_callback: Optional[Callable[[str, bool], None]] = None,
                                 progress_callback: Optional[Callable[[float, str], None]] = None,
                                 prediction_type: str = 'volatility'
                                 ) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[List[str]]]:
    try:
        predictor = StockVolatilityPredictor(forecast_horizon=forecast_horizon)
        return predictor.run_full_training_pipeline(
            processed_files, target_threshold, test_size,
            status_callback, progress_callback, prediction_type=prediction_type
        )
    except Exception as e_global:
        if status_callback: status_callback(f"Lỗi nghiêm trọng trong pipeline huấn luyện ({prediction_type}): {e_global}", True)
        print(f"Lỗi nghiêm trọng trong pipeline huấn luyện ({prediction_type}): {e_global}")
        traceback.print_exc()
        return None, None, None

if __name__ == '__main__':
    print("--- Running StockVolatilityPredictor Standalone Test ---")
    def cli_status_callback(message: str, is_error: bool = False): print(f"[{'ERROR' if is_error else 'INFO'}] {message}")
    def cli_progress_callback(progress: float, message: str): print(f"Progress: {progress*100:.1f}% - {message}")

    sample_processed_files = []
    if os.path.exists(PROCESSED_DATA_DIR):
        all_files = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed_data.csv')]
        if all_files: sample_processed_files = all_files[:min(1, len(all_files))] 
    if not sample_processed_files:
        cli_status_callback("No processed data files found for testing. Exiting.", True); sys.exit(1)
    cli_status_callback(f"Using files for test: {[os.path.basename(f) for f in sample_processed_files]}", False)

    try:
        model_file_path, final_metrics, final_feature_cols = train_stock_prediction_model(
            processed_files=sample_processed_files, forecast_horizon=5,
            target_threshold=0.015, test_size=0.2, 
            status_callback=cli_status_callback, progress_callback=cli_progress_callback, 
            prediction_type='volatility'
        )
        if model_file_path and final_metrics:
            cli_status_callback(f"--- Training (Volatility) Complete ---", False)
            cli_status_callback(f"Model saved to: {model_file_path}", False)
            cli_status_callback(f"Final Metrics (Volatility): {final_metrics}", False)
            if final_feature_cols:
                cli_status_callback(f"Number of features used: {len(final_feature_cols)}", False)
                cli_status_callback(f"Sample features: {final_feature_cols[:5]}...", False)
            accuracy_val = final_metrics.get('accuracy', 0)
            if accuracy_val >= 0.85: cli_status_callback(f"🎉 TARGET ACHIEVED (Volatility)! Accuracy: {accuracy_val*100:.2f}%", False)
            else: cli_status_callback(f"📈 Accuracy (Volatility): {accuracy_val*100:.2f}%.", False)
        else: cli_status_callback("Training (Volatility) failed or did not return results.", True)
    except Exception as main_e:
        cli_status_callback(f"Standalone test (Volatility) failed with error: {main_e}", True); traceback.print_exc()
    print("--- StockVolatilityPredictor Standalone Test Complete ---")