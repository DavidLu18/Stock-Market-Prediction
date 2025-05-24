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
import xgboost as xgb
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
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
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

# Tạo thư mục models nếu chưa có
for dir_path in [MODEL_DIR]:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}")

class StockTrendPredictor:
    """
    Lớp dự đoán xu hướng cổ phiếu sử dụng XGBoost
    Mục tiêu: Đạt độ chính xác trên 80% cho dự đoán 5 ngày
    """

    def __init__(self, forecast_horizon: int = 5):
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.target_column = None
        self.model_info = {}

    def _status_update(self, message: str, status_callback: Optional[Callable[[str, bool], None]] = None, is_error: bool = False):
        """Cập nhật trạng thái"""
        if status_callback:
            status_callback(message, is_error)
        else:
            level = "ERROR" if is_error else "INFO"
            print(f"[{level}] {message}")

    def _progress_update(self, progress: float, message: str, progress_callback: Optional[Callable[[float, str], None]] = None):
        """Cập nhật tiến độ"""
        if progress_callback:
            progress_callback(progress, message)
        else:
            print(f"Progress: {progress*100:.1f}% - {message}")

    def load_and_combine_data(self, processed_files: List[str], status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        """Tải và kết hợp dữ liệu từ các file đã xử lý"""
        self._status_update("Đang tải và kết hợp dữ liệu...", status_callback)

        all_data = []
        for file_path in processed_files:
            try:
                if not os.path.exists(file_path):
                    self._status_update(f"File không tồn tại: {file_path}", status_callback, True)
                    continue

                df = pd.read_csv(file_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)

                # Thêm tên ticker từ tên file
                ticker = os.path.basename(file_path).split('_processed_data.csv')[0]
                df['Ticker'] = ticker

                all_data.append(df)
                self._status_update(f"Đã tải {len(df)} rows từ {ticker}", status_callback)

            except Exception as e:
                self._status_update(f"Lỗi khi tải {file_path}: {e}", status_callback, True)
                continue

        if not all_data:
            raise ValueError("Không có dữ liệu nào được tải thành công")

        # Kết hợp tất cả dữ liệu
        combined_df = pd.concat(all_data, ignore_index=False, sort=False)
        combined_df.sort_index(inplace=True)

        self._status_update(f"Đã kết hợp {len(combined_df)} rows từ {len(all_data)} files", status_callback)
        return combined_df

    def engineer_advanced_features(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        """Tạo thêm các đặc trưng nâng cao cho XGBoost"""
        self._status_update("Đang tạo đặc trưng nâng cao...", status_callback)

        df_featured = df.copy()

        # Kiểm tra các cột cần thiết
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in df_featured.columns]
        if missing_cols:
            self._status_update(f"Thiếu các cột cần thiết: {missing_cols}. Sẽ bỏ qua một số features.", status_callback)

        # Chỉ thực hiện feature engineering nếu có đủ dữ liệu cơ bản
        if 'Close' not in df_featured.columns:
            self._status_update("Không có cột 'Close'. Không thể tạo features.", status_callback, True)
            return df_featured

        # 1. Price momentum features - Mở rộng window sizes
        periods = [2, 3, 5, 7, 10, 14, 21, 30, 50]
        for period in periods:
            df_featured[f'Price_Momentum_{period}D'] = (df_featured['Close'] / df_featured['Close'].shift(period) - 1) * 100

        # Thêm momentum acceleration (sau khi tất cả momentum đã được tạo)
        for period in periods:
            if period > 2:
                shorter_period = period // 2
                # Tìm period gần nhất có sẵn
                available_periods = [p for p in periods if p <= shorter_period]
                if available_periods:
                    reference_period = max(available_periods)
                    if f'Price_Momentum_{reference_period}D' in df_featured.columns:
                        df_featured[f'Momentum_Accel_{period}D'] = df_featured[f'Price_Momentum_{period}D'] - df_featured[f'Price_Momentum_{reference_period}D']

        # 2. Volume features - Mở rộng
        if 'Volume' in df_featured.columns:
            for period in [3, 5, 7, 10, 14, 20, 30]:
                df_featured[f'Volume_SMA_{period}'] = df_featured['Volume'].rolling(period).mean()
                # Avoid division by zero
                volume_sma_col = f'Volume_SMA_{period}'
                if volume_sma_col in df_featured.columns:
                    df_featured[f'Volume_Ratio_{period}'] = df_featured['Volume'] / df_featured[volume_sma_col].replace(0, np.nan)
                df_featured[f'Volume_Std_{period}'] = df_featured['Volume'].rolling(period).std()
                # Volume momentum - avoid division by zero
                volume_shifted = df_featured['Volume'].shift(period).replace(0, np.nan)
                df_featured[f'Volume_Momentum_{period}'] = (df_featured['Volume'] / volume_shifted - 1) * 100
        else:
            self._status_update("Không có cột 'Volume'. Bỏ qua volume features.", status_callback)

        # 3. Advanced volatility features
        returns = df_featured['Close'].pct_change()
        for period in [3, 5, 7, 10, 14, 20, 30, 50]:
            # Standard volatility
            df_featured[f'Volatility_{period}D'] = returns.rolling(period).std() * np.sqrt(252)
            # Downside volatility (negative returns only)
            negative_returns = returns.where(returns < 0, 0)
            df_featured[f'Downside_Vol_{period}D'] = negative_returns.rolling(period).std() * np.sqrt(252)
            # Volatility of volatility
            vol_series = returns.rolling(period).std()
            df_featured[f'Vol_of_Vol_{period}D'] = vol_series.rolling(5).std()
            # GARCH-like features
            df_featured[f'Volatility_MA_{period}'] = df_featured[f'Volatility_{period}D'].rolling(5).mean()

        # 4. Enhanced technical indicator features
        if 'RSI_14' in df_featured.columns:
            df_featured['RSI_Oversold'] = (df_featured['RSI_14'] < 30).astype(int)
            df_featured['RSI_Overbought'] = (df_featured['RSI_14'] > 70).astype(int)
            df_featured['RSI_Momentum'] = df_featured['RSI_14'].diff()
            df_featured['RSI_MA'] = df_featured['RSI_14'].rolling(5).mean()
            # RSI divergence proxy
            df_featured['Price_RSI_Divergence'] = (df_featured['Close'].pct_change() * df_featured['RSI_14'].diff()).rolling(5).mean()

        if 'MACD' in df_featured.columns and 'MACD_Signal' in df_featured.columns:
            df_featured['MACD_Above_Signal'] = (df_featured['MACD'] > df_featured['MACD_Signal']).astype(int)
            df_featured['MACD_Signal_Strength'] = abs(df_featured['MACD'] - df_featured['MACD_Signal'])
            df_featured['MACD_Momentum'] = df_featured['MACD'].diff()
            df_featured['MACD_Signal_Momentum'] = df_featured['MACD_Signal'].diff()

        # 5. Enhanced Bollinger Bands features
        if all(col in df_featured.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
            # Avoid division by zero in BB calculations
            bb_range = (df_featured['BB_Upper'] - df_featured['BB_Lower']).replace(0, np.nan)
            df_featured['BB_Position'] = (df_featured['Close'] - df_featured['BB_Lower']) / bb_range
            df_featured['BB_Width'] = bb_range / df_featured['Close'].replace(0, np.nan)

            # BB_Squeeze calculation - check if BB_Width exists first
            if 'BB_Width' in df_featured.columns:
                bb_width_quantile = df_featured['BB_Width'].rolling(20).quantile(0.2)
                df_featured['BB_Squeeze'] = (df_featured['BB_Width'] < bb_width_quantile).astype(int)

            df_featured['BB_Breakout_Up'] = (df_featured['Close'] > df_featured['BB_Upper']).astype(int)
            df_featured['BB_Breakout_Down'] = (df_featured['Close'] < df_featured['BB_Lower']).astype(int)

        # 6. Multi-timeframe SMA features
        sma_periods = [5, 10, 20, 50, 100, 200]
        for period in sma_periods:
            if f'SMA_{period}' in df_featured.columns:
                sma_col = f'SMA_{period}'
                df_featured[f'Price_Above_SMA_{period}'] = (df_featured['Close'] > df_featured[sma_col]).astype(int)
                # Avoid division by zero
                df_featured[f'Price_Distance_SMA_{period}'] = (df_featured['Close'] / df_featured[sma_col].replace(0, np.nan) - 1) * 100
                # SMA slope
                df_featured[f'SMA_{period}_Slope'] = df_featured[sma_col].diff(5)

        # SMA confluence (sau khi tất cả SMA features đã được tạo)
        for period in sma_periods:
            if period < 200 and f'SMA_{period}' in df_featured.columns:
                shorter_period = max(5, period // 2)
                # Tìm period gần nhất có sẵn
                available_shorter_periods = [p for p in sma_periods if p <= shorter_period]
                if available_shorter_periods:
                    reference_period = max(available_shorter_periods)
                    if f'SMA_{reference_period}' in df_featured.columns:
                        df_featured[f'SMA_Confluence_{reference_period}_{period}'] = (df_featured[f'SMA_{reference_period}'] > df_featured[f'SMA_{period}']).astype(int)

        # 7. Enhanced price action features
        if all(col in df_featured.columns for col in ['High', 'Low', 'Open', 'Close']):
            # Avoid division by zero in price action calculations
            df_featured['High_Low_Ratio'] = (df_featured['High'] - df_featured['Low']) / df_featured['Close'].replace(0, np.nan)
            df_featured['Open_Close_Ratio'] = (df_featured['Close'] - df_featured['Open']) / df_featured['Open'].replace(0, np.nan)
            df_featured['Upper_Shadow'] = (df_featured['High'] - np.maximum(df_featured['Open'], df_featured['Close'])) / df_featured['Close'].replace(0, np.nan)
            df_featured['Lower_Shadow'] = (np.minimum(df_featured['Open'], df_featured['Close']) - df_featured['Low']) / df_featured['Close'].replace(0, np.nan)
            df_featured['Body_Size'] = abs(df_featured['Close'] - df_featured['Open']) / df_featured['Close'].replace(0, np.nan)

            # Gap features
            df_featured['Gap_Up'] = ((df_featured['Open'] > df_featured['High'].shift(1)) & (df_featured['Open'] > df_featured['Close'].shift(1))).astype(int)
            df_featured['Gap_Down'] = ((df_featured['Open'] < df_featured['Low'].shift(1)) & (df_featured['Open'] < df_featured['Close'].shift(1))).astype(int)
        else:
            self._status_update("Thiếu OHLC data. Bỏ qua price action features.", status_callback)

        # 8. Multiple timeframe lag features
        important_features = ['Close', 'Volume', 'RSI_14', 'MACD', 'BB_Position']
        for feature in important_features:
            if feature in df_featured.columns:
                for lag in [1, 2, 3, 5, 7, 10]:
                    df_featured[f'{feature}_Lag_{lag}'] = df_featured[feature].shift(lag)

                # Feature momentum and acceleration
                df_featured[f'{feature}_Momentum_1'] = df_featured[feature].diff()
                df_featured[f'{feature}_Momentum_3'] = df_featured[feature].diff(3)
                df_featured[f'{feature}_Acceleration'] = df_featured[f'{feature}_Momentum_1'].diff()

        # 9. Enhanced rolling statistics
        for period in [3, 5, 7, 10, 14, 20]:
            df_featured[f'Close_Rolling_Max_{period}'] = df_featured['Close'].rolling(period).max()
            df_featured[f'Close_Rolling_Min_{period}'] = df_featured['Close'].rolling(period).min()
            df_featured[f'Close_Rolling_Range_{period}'] = df_featured[f'Close_Rolling_Max_{period}'] - df_featured[f'Close_Rolling_Min_{period}']
            df_featured[f'Close_Rolling_Std_{period}'] = df_featured['Close'].rolling(period).std()

            # Position within range - avoid division by zero
            range_col = f'Close_Rolling_Range_{period}'
            df_featured[f'Price_Position_Range_{period}'] = (df_featured['Close'] - df_featured[f'Close_Rolling_Min_{period}']) / df_featured[range_col].replace(0, np.nan)

            # Rolling rank (percentile position)
            df_featured[f'Price_Rank_{period}'] = df_featured['Close'].rolling(period).rank(pct=True)

        # 10. Market microstructure features
        if all(col in df_featured.columns for col in ['High', 'Low', 'Close']):
            df_featured['VWAP_Distance'] = (df_featured['Close'] - ((df_featured['High'] + df_featured['Low'] + df_featured['Close']) / 3))
            # Avoid division by zero in price efficiency
            high_low_diff = (df_featured['High'] - df_featured['Low']).replace(0, np.nan)
            df_featured['Price_Efficiency'] = abs(df_featured['Close'].diff()) / high_low_diff

        # 11. Regime detection features
        for period in [10, 20, 50]:
            # Trend strength
            df_featured[f'Trend_Strength_{period}'] = df_featured['Close'].rolling(period).apply(
                lambda x: np.corrcoef(x, np.arange(len(x)))[0, 1] if len(x) == period else np.nan
            )

            # Choppiness index
            if all(col in df_featured.columns for col in ['High', 'Low', 'Close']):
                true_range = np.maximum(
                    df_featured['High'] - df_featured['Low'],
                    np.maximum(
                        abs(df_featured['High'] - df_featured['Close'].shift(1)),
                        abs(df_featured['Low'] - df_featured['Close'].shift(1))
                    )
                )
                atr = true_range.rolling(period).mean()
                price_change = abs(df_featured['Close'] - df_featured['Close'].shift(period))

                # Avoid division by zero and log of zero/negative numbers
                atr_sum = atr.rolling(period).sum()
                price_change_safe = price_change.replace(0, np.nan)

                # Only calculate choppiness for valid ratios
                ratio = atr_sum / price_change_safe
                ratio_safe = ratio[(ratio > 0) & (ratio.notna())]

                if len(ratio_safe) > 0:
                    choppiness = 100 * np.log10(ratio) / np.log10(period)
                    # Replace invalid values with NaN
                    choppiness = choppiness.replace([np.inf, -np.inf], np.nan)
                    df_featured[f'Choppiness_{period}'] = choppiness
                else:
                    df_featured[f'Choppiness_{period}'] = np.nan

        # 12. Cross-asset features (if multiple tickers in dataset)
        if 'Ticker' in df_featured.columns:
            try:
                # Market-wide features - safer implementation
                market_close = df_featured.groupby(df_featured.index)['Close'].mean()

                # Ensure index alignment
                market_close_aligned = market_close.reindex(df_featured.index, method='ffill')
                df_featured['Market_Relative_Performance'] = (df_featured['Close'] / market_close_aligned.replace(0, np.nan) - 1).fillna(0)

                # Sector rotation features (simplified) - safer implementation
                close_pct_change = df_featured['Close'].pct_change().rolling(20).mean()
                market_pct_change = market_close_aligned.pct_change().rolling(20).mean()

                df_featured['Relative_Strength_vs_Market'] = (close_pct_change - market_pct_change).fillna(0)

            except Exception as e_cross_asset: # Catch specific exception
                self._status_update(f"Cross-asset feature calculation failed: {e_cross_asset}", status_callback, True)


        # 13. Time-based features
        try:
            df_featured['Day_of_Week'] = df_featured.index.dayofweek
            df_featured['Month'] = df_featured.index.month
            df_featured['Quarter'] = df_featured.index.quarter

            # Seasonal effects
            df_featured['Is_Monday'] = (df_featured['Day_of_Week'] == 0).astype(int)
            df_featured['Is_Friday'] = (df_featured['Day_of_Week'] == 4).astype(int)
            df_featured['Is_Month_End'] = (df_featured.index.day > 25).astype(int)
            df_featured['Is_Quarter_End'] = ((df_featured.index.month % 3 == 0) & (df_featured.index.day > 25)).astype(int)
        except Exception as e_time_features: # Catch specific exception
            self._status_update(f"Time-based feature calculation failed: {e_time_features}", status_callback, True)
            # If time-based features fail, create default values
            df_featured['Day_of_Week'] = 0
            df_featured['Month'] = 1
            df_featured['Quarter'] = 1
            df_featured['Is_Monday'] = 0
            df_featured['Is_Friday'] = 0
            df_featured['Is_Month_End'] = 0
            df_featured['Is_Quarter_End'] = 0

        # Clean up any infinite or extreme values at the end
        df_featured = df_featured.replace([np.inf, -np.inf], np.nan)

        # Fill remaining NaN values with more conservative approach
        for col in df_featured.columns:
            if df_featured[col].dtype in ['float64', 'float32']:
                # For numeric columns, use forward fill then backward fill, then 0
                df_featured[col] = df_featured[col].ffill().bfill().fillna(0)

        self._status_update(f"Đã tạo {len(df_featured.columns) - len(df.columns)} đặc trưng mới", status_callback)
        return df_featured

    def engineer_ultra_advanced_features(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        """Tạo features siêu nâng cao để đạt accuracy > 80%"""
        self._status_update("Đang tạo features siêu nâng cao để tối ưu accuracy...", status_callback)

        # Start with basic advanced features
        df_featured = self.engineer_advanced_features(df, status_callback)

        # Add deep learning inspired features
        df_featured = self.engineer_deep_learning_inspired_features(df_featured, status_callback)

        # 1. Market Microstructure Features
        if all(col in df_featured.columns for col in ['High', 'Low', 'Close', 'Open', 'Volume']):
            # Micro price movements
            df_featured['Micro_Price_Change'] = (df_featured['Close'] - df_featured['Open']) / df_featured['Open'].replace(0, np.nan)
            df_featured['Intraday_Volatility'] = (df_featured['High'] - df_featured['Low']) / df_featured['Open'].replace(0, np.nan)

            # Order flow imbalance proxy
            df_featured['Buy_Pressure'] = (df_featured['Close'] - df_featured['Low']) / (df_featured['High'] - df_featured['Low']).replace(0, np.nan)
            df_featured['Sell_Pressure'] = (df_featured['High'] - df_featured['Close']) / (df_featured['High'] - df_featured['Low']).replace(0, np.nan)
            df_featured['Order_Flow_Imbalance'] = df_featured['Buy_Pressure'] - df_featured['Sell_Pressure']

            # Volume-weighted price levels
            df_featured['VWAP'] = (df_featured['Volume'] * (df_featured['High'] + df_featured['Low'] + df_featured['Close']) / 3).rolling(20).sum() / df_featured['Volume'].rolling(20).sum()
            df_featured['Price_VWAP_Ratio'] = df_featured['Close'] / df_featured['VWAP'].replace(0, np.nan)

            # Accumulation/Distribution enhanced
            df_featured['Money_Flow_Multiplier'] = ((df_featured['Close'] - df_featured['Low']) - (df_featured['High'] - df_featured['Close'])) / (df_featured['High'] - df_featured['Low']).replace(0, np.nan)
            df_featured['Money_Flow_Volume'] = df_featured['Money_Flow_Multiplier'] * df_featured['Volume']
            df_featured['ADL_Enhanced'] = df_featured['Money_Flow_Volume'].cumsum()

        # 2. Advanced Statistical Features
        if 'Close' in df_featured.columns:
            returns = df_featured['Close'].pct_change()

            # Higher moments
            for window in [5, 10, 20, 50]:
                df_featured[f'Skewness_{window}'] = returns.rolling(window).skew()
                df_featured[f'Kurtosis_{window}'] = returns.rolling(window).kurt()

                # Value at Risk (VaR) and Conditional VaR
                df_featured[f'VaR_95_{window}'] = returns.rolling(window).quantile(0.05)
                df_featured[f'CVaR_95_{window}'] = returns.rolling(window).apply(lambda x: x[x <= x.quantile(0.05)].mean())

            # Entropy of returns (market uncertainty)
            def calculate_entropy(series):
                if len(series) < 2:
                    return 0
                hist, _ = np.histogram(series.dropna(), bins=10)
                hist = hist[hist > 0]
                if len(hist) == 0:
                    return 0
                probs = hist / hist.sum()
                return -np.sum(probs * np.log(probs + 1e-10))

            for window in [10, 20, 50]:
                df_featured[f'Return_Entropy_{window}'] = returns.rolling(window).apply(calculate_entropy)

        # 3. Market Regime Detection Features
        if 'Close' in df_featured.columns:
            # Bull/Bear market indicator
            df_featured['SMA_50'] = df_featured['Close'].rolling(50).mean()
            df_featured['SMA_200'] = df_featured['Close'].rolling(200).mean()
            df_featured['Golden_Cross'] = (df_featured['SMA_50'] > df_featured['SMA_200']).astype(int)

            # Market trend strength
            close_prices = df_featured['Close']
            for window in [20, 50, 100]:
                # Linear regression trend
                def calculate_trend_strength(prices):
                    if len(prices) < 2:
                        return 0
                    x = np.arange(len(prices))
                    if np.std(x) == 0 or np.std(prices) == 0:
                        return 0
                    return np.corrcoef(x, prices)[0, 1]

                df_featured[f'Trend_Strength_{window}'] = close_prices.rolling(window).apply(calculate_trend_strength)

                # Hurst exponent for trend persistence
                def calculate_hurst(ts):
                    if len(ts) < 20:
                        return 0.5
                    lags = range(2, min(20, len(ts)//2))
                    tau = []
                    for lag in lags:
                        try:
                            differences = ts[lag:].values - ts[:-lag].values
                            tau.append(np.std(differences))
                        except:
                            tau.append(np.nan)

                    tau = np.array([t for t in tau if not np.isnan(t)])
                    if len(tau) < 2:
                        return 0.5

                    lags = np.array(list(range(2, 2 + len(tau))))
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return poly[0] / 2.0 # Hurst exponent is H = slope / 2 for R/S analysis or related measures

                df_featured[f'Hurst_Exponent_{window}'] = close_prices.rolling(window).apply(calculate_hurst)

        # 4. Inter-market Relationships
        if 'VIX' in df_featured.columns:
            # VIX-based fear indicators
            df_featured['VIX_MA_20'] = df_featured['VIX'].rolling(20).mean()
            df_featured['VIX_Spike'] = (df_featured['VIX'] > df_featured['VIX_MA_20'] * 1.2).astype(int)
            df_featured['VIX_Price_Divergence'] = df_featured['Close'].pct_change().rolling(5).mean() * df_featured['VIX'].pct_change().rolling(5).mean()

        # 5. Seasonality and Calendar Effects
        if isinstance(df_featured.index, pd.DatetimeIndex):
            # Day of month effects
            df_featured['Is_Month_Start'] = (df_featured.index.day <= 5).astype(int)
            df_featured['Is_Month_End'] = (df_featured.index.day >= 25).astype(int)
            df_featured['Is_Quarter_End'] = ((df_featured.index.month % 3 == 0) & (df_featured.index.day >= 25)).astype(int)

            # Options expiration effects (3rd Friday)
            def is_third_friday(date):
                if date.weekday() != 4:  # Not Friday
                    return False
                return 15 <= date.day <= 21

            df_featured['Is_Options_Expiry'] = df_featured.index.map(is_third_friday).astype(int)

            # Holiday effects (simplified)
            df_featured['Days_To_Month_End'] = df_featured.index.to_series().apply(lambda x: (x + pd.offsets.MonthEnd(0) - x).days)

        # 6. Pattern Recognition Features
        if 'Close' in df_featured.columns:
            close_prices = df_featured['Close']

            # Local peaks and troughs
            for window in [5, 10, 20]:
                # Find peaks
                def find_local_extrema(prices, is_peak=True):
                    if len(prices) < 3:
                        return 0
                    mid_idx = len(prices) // 2
                    if is_peak:
                        return int(prices.iloc[mid_idx] == prices.max())
                    else:
                        return int(prices.iloc[mid_idx] == prices.min())

                df_featured[f'Is_Local_Peak_{window}'] = close_prices.rolling(window, center=True).apply(lambda x: find_local_extrema(x, True))
                df_featured[f'Is_Local_Trough_{window}'] = close_prices.rolling(window, center=True).apply(lambda x: find_local_extrema(x, False))

            # Support/Resistance levels
            for lookback in [20, 50]:
                df_featured[f'Distance_From_High_{lookback}'] = (close_prices - close_prices.rolling(lookback).max()) / close_prices
                df_featured[f'Distance_From_Low_{lookback}'] = (close_prices - close_prices.rolling(lookback).min()) / close_prices

        # 7. Advanced Volume Analysis
        if 'Volume' in df_featured.columns:
            volume = df_featured['Volume']

            # Volume profile
            for window in [5, 10, 20]:
                df_featured[f'Volume_Rank_{window}'] = volume.rolling(window).rank(pct=True)

                # Volume-price correlation
                if 'Close' in df_featured.columns:
                    df_featured[f'Volume_Price_Corr_{window}'] = volume.rolling(window).corr(df_featured['Close'])

                    # On-Balance Volume variations
                    price_change = df_featured['Close'].diff()
                    df_featured[f'OBV_Momentum_{window}'] = (volume * np.sign(price_change)).rolling(window).sum()

        # 8. Risk-Adjusted Returns
        if 'Close' in df_featured.columns:
            returns = df_featured['Close'].pct_change()

            for window in [10, 20, 50]:
                # Sharpe ratio (simplified, assuming 0 risk-free rate)
                df_featured[f'Sharpe_Ratio_{window}'] = returns.rolling(window).mean() / returns.rolling(window).std().replace(0, np.nan)

                # Sortino ratio (downside deviation)
                downside_returns = returns.where(returns < 0, 0)
                downside_std = downside_returns.rolling(window).std()
                df_featured[f'Sortino_Ratio_{window}'] = returns.rolling(window).mean() / downside_std.replace(0, np.nan)

                # Calmar ratio (return over max drawdown)
                rolling_max = df_featured['Close'].rolling(window).max()
                drawdown = (df_featured['Close'] - rolling_max) / rolling_max
                max_drawdown = drawdown.rolling(window).min()
                df_featured[f'Calmar_Ratio_{window}'] = returns.rolling(window).mean() / abs(max_drawdown).replace(0, np.nan)

        # 9. Liquidity Features
        if all(col in df_featured.columns for col in ['High', 'Low', 'Volume']):
            # Amihud illiquidity
            df_featured['Amihud_Illiquidity'] = abs(df_featured['Close'].pct_change()) / (df_featured['Volume'] * df_featured['Close']).replace(0, np.nan)

            # Roll's implicit spread estimator
            if 'Close' in df_featured.columns:
                price_changes = df_featured['Close'].diff()
                df_featured['Roll_Spread'] = 2 * np.sqrt(abs(price_changes.rolling(20).cov(price_changes.shift(1))))

        # 10. Machine Learning-based Features
        # Autoencoder-style features (using PCA as proxy)
        numeric_cols = df_featured.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 10:
            from sklearn.decomposition import PCA

            # Select recent important features for PCA
            important_features = ['Close', 'Volume', 'RSI_14', 'MACD', 'ATR_14', 'ADX_14']
            available_features = [f for f in important_features if f in df_featured.columns]

            if len(available_features) >= 3:
                # Prepare data for PCA
                pca_data = df_featured[available_features].dropna()
                if len(pca_data) > 50:
                    # Standardize
                    pca_data_std = (pca_data - pca_data.mean()) / pca_data.std().replace(0, 1) # Avoid division by zero for constant columns

                    # Apply PCA
                    pca = PCA(n_components=min(3, len(available_features)))
                    pca_transformed = pca.fit_transform(pca_data_std.fillna(0))

                    # Add PCA components as features
                    for i in range(pca_transformed.shape[1]):
                        pca_feature = pd.Series(index=pca_data.index, data=pca_transformed[:, i])
                        df_featured[f'PCA_Component_{i+1}'] = pca_feature.reindex(df_featured.index)

        # Clean up any infinite or extreme values
        df_featured = df_featured.replace([np.inf, -np.inf], np.nan)

        # Advanced NaN handling with forward fill, backward fill, then interpolation
        for col in df_featured.columns:
            if df_featured[col].dtype in ['float64', 'float32']:
                # Use different strategies for different types of features
                if 'entropy' in col.lower() or 'hurst' in col.lower():
                    # For complex calculations, use median
                    df_featured[col] = df_featured[col].fillna(df_featured[col].median())
                else:
                    # For others, use interpolation
                    df_featured[col] = df_featured[col].interpolate(method='linear', limit_direction='both')
                    df_featured[col] = df_featured[col].ffill().bfill().fillna(0)

        original_features = len(df.columns)
        total_features = len(df_featured.columns)
        new_features = total_features - original_features

        self._status_update(f"🚀 Đã tạo {new_features} features siêu nâng cao (tổng: {total_features} features)", status_callback)
        return df_featured

    def create_target_variable(self, df: pd.DataFrame, target_threshold: float = 0.02, status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        """Tạo biến mục tiêu cho dự đoán xu hướng 5 ngày"""
        self._status_update(f"Đang tạo biến mục tiêu với ngưỡng {target_threshold*100:.1f}%...", status_callback)

        df_target = df.copy()

        # Tính toán giá trị tương lai
        future_return = (df_target['Close'].shift(-self.forecast_horizon) / df_target['Close'] - 1)

        # Tạo biến phân loại: 1 nếu tăng trên threshold, 0 nếu không
        df_target['Future_Return'] = future_return
        df_target['Target'] = (future_return > target_threshold).astype(int)

        # Loại bỏ các hàng không có đủ dữ liệu tương lai
        df_target = df_target.dropna(subset=['Target'])

        target_distribution = df_target['Target'].value_counts()
        self._status_update(f"Phân phối target: Up(1): {target_distribution.get(1, 0)}, Down(0): {target_distribution.get(0, 0)}", status_callback)

        return df_target

    def prepare_features(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Chuẩn bị đặc trưng để huấn luyện"""
        self._status_update("Đang chuẩn bị đặc trưng...", status_callback)

        # Chọn các cột số để làm đặc trưng
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Loại bỏ các cột không phù hợp
        exclude_columns = ['Target', 'Future_Return', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] # Adj Close from yfinance, "Adj_Close" might be our internal one.
        feature_columns = [col for col in numeric_columns if col not in exclude_columns and col.lower() != 'adj_close'] # Case-insensitive check for adj_close

        # Loại bỏ các cột có quá nhiều NaN
        for col in feature_columns.copy():
            if df[col].isna().sum() / len(df) > 0.5:  # Nếu >50% là NaN
                feature_columns.remove(col)
                self._status_update(f"Loại bỏ {col} do quá nhiều NaN", status_callback)

        # Chuẩn bị DataFrame đặc trưng
        X = df[feature_columns].copy()

        # Xử lý NaN - sử dụng method mới thay vì deprecated
        X = X.ffill().bfill().fillna(0)

        # Loại bỏ inf values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        self._status_update(f"Đã chuẩn bị {len(feature_columns)} đặc trưng", status_callback)
        return X, feature_columns

    def advanced_feature_selection(self, X: pd.DataFrame, y: pd.Series, status_callback: Optional[Callable[[str, bool], None]] = None) -> List[str]:
        """Advanced feature selection với multiple methods để đạt accuracy > 80%"""
        self._status_update("Đang thực hiện Advanced Feature Selection...", status_callback)

        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
        from sklearn.ensemble import RandomForestClassifier

        # Initialize feature importance dictionaries
        feature_scores = {}
        for col in X.columns:
            feature_scores[col] = []

        # Method 1: Random Forest Feature Importance
        try:
            rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
            rf_selector.fit(X, y)

            rf_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_selector.feature_importances_
            }).sort_values('importance', ascending=False)

            # Normalize scores to 0-1
            max_score = rf_importance['importance'].max() if not rf_importance.empty else 1.0
            if max_score == 0: max_score = 1.0 # Avoid division by zero
            for idx, row in rf_importance.iterrows():
                feature_scores[row['feature']].append(row['importance'] / max_score)

            self._status_update("✅ Random Forest feature importance calculated", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ RF feature selection failed: {e}", status_callback)

        # Method 2: Mutual Information
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_df = pd.DataFrame({
                'feature': X.columns,
                'mi_score': mi_scores
            }).sort_values('mi_score', ascending=False)

            # Normalize scores to 0-1
            max_score = mi_df['mi_score'].max() if not mi_df.empty else 1.0
            if max_score > 0:
                for idx, row in mi_df.iterrows():
                    feature_scores[row['feature']].append(row['mi_score'] / max_score)
            else: # If all mi_scores are 0 or less
                for feature_name in mi_df['feature']:
                    feature_scores[feature_name].append(0.0)


            self._status_update("✅ Mutual Information scores calculated", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ MI feature selection failed: {e}", status_callback)

        # Method 3: F-statistics
        try:
            f_scores, _ = f_classif(X, y)
            f_df = pd.DataFrame({
                'feature': X.columns,
                'f_score': f_scores
            }).sort_values('f_score', ascending=False)

            # Normalize scores to 0-1
            max_score = f_df['f_score'].max() if not f_df.empty else 1.0
            if max_score > 0:
                for idx, row in f_df.iterrows():
                    feature_scores[row['feature']].append(row['f_score'] / max_score)
            else: # If all f_scores are 0 or less
                for feature_name in f_df['feature']:
                    feature_scores[feature_name].append(0.0)


            self._status_update("✅ F-statistics scores calculated", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ F-statistics failed: {e}", status_callback)

        # Method 4: XGBoost feature importance
        try:
            xgb_selector = xgb.XGBClassifier(
                max_depth=6,
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            xgb_selector.fit(X, y)

            xgb_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': xgb_selector.feature_importances_
            }).sort_values('importance', ascending=False)

            # Normalize scores to 0-1
            max_score = xgb_importance['importance'].max() if not xgb_importance.empty else 1.0
            if max_score == 0: max_score = 1.0 # Avoid division by zero
            for idx, row in xgb_importance.iterrows():
                feature_scores[row['feature']].append(row['importance'] / max_score)

            self._status_update("✅ XGBoost feature importance calculated", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ XGBoost feature selection failed: {e}", status_callback)

        # Combine all scores using ensemble approach
        final_scores = {}
        for feature, scores in feature_scores.items():
            if scores:  # If feature has any scores
                # Use mean of available scores
                final_scores[feature] = np.mean(scores)
            else:
                final_scores[feature] = 0.0

        # Sort features by combined score
        sorted_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top features using adaptive threshold
        total_features = len(sorted_features)

        # Dynamic feature selection based on score distribution
        scores_array = np.array([score for _, score in sorted_features if score > 0]) # Consider only positive scores for percentile

        if len(scores_array) > 0:
            # Use percentile-based selection
            score_threshold = np.percentile(scores_array, 70)  # Top 30% of positive scoring features

            selected_features = []
            for feature, score in sorted_features:
                if score >= score_threshold and score > 0 and len(selected_features) < 100:  # Max 100 features, must have positive score
                    selected_features.append(feature)

            # Ensure minimum number of features if threshold is too high or scores too low
            if len(selected_features) < 20:
                # Take top N features regardless of threshold if initial selection is too small
                selected_features = [feature for feature, score in sorted_features[:min(20, len(sorted_features))]]
        else:
            # Fallback: take top 50 features if no scores are positive or list is empty
            selected_features = [feature for feature, _ in sorted_features[:min(50, len(sorted_features))]]


        self._status_update(f"🎯 Đã chọn {len(selected_features)} features với ensemble selection", status_callback)

        # Display top 10 features for debugging
        top_10_features = [(feature, score) for feature, score in sorted_features[:10]]
        self._status_update(f"Top 10 features: {[f[0] for f in top_10_features]}", status_callback)

        return selected_features if selected_features else X.columns.tolist()[:20] # Fallback if selection is empty


    def advanced_data_preprocessing(self, X: pd.DataFrame, y: pd.Series, status_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Advanced data preprocessing để cải thiện model performance"""
        self._status_update("Đang thực hiện Advanced Data Preprocessing...", status_callback)

        from sklearn.preprocessing import StandardScaler, RobustScaler

        X_processed = X.copy()
        y_processed = y.copy()

        # 1. Remove features with very low variance
        try:
            from sklearn.feature_selection import VarianceThreshold
            variance_selector = VarianceThreshold(threshold=0.01)
            X_temp = variance_selector.fit_transform(X_processed)
            X_processed = pd.DataFrame(
                X_temp,
                columns=X_processed.columns[variance_selector.get_support()],
                index=X_processed.index
            )
            self._status_update(f"✅ Removed low variance features: {X.shape[1]} -> {X_processed.shape[1]}", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Variance filtering failed: {e}", status_callback)

        # 2. Remove highly correlated features
        try:
            correlation_matrix = X_processed.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )

            # Find features with correlation greater than 0.95
            high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

            if high_corr_features:
                X_processed = X_processed.drop(columns=high_corr_features)
                self._status_update(f"✅ Removed {len(high_corr_features)} highly correlated features", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Correlation filtering failed: {e}", status_callback)

        # 3. Outlier detection and treatment
        try:
            from sklearn.ensemble import IsolationForest

            # Detect outliers using Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42) # contamination can be tuned
            outlier_labels = iso_forest.fit_predict(X_processed)

            # Remove outliers - ensure X and y are aligned
            outlier_mask = outlier_labels == 1
            X_processed = X_processed[outlier_mask]
            y_processed = y_processed[outlier_mask] # Apply mask to y as well

            # Reset index to ensure alignment
            X_processed = X_processed.reset_index(drop=True)
            y_processed = y_processed.reset_index(drop=True)

            outliers_removed = np.sum(outlier_labels == -1)
            self._status_update(f"✅ Removed {outliers_removed} outliers. Remaining: {len(X_processed)} samples", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Outlier removal failed: {e}", status_callback)

        # 4. Handle class imbalance with SMOTE
        if IMBLEARN_AVAILABLE:
            try:
                class_distribution = y_processed.value_counts()
                self._status_update(f"Original class distribution: {dict(class_distribution)}", status_callback)

                # Only apply SMOTE if there's significant imbalance
                minority_class_ratio = min(class_distribution) / max(class_distribution) if max(class_distribution) > 0 else 1.0

                if minority_class_ratio < 0.7:  # If minority class is less than 70% of majority
                    # Use SMOTETomek which combines over-sampling and under-sampling
                    # Adjust k_neighbors for SMOTE part if minority class is too small
                    smote_k_neighbors = min(5, min(class_distribution)-1 if min(class_distribution) > 1 else 1)
                    if smote_k_neighbors < 1: # Should not happen if minority_class_ratio check is done correctly
                        self._status_update("Not enough samples in minority class for SMOTE. Skipping balancing.", status_callback)
                    else:
                        smote_tomek = SMOTETomek(random_state=42, smote=SMOTE(random_state=42, k_neighbors=smote_k_neighbors))

                        # Store original column names
                        original_columns = X_processed.columns.tolist()

                        X_resampled, y_resampled = smote_tomek.fit_resample(X_processed, y_processed)

                        # Convert back to DataFrame/Series with proper index
                        X_processed = pd.DataFrame(
                            X_resampled,
                            columns=original_columns,
                            index=range(len(X_resampled))
                        )
                        y_processed = pd.Series(y_resampled, name=y.name, index=range(len(y_resampled)))

                        new_class_distribution = y_processed.value_counts()
                        self._status_update(f"✅ Balanced classes: {dict(new_class_distribution)}. Total samples: {len(X_processed)}", status_callback)
                else:
                    self._status_update("Classes already balanced, skipping SMOTE", status_callback)

            except Exception as e:
                self._status_update(f"⚠️ Class balancing failed: {e}", status_callback)
        else:
            self._status_update("imblearn not available, skipping class balancing", status_callback)

        # 5. Advanced scaling with robust methods
        try:
            # Use RobustScaler which is less sensitive to outliers
            robust_scaler = RobustScaler()
            X_processed_scaled = robust_scaler.fit_transform(X_processed)
            X_processed = pd.DataFrame(
                X_processed_scaled,
                columns=X_processed.columns,
                index=X_processed.index  # Keep original index
            )
            self._status_update("✅ Applied robust scaling", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Robust scaling failed: {e}", status_callback)

        final_shape = X_processed.shape
        self._status_update(f"🎯 Preprocessing completed: {final_shape[0]} samples, {final_shape[1]} features", status_callback)

        # Final check: ensure X and y have same length and index alignment
        if len(X_processed) != len(y_processed):
            self._status_update(f"⚠️ Alignment issue detected. X: {len(X_processed)}, y: {len(y_processed)}. Fixing...", status_callback)
            min_length = min(len(X_processed), len(y_processed))
            X_processed = X_processed.iloc[:min_length].reset_index(drop=True)
            y_processed = y_processed.iloc[:min_length].reset_index(drop=True)
            self._status_update(f"✅ Fixed alignment. Final: {len(X_processed)} samples", status_callback)

        return X_processed, pd.Series(y_processed, name=y.name)


    def train_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                           status_callback: Optional[Callable[[str, bool], None]] = None) -> xgb.XGBClassifier:
        """Huấn luyện mô hình XGBoost được tối ưu hóa với Bayesian hyperparameter tuning"""
        self._status_update("Đang thực hiện Advanced Hyperparameter Tuning cho XGBoost...", status_callback)

        # Try to use Optuna for Bayesian optimization if available
        try:
            import optuna
            OPTUNA_AVAILABLE = True
            self._status_update("✅ Using Optuna for Bayesian hyperparameter optimization", status_callback)
        except ImportError:
            OPTUNA_AVAILABLE = False
            self._status_update("⚠️ Optuna not available, using GridSearchCV instead", status_callback)

        if OPTUNA_AVAILABLE:
            # Define objective function for Optuna
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 4, 15),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000), # Optuna suggests n_estimators
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0),
                    'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'tree_method': 'hist',
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0,
                    # MODIFICATION: Move early_stopping_rounds to constructor parameters
                    'early_stopping_rounds': 50
                }

                model = xgb.XGBClassifier(**params)
                
                # MODIFICATION: Remove early_stopping_rounds from fit()
                # It's now part of the constructor params for older XGBoost versions
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False # verbose=False in fit, verbosity=0 in constructor for less output
                )

                y_pred_proba_val = model.predict_proba(X_val)[:, 1]
                auc_score = roc_auc_score(y_val, y_pred_proba_val)
                
                return auc_score

            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=50, show_progress_bar=False, n_jobs=1) 

            best_params_from_optuna = study.best_params # These are suggestions for constructor
            # Remove 'early_stopping_rounds' if it was part of best_params_from_optuna suggestions,
            # as we'll handle it explicitly for the final model.
            # However, since we hardcoded it in `params` dict, it won't be in `study.best_params`.
            # So, `best_params_from_optuna` will contain parameters Optuna actually tuned.


            self._status_update(f"Best trial score (AUC): {study.best_value:.4f}", status_callback)
            self._status_update(f"Best parameters suggested by Optuna: {best_params_from_optuna}", status_callback)
            
            # Prepare parameters for the final model, including the fixed early_stopping_rounds
            final_model_params = {
                **best_params_from_optuna, # Use what Optuna tuned
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': -1,
                'early_stopping_rounds': 50 # Add fixed early_stopping_rounds for constructor
            }
            # If Optuna did not suggest 'n_estimators' (e.g. if it was fixed), ensure it's set for the final model
            if 'n_estimators' not in final_model_params:
                final_model_params['n_estimators'] = 1000 # A default reasonable max

            best_model = xgb.XGBClassifier(**final_model_params)

            self._status_update("Training final XGBoost model with optimized parameters and constructor early stopping...", status_callback)
            best_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)], # eval_set is crucial for early stopping
                verbose=False
            )

        else: # Fallback to GridSearchCV
            # (Code for GridSearchCV remains unchanged)
            # Ensure GridSearchCV also handles early_stopping_rounds appropriately if its XGBClassifier is used.
            # For GridSearchCV, you'd typically pass early_stopping_rounds to the fit_params
            # of GridSearchCV if the XGBoost version supports it in fit(), or to the XGBClassifier
            # constructor if the version is older.
            # Given the error, if GridSearchCV is used, early_stopping_rounds should also be
            # part of the XGBClassifier constructor params within the GridSearchCV setup.

            param_grid = {
                'max_depth': [6, 8, 10],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [300, 500, 800], # Max n_estimators
                # early_stopping_rounds would be fixed here if passed to constructor
            }

            xgb_base = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                tree_method='hist',
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                early_stopping_rounds=50 # Add here for constructor if using older XGBoost
            )

            tscv = TimeSeriesSplit(n_splits=5)
            grid_search = GridSearchCV(
                estimator=xgb_base,
                param_grid=param_grid,
                cv=tscv,
                scoring='roc_auc', # Using roc_auc for consistency
                n_jobs=-1, # Parallelize GridSearchCV itself
                verbose=0
            )
            self._status_update("Performing GridSearchCV (this may take a while)...", status_callback)
            # For GridSearchCV with early stopping, eval_set needs to be passed to fit
            # This is typically done via fit_params in GridSearchCV
            # However, if early_stopping_rounds is in constructor, it uses eval_set from fit.
            # This part can get tricky with GridSearchCV and older XGBoost versions.
            # For simplicity with current error, assuming early_stopping_rounds in constructor is the way.
            grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


            best_params_from_grid = grid_search.best_params_
            self._status_update(f"Best CV Score (AUC): {grid_search.best_score_:.4f}", status_callback)
            self._status_update(f"Best parameters from GridSearchCV: {best_params_from_grid}", status_callback)
            
            # best_model = grid_search.best_estimator_ # This already has early_stopping_rounds from xgb_base
            # Re-initialize with best params and fixed early_stopping_rounds for clarity and safety
            final_model_params_grid = {
                **best_params_from_grid,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': -1,
                'early_stopping_rounds': 50
            }
            best_model = xgb.XGBClassifier(**final_model_params_grid)
            self._status_update("Training final XGBoost model with GridSearchCV optimized parameters...", status_callback)
            best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


        # Common evaluation part for both Optuna and GridSearchCV paths
        val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred_proba)
        val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred)

        self._status_update(f"Final Model Validation - AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}, F1-Score: {val_f1:.4f}", status_callback)
        if hasattr(best_model, 'best_iteration') and best_model.best_iteration is not None: # XGBoost with early stopping in constructor stores it
            self._status_update(f"Best iteration from final model: {best_model.best_iteration}", status_callback)
        elif hasattr(best_model, 'best_ntree_limit') and best_model.best_ntree_limit is not None: # Older attribute
             self._status_update(f"Best ntree_limit from final model: {best_model.best_ntree_limit}", status_callback)

        self._status_update("XGBoost training completed with advanced optimization", status_callback)
        return best_model


    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                             status_callback: Optional[Callable[[str, bool], None]] = None):
        """Tạo ensemble model kết hợp nhiều algorithms để tăng accuracy"""
        self._status_update("Đang tạo ensemble model...", status_callback)

        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression

        models_to_ensemble = []

        # 1. XGBoost với parameters tối ưu
        try:
            xgb_model = self.train_xgboost_model(X_train, y_train, X_val, y_val, status_callback)
            models_to_ensemble.append(('xgb', xgb_model))
            self._status_update("✅ XGBoost model trained successfully", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ XGBoost failed: {e}", status_callback)

        # 2. Random Forest
        try:
            rf_model = RandomForestClassifier(
                n_estimators=200,  # Reduced from 300
                max_depth=12,  # Reduced from 15
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # Single thread
            )
            rf_model.fit(X_train, y_train)
            models_to_ensemble.append(('rf', rf_model))
            self._status_update("✅ Random Forest model trained successfully", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Random Forest failed: {e}", status_callback)

        # 3. Gradient Boosting
        try:
            gb_model = GradientBoostingClassifier(
                n_estimators=100,  # Reduced from 200
                learning_rate=0.1,  # Increased from 0.05
                max_depth=6,  # Reduced from 8
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            models_to_ensemble.append(('gb', gb_model))
            self._status_update("✅ Gradient Boosting model trained successfully", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Gradient Boosting failed: {e}", status_callback)

        # 4. Logistic Regression
        try:
            lr_model = LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=500,  # Reduced from 1000
                solver='liblinear'  # More stable solver
            )
            lr_model.fit(X_train, y_train)
            models_to_ensemble.append(('lr', lr_model))
            self._status_update("✅ Logistic Regression model trained successfully", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Logistic Regression failed: {e}", status_callback)

        # Check if we have at least one successful model
        if not models_to_ensemble:
            self._status_update("❌ All models failed. Creating simple fallback model.", status_callback, True)
            # Create a simple fallback model
            fallback_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=1
            )
            fallback_model.fit(X_train, y_train)
            return fallback_model

        # If only one model succeeded, return it directly
        if len(models_to_ensemble) == 1:
            self._status_update(f"Chỉ có 1 model thành công. Sử dụng {models_to_ensemble[0][0]} model.", status_callback)
            return models_to_ensemble[0][1]

        # Create Voting Classifier with successful models
        try:
            ensemble_model = VotingClassifier(
                estimators=models_to_ensemble,
                voting='soft',  # Use probability averaging
                n_jobs=1  # Single thread
            )

            ensemble_model.fit(X_train, y_train)

            # Evaluate individual models
            self._status_update("Đánh giá individual models:", status_callback)
            best_model = None
            best_score = 0

            for name, model in models_to_ensemble:
                try:
                    val_score = model.score(X_val, y_val)
                    self._status_update(f"  {name}: {val_score:.4f}", status_callback)
                    if val_score > best_score:
                        best_score = val_score
                        best_model = model
                except Exception as e:
                    self._status_update(f"  {name}: evaluation failed - {e}", status_callback)

            # Compare ensemble vs best individual
            try:
                ensemble_score = ensemble_model.score(X_val, y_val)
                self._status_update(f"Ensemble Score: {ensemble_score:.4f}", status_callback)

                if ensemble_score > best_score and best_model is not None : # Ensure best_model is not None
                    self._status_update(f"🏆 Sử dụng Ensemble Model (Score: {ensemble_score:.4f})", status_callback)
                    return ensemble_model
                elif best_model is not None:
                    self._status_update(f"🏆 Sử dụng Best Individual Model (Score: {best_score:.4f})", status_callback)
                    return best_model
                else: # Fallback if best_model is somehow None
                    self._status_update(f"Using Ensemble as fallback (Score: {ensemble_score:.4f})", status_callback)
                    return ensemble_model


            except Exception as e:
                self._status_update(f"Ensemble evaluation failed: {e}. Using best individual model.", status_callback)
                return best_model if best_model else models_to_ensemble[0][1]

        except Exception as e:
            self._status_update(f"Ensemble creation failed: {e}. Using best individual model.", status_callback)
            # Return the first successful model as fallback
            return models_to_ensemble[0][1]

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      status_callback: Optional[Callable[[str, bool], None]] = None) -> Dict[str, float]:
        """Đánh giá mô hình chi tiết - hỗ trợ cả XGBoost và Ensemble models"""
        self._status_update("Đang đánh giá mô hình...", status_callback)

        # Dự đoán
        y_pred = model.predict(X_test)

        # Handle ensemble models that might not have predict_proba for all estimators
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError: # predict_proba might not exist
             # Fallback for models without predict_proba by creating dummy probabilities
             y_pred_proba = np.zeros(len(y_test))
             if hasattr(model, 'decision_function'): # For SVM or similar
                 try:
                     decision_values = model.decision_function(X_test)
                     # Normalize decision values to [0,1] range (crude sigmoid-like)
                     y_pred_proba = 1 / (1 + np.exp(-decision_values))
                 except: pass # Keep as zeros if decision_function also fails
             else: # if no decision_function, use class predictions as pseudo-probabilities
                 y_pred_proba = y_pred.astype(float)

        # Tính toán các metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 and not np.all(y_pred_proba == y_pred_proba[0]) else 0.5
        }

        # In kết quả với highlight nếu đạt target
        accuracy_pct = metrics['accuracy'] * 100
        target_reached = "🎉 ĐÃ ĐẠT MỤC TIÊU!" if accuracy_pct >= 80 else "⚠️ Chưa đạt 80%"

        self._status_update(f"=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===", status_callback)
        self._status_update(f"  Accuracy: {metrics['accuracy']:.4f} ({accuracy_pct:.2f}%) - {target_reached}", status_callback)
        self._status_update(f"  Precision: {metrics['precision']:.4f}", status_callback)
        self._status_update(f"  Recall: {metrics['recall']:.4f}", status_callback)
        self._status_update(f"  F1-Score: {metrics['f1_score']:.4f}", status_callback)
        self._status_update(f"  ROC-AUC: {metrics['roc_auc']:.4f}", status_callback)

        return metrics

    def save_model(self, model, scaler: StandardScaler, feature_columns: List[str],
                  metrics: Dict[str, float], target_threshold: float,
                  status_callback: Optional[Callable[[str, bool], None]] = None) -> str:
        """Lưu mô hình và metadata - hỗ trợ cả XGBoost và Ensemble models"""
        self._status_update("Đang lưu mô hình...", status_callback)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        accuracy_pct = int(metrics['accuracy'] * 100)

        # Xác định loại model
        model_type = "Ensemble" if hasattr(model, 'estimators_') or hasattr(model, 'estimators') or hasattr(model, 'models_data') else "XGBoost"
        if isinstance(model, xgb.XGBClassifier): model_type = "XGBoost" # More specific check


        # Tên file
        model_filename = f"{model_type.lower()}_stock_predictor_{self.forecast_horizon}d_{accuracy_pct}pct_{timestamp}.joblib"
        scaler_filename = f"scaler_{self.forecast_horizon}d_{timestamp}.joblib"
        info_filename = f"model_info_{self.forecast_horizon}d_{timestamp}.json"

        model_path = os.path.join(MODEL_DIR, model_filename)
        scaler_path = os.path.join(MODEL_DIR, scaler_filename)
        info_path = os.path.join(MODEL_DIR, info_filename)

        # Lưu mô hình và scaler
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Lưu thông tin mô hình
        model_info = {
            'model_type': model_type,
            'forecast_horizon_days': self.forecast_horizon,
            'target_threshold': target_threshold,
            'feature_columns': feature_columns,
            'metrics': metrics,
            'model_filename': model_filename,
            'scaler_filename': scaler_filename,
            'training_timestamp': timestamp,
            'feature_count': len(feature_columns),
            'accuracy_target_reached': metrics['accuracy'] >= 0.8
        }

        # Thêm thông tin model parameters nếu có
        if hasattr(model, 'get_params'):
            try:
                # Handle non-serializable parts of ensemble parameters
                params = model.get_params(deep=False) # Get only top-level params
                serializable_params = {}
                for k, v in params.items():
                    if isinstance(v, (list, tuple)) and any(not isinstance(item, (str, int, float, bool, type(None))) for item in v):
                        serializable_params[k] = [str(type(item)) for item in v] # Store types or representative strings
                    elif not isinstance(v, (str, int, float, bool, type(None), dict)): # dicts should be fine
                        serializable_params[k] = str(type(v))
                    else:
                        serializable_params[k] = v
                model_info['model_params'] = serializable_params
            except Exception as param_e:
                 model_info['model_params'] = f"Parameters not fully serializable: {param_e}"
        elif model_type == "Ensemble" and hasattr(model, 'estimators_'): # For VotingClassifier
             estimators_info = []
             for name, est_model in model.estimators_:
                 est_params = est_model.get_params(deep=False) if hasattr(est_model, 'get_params') else {"type": str(type(est_model))}
                 estimators_info.append({"name": name, "type": str(type(est_model)), "params": est_params})
             model_info['model_params'] = {"estimators": estimators_info}
        elif model_type == "Ensemble" and hasattr(model, 'estimators'): # For StackingClassifier
             estimators_info = []
             if hasattr(model, 'estimators'): # StackingClassifier uses 'estimators' attribute
                 for name, est_model in model.estimators:
                     est_params = est_model.get_params(deep=False) if hasattr(est_model, 'get_params') else {"type": str(type(est_model))}
                     estimators_info.append({"name": name, "type": str(type(est_model)), "params": est_params})
             if hasattr(model, 'final_estimator_') and model.final_estimator_ is not None:
                 final_est_params = model.final_estimator_.get_params(deep=False) if hasattr(model.final_estimator_, 'get_params') else {"type": str(type(model.final_estimator_))}
                 estimators_info.append({"name": "final_estimator", "type": str(type(model.final_estimator_)), "params": final_est_params})

             model_info['model_params'] = {"estimators": estimators_info}


        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4, default=str) # Use default=str for non-serializable

        self._status_update(f"Đã lưu {model_type} model: {model_filename}", status_callback)
        return model_path

    def create_advanced_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                                         status_callback: Optional[Callable[[str, bool], None]] = None):
        """Tạo advanced stacking ensemble để đạt accuracy > 80%"""
        self._status_update("Đang tạo Advanced Stacking Ensemble...", status_callback)

        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import StackingClassifier
        from sklearn.model_selection import StratifiedKFold

        # Base models with optimized parameters
        base_models = []

        # 1. XGBoost with best params
        try:
            xgb_model = xgb.XGBClassifier(
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                learning_rate=0.05,
                n_estimators=800,
                reg_alpha=0.1,
                reg_lambda=1.5,
                objective='binary:logistic',
                eval_metric='logloss', # Changed from auc for stacking stability
                random_state=42,
                verbosity=0,
                n_jobs=1 # Ensure single thread for stability within ensemble
            )
            base_models.append(('xgb', xgb_model))
            self._status_update("✅ XGBoost model configured", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ XGBoost failed: {e}", status_callback)

        # 2. Enhanced Random Forest
        try:
            rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=1,
                class_weight='balanced'
            )
            base_models.append(('rf', rf_model))
            self._status_update("✅ Enhanced Random Forest configured", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Random Forest failed: {e}", status_callback)

        # 3. Extra Trees (Extremely Randomized Trees)
        try:
            et_model = ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=1,
                class_weight='balanced'
            )
            base_models.append(('et', et_model))
            self._status_update("✅ Extra Trees configured", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Extra Trees failed: {e}", status_callback)

        # 4. Gradient Boosting with optimal params
        try:
            gb_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=42
            )
            base_models.append(('gb', gb_model))
            self._status_update("✅ Gradient Boosting configured", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Gradient Boosting failed: {e}", status_callback)

        # 5. AdaBoost
        try:
            ada_model = AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.8,
                random_state=42
            )
            base_models.append(('ada', ada_model))
            self._status_update("✅ AdaBoost configured", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ AdaBoost failed: {e}", status_callback)

        # 6. Support Vector Machine
        try:
            svm_model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            base_models.append(('svm', svm_model))
            self._status_update("✅ SVM configured", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ SVM failed: {e}", status_callback)

        # 7. Naive Bayes
        try:
            nb_model = GaussianNB()
            base_models.append(('nb', nb_model))
            self._status_update("✅ Naive Bayes configured", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Naive Bayes failed: {e}", status_callback)

        # Check if we have enough models
        if len(base_models) < 3:
            self._status_update("❌ Không đủ base models. Fallback to simple ensemble.", status_callback, True)
            return self.create_ensemble_model(X_train, y_train, X_val, y_val, status_callback)

        # Meta-learner: Advanced logistic regression with regularization
        meta_learner = LogisticRegression(
            C=0.1,
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )

        # Create Stacking Classifier with cross-validation
        try:
            stacking_classifier = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                stack_method='predict_proba',
                n_jobs=1, # Ensure single thread for stability
                verbose=0
            )

            self._status_update(f"Đang train Stacking Ensemble với {len(base_models)} base models...", status_callback)
            stacking_classifier.fit(X_train, y_train)

            # Evaluate stacking model
            stacking_score = stacking_classifier.score(X_val, y_val)
            self._status_update(f"🚀 Stacking Ensemble Score: {stacking_score:.4f}", status_callback)

            # Compare with individual base models
            best_individual_score = 0
            best_individual_model = None

            for name, model_instance in base_models: # Renamed model to model_instance to avoid conflict with outer scope
                try:
                    # Models are already configured, just need to fit if not done by StackingClassifier internally for comparison
                    # However, StackingClassifier does fit them, so we can just score them if they were part of the stack.
                    # If we want to score them independently, they need to be fit on X_train, y_train
                    temp_model_to_score = model_instance # Use the instance from base_models
                    if not hasattr(temp_model_to_score, "classes_"): # Check if it's fitted
                        temp_model_to_score.fit(X_train, y_train) # Fit if not already
                    score = temp_model_to_score.score(X_val, y_val)
                    self._status_update(f"  {name}: {score:.4f}", status_callback)

                    if score > best_individual_score:
                        best_individual_score = score
                        best_individual_model = temp_model_to_score
                except Exception as e:
                    self._status_update(f"  {name}: training/evaluation failed - {e}", status_callback)


            # Return the best performing model
            if stacking_score > best_individual_score and best_individual_model is not None:
                self._status_update(f"🏆 Sử dụng Stacking Ensemble (Score: {stacking_score:.4f})", status_callback)
                return stacking_classifier
            elif best_individual_model is not None:
                self._status_update(f"🏆 Sử dụng Best Individual Model (Score: {best_individual_score:.4f})", status_callback)
                return best_individual_model
            else: # Fallback if no individual model scored better and best_individual_model is None
                 self._status_update(f"Stacking or individual models failed to improve. Returning Stacking Ensemble as default (Score: {stacking_score:.4f})", status_callback)
                 return stacking_classifier


        except Exception as e:
            self._status_update(f"Stacking failed: {e}. Using best individual model.", status_callback)

            # Fallback: train and return best individual model
            best_score = 0
            best_model_fallback = None # Renamed to avoid conflict

            for name, model_instance_fallback in base_models: # Renamed model to model_instance_fallback
                try:
                    if not hasattr(model_instance_fallback, "classes_"): # Fit if not already
                        model_instance_fallback.fit(X_train, y_train)
                    score = model_instance_fallback.score(X_val, y_val)
                    if score > best_score:
                        best_score = score
                        best_model_fallback = model_instance_fallback
                except:
                    continue

            return best_model_fallback if best_model_fallback else (base_models[0][1] if base_models else None)


    def create_meta_learning_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                                      status_callback: Optional[Callable[[str, bool], None]] = None):
        """Tạo Meta-Learning Ensemble với nhiều strategies để đạt accuracy > 80%"""
        self._status_update("Đang tạo Meta-Learning Ensemble...", status_callback)

        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import StratifiedKFold

        # Meta-learning: Train multiple diverse models with different philosophies
        meta_models = {}

        # 1. Ensemble focused on different aspects
        models_config = {
            'momentum_focused': {
                'model': RandomForestClassifier(
                    n_estimators=500,
                    max_depth=20,
                    min_samples_split=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=1
                ),
                'feature_filter': lambda df_cols: [col for col in df_cols if 'momentum' in col.lower() or 'trend' in col.lower()]
            },
            'volatility_focused': {
                'model': GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=8,
                    random_state=42
                ),
                'feature_filter': lambda df_cols: [col for col in df_cols if 'vol' in col.lower() or 'std' in col.lower()]
            },
            'volume_focused': {
                'model': ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=15,
                    random_state=42,
                    n_jobs=1
                ),
                'feature_filter': lambda df_cols: [col for col in df_cols if 'volume' in col.lower() or 'obv' in col.lower()]
            },
            'deep_learning_focused': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(100, 50, 25),
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42
                ),
                'feature_filter': lambda df_cols: [col for col in df_cols if any(keyword in col.lower() for keyword in ['lstm', 'cnn', 'attention', 'residual', 'batch'])]
            },
            'technical_focused': {
                'model': xgb.XGBClassifier(
                    max_depth=10,
                    n_estimators=800,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    random_state=42,
                    verbosity=0,
                    n_jobs=1
                ),
                'feature_filter': lambda df_cols: [col for col in df_cols if any(keyword in col.lower() for keyword in ['rsi', 'macd', 'bb', 'sma', 'ema'])]
            }
        }

        trained_models = []

        # Train specialized models
        for model_name, config in models_config.items():
            try:
                self._status_update(f"Đang train {model_name} model...", status_callback)

                # Filter features based on specialization
                relevant_features = config['feature_filter'](X_train.columns)
                if not relevant_features:
                    self._status_update(f"⚠️ Không tìm thấy features cho {model_name}", status_callback)
                    continue

                # Take top features if too many
                if len(relevant_features) > 50:
                    relevant_features = relevant_features[:50]

                # Train on relevant features
                X_train_filtered = X_train[relevant_features]
                X_val_filtered = X_val[relevant_features]

                model = config['model']
                model.fit(X_train_filtered, y_train)

                # Evaluate
                val_score = model.score(X_val_filtered, y_val)
                self._status_update(f"  {model_name}: {val_score:.4f} ({len(relevant_features)} features)", status_callback)

                trained_models.append({
                    'name': model_name,
                    'model': model,
                    'features': relevant_features,
                    'score': val_score
                })

            except Exception as e:
                self._status_update(f"⚠️ {model_name} failed: {e}", status_callback)
                continue

        if not trained_models:
            self._status_update("❌ No meta-models trained successfully", status_callback, True)
            return None

        # Create meta-ensemble with weighted voting
        class WeightedMetaEnsemble:
            def __init__(self, models_data):
                self.models_data = models_data
                # Calculate weights based on validation scores
                scores = [m['score'] for m in models_data if m['score'] is not None and not np.isnan(m['score'])] # Filter out None/NaN scores
                if not scores: # If all scores are None/NaN
                    self.weights = [1/len(models_data)] * len(models_data) if models_data else []
                else:
                    total_score = sum(scores)
                    self.weights = [ (m['score'] / total_score if m['score'] is not None and not np.isnan(m['score']) and total_score > 0 else (1/len(scores) if scores else 0) ) for m in models_data]


            def predict(self, X):
                if not self.models_data or not self.weights or len(self.models_data) != len(self.weights):
                    return np.array([]) # Or raise error

                predictions_weighted_sum = np.zeros(len(X))
                total_weight_applied = 0

                for i, model_data in enumerate(self.models_data):
                    if i >= len(self.weights): continue # Safety break

                    model = model_data['model']
                    features = model_data['features']
                    weight = self.weights[i]

                    if not features: continue # Skip if no features for this model

                    X_filtered = X[features]
                    pred = model.predict(X_filtered)
                    predictions_weighted_sum += pred * weight
                    total_weight_applied += weight

                # Weighted majority voting
                if total_weight_applied == 0: return (np.zeros(len(X))).astype(int) # Avoid division by zero
                return (predictions_weighted_sum / total_weight_applied > 0.5).astype(int)


            def predict_proba(self, X):
                if not self.models_data or not self.weights or len(self.models_data) != len(self.weights):
                     # Return a 2D array with a default probability like 0.5 for both classes
                     return np.full((len(X), 2), 0.5) if X is not None and len(X) > 0 else np.array([])


                probabilities_weighted_sum = np.zeros((len(X), 2))
                total_weight_applied = 0

                for i, model_data in enumerate(self.models_data):
                    if i >= len(self.weights): continue

                    model = model_data['model']
                    features = model_data['features']
                    weight = self.weights[i]

                    if not features: continue

                    X_filtered = X[features]

                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_filtered)
                    else: # Fallback if no predict_proba
                        pred_class = model.predict(X_filtered)
                        proba = np.zeros((len(pred_class), 2))
                        proba[np.arange(len(pred_class)), pred_class.astype(int)] = 1.0


                    probabilities_weighted_sum += proba * weight
                    total_weight_applied += weight

                # Weighted average of probabilities
                if total_weight_applied == 0: return np.full((len(X), 2), 0.5) # Avoid division by zero
                return probabilities_weighted_sum / total_weight_applied


            def score(self, X, y):
                predictions = self.predict(X)
                if len(predictions) == 0: return 0.0 # Handle empty predictions
                return np.mean(predictions == y)

        # Create the ensemble
        meta_ensemble = WeightedMetaEnsemble(trained_models)

        # Evaluate meta-ensemble
        ensemble_score = meta_ensemble.score(X_val, y_val)
        self._status_update(f"🚀 Meta-Learning Ensemble Score: {ensemble_score:.4f}", status_callback)

        return meta_ensemble

    def adaptive_ensemble_selection(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                                   status_callback: Optional[Callable[[str, bool], None]] = None):
        """Adaptive Ensemble Selection - chọn ensemble tốt nhất dựa trên validation performance"""
        self._status_update("Đang thực hiện Adaptive Ensemble Selection...", status_callback)

        ensemble_candidates = []

        # 1. Stacking Ensemble
        try:
            stacking_model = self.create_advanced_stacking_ensemble(X_train, y_train, X_val, y_val, status_callback)
            if stacking_model:
                stacking_score = stacking_model.score(X_val, y_val)
                ensemble_candidates.append({
                    'name': 'Stacking Ensemble',
                    'model': stacking_model,
                    'score': stacking_score
                })
                self._status_update(f"✅ Stacking Ensemble: {stacking_score:.4f}", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Stacking Ensemble failed: {e}", status_callback)

        # 2. Meta-Learning Ensemble
        try:
            meta_model = self.create_meta_learning_ensemble(X_train, y_train, X_val, y_val, status_callback)
            if meta_model:
                meta_score = meta_model.score(X_val, y_val)
                ensemble_candidates.append({
                    'name': 'Meta-Learning Ensemble',
                    'model': meta_model,
                    'score': meta_score
                })
                self._status_update(f"✅ Meta-Learning Ensemble: {meta_score:.4f}", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Meta-Learning Ensemble failed: {e}", status_callback)

        # 3. Simple Voting Ensemble (fallback)
        try:
            voting_model = self.create_ensemble_model(X_train, y_train, X_val, y_val, status_callback)
            if voting_model:
                voting_score = voting_model.score(X_val, y_val)
                ensemble_candidates.append({
                    'name': 'Voting Ensemble',
                    'model': voting_model,
                    'score': voting_score
                })
                self._status_update(f"✅ Voting Ensemble: {voting_score:.4f}", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Voting Ensemble failed: {e}", status_callback)

        # 4. Single XGBoost with optimal params
        try:
            xgb_model = self.train_xgboost_model(X_train, y_train, X_val, y_val, status_callback)
            if xgb_model:
                xgb_score = xgb_model.score(X_val, y_val)
                ensemble_candidates.append({
                    'name': 'Optimized XGBoost',
                    'model': xgb_model,
                    'score': xgb_score
                })
                self._status_update(f"✅ Optimized XGBoost: {xgb_score:.4f}", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Optimized XGBoost failed: {e}", status_callback)

        # Select best ensemble
        if not ensemble_candidates:
            self._status_update("❌ No ensemble candidates available. Training a default XGBoost model.", status_callback, True)
            # Fallback to a default XGBoost if all ensembles fail
            default_xgb = xgb.XGBClassifier(random_state=42, n_jobs=1, verbosity=0)
            default_xgb.fit(X_train, y_train)
            self._status_update("Trained default XGBoost as fallback.", status_callback)
            return default_xgb


        best_ensemble = max(ensemble_candidates, key=lambda x: x['score'])
        best_score = best_ensemble['score']
        best_name = best_ensemble['name']
        best_model = best_ensemble['model']

        self._status_update(f"🏆 Best Ensemble: {best_name} (Score: {best_score:.4f})", status_callback)

        # Show comparison
        self._status_update("📊 Ensemble Comparison:", status_callback)
        for candidate in sorted(ensemble_candidates, key=lambda x: x['score'], reverse=True):
            emoji = "🥇" if candidate == best_ensemble else "🥈" if candidate['score'] > 0.75 else "🥉"
            self._status_update(f"  {emoji} {candidate['name']}: {candidate['score']:.4f}", status_callback)

        return best_model

    def smart_class_balancing(self, X: pd.DataFrame, y: pd.Series, strategy: str = 'adaptive',
                             status_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Smart Class Balancing với adaptive strategies"""
        self._status_update("Đang thực hiện Smart Class Balancing...", status_callback)

        if not IMBLEARN_AVAILABLE:
            self._status_update("⚠️ imblearn not available, skipping class balancing", status_callback)
            return X, y

        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
        from imblearn.combine import SMOTETomek, SMOTEENN

        # Analyze class distribution
        class_counts = y.value_counts()
        total_samples = len(y)
        minority_ratio = min(class_counts) / max(class_counts) if max(class_counts) > 0 else 1.0


        self._status_update(f"Original distribution: {dict(class_counts)} (Ratio: {minority_ratio:.3f})", status_callback)

        # Choose strategy based on data characteristics
        if strategy == 'adaptive':
            if minority_ratio > 0.8:
                strategy = 'none'  # Already balanced
            elif minority_ratio > 0.5:
                strategy = 'tomek'  # Light balancing
            elif minority_ratio > 0.3:
                strategy = 'smote_tomek'  # Moderate balancing
            elif total_samples > 10000:
                strategy = 'adasyn'  # Adaptive for large datasets
            else:
                strategy = 'borderline_smote'  # Focused balancing for small datasets

        self._status_update(f"Selected balancing strategy: {strategy}", status_callback)

        try:
            if strategy == 'none':
                return X, y

            # Store original column names and index
            original_columns = X.columns.tolist()

            # Adjust k_neighbors for SMOTE based on minority class size
            k_neighbors_val = min(5, min(class_counts) - 1 if min(class_counts) > 1 else 1)
            if k_neighbors_val < 1: # Should ideally not happen if class_counts is not empty
                 self._status_update("Not enough samples in minority class for k-neighbors based balancing. Skipping.", status_callback)
                 return X, y

            # Apply selected strategy
            if strategy == 'smote':
                balancer = SMOTE(random_state=42, k_neighbors=k_neighbors_val)
            elif strategy == 'adasyn':
                balancer = ADASYN(random_state=42, n_neighbors=k_neighbors_val)
            elif strategy == 'borderline_smote':
                balancer = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors_val)
            elif strategy == 'smote_tomek':
                balancer = SMOTETomek(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k_neighbors_val))
            elif strategy == 'smote_enn':
                balancer = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k_neighbors_val))
            elif strategy == 'tomek':
                balancer = TomekLinks()
            else:
                self._status_update(f"⚠️ Unknown strategy: {strategy}, using SMOTE", status_callback)
                balancer = SMOTE(random_state=42, k_neighbors=k_neighbors_val)

            # Apply balancing
            X_balanced, y_balanced = balancer.fit_resample(X, y)

            # Convert back to DataFrame/Series with proper structure
            X_balanced = pd.DataFrame(
                X_balanced,
                columns=original_columns,
                index=range(len(X_balanced)) # Reset index for balanced data
            )
            y_balanced = pd.Series(y_balanced, name=y.name, index=range(len(y_balanced))) # Reset index

            # Report results
            new_class_counts = y_balanced.value_counts()
            new_ratio = min(new_class_counts) / max(new_class_counts) if max(new_class_counts) > 0 else 1.0

            self._status_update(f"✅ Balanced distribution: {dict(new_class_counts)} (Ratio: {new_ratio:.3f})", status_callback)
            self._status_update(f"Sample count: {len(X)} → {len(X_balanced)}", status_callback)

            return X_balanced, y_balanced

        except Exception as e:
            self._status_update(f"⚠️ Class balancing failed: {e}. Using original data.", status_callback)
            return X, y

    def run_full_training_pipeline(self, processed_files: List[str], target_threshold: float = 0.02,
                                 test_size: float = 0.2, status_callback: Optional[Callable[[str, bool], None]] = None,
                                 progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[str, Dict[str, float], List[str]]:
        """Chạy toàn bộ quy trình huấn luyện với optimizations để đạt >80% accuracy"""
        try:
            # 1. Tải dữ liệu
            self._progress_update(0.05, "Tải và kiểm tra dữ liệu...", progress_callback)
            df = self.load_and_combine_data(processed_files, status_callback)

            # Data quality checks
            initial_rows = len(df)
            self._status_update(f"Dữ liệu ban đầu: {initial_rows:,} rows", status_callback)

            # 2. Tạo đặc trưng nâng cao
            self._progress_update(0.15, "Tạo đặc trưng nâng cao (ultra advanced features)...", progress_callback)
            df = self.engineer_ultra_advanced_features(df, status_callback)

            # 2.5. Tối ưu target threshold để đạt accuracy tốt nhất
            self._progress_update(0.20, "Tối ưu target threshold...", progress_callback)
            optimized_threshold = self.optimize_target_threshold(df, status_callback)
            if optimized_threshold != target_threshold:
                self._status_update(f"Threshold được tối ưu từ {target_threshold} thành {optimized_threshold}", status_callback)
                target_threshold = optimized_threshold

            # 3. Tạo biến mục tiêu
            self._progress_update(0.25, "Tạo biến mục tiêu và kiểm tra class balance...", progress_callback)
            df = self.create_target_variable(df, target_threshold, status_callback)

            # Check class balance
            target_distribution = df['Target'].value_counts(normalize=True)
            self._status_update(f"Class balance: Up={target_distribution.get(1, 0):.2%}, Down={target_distribution.get(0, 0):.2%}", status_callback)

            # 4. Chuẩn bị đặc trưng với quality control
            self._progress_update(0.35, "Chuẩn bị và lọc đặc trưng chất lượng cao...", progress_callback)
            X, feature_columns_initial = self.prepare_features(df, status_callback) # Renamed to avoid conflict
            y = df['Target']

            # Remove rows with too many NaN values
            row_nan_threshold = 0.3  # Max 30% NaN per row
            good_rows_mask = (X.isna().sum(axis=1) / X.shape[1]) <= row_nan_threshold if X.shape[1] > 0 else pd.Series(True, index=X.index)
            X = X[good_rows_mask]
            y = y[good_rows_mask] # Apply same mask to y

            self._status_update(f"Sau data cleaning: {len(X):,} rows, {X.shape[1]} features", status_callback)

            # 4.5. Advanced Data Preprocessing để cải thiện accuracy
            self._progress_update(0.40, "Advanced data preprocessing (outliers, variance, correlation)...", progress_callback)
            X_processed, y_processed = self.advanced_data_preprocessing(X, y, status_callback)

            # 4.7. Smart Class Balancing
            self._progress_update(0.42, "Smart Class Balancing...", progress_callback)
            X_processed, y_processed = self.smart_class_balancing(X_processed, y_processed, 'adaptive', status_callback)

            # 5. Chọn đặc trưng quan trọng với multiple methods
            self._progress_update(0.45, "Advanced feature selection...", progress_callback)
            selected_features = self.advanced_feature_selection(X_processed, y_processed, status_callback)
            X_selected = X_processed[selected_features]

            # Verify alignment (should be aligned after advanced preprocessing)
            if len(X_selected) != len(y_processed):
                self._status_update(f"⚠️ Unexpected alignment issue after feature selection. X: {len(X_selected)}, y: {len(y_processed)}", status_callback, True)
                min_length = min(len(X_selected), len(y_processed))
                X_selected = X_selected.iloc[:min_length].reset_index(drop=True)
                y_processed = y_processed.iloc[:min_length].reset_index(drop=True)
                self._status_update(f"✅ Fixed alignment. Using {min_length} samples", status_callback)

            self._status_update(f"After preprocessing and feature selection: {len(X_selected):,} samples, {len(selected_features)} features", status_callback)

            # 6. Chia dữ liệu với stratification
            self._progress_update(0.55, "Chia dữ liệu với stratification...", progress_callback)

            # Ensure minimum samples per class for stratification
            min_class_size = min(y_processed.value_counts()) if not y_processed.empty else 0
            if min_class_size < 2 : # Stratify needs at least 2 samples per class for split
                # If not enough, proceed without stratification or raise specific error
                if len(y_processed) < 10: # Arbitrary small number
                    raise ValueError(f"Dataset too small or class imbalance too extreme after processing ({len(y_processed)} samples, min_class_size: {min_class_size}). Cannot proceed with training.")
                self._status_update(f"Warning: Low samples in minority class ({min_class_size}). Proceeding without stratification for train/test split.", status_callback, True)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y_processed, test_size=test_size, random_state=42
                )
            else:
                 X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y_processed, test_size=test_size, random_state=42, stratify=y_processed
                )


            # Chia thêm validation set từ training set, also consider stratification
            min_class_size_train = min(y_train.value_counts()) if not y_train.empty else 0
            if min_class_size_train < 2:
                 if len(y_train) < 10:
                     raise ValueError(f"Training set too small or class imbalance too extreme ({len(y_train)} samples, min_class_size_train: {min_class_size_train}). Cannot proceed.")
                 self._status_update(f"Warning: Low samples in minority class in training set ({min_class_size_train}). Proceeding without stratification for train/validation split.", status_callback, True)
                 X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42 # Test size for validation split
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train # Test size for validation split
                )


            self._status_update(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}", status_callback)

            # 7. Chuẩn hóa dữ liệu với robust scaling
            self._progress_update(0.65, "Chuẩn hóa dữ liệu...", progress_callback)
            scaler = StandardScaler() # Using StandardScaler as it's common for XGBoost
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # Chuyển về DataFrame với proper column names
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=selected_features, index=X_val.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)

            # 8. Huấn luyện ensemble model với Adaptive Selection
            self._progress_update(0.75, "Adaptive Ensemble Selection để tìm model tốt nhất...", progress_callback)
            model = self.adaptive_ensemble_selection(X_train_scaled, y_train, X_val_scaled, y_val, status_callback)

            # Fallback nếu adaptive selection fails
            if model is None:
                self._status_update("⚠️ Adaptive selection failed, using fallback XGBoost", status_callback)
                model = self.train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val, status_callback)

            # 9. Đánh giá mô hình chi tiết
            self._progress_update(0.90, "Đánh giá mô hình trên test set...", progress_callback)
            metrics = self.evaluate_model(model, X_test_scaled, y_test, status_callback)

            # 10. Lưu mô hình
            self._progress_update(0.95, "Lưu mô hình và metadata...", progress_callback)
            model_path = self.save_model(model, scaler, selected_features, metrics, target_threshold, status_callback)

            # Lưu thông tin cho class
            self.model = model
            self.scaler = scaler
            self.feature_columns = selected_features
            self.target_column = 'Target'

            self._progress_update(1.0, "🎉 Hoàn thành pipeline!", progress_callback)

            # Final summary
            accuracy_pct = metrics['accuracy'] * 100
            if accuracy_pct >= 80:
                self._status_update(f"🏆 SUCCESS! Đã đạt mục tiêu accuracy: {accuracy_pct:.2f}% >= 80%", status_callback)
            else:
                self._status_update(f"📈 Accuracy: {accuracy_pct:.2f}% - Gần đạt mục tiêu 80%", status_callback)
                self._status_update("💡 Tip: Thử thêm dữ liệu hoặc điều chỉnh target_threshold", status_callback)

            return model_path, metrics, selected_features # Return selected_features

        except Exception as e:
            self._status_update(f"❌ Lỗi trong quá trình huấn luyện: {e}", status_callback, True)
            # traceback.print_exc() # Print full traceback for debugging
            raise e

    def optimize_target_threshold(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> float:
        """Tối ưu threshold để đạt accuracy tốt nhất"""
        self._status_update("Đang tối ưu target threshold...", status_callback)

        if 'Close' not in df.columns:
            return 0.02  # Default threshold

        thresholds_to_test = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
        best_threshold = 0.02
        best_balance_score = -1 # Initialize to a value that will be beaten

        for threshold in thresholds_to_test:
            try:
                # Calculate future returns
                future_return = (df['Close'].shift(-self.forecast_horizon) / df['Close'] - 1)
                target = (future_return > threshold).astype(int)

                # Remove NaN values
                valid_mask = future_return.notna()
                target_clean = target[valid_mask]

                if len(target_clean) == 0:
                    continue

                # Calculate class distribution
                class_counts = target_clean.value_counts()
                if len(class_counts) < 2: # Need both classes to calculate balance
                    continue

                # Calculate balance score (higher is better, max when 50-50 split)
                class_0_ratio = class_counts.get(0, 0) / len(target_clean)
                class_1_ratio = class_counts.get(1, 0) / len(target_clean)

                # Penalize extreme imbalances, reward balanced classes
                balance_score = 1 - abs(class_0_ratio - class_1_ratio)

                # Also consider minimum class size
                min_class_size = min(class_counts.get(0,0), class_counts.get(1,0))
                if min_class_size < 50:  # Penalize if too few samples in minority class
                    balance_score *= (min_class_size / 50)

                self._status_update(f"Threshold {threshold}: Class 0: {class_0_ratio:.2%}, Class 1: {class_1_ratio:.2%}, Score: {balance_score:.3f}", status_callback)

                if balance_score > best_balance_score:
                    best_balance_score = balance_score
                    best_threshold = threshold

            except Exception as e:
                self._status_update(f"Error testing threshold {threshold}: {e}", status_callback)
                continue

        self._status_update(f"🎯 Optimal threshold: {best_threshold} (Balance Score: {best_balance_score:.3f})", status_callback)
        return best_threshold

    def engineer_deep_learning_inspired_features(self, df: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        """Tạo features được lấy cảm hứng từ deep learning để đạt accuracy > 80%"""
        self._status_update("Đang tạo Deep Learning Inspired Features...", status_callback)

        df_deep = df.copy()

        # 1. Attention-like Features (Multi-head attention simulation)
        if 'Close' in df_deep.columns:
            try:
                close_prices = df_deep['Close']

                # Multi-scale attention weights
                for window in [5, 10, 20, 50]:
                    # Attention weights based on volatility
                    returns = close_prices.pct_change()
                    volatility = returns.rolling(window).std().fillna(0) # Fill NaN for robust exp

                    # Softmax-like normalization
                    exp_volatility = np.exp(volatility)
                    sum_exp_volatility_rolling = exp_volatility.rolling(window).sum().replace(0, 1e-9) # Avoid div by zero
                    attention_weights = exp_volatility / sum_exp_volatility_rolling


                    # Weighted price features
                    df_deep[f'Attention_Price_{window}'] = (close_prices * attention_weights).rolling(window).sum()
                    df_deep[f'Attention_Volume_{window}'] = (df_deep.get('Volume', pd.Series(0, index=df_deep.index)) * attention_weights).rolling(window).sum() if 'Volume' in df_deep.columns else 0


                self._status_update("✅ Attention-like features created", status_callback)
            except Exception as e:
                self._status_update(f"⚠️ Attention features failed: {e}", status_callback)

        # 2. CNN-inspired Local Pattern Features
        if 'Close' in df_deep.columns:
            try:
                close_prices = df_deep['Close']

                # 1D Convolution-like features with different kernels
                kernels = {
                    'trend_up': [1, 2, 3, 2, 1],      # Upward trend detector
                    'trend_down': [1, 2, -3, -2, -1], # Downward trend detector
                    'peak': [1, 2, 0, -2, -1],        # Peak detector
                    'valley': [-1, -2, 0, 2, 1],      # Valley detector
                    'momentum': [-2, -1, 0, 1, 2]     # Momentum detector
                }

                for kernel_name, kernel in kernels.items():
                    kernel_size = len(kernel)
                    # Apply convolution using pandas rolling apply for simplicity and to handle NaNs from shifts
                    df_deep[f'CNN_{kernel_name}'] = close_prices.rolling(window=kernel_size).apply(lambda x: np.sum(x * kernel) if len(x) == kernel_size else 0, raw=True).fillna(0)


                self._status_update("✅ CNN-inspired features created", status_callback)
            except Exception as e:
                self._status_update(f"⚠️ CNN features failed: {e}", status_callback)

        # 3. LSTM-inspired Sequential Features
        if 'Close' in df_deep.columns:
            try:
                close_prices = df_deep['Close']

                # Simulated LSTM cell states
                for seq_len in [10, 20, 50]:
                    # Forget gate simulation (based on volatility)
                    returns = close_prices.pct_change().fillna(0)
                    volatility = returns.rolling(seq_len).std().fillna(0)
                    forget_gate = 1 / (1 + np.exp(-volatility))  # Sigmoid activation

                    # Input gate simulation (based on volume changes)
                    if 'Volume' in df_deep.columns:
                        volume_change = df_deep['Volume'].pct_change().fillna(0)
                        input_gate = 1 / (1 + np.exp(-volume_change.rolling(seq_len).mean().fillna(0)))
                    else:
                        input_gate = 0.5

                    # Cell state simulation
                    cell_state = close_prices.rolling(seq_len).mean().fillna(0) * forget_gate + close_prices * input_gate
                    df_deep[f'LSTM_Cell_{seq_len}'] = cell_state

                    # Hidden state simulation
                    output_gate = 1 / (1 + np.exp(-returns.rolling(seq_len).mean().fillna(0)))
                    hidden_state = np.tanh(cell_state) * output_gate
                    df_deep[f'LSTM_Hidden_{seq_len}'] = hidden_state

                self._status_update("✅ LSTM-inspired features created", status_callback)
            except Exception as e:
                self._status_update(f"⚠️ LSTM features failed: {e}", status_callback)

        # 4. Transformer-inspired Positional Encoding
        try:
            seq_length = len(df_deep)
            d_model = 64  # Embedding dimension

            # Sinusoidal positional encoding
            position = np.arange(seq_length).reshape(-1, 1)
            div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

            # Create positional encoding features
            pos_enc_sin = np.sin(position * div_term[:min(32, len(div_term))])
            pos_enc_cos = np.cos(position * div_term[:min(32, len(div_term))])

            # Add as features (use only first few dimensions)
            for i in range(min(8, pos_enc_sin.shape[1])):
                df_deep[f'Pos_Enc_Sin_{i}'] = pos_enc_sin[:, i]
                df_deep[f'Pos_Enc_Cos_{i}'] = pos_enc_cos[:, i]

            self._status_update("✅ Positional encoding features created", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Positional encoding failed: {e}", status_callback)

        # 5. Residual Connection Inspired Features
        if 'Close' in df_deep.columns:
            try:
                close_prices = df_deep['Close']

                # Multi-layer residual features
                layer_1 = close_prices.rolling(5).mean().fillna(method='bfill').fillna(method='ffill')  # First "layer"
                layer_2 = layer_1.rolling(5).mean().fillna(method='bfill').fillna(method='ffill')      # Second "layer"
                layer_3 = layer_2.rolling(5).mean().fillna(method='bfill').fillna(method='ffill')      # Third "layer"


                # Residual connections
                df_deep['Residual_1_0'] = layer_1 + close_prices  # Skip connection
                df_deep['Residual_2_1'] = layer_2 + layer_1       # Skip connection
                df_deep['Residual_3_0'] = layer_3 + close_prices  # Long skip connection

                # Batch normalization inspired features
                for window in [10, 20]:
                    mean = close_prices.rolling(window).mean()
                    std = close_prices.rolling(window).std().replace(0, 1e-9) # Avoid division by zero
                    df_deep[f'BatchNorm_{window}'] = (close_prices - mean) / std


                self._status_update("✅ Residual connection features created", status_callback)
            except Exception as e:
                self._status_update(f"⚠️ Residual features failed: {e}", status_callback)

        # 6. GAN-inspired Discriminator Features
        if 'Close' in df_deep.columns and 'Volume' in df_deep.columns:
            try:
                # Real vs Fake pattern detection (inspired by GANs)
                close_prices = df_deep['Close']
                volume = df_deep['Volume']

                # Pattern authenticity scores
                for window in [10, 20]:
                    # Price-volume relationship authenticity
                    price_change = close_prices.pct_change().fillna(0)
                    volume_change = volume.pct_change().fillna(0)


                    # Correlation as authenticity measure
                    correlation = price_change.rolling(window).corr(volume_change).fillna(0)
                    df_deep[f'Pattern_Authenticity_{window}'] = correlation

                    # Volatility authenticity (detect unusual patterns)
                    volatility = price_change.rolling(window).std().fillna(0)
                    vol_mean = volatility.rolling(50).mean().fillna(0)
                    vol_std = volatility.rolling(50).std().replace(0, 1e-9) # Avoid division by zero
                    vol_z_score = (volatility - vol_mean) / vol_std
                    df_deep[f'Volatility_Authenticity_{window}'] = vol_z_score.fillna(0)


                self._status_update("✅ GAN-inspired features created", status_callback)
            except Exception as e:
                self._status_update(f"⚠️ GAN features failed: {e}", status_callback)

        # 7. Autoencoder-inspired Reconstruction Features
        if 'Close' in df_deep.columns:
            try:
                close_prices = df_deep['Close']

                # Dimension reduction and reconstruction (PCA-like)
                for window in [20, 50]:
                    # Rolling PCA simulation
                    # Create a matrix of lagged means for PCA simulation
                    lag_cols = []
                    for i in range(5): # Number of lagged features for PCA
                        lag_col_name = f'close_lag_mean_{i}_win{window}'
                        df_deep[lag_col_name] = close_prices.shift(i).rolling(window).mean().fillna(method='bfill').fillna(method='ffill')
                        lag_cols.append(lag_col_name)

                    price_matrix_df = df_deep[lag_cols]

                    # Simple reconstruction error
                    mean_reconstruction = price_matrix_df.mean(axis=1)
                    reconstruction_error = np.abs(close_prices - mean_reconstruction)
                    df_deep[f'Reconstruction_Error_{window}'] = reconstruction_error

                    # Compressed representation
                    df_deep[f'Compressed_Rep_{window}'] = mean_reconstruction

                    # Clean up temporary lag columns
                    df_deep.drop(columns=lag_cols, inplace=True)


                self._status_update("✅ Autoencoder-inspired features created", status_callback)
            except Exception as e:
                self._status_update(f"⚠️ Autoencoder features failed: {e}", status_callback)

        # Clean up NaN and infinite values
        df_deep = df_deep.replace([np.inf, -np.inf], np.nan)
        for col in df_deep.columns:
            if df_deep[col].dtype in ['float64', 'float32']:
                df_deep[col] = df_deep[col].ffill().bfill().fillna(0)

        new_features = len(df_deep.columns) - len(df.columns)
        self._status_update(f"🚀 Đã tạo {new_features} Deep Learning Inspired Features", status_callback)

        return df_deep

    def create_ultra_advanced_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                                      status_callback: Optional[Callable[[str, bool], None]] = None):
        """Tạo Ultra Advanced Ensemble với XGBoost, LightGBM, CatBoost và Deep Learning"""
        self._status_update("Đang tạo Ultra Advanced Ensemble để đạt accuracy > 80%...", status_callback)

        from sklearn.ensemble import StackingClassifier
        from sklearn.model_selection import StratifiedKFold

        base_models = []

        # 1. Optimized XGBoost
        self._status_update("Training Optimized XGBoost...", status_callback)
        xgb_model = self.train_xgboost_model(X_train, y_train, X_val, y_val, status_callback)
        base_models.append(('xgb', xgb_model))

        # 2. LightGBM with Bayesian Optimization
        if LIGHTGBM_AVAILABLE:
            self._status_update("Training LightGBM with advanced optimization...", status_callback)
            try:
                lgb_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 127,
                    'max_depth': -1,
                    'learning_rate': 0.05,
                    'n_estimators': 1000,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'n_jobs': -1, # Use all available cores
                    'verbose': -1,
                    'importance_type': 'gain',
                    'min_child_samples': 20,
                    'min_split_gain': 0.01,
                    'bagging_freq': 5,
                    'bagging_fraction': 0.8,
                    'feature_fraction': 0.8,
                    'lambda_l1': 0.1,
                    'lambda_l2': 0.1,
                    'min_data_in_leaf': 20,
                    'max_bin': 255,
                    # 'early_stopping_rounds': 100 # early_stopping_rounds is a fit param
                }

                lgb_model = lgb.LGBMClassifier(**lgb_params)
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='binary_logloss',
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
                )
                base_models.append(('lgb', lgb_model))
                self._status_update("✅ LightGBM trained successfully", status_callback)
            except Exception as e:
                self._status_update(f"⚠️ LightGBM failed: {e}", status_callback)

        # 3. CatBoost with automatic categorical feature handling
        if CATBOOST_AVAILABLE:
            self._status_update("Training CatBoost with automatic feature handling...", status_callback)
            try:
                cat_params = {
                    'iterations': 1000,
                    'learning_rate': 0.05,
                    'depth': 8,
                    'l2_leaf_reg': 3,
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC',
                    'random_seed': 42,
                    'bagging_temperature': 0.8,
                    'od_type': 'Iter',
                    'od_wait': 50,
                    'verbose': False,
                    'allow_const_label': True, # Important if target becomes constant after splits
                    'task_type': 'CPU', # Can be 'GPU' if available
                    'thread_count': -1, # Use all available cores
                    'border_count': 254,
                    'bootstrap_type': 'Bayesian',
                    'posterior_sampling': True,
                    'sampling_unit': 'Object',
                    'grow_policy': 'Lossguide',
                    'min_data_in_leaf': 20,
                    'max_leaves': 64,
                    'rsm': 0.8,
                    'nan_mode': 'Min', # How to handle NaNs
                    'input_borders': None,
                    'boosting_type': 'Ordered', # Or 'Plain'
                    'permutation_count': 4
                }

                cat_model = cb.CatBoostClassifier(**cat_params)
                cat_model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
                base_models.append(('catboost', cat_model))
                self._status_update("✅ CatBoost trained successfully", status_callback)
            except Exception as e:
                self._status_update(f"⚠️ CatBoost failed: {e}", status_callback)

        # 4. Enhanced Random Forest with optimal hyperparameters
        self._status_update("Training Enhanced Random Forest...", status_callback)
        try:
            rf_model = RandomForestClassifier(
                n_estimators=1000,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced_subsample',
                criterion='entropy',
                bootstrap=True,
                oob_score=True,
                warm_start=False,
                max_samples=0.8
            )
            rf_model.fit(X_train, y_train)
            base_models.append(('rf', rf_model))
            self._status_update("✅ Random Forest trained successfully", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Random Forest failed: {e}", status_callback)

        # 5. Extra Trees with diverse configuration
        self._status_update("Training Extra Trees Classifier...", status_callback)
        try:
            et_model = ExtraTreesClassifier(
                n_estimators=800,
                max_depth=None, # Allow full growth
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                criterion='gini',
                bootstrap=False # Default for ExtraTrees
            )
            et_model.fit(X_train, y_train)
            base_models.append(('et', et_model))
            self._status_update("✅ Extra Trees trained successfully", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Extra Trees failed: {e}", status_callback)

        # 6. Advanced Neural Network
        self._status_update("Training Deep Neural Network...", status_callback)
        try:
            nn_model = MLPClassifier(
                hidden_layer_sizes=(200, 100, 50, 25), # Deeper network
                activation='relu',
                solver='adam',
                alpha=0.001, # L2 regularization
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000, # More iterations
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2, # Use part of training for validation
                n_iter_no_change=20, # Stop if no improvement
                tol=0.0001,
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                batch_size='auto' # Or a fixed size like 256
            )
            # Scale data for neural network
            from sklearn.preprocessing import StandardScaler
            scaler_nn = StandardScaler()
            X_train_scaled_nn = scaler_nn.fit_transform(X_train) # Fit on train only
            # X_val_scaled_nn = scaler_nn.transform(X_val) # NN will use validation_fraction from X_train_scaled_nn

            nn_model.fit(X_train_scaled_nn, y_train)

            # Wrap in a custom classifier to handle scaling for Stacking
            class ScaledNNClassifier:
                def __init__(self, nn_model, scaler):
                    self.nn_model = nn_model
                    self.scaler = scaler
                def predict(self, X):
                    X_scaled = self.scaler.transform(X)
                    return self.nn_model.predict(X_scaled)
                def predict_proba(self, X):
                    X_scaled = self.scaler.transform(X)
                    return self.nn_model.predict_proba(X_scaled)
                def fit(self, X, y): # Stacking requires fit method
                    X_scaled_fit = self.scaler.fit_transform(X)
                    self.nn_model.fit(X_scaled_fit, y)
                    return self
                def get_params(self, deep=True): # Stacking requires get_params
                    return self.nn_model.get_params(deep=deep)


            scaled_nn = ScaledNNClassifier(nn_model, scaler_nn)
            base_models.append(('nn', scaled_nn))
            self._status_update("✅ Neural Network trained successfully", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Neural Network failed: {e}", status_callback)

        # 7. Gradient Boosting with different loss function
        self._status_update("Training Gradient Boosting with exponential loss...", status_callback)
        try:
            gb_model = GradientBoostingClassifier(
                loss='exponential', # More emphasis on misclassified points
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42,
                max_features='sqrt',
                verbose=0,
                warm_start=False,
                validation_fraction=0.2, # Use for early stopping
                n_iter_no_change=20,
                tol=0.0001
            )
            gb_model.fit(X_train, y_train)
            base_models.append(('gb_exp', gb_model))
            self._status_update("✅ Gradient Boosting trained successfully", status_callback)
        except Exception as e:
            self._status_update(f"⚠️ Gradient Boosting failed: {e}", status_callback)

        # Check if we have enough models
        if len(base_models) < 3:
            self._status_update("❌ Không đủ base models cho Ultra Advanced Ensemble. Fallback to simpler ensemble.", status_callback, True)
            # If 'e' was from the last failed try-except block, it might be relevant.
            # However, it's safer to assume 'e' is not generally defined here.
            # print(f"❌ Lỗi: Not enough base models. Last error might have been {e if 'e' in locals() else 'unknown'}")
            # traceback.print_exc() # This might print "NoneType" if no active exception
            return self.adaptive_ensemble_selection(X_train, y_train, X_val, y_val, status_callback) # Fallback

        # Meta-learner: Calibrated XGBoost or Logistic Regression
        self._status_update("Configuring Meta-Learner...", status_callback)
        meta_learner = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=123, # Different seed for meta-learner
            n_jobs=1, verbosity=0
        )

        # Create Stacking Classifier
        self._status_update("Creating Stacking Classifier...", status_callback)
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), # Stratified CV
            stack_method='predict_proba', # Use probabilities for meta-learner
            n_jobs=1, # Single job for stacking stability
            verbose=0
        )

        self._status_update(f"Training Ultra Advanced Stacking Ensemble with {len(base_models)} base models...", status_callback)
        stacking_model.fit(X_train, y_train) # Train on the full training set

        ultra_score = stacking_model.score(X_val, y_val)
        self._status_update(f"🚀 Ultra Advanced Ensemble Score on Validation: {ultra_score:.4f}", status_callback)

        return stacking_model


# --- Global function to be called by app.py ---
def train_stock_prediction_model(processed_files: List[str], forecast_horizon: int,
                                 target_threshold: float, test_size: float = 0.2,
                                 status_callback: Optional[Callable[[str, bool], None]] = None,
                                 progress_callback: Optional[Callable[[float, str], None]] = None
                                 ) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[List[str]]]: # Added List[str] for features
    """
    Main function to run the full training pipeline.
    Returns: (model_path, metrics_dict, feature_columns_list)
    """
    try:
        predictor = StockTrendPredictor(forecast_horizon=forecast_horizon)
        model_path, metrics, feature_columns = predictor.run_full_training_pipeline(
            processed_files, target_threshold, test_size,
            status_callback, progress_callback
        )
        return model_path, metrics, feature_columns
    except Exception as e_global:
        if status_callback:
            status_callback(f"Lỗi nghiêm trọng trong pipeline huấn luyện: {e_global}", True)
        print(f"Lỗi nghiêm trọng trong pipeline huấn luyện: {e_global}")
        traceback.print_exc()
        return None, None, None


if __name__ == '__main__':
    print("--- Running StockTrendPredictor Standalone Test ---")

    def cli_status_callback(message: str, is_error: bool = False):
        level = "ERROR" if is_error else "INFO"
        print(f"[{level}] {message}")

    def cli_progress_callback(progress: float, message: str):
        print(f"Progress: {progress*100:.1f}% - {message}")

    # Example usage: Find some processed files to test with
    sample_processed_files = []
    if os.path.exists(PROCESSED_DATA_DIR):
        all_files = [os.path.join(PROCESSED_DATA_DIR, f)
                     for f in os.listdir(PROCESSED_DATA_DIR)
                     if f.endswith('_processed_data.csv')]
        if all_files:
            sample_processed_files = all_files[:min(3, len(all_files))] # Use up to 3 files

    if not sample_processed_files:
        cli_status_callback("No processed data files found for testing in 'data/processed/'. Exiting.", True)
        sys.exit(1)

    cli_status_callback(f"Using files for test: {[os.path.basename(f) for f in sample_processed_files]}", False)

    try:
        model_file_path, final_metrics, final_feature_cols = train_stock_prediction_model(
            processed_files=sample_processed_files,
            forecast_horizon=5,
            target_threshold=0.02, # 2% increase target
            test_size=0.2,
            status_callback=cli_status_callback,
            progress_callback=cli_progress_callback
        )

        if model_file_path and final_metrics:
            cli_status_callback(f"--- Training Complete ---", False)
            cli_status_callback(f"Model saved to: {model_file_path}", False)
            cli_status_callback(f"Final Metrics: {final_metrics}", False)
            cli_status_callback(f"Number of features used: {len(final_feature_cols)}", False)
            cli_status_callback(f"Sample features: {final_feature_cols[:5]}...", False)

            accuracy_val = final_metrics.get('accuracy', 0)
            if accuracy_val >= 0.8:
                 cli_status_callback(f"🎉 TARGET ACHIEVED! Accuracy: {accuracy_val*100:.2f}%", False)
            else:
                 cli_status_callback(f"📈 Accuracy: {accuracy_val*100:.2f}%. Consider further tuning or more data.", False)

        else:
            cli_status_callback("Training failed to produce a model or metrics.", True)

    except Exception as main_e:
        cli_status_callback(f"Standalone test failed with error: {main_e}", True)
        traceback.print_exc()

    print("--- StockTrendPredictor Standalone Test Complete ---")