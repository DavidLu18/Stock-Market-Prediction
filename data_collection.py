import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback
import requests # for yfinance retries
import json
from typing import List, Dict, Tuple, Optional, Any, Callable

import yfinance as yf
import talib # Assuming TALIB_AVAILABLE is handled by the calling script or checked here if necessary
from pytrends.request import TrendReq 
# import praw # Reddit (PRAW is imported in app.py, DataCollector will use passed client if available)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas_datareader.data as pdr 

# --- Directory Setup ---
SCRIPT_DIR_DC = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR_DC = SCRIPT_DIR_DC # Assuming script is in root or adjust as needed
DATA_DIR_DC = os.path.join(ROOT_DIR_DC, 'data')
RAW_DATA_DIR_DC = os.path.join(DATA_DIR_DC, 'raw')
PROCESSED_DATA_DIR_DC = os.path.join(DATA_DIR_DC, 'processed')

for dir_path in [DATA_DIR_DC, RAW_DATA_DIR_DC, PROCESSED_DATA_DIR_DC]:
    if not os.path.exists(dir_path):
        try: os.makedirs(dir_path); print(f"(DataCollector) Created directory: {dir_path}")
        except OSError as e: print(f"(DataCollector) Error creating directory {dir_path}: {e}")


class DataCollector:
    def __init__(self):
        self.companies: List[Dict[str, str]] = []
        self.reddit_client: Optional[Any] = None # praw.Reddit instance
        self.vader_analyzer: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
        try:
            self.pytrends: Optional[TrendReq] = TrendReq(hl='en-US', tz=360, retries=3, backoff_factor=0.5)
        except Exception as e:
            print(f"Warning: Failed to initialize TrendReq: {e}. Google Trends might be unavailable.")
            self.pytrends = None

        self.fred_api_key: Optional[str] = os.environ.get('FRED_API_KEY')

        # Standardized column names
        self.COL_DATE: str = 'Date'
        self.COL_TICKER: str = 'Ticker'
        self.COL_OPEN: str = 'Open'
        self.COL_HIGH: str = 'High'
        self.COL_LOW: str = 'Low'
        self.COL_CLOSE: str = 'Close'
        self.COL_ADJ_CLOSE: str = 'Adj Close'
        self.COL_VOLUME: str = 'Volume'

        self.MARKET_INDICATORS_MAP: Dict[str, str] = {
            'VIX': '^VIX',      # CBOE Volatility Index
            'TNX': '^TNX',      # 10-Year Treasury Yield
            'GSPC': '^GSPC'     # S&P 500 Index (as a general market proxy)
        }
        self.FRED_SERIES_MAP: Dict[str, str] = {
            'FEDFUNDS': 'FEDFUNDS', # Effective Federal Funds Rate
            # Add more FRED series here if desired, e.g., 'UNRATE' for Unemployment
        }
        self.sp500_companies_df: Optional[pd.DataFrame] = None

    def _status_update(self, message: str, status_callback: Optional[Callable[[str, bool], None]], is_error: bool = False, indent_level: int = 0) -> None:
        prefix: str = "  " * indent_level
        if status_callback:
            status_callback(f"{prefix}{message}", is_error)
        else:
            level: str = "ERROR" if is_error else "INFO"
            print(f"{prefix}[{level}] {message}")

    def _progress_update(self, current_val: float, total_val: float, task_name: str, progress_callback: Optional[Callable[[float, str], None]]) -> None:
        if progress_callback:
            progress: float = (current_val / total_val) if total_val > 0 else 0.0
            progress_callback(progress, f"{task_name}: {current_val:.1f}/{total_val:.1f} ({progress*100:.1f}%)")


    def _save_raw_data(self, df: pd.DataFrame, filename: str, sub_dir: str = "", status_callback: Optional[Callable[[str, bool], None]] = None, is_json: bool = False) -> Optional[str]:
        dir_path: str = os.path.join(RAW_DATA_DIR_DC, sub_dir)
        if not os.path.exists(dir_path):
            try: os.makedirs(dir_path)
            except OSError as e: self._status_update(f"Error creating raw subdir {dir_path}: {e}", status_callback, True); return None

        filepath: str = os.path.join(dir_path, filename)
        try:
            df_to_save = df.copy()
            if isinstance(df_to_save.index, pd.DatetimeIndex) and df_to_save.index.tz is not None:
                df_to_save.index = df_to_save.index.tz_localize(None)

            if is_json:
                if df_to_save.index.name == self.COL_DATE or isinstance(df_to_save.index, pd.DatetimeIndex): # If index is date-like
                    df_to_save = df_to_save.reset_index()
                # Ensure Date column is string for JSON if it exists
                if self.COL_DATE in df_to_save.columns and pd.api.types.is_datetime64_any_dtype(df_to_save[self.COL_DATE]):
                    df_to_save[self.COL_DATE] = df_to_save[self.COL_DATE].dt.strftime('%Y-%m-%dT%H:%M:%S')
                df_to_save.to_json(filepath, orient='records', lines=True, date_format='iso') # date_format for any remaining datetime objects
            else: 
                # For CSV, if index is DatetimeIndex and named 'Date', save it. Otherwise, don't save index if it's just a range.
                save_index = isinstance(df_to_save.index, pd.DatetimeIndex) or df_to_save.index.name == self.COL_DATE
                df_to_save.to_csv(filepath, index=save_index)
            return filepath
        except Exception as e:
            self._status_update(f"Error saving raw data to {filepath}: {e}", status_callback, True)
            return None

    def _load_raw_data(self, filename: str, sub_dir: str = "", status_callback: Optional[Callable[[str, bool], None]] = None, 
                       parse_dates_col_name: Optional[str] = None, index_col_name: Optional[str] = None, is_json: bool = False) -> Optional[pd.DataFrame]:
        filepath: str = os.path.join(RAW_DATA_DIR_DC, sub_dir, filename)
        if not os.path.exists(filepath):
            return None
        try:
            df: Optional[pd.DataFrame] = None
            if is_json:
                df = pd.read_json(filepath, orient='records', lines=True)
                if parse_dates_col_name and parse_dates_col_name in df.columns:
                     df[parse_dates_col_name] = pd.to_datetime(df[parse_dates_col_name])
                if index_col_name and index_col_name in df.columns:
                     df = df.set_index(index_col_name)
            else: 
                # For CSV, parse 'Date' if it's specified as index or parse_dates target
                parse_dates_list = [parse_dates_col_name] if parse_dates_col_name and parse_dates_col_name != index_col_name else None
                if index_col_name == self.COL_DATE and parse_dates_list is None: # If Date is index, ensure it's parsed
                    parse_dates_list = [self.COL_DATE] 
                
                # If 'Date' is the index_col, it will be parsed. If 'Date' is in parse_dates_list, it will be parsed.
                df = pd.read_csv(filepath, index_col=index_col_name, parse_dates=parse_dates_list or True) # parse_dates=True attempts to parse index

            if df is None: return None

            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                if df.index.name != self.COL_DATE and self.COL_DATE not in df.columns : df.index.name = self.COL_DATE
            
            if self.COL_DATE in df.columns and df.index.name != self.COL_DATE:
                if pd.api.types.is_datetime64_any_dtype(df[self.COL_DATE]) and df[self.COL_DATE].dt.tz is not None:
                    df[self.COL_DATE] = df[self.COL_DATE].dt.tz_localize(None)
                if index_col_name is None and parse_dates_col_name == self.COL_DATE :
                    df = df.set_index(self.COL_DATE)
            return df
        except Exception as e:
            self._status_update(f"Error loading raw data from {filepath}: {e}", status_callback, True)
            return None

    def fetch_and_save_yfinance_data(self, ticker_symbol: str, start_date_str: str, end_date_str: str,
                                     data_type_name: str = "stock", status_callback: Optional[Callable[[str, bool], None]] = None, retries: int = 3, delay: int = 5) -> Optional[pd.DataFrame]:
        self._status_update(f"Fetching {data_type_name} data for {ticker_symbol} ({start_date_str} to {end_date_str})", status_callback, indent_level=1)
        yf_ticker_obj = yf.Ticker(ticker_symbol.replace('.', '-')) # yfinance prefers '-' for dots in tickers like BRK.B -> BRK-B
        end_date_yf = (pd.to_datetime(end_date_str) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') # yfinance end is exclusive

        for attempt in range(retries):
            try:
                df = yf_ticker_obj.history(start=start_date_str, end=end_date_yf, interval="1d",
                                           auto_adjust=False, actions=True) # auto_adjust=False gives Adj Close separately
                if df.empty:
                    if attempt < retries - 1: time.sleep(delay); continue
                    self._status_update(f"No data for {ticker_symbol} after {retries} retries.", status_callback, True, indent_level=1)
                    return None

                if df.index.tz is not None: 
                    df.index = df.index.tz_localize(None)
                df.index.name = self.COL_DATE
                df.columns = [col.replace(' ', '_') for col in df.columns] # Sanitize column names

                rename_map = {'Adj_Close': self.COL_ADJ_CLOSE, 'Stock_Splits': 'Stock_Splits'} # Ensure our standard names
                df.rename(columns=rename_map, inplace=True)

                # Ensure essential OHLCV columns exist
                expected_cols = [self.COL_OPEN, self.COL_HIGH, self.COL_LOW, self.COL_CLOSE, self.COL_ADJ_CLOSE, self.COL_VOLUME]
                for col in expected_cols:
                    if col not in df.columns: df[col] = np.nan # Add if missing, to be filled later

                raw_filename = f"{ticker_symbol.upper()}_{data_type_name}_raw.csv"
                self._save_raw_data(df, raw_filename, sub_dir=data_type_name, status_callback=status_callback)
                return df

            except requests.exceptions.ConnectionError as e: # More specific yfinance connection error
                self._status_update(f"Connection error for {ticker_symbol} (attempt {attempt+1}/{retries}): {e}", status_callback, True, indent_level=1)
                if attempt < retries - 1: time.sleep(delay)
                else: self._status_update(f"Failed to fetch {ticker_symbol} after {retries} retries due to ConnectionError.", status_callback, True, indent_level=1); return None
            except Exception as e: # Catch other potential yfinance errors
                self._status_update(f"yfinance error for {ticker_symbol} (attempt {attempt+1}/{retries}): {e}", status_callback, True, indent_level=1)
                if attempt < retries - 1: time.sleep(delay)
                else: self._status_update(f"Failed to fetch {ticker_symbol} after {retries} retries: {e}", status_callback, True, indent_level=1); return None
        return None
        
    def fetch_and_save_fred_data(self, series_map: Dict[str, str], start_date_dt: datetime, end_date_dt: datetime, status_callback: Optional[Callable[[str, bool], None]] = None) -> Dict[str, pd.DataFrame]:
        self._status_update(f"Fetching FRED data for series: {', '.join(series_map.keys())}", status_callback, indent_level=1)
        all_fred_data: Dict[str, pd.DataFrame] = {}
        for series_name, fred_id in series_map.items():
            try:
                df_series = pdr.get_data_fred(fred_id, start_date_dt, end_date_dt, api_key=self.fred_api_key)
                if df_series.empty:
                    self._status_update(f"No data for FRED series {fred_id}.", status_callback, True, indent_level=2)
                    continue

                if df_series.index.tz is not None: 
                    df_series.index = df_series.index.tz_localize(None)
                df_series.index.name = self.COL_DATE
                df_series.rename(columns={fred_id: series_name}, inplace=True) # Rename column to our internal name

                raw_filename = f"{series_name}_fred_raw.csv"
                self._save_raw_data(df_series, raw_filename, sub_dir="fred", status_callback=status_callback)
                all_fred_data[series_name] = df_series
            except Exception as e:
                self._status_update(f"Error fetching FRED series {fred_id}: {e}", status_callback, True, indent_level=2)
        return all_fred_data

    def setup_reddit_api(self, client_id: str, client_secret: str, user_agent: str, status_callback: Callable[[str, bool], None] = print) -> bool:
        if not all([client_id, client_secret, user_agent]):
            self._status_update("Reddit API credentials incomplete. Reddit collection disabled.", status_callback, is_error=True)
            self.reddit_client = None
            return False
        try:
            import praw # Local import
            self.reddit_client = praw.Reddit(
                client_id=client_id, client_secret=client_secret,
                user_agent=user_agent, read_only=True
            )
            # Test connection (optional, PRAW might do lazy init)
            # self.reddit_client.user.me() # This would require non-read-only scope or fail
            self._status_update("Reddit API client object created (connection not fully tested yet).", status_callback)
            return True
        except ImportError:
            self._status_update("PRAW library not installed. Reddit collection disabled.", status_callback, is_error=True)
            self.reddit_client = None
            return False
        except Exception as e: # Catch PRAW specific exceptions if known, e.g., praw.exceptions.APIException
            self._status_update(f"Error initializing Reddit API client: {e}", status_callback, is_error=True)
            self.reddit_client = None
            return False

    def fetch_and_save_reddit_sentiment(self, ticker_symbol: str, company_name: str, start_date_dt: datetime, end_date_dt: datetime,
                                        status_callback: Optional[Callable[[str, bool], None]] = None, subreddits: Optional[List[str]] = None, limit_per_subreddit: int = 50) -> Optional[pd.DataFrame]:
        if not self.reddit_client:
            self._status_update("Reddit client not initialized. Skipping Reddit sentiment.", status_callback, True, indent_level=1)
            return None

        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket', 'SecurityAnalysis', 'options', 'finance'] # Added finance

        self._status_update(f"Fetching Reddit sentiment for {ticker_symbol} (Keywords: {ticker_symbol}, {company_name})", status_callback, indent_level=1)

        search_terms: List[str] = [ticker_symbol.lower(), f"${ticker_symbol.lower()}"] # Common ways to refer to tickers
        if company_name and company_name.lower() != ticker_symbol.lower():
            # Add parts of the company name as search terms
            name_parts = company_name.lower().replace('inc.', '').replace('corp.', '').replace('.', '').replace(',', '').strip().split()
            if len(name_parts) > 0: search_terms.append(name_parts[0]) # First word
            if len(name_parts) > 1: search_terms.append(" ".join(name_parts[:2])) # First two words
        search_terms = list(set(t for t in search_terms if len(t) > 1)) # Unique terms, min length 2

        all_posts_data: List[Dict[str, Any]] = []
        start_timestamp: int = int(start_date_dt.timestamp())
        end_timestamp: int = int(end_date_dt.timestamp())

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                # PRAW search query structure: (term1 OR term2)
                query: str = f"({' OR '.join(search_terms)}) timestamp:{start_timestamp}..{end_timestamp}"
                
                posts_collected_count: int = 0
                # Search submissions (posts)
                for post in subreddit.search(query, sort='new', syntax='cloudsearch', time_filter='all', limit=limit_per_subreddit):
                    # Double check timestamp, as Reddit search can be a bit fuzzy with time_filter='all'
                    if not (start_timestamp <= post.created_utc <= end_timestamp): continue

                    text_content: str = post.title + " " + post.selftext # Combine title and body for sentiment
                    sentiment_scores: Dict[str, float] = self.vader_analyzer.polarity_scores(text_content)
                    all_posts_data.append({
                        'ticker': ticker_symbol, 'created_utc': post.created_utc,
                        self.COL_DATE: datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'), 
                        'subreddit': subreddit_name, 'title': post.title, 'score': post.score,
                        'num_comments': post.num_comments, 'compound': sentiment_scores['compound'],
                        'positive': sentiment_scores['pos'], 'neutral': sentiment_scores['neu'],
                        'negative': sentiment_scores['neg'], 'id': post.id
                    })
                    posts_collected_count +=1
            except Exception as e: # Catch prawcore exceptions or others
                self._status_update(f"Error searching r/{subreddit_name} for {ticker_symbol}: {e}", status_callback, True, indent_level=2)

        if not all_posts_data:
            self._status_update(f"No Reddit posts found for {ticker_symbol} in the given period.", status_callback, indent_level=1)
            return None

        df_posts = pd.DataFrame(all_posts_data)
        df_posts[self.COL_DATE] = pd.to_datetime(df_posts[self.COL_DATE]).dt.normalize() 

        raw_filename_posts = f"{ticker_symbol.upper()}_reddit_posts_raw.json" # Save detailed posts as JSON
        self._save_raw_data(df_posts, raw_filename_posts, sub_dir="reddit", status_callback=status_callback, is_json=True)

        if df_posts[self.COL_DATE].dt.tz is not None:
            df_posts[self.COL_DATE] = df_posts[self.COL_DATE].dt.tz_localize(None)

        agg_sentiment_df = df_posts.groupby(self.COL_DATE).agg(
            Compound=('compound', 'mean'), Positive=('positive', 'mean'),
            Neutral=('neutral', 'mean'), Negative=('negative', 'mean'),
            Count=('id', 'count') # Number of posts per day
        ).reset_index() 

        raw_filename_agg = f"{ticker_symbol.upper()}_reddit_sentiment_agg_raw.csv" # Save aggregated sentiment as CSV
        self._save_raw_data(agg_sentiment_df, raw_filename_agg, sub_dir="reddit", status_callback=status_callback)

        return agg_sentiment_df.set_index(self.COL_DATE) # Return with Date as index for merging

    def fetch_and_save_google_trends(self, ticker_symbol: str, company_name: str, start_date_str: str, end_date_str: str, status_callback: Optional[Callable[[str, bool], None]] = None) -> Optional[pd.DataFrame]:
        if not self.pytrends:
            self._status_update("Pytrends client not initialized. Skipping Google Trends.", status_callback, True, indent_level=1)
            return None

        keywords: List[str] = [ticker_symbol] # Always include ticker
        if company_name and company_name.lower() != ticker_symbol.lower():
            simple_name: str = company_name.split(' ')[0] # First word of company name
            if len(simple_name) > 2 and simple_name.lower() not in ticker_symbol.lower() : # Avoid redundancy if ticker is like 'MSFT' and name 'Microsoft'
                 keywords.append(simple_name)
        keywords = list(set(keywords)) # Unique keywords

        self._status_update(f"Fetching Google Trends for {ticker_symbol} (Keywords: {', '.join(keywords)})", status_callback, indent_level=1)
        timeframe: str = f"{start_date_str} {end_date_str}"

        df_trend_primary: Optional[pd.DataFrame] = None
        for kw_idx, kw in enumerate(keywords):
            try:
                self.pytrends.build_payload(kw_list=[kw], cat=0, timeframe=timeframe, geo='US', gprop='') # Focus on US
                df_trend_kw: pd.DataFrame = self.pytrends.interest_over_time()

                if df_trend_kw.empty or kw not in df_trend_kw.columns:
                    continue

                if df_trend_kw.index.tz is not None:
                    df_trend_kw.index = df_trend_kw.index.tz_localize(None)
                df_trend_kw.index.name = self.COL_DATE

                raw_filename = f"{ticker_symbol.upper()}_trends_{kw.replace(' ','_')}_raw.csv"
                self._save_raw_data(df_trend_kw, raw_filename, sub_dir="trends", status_callback=status_callback)

                if kw_idx == 0: # Use the first keyword's trend as the primary one
                    df_trend_primary = df_trend_kw[[kw]].rename(columns={kw: "Google_Trends"})
                    if 'isPartial' in df_trend_kw.columns: # Check if 'isPartial' column exists
                         df_trend_primary['Trends_IsPartial'] = df_trend_kw['isPartial'].astype(int) # Convert boolean to int for easier processing later
            except requests.exceptions.Timeout:
                 self._status_update(f"Timeout fetching Google Trends for keyword '{kw}'. Skipping this keyword.", status_callback, True, indent_level=2)
                 time.sleep(10) # Wait a bit before next keyword
            except Exception as e: # Catch other pytrends errors (e.g., 429 Too Many Requests)
                self._status_update(f"Error fetching Google Trends for keyword '{kw}': {e}", status_callback, True, indent_level=2)
                if "response code 429" in str(e).lower() or "Too Many Requests" in str(e):
                    self._status_update("Rate limit hit for Google Trends. Waiting 60s.", status_callback, True, indent_level=2)
                    time.sleep(60) # Wait longer for rate limits
                else: time.sleep(5) # Shorter wait for other errors

        if df_trend_primary is None:
            self._status_update(f"No Google Trends data collected for {ticker_symbol} after trying all keywords.", status_callback, indent_level=1)
        return df_trend_primary

    def _calculate_basic_technical_indicators(self, df_ohlcv: pd.DataFrame, status_callback: Optional[Callable[[str, bool], None]] = None, ticker: str = "N/A") -> pd.DataFrame:
        df = df_ohlcv.copy()

        required_cols: List[str] = [self.COL_OPEN, self.COL_HIGH, self.COL_LOW, self.COL_CLOSE, self.COL_VOLUME, self.COL_ADJ_CLOSE]
        for col in required_cols:
            if col not in df.columns: df[col] = np.nan # Add if missing
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill NaNs in OHLCV before TA calculation
        df[required_cols] = df[required_cols].ffill().bfill() 
        df[self.COL_VOLUME].fillna(0, inplace=True) # Volume specifically can be 0
        df.dropna(subset=[self.COL_CLOSE], inplace=True) # Drop if Close is still NaN (shouldn't happen after bfill if any data)

        if df.empty or len(df) < 20: # TA-Lib often needs ~2x period for some indicators
            self._status_update(f"Data too short for TA calculation for {ticker} after cleaning ({len(df)} rows). Populating TA columns with NaN.", status_callback, True, indent_level=2)
            # Define a standard set of TA columns that might be expected downstream
            ta_cols: List[str] = [
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
                'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
                'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_Upper', 'BB_Middle', 'BB_Lower', 'SlowK', 'SlowD', # Stochastic
                'ADX_14', 'PLUS_DI_14', 'MINUS_DI_14', 'OBV', 'ATR_14', 'WILLR_14', # Williams %R
                'ROC_5', 'ROC_10', 'ROC_20', 'CCI_20', 'MFI_14', # Commodity Channel Index, Money Flow Index
                'AD_Line', 'ADOSC', 'TRIX_30' # Accumulation/Distribution Line, AD Oscillator, TRIX
            ]
            for ta_col in ta_cols: df[ta_col] = np.nan
            return df

        # Ensure data types are float for TA-Lib
        op: np.ndarray = df[self.COL_OPEN].values.astype(float)
        hi: np.ndarray = df[self.COL_HIGH].values.astype(float)
        lo: np.ndarray = df[self.COL_LOW].values.astype(float)
        cl: np.ndarray = df[self.COL_CLOSE].values.astype(float)
        vo: np.ndarray = df[self.COL_VOLUME].values.astype(float)
        
        # Use a global TALIB_AVAILABLE check if preferred, or pass as arg
        # For now, assuming talib is available if this function is reached with sufficient data
        try:
            import talib # Ensure talib is accessible here
            TALIB_LOADED_LOCALLY = True
        except ImportError:
            self._status_update(f"TA-Lib could not be imported in _calculate_basic_technical_indicators for {ticker}. No TA calculated.", status_callback, True, indent_level=2)
            TALIB_LOADED_LOCALLY = False
            # Populate with NaNs as above if TALIB is not available here
            ta_cols = [ # Duplicating list for safety, could be a class member
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
                'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SlowK', 'SlowD',
                'ADX_14', 'PLUS_DI_14', 'MINUS_DI_14', 'OBV', 'ATR_14', 'WILLR_14', 'ROC_5', 'ROC_10', 'ROC_20', 
                'CCI_20', 'MFI_14', 'AD_Line', 'ADOSC', 'TRIX_30'
            ]
            for ta_col in ta_cols: df[ta_col] = np.nan
            return df


        with np.errstate(divide='ignore', invalid='ignore'): # Suppress expected warnings from TA-Lib with insufficient data at head/tail
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{period}'] = talib.SMA(cl, timeperiod=period)
                df[f'EMA_{period}'] = talib.EMA(cl, timeperiod=period)
            df['RSI_14'] = talib.RSI(cl, timeperiod=14)
            macd, macdsignal, macdhist = talib.MACD(cl, fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd, macdsignal, macdhist
            upper, middle, lower = talib.BBANDS(cl, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = upper, middle, lower
            slowk, slowd = talib.STOCH(hi, lo, cl, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            df['SlowK'], df['SlowD'] = slowk, slowd
            df['ADX_14'] = talib.ADX(hi, lo, cl, timeperiod=14)
            df['PLUS_DI_14'] = talib.PLUS_DI(hi, lo, cl, timeperiod=14)
            df['MINUS_DI_14'] = talib.MINUS_DI(hi, lo, cl, timeperiod=14)
            df['OBV'] = talib.OBV(cl, vo)
            df['ATR_14'] = talib.ATR(hi, lo, cl, timeperiod=14)
            df['WILLR_14'] = talib.WILLR(hi, lo, cl, timeperiod=14)
            for period in [5, 10, 20]: df[f'ROC_{period}'] = talib.ROC(cl, timeperiod=period)
            df['CCI_20'] = talib.CCI(hi, lo, cl, timeperiod=20)
            df['MFI_14'] = talib.MFI(hi, lo, cl, vo, timeperiod=14)
            df['AD_Line'] = talib.AD(hi, lo, cl, vo) # Chaikin A/D Line
            df['ADOSC'] = talib.ADOSC(hi, lo, cl, vo, fastperiod=3, slowperiod=10) # Chaikin A/D Oscillator
            df['TRIX_30'] = talib.TRIX(cl, timeperiod=30)
        return df

    def _engineer_base_features(self, df: pd.DataFrame, ticker: str = "N/A", status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        df_eng = df.copy()
        adj_cl: pd.Series = df_eng[self.COL_ADJ_CLOSE]

        df_eng['Log_Return'] = np.log(adj_cl / adj_cl.shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        for lag in [1, 2, 3, 5, 10, 20]: # Common lag periods
            df_eng[f'Log_Return_Lag_{lag}'] = df_eng['Log_Return'].shift(lag)

        raw_vol: pd.Series = df_eng[self.COL_VOLUME]
        df_eng['Log_Volume'] = np.log(raw_vol.replace(0, 1)) # Replace 0 with 1 before log to avoid -inf
        for lag in [1, 2, 3, 5]:
            df_eng[f'Volume_Lag_{lag}'] = raw_vol.shift(lag)
            df_eng[f'Log_Volume_Lag_{lag}'] = df_eng['Log_Volume'].shift(lag)

        for window in [5, 10, 20, 60]: # Volatility over different windows
            df_eng[f'Volatility_{window}D'] = df_eng['Log_Return'].rolling(window=window, min_periods=max(1,window//2)).std(ddof=0) * np.sqrt(window) # Annualized for window

        # Price ratios
        df_eng['Close_Open_Ratio'] = (df_eng[self.COL_CLOSE] / df_eng[self.COL_OPEN].replace(0,np.nan)) -1
        df_eng['High_Low_Ratio'] = (df_eng[self.COL_HIGH] / df_eng[self.COL_LOW].replace(0,np.nan)) -1 # Range relative to low
        df_eng['High_Close_Ratio'] = (df_eng[self.COL_HIGH] / df_eng[self.COL_CLOSE].replace(0,np.nan)) -1 # Wick size
        df_eng['Low_Close_Ratio'] = (df_eng[self.COL_LOW] / df_eng[self.COL_CLOSE].replace(0,np.nan)) -1 # Wick size

        if isinstance(df_eng.index, pd.DatetimeIndex):
            df_eng['Day_Of_Week'] = df_eng.index.dayofweek.astype(float)
            df_eng['Day_Of_Month'] = df_eng.index.day.astype(float)
            df_eng['Day_Of_Year'] = df_eng.index.dayofyear.astype(float)
            df_eng['Week_Of_Year'] = (df_eng.index.isocalendar().week if hasattr(df_eng.index, 'isocalendar') else df_eng.index.weekofyear).astype(float)
            df_eng['Month'] = df_eng.index.month.astype(float)
            df_eng['Quarter'] = df_eng.index.quarter.astype(float)
            df_eng['Year'] = df_eng.index.year.astype(float)
        else:
            self._status_update(f"Index is not DatetimeIndex for {ticker}. Cannot create time features.", status_callback, True, indent_level=2)
        return df_eng

    def _ensure_timezone_naive(self, df: Optional[pd.DataFrame], df_name: str = "DataFrame", status_callback: Optional[Callable[[str, bool], None]] = None) -> Optional[pd.DataFrame]:
        if df is not None and isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def _merge_all_data_sources(self, df_stock_featured: pd.DataFrame, dict_market_indices: Dict[str, pd.DataFrame], dict_fred_data: Dict[str, pd.DataFrame],
                                df_reddit_sentiment: Optional[pd.DataFrame], df_google_trends: Optional[pd.DataFrame], ticker: str = "N/A", status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        df_merged = self._ensure_timezone_naive(df_stock_featured.copy(), "stock_featured", status_callback)

        if dict_market_indices:
            for indicator_name, df_indicator_orig in dict_market_indices.items():
                df_indicator = self._ensure_timezone_naive(df_indicator_orig.copy() if df_indicator_orig is not None else None, indicator_name, status_callback)
                if df_indicator is not None and not df_indicator.empty:
                    # Prefer 'Adj Close' for indices if available, else 'Close'
                    col_to_merge = self.COL_ADJ_CLOSE if self.COL_ADJ_CLOSE in df_indicator.columns else self.COL_CLOSE
                    if col_to_merge not in df_indicator.columns and len(df_indicator.columns)==1: # If only one column, use that
                        col_to_merge = df_indicator.columns[0]

                    if col_to_merge in df_indicator.columns:
                        if indicator_name == 'GSPC': # For S&P 500, calculate market return
                            gspc_adj_close = df_indicator[col_to_merge]
                            market_return = np.log(gspc_adj_close / gspc_adj_close.shift(1).replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
                            df_indicator_to_merge = pd.DataFrame(market_return).rename(columns={col_to_merge: 'Market_Return'})
                        else: # For other indices like VIX, TNX, use their value directly
                            df_indicator_to_merge = df_indicator[[col_to_merge]].rename(columns={col_to_merge: indicator_name})
                        df_merged = pd.merge(df_merged, df_indicator_to_merge, left_index=True, right_index=True, how='left', suffixes=('', f'_{indicator_name}_val')) # Suffix if name collision
                    else: self._status_update(f"Relevant column not found in {indicator_name} data for merging.", status_callback, True, indent_level=2)

        if dict_fred_data:
            for series_name, df_series_orig in dict_fred_data.items():
                df_series = self._ensure_timezone_naive(df_series_orig.copy() if df_series_orig is not None else None, series_name, status_callback)
                if df_series is not None and not df_series.empty:
                    if series_name in df_series.columns:
                        df_merged = pd.merge(df_merged, df_series[[series_name]], left_index=True, right_index=True, how='left', suffixes=('', f'_{series_name}_val'))

        df_reddit_sentiment = self._ensure_timezone_naive(df_reddit_sentiment.copy() if df_reddit_sentiment is not None else None, "Reddit_Sentiment", status_callback)
        if df_reddit_sentiment is not None and not df_reddit_sentiment.empty:
            df_merged = pd.merge(df_merged, df_reddit_sentiment, left_index=True, right_index=True, how='left', suffixes=('', '_Reddit'))

        df_google_trends = self._ensure_timezone_naive(df_google_trends.copy() if df_google_trends is not None else None, "Google_Trends", status_callback)
        if df_google_trends is not None and not df_google_trends.empty:
            df_merged = pd.merge(df_merged, df_google_trends, left_index=True, right_index=True, how='left', suffixes=('', '_Trends'))

        # Columns that typically get forward-filled after merging (values persist until new one comes)
        cols_to_ffill_after_merge: List[str] = [
            'VIX', 'TNX', 'Market_Return', 'FEDFUNDS', # Market and economic data
            'Compound', 'Positive', 'Neutral', 'Negative', 'Count', # Reddit sentiment
            'Google_Trends', 'Trends_IsPartial' # Google Trends
        ]
        for col in cols_to_ffill_after_merge:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].ffill()
        return df_merged

    def _final_data_cleaning_and_typing(self, df: pd.DataFrame, ticker: str = "N/A", status_callback: Optional[Callable[[str, bool], None]] = None) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle infinities first

        # Prioritize ffill then bfill for most features to carry forward known info, then backfill for start of series
        # Finally, fill any remaining NaNs with 0 (neutral value for many features)
        df_clean = df_clean.ffill().bfill().fillna(0)

        # Type conversion to save memory and ensure consistency
        for col in df_clean.columns:
            if df_clean[col].dtype == np.float64:
                df_clean[col] = df_clean[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(df_clean[col].dtype) and not col.startswith('Is_'): # Don't convert 'Is_Partial' if it's int boolean
                 # Check if integer values can safely fit in float32 if they represent categories or counts
                 if df_clean[col].min() >= np.finfo(np.float32).min and df_clean[col].max() <= np.finfo(np.float32).max:
                     df_clean[col] = df_clean[col].astype(np.float32)

        if df_clean.isna().any().any():
            self._status_update(f"Warning: NaNs still present in data for {ticker} after final cleaning. Columns: {df_clean.columns[df_clean.isna().any()].tolist()}", status_callback, True, indent_level=2)
        return df_clean

    def _save_processed_data(self, df_processed: pd.DataFrame, ticker: str, status_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[Optional[str], List[str]]:
        filename: str = f"{ticker.upper()}_processed_data.csv"
        filepath: str = os.path.join(PROCESSED_DATA_DIR_DC, filename)
        try:
            df_to_save = df_processed.copy()
            # Ensure Date is a column for CSV, not index, for consistency with how app.py might load it initially
            if isinstance(df_to_save.index, pd.DatetimeIndex) and df_to_save.index.name == self.COL_DATE:
                if df_to_save.index.tz is not None: 
                    df_to_save.index = df_to_save.index.tz_localize(None)
                df_to_save = df_to_save.reset_index() # Make Date a column
            
            # Ensure Date column is tz-naive if it exists
            if self.COL_DATE in df_to_save.columns:
                if pd.api.types.is_datetime64_any_dtype(df_to_save[self.COL_DATE]) and df_to_save[self.COL_DATE].dt.tz is not None:
                    df_to_save[self.COL_DATE] = df_to_save[self.COL_DATE].dt.tz_localize(None)

            df_to_save.to_csv(filepath, index=False) # Always save without index if Date is a column
            self._status_update(f"Saved processed data for {ticker} to {filepath}", status_callback, indent_level=1)
            return filepath, df_to_save.columns.tolist()
        except Exception as e:
            self._status_update(f"Error saving processed data for {ticker} to {filepath}: {e}", status_callback, True, indent_level=1)
            return None, []

    def run_full_pipeline(self, companies_to_process: List[Dict[str,str]], start_date_str: str, end_date_str: str,
                          use_market_indices: bool = True, use_fred_data: bool = True,
                          use_reddit_sentiment: bool = False, use_google_trends: bool = False,
                          progress_callback: Optional[Callable[[float, str], None]] = None, status_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[List[str], List[str]]:
        self._status_update(f"Starting data collection pipeline for {len(companies_to_process)} companies.", status_callback)
        start_time_pipeline: float = time.time()
        start_date_dt: datetime = pd.to_datetime(start_date_str)
        end_date_dt: datetime = pd.to_datetime(end_date_str)

        market_indices_data_frames: Dict[str, pd.DataFrame] = {}
        if use_market_indices:
            self._status_update("--- Collecting Market Indices ---", status_callback)
            total_indices = len(self.MARKET_INDICATORS_MAP); idx_count = 0
            for indicator_name, yf_symbol in self.MARKET_INDICATORS_MAP.items():
                idx_count += 1
                if progress_callback: progress_callback(idx_count / total_indices * 0.1, f"Market Indices ({indicator_name})") 
                
                raw_filename = f"{yf_symbol.upper()}_market_index_raw.csv"
                df = self._load_raw_data(raw_filename, sub_dir="market_indices", index_col_name=self.COL_DATE, parse_dates_col_name=self.COL_DATE, status_callback=status_callback)
                if df is None:
                    df = self.fetch_and_save_yfinance_data(yf_symbol, start_date_str, end_date_str,
                                                           data_type_name="market_index", status_callback=status_callback)
                if df is not None: market_indices_data_frames[indicator_name] = df

        fred_series_data_frames: Dict[str, pd.DataFrame] = {}
        if use_fred_data:
            self._status_update("--- Collecting FRED Economic Data ---", status_callback)
            if progress_callback: progress_callback(0.1, f"FRED Data") 
            
            temp_fred_dfs: Dict[str, pd.DataFrame] = {}
            all_fred_loaded_from_cache: bool = True
            for series_name, fred_id in self.FRED_SERIES_MAP.items():
                raw_filename_fred = f"{series_name}_fred_raw.csv"
                df_fred = self._load_raw_data(raw_filename_fred, sub_dir="fred", index_col_name=self.COL_DATE, parse_dates_col_name=self.COL_DATE, status_callback=status_callback)
                if df_fred is None: all_fred_loaded_from_cache = False; break # If one is missing, fetch all
                temp_fred_dfs[series_name] = df_fred
            
            if all_fred_loaded_from_cache and temp_fred_dfs:
                fred_series_data_frames = temp_fred_dfs
                self._status_update("All FRED data loaded from cache.", status_callback, indent_level=1)
            else:
                fred_series_data_frames = self.fetch_and_save_fred_data(self.FRED_SERIES_MAP, start_date_dt, end_date_dt, status_callback=status_callback)
            if progress_callback: progress_callback(0.15, f"FRED Data Complete") 

        self._status_update(f"\n--- Processing Individual Tickers ({len(companies_to_process)} total) ---", status_callback)
        successfully_processed_tickers: List[str] = []
        all_feature_columns: List[str] = [] # From the last successfully processed ticker
        base_progress: float = 0.15 
        total_ticker_progress: float = 1.0 - base_progress

        for i, company_info in enumerate(companies_to_process):
            ticker: str = company_info['ticker']
            company_name_for_search: str = company_info.get('name', ticker)
            
            current_ticker_progress_start: float = base_progress + (i / len(companies_to_process) * total_ticker_progress)
            current_ticker_progress_end: float = base_progress + ((i + 1) / len(companies_to_process) * total_ticker_progress)
            # Define stages within ticker processing for finer progress updates
            num_ticker_stages = 6 
            ticker_progress_step: float = (current_ticker_progress_end - current_ticker_progress_start) / num_ticker_stages

            self._status_update(f"Processing {ticker} ({i+1}/{len(companies_to_process)})", status_callback)
            if progress_callback: progress_callback(current_ticker_progress_start, f"Ticker {ticker} ({i+1}/{len(companies_to_process)}) - Fetching Stock")

            stock_raw_filename = f"{ticker.upper()}_stock_raw.csv"
            df_stock_raw = self._load_raw_data(stock_raw_filename, sub_dir="stock", index_col_name=self.COL_DATE, parse_dates_col_name=self.COL_DATE, status_callback=status_callback)
            if df_stock_raw is None:
                df_stock_raw = self.fetch_and_save_yfinance_data(ticker, start_date_str, end_date_str,
                                                                data_type_name="stock", status_callback=status_callback)
            if df_stock_raw is None or df_stock_raw.empty:
                self._status_update(f"Failed to load/fetch stock data for {ticker}. Skipping.", status_callback, True); continue

            if progress_callback: progress_callback(current_ticker_progress_start + ticker_progress_step, f"Ticker {ticker} - Calculating TA")
            df_with_tas = self._calculate_basic_technical_indicators(df_stock_raw, status_callback=status_callback, ticker=ticker)

            if progress_callback: progress_callback(current_ticker_progress_start + 2*ticker_progress_step, f"Ticker {ticker} - Engineering Base Features")
            df_featured_base = self._engineer_base_features(df_with_tas, ticker=ticker, status_callback=status_callback)

            df_reddit: Optional[pd.DataFrame] = None
            if use_reddit_sentiment and self.reddit_client: # Check if client was successfully setup
                if progress_callback: progress_callback(current_ticker_progress_start + 3*ticker_progress_step, f"Ticker {ticker} - Fetching Reddit")
                reddit_agg_raw_filename = f"{ticker.upper()}_reddit_sentiment_agg_raw.csv" # Aggregated is CSV
                df_reddit_loaded = self._load_raw_data(reddit_agg_raw_filename, sub_dir="reddit", index_col_name=self.COL_DATE, parse_dates_col_name=self.COL_DATE, status_callback=status_callback, is_json=False) 
                if df_reddit_loaded is None:
                    df_reddit = self.fetch_and_save_reddit_sentiment(ticker, company_name_for_search, start_date_dt, end_date_dt, status_callback=status_callback)
                else: df_reddit = df_reddit_loaded

            df_trends: Optional[pd.DataFrame] = None
            if use_google_trends:
                if progress_callback: progress_callback(current_ticker_progress_start + 4*ticker_progress_step, f"Ticker {ticker} - Fetching Trends")
                primary_kw_trends: str = ticker.upper() # Primary keyword for trend filename convention
                trends_raw_filename = f"{ticker.upper()}_trends_{primary_kw_trends.replace(' ','_')}_raw.csv"
                df_trends_loaded = self._load_raw_data(trends_raw_filename, sub_dir="trends", index_col_name=self.COL_DATE, parse_dates_col_name=self.COL_DATE, status_callback=status_callback)
                if df_trends_loaded is not None: # Try to reconstruct the primary trend df
                    if "Google_Trends" in df_trends_loaded.columns: df_trends = df_trends_loaded[["Google_Trends"]]
                    elif primary_kw_trends in df_trends_loaded.columns: df_trends = df_trends_loaded[[primary_kw_trends]].rename(columns={primary_kw_trends: "Google_Trends"})
                    if df_trends is not None and 'Trends_IsPartial' in df_trends_loaded.columns : df_trends['Trends_IsPartial'] = df_trends_loaded['Trends_IsPartial']
                
                if df_trends is None: # If not loaded or load failed to find correct column
                    df_trends = self.fetch_and_save_google_trends(ticker, company_name_for_search, start_date_str, end_date_str, status_callback=status_callback)

            if progress_callback: progress_callback(current_ticker_progress_start + 5*ticker_progress_step, f"Ticker {ticker} - Merging & Cleaning")
            df_merged = self._merge_all_data_sources(df_featured_base, market_indices_data_frames,
                                                    fred_series_data_frames, df_reddit, df_trends,
                                                    ticker=ticker, status_callback=status_callback)
            df_final_pre_advanced_feat = self._final_data_cleaning_and_typing(df_merged, ticker=ticker, status_callback=status_callback)

            if df_final_pre_advanced_feat.empty:
                self._status_update(f"Data for {ticker} became empty after cleaning. Skipping.", status_callback, True); continue

            if self.COL_TICKER not in df_final_pre_advanced_feat.columns: # Ensure Ticker column exists
                 df_final_pre_advanced_feat[self.COL_TICKER] = ticker

            if progress_callback: progress_callback(current_ticker_progress_end, f"Ticker {ticker} - Saving Processed") # End of this ticker's progress
            processed_filepath, feature_cols = self._save_processed_data(df_final_pre_advanced_feat, ticker, status_callback=status_callback)
            if processed_filepath:
                successfully_processed_tickers.append(ticker)
                all_feature_columns = feature_cols 

        duration_pipeline: float = time.time() - start_time_pipeline
        self._status_update(f"\nData collection pipeline finished in {duration_pipeline:.2f} seconds.", status_callback)
        self._status_update(f"Successfully processed {len(successfully_processed_tickers)} tickers: {', '.join(successfully_processed_tickers)}", status_callback)
        return successfully_processed_tickers, all_feature_columns

    def load_and_set_companies_list(self, num_companies: Optional[int] = None, status_callback: Callable[[str, bool], None] = print) -> List[Dict[str, str]]:
        latest_file: Optional[str] = None; latest_num_in_file: int = 0
        if os.path.exists(DATA_DIR_DC):
            try:
                all_top_files: List[str] = [f for f in os.listdir(DATA_DIR_DC) if f.startswith('top_') and f.endswith('_companies.csv')]
                if all_top_files:
                    nums: List[int] = []
                    for f_name in all_top_files:
                        try: parts = f_name.split('_'); num_str = parts[1]; nums.append(int(num_str))
                        except (IndexError, ValueError): continue # Skip malformed filenames
                    if nums: latest_num_in_file = max(nums)
                if latest_num_in_file > 0:
                    latest_file = os.path.join(DATA_DIR_DC, f'top_{latest_num_in_file}_companies.csv')
            except Exception as e: self._status_update(f"Warn: Error finding companies file: {e}", status_callback)

        df_companies: Optional[pd.DataFrame] = None
        if latest_file and os.path.exists(latest_file):
            try:
                df_temp = pd.read_csv(latest_file)
                # Standardize column names to 'ticker' and 'name'
                if 'Symbol' in df_temp.columns and 'Name' in df_temp.columns:
                    df_companies = df_temp[['Symbol', 'Name']].rename(columns={'Symbol': 'ticker', 'Name': 'name'})
                elif 'ticker' in df_temp.columns and 'name' in df_temp.columns: # If already in desired format
                    df_companies = df_temp[['ticker', 'name']]
                
                if df_companies is not None:
                    self._status_update(f"Loaded {len(df_companies)} companies from {os.path.basename(latest_file)}", status_callback)
            except Exception as e: self._status_update(f"Error reading {os.path.basename(latest_file)}: {e}", status_callback, True)

        if df_companies is None: # Fallback to default list
            self._status_update("Warn: Using default company list as no suitable file found or error in loading.", status_callback)
            default_companies_list: List[Dict[str, str]] = [
                {'ticker': 'AAPL', 'name': 'Apple Inc.'}, {'ticker': 'MSFT', 'name': 'Microsoft Corp.'},
                {'ticker': 'GOOGL', 'name': 'Alphabet Inc. (A)'},{'ticker': 'GOOG', 'name': 'Alphabet Inc. (C)'},
                {'ticker': 'AMZN', 'name': 'Amazon.com, Inc.'}, {'ticker': 'NVDA', 'name': 'NVIDIA Corp.'},
                {'ticker': 'META', 'name': 'Meta Platforms, Inc.'}, {'ticker': 'TSLA', 'name': 'Tesla, Inc.'}
            ] # Add more diverse defaults if needed
            df_companies = pd.DataFrame(default_companies_list)

        if num_companies is not None and df_companies is not None:
            df_companies = df_companies.head(num_companies)

        if df_companies is not None:
            self.companies = df_companies.to_dict('records')
        else: # Should not happen if default list is used
            self.companies = []
        return self.companies

# --- Main execution for testing DataCollector directly ---
if __name__ == "__main__":
    print("--- Running DataCollector Standalone Test ---")

    def cli_status_callback_main(message: str, is_error: bool = False) -> None:
        level = "ERROR" if is_error else "INFO"
        print(f"[{level}] {message}")

    def cli_progress_callback_main(progress_value: float, text_message: Optional[str] = None) -> None:
        bar_length: int = 30
        filled_length: int = int(bar_length * progress_value)
        bar: str = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\rProgress: |{bar}| {progress_value*100:.1f}% ({text_message or ""})      ', end='')
        if progress_value >= 1.0:
            print() 


    dc = DataCollector()
    # Test with a small number of companies from the list
    test_companies_list = dc.load_and_set_companies_list(num_companies=2, status_callback=cli_status_callback_main)
    if not test_companies_list:
        cli_status_callback_main("No companies to process for test. Exiting.", True)
        sys.exit(1)

    cli_status_callback_main(f"Test companies: {[c['ticker'] for c in test_companies_list]}", is_error=False)

    start_date_test_str = (datetime.now() - timedelta(days=3*365 + 90)).strftime('%Y-%m-%d') # Approx 3 years of data
    end_date_test_str = datetime.now().strftime('%Y-%m-%d')

    # Running the pipeline
    processed_tickers_list_res, final_features_res = dc.run_full_pipeline(
        companies_to_process=test_companies_list,
        start_date_str=start_date_test_str,
        end_date_str=end_date_test_str,
        use_market_indices=True,
        use_fred_data=True,
        use_reddit_sentiment=False, # Set to True and configure API in app.py/env for full test
        use_google_trends=True,
        progress_callback=cli_progress_callback_main,
        status_callback=cli_status_callback_main
    )

    print("\n--- DataCollector Standalone Test Complete ---")
    if processed_tickers_list_res:
        print(f"Successfully processed: {', '.join(processed_tickers_list_res)}")
        if final_features_res:
            print(f"Representative features from last processed file ({len(final_features_res)} total):")
            print(f"  {', '.join(final_features_res[:10])}{'...' if len(final_features_res) > 10 else ''}")

        # Verification step
        if os.path.exists(PROCESSED_DATA_DIR_DC) and processed_tickers_list_res:
            try:
                test_ticker_to_load_verify = processed_tickers_list_res[0]
                df_verify_load = pd.read_csv(os.path.join(PROCESSED_DATA_DIR_DC, f"{test_ticker_to_load_verify}_processed_data.csv"))
                print(f"\nVerified: Loaded {test_ticker_to_load_verify}_processed_data.csv, Shape: {df_verify_load.shape}")
                
                print("\nNaN check in verified file (sum per column for columns with NaNs):")
                nan_summary_verify = df_verify_load.isna().sum()
                print(nan_summary_verify[nan_summary_verify > 0])
                if not nan_summary_verify[nan_summary_verify > 0].empty:
                     print("\nWARNING: NaNs found in processed file. This indicates an issue in the cleaning/filling logic.")
                else:
                     print("  No NaNs found in the processed file. Good!")

            except Exception as e_verify:
                print(f"Error verifying processed file: {e_verify}")
                traceback.print_exc()
    else:
        print("No tickers were processed successfully in the test.")