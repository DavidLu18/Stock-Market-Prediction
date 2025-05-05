import os
import pandas as pd
import yfinance as yf
import praw
import numpy as np
import requests
# No sklearn imports here anymore (moved to model_training)
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.preprocessing import PolynomialFeatures
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import talib # Keep TA-Lib for primary indicator calculation
import time
import datetime
import traceback
from pytrends.request import TrendReq
from bs4 import BeautifulSoup
import pandas_datareader.data as web
import warnings
warnings.filterwarnings('ignore')

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models') # Define MODEL_DIR for consistency if needed

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    if not os.path.exists(dir_path):
        try: os.makedirs(dir_path)
        except OSError as e: print(f"Warning: Could not create directory {dir_path}: {e}")

class DataCollector:
    def __init__(self):
        self.companies = []
        self.reddit = None
        try: # Handle potential NLTK data download issue
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            print("NLTK VADER lexicon not found. Downloading...")
            import nltk
            try:
                nltk.download('vader_lexicon')
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                print(f"Error downloading VADER lexicon: {e}. Sentiment analysis will be limited.")
                self.sentiment_analyzer = None # Set to None if download fails
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36' # Updated User-Agent
        })
        self.data_dir = DATA_DIR
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR

    def setup_reddit_api(self, client_id, client_secret, user_agent):
        """Initialize Reddit API client"""
        # (Giữ nguyên logic, chỉ thêm kiểm tra self.sentiment_analyzer)
        if not self.sentiment_analyzer:
            print("Warning: Sentiment Analyzer not available, Reddit sentiment scores will be limited.")
        try:
            if not client_id or not client_secret or not user_agent: return False
            self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
            _ = self.reddit.subreddit("wallstreetbets").display_name # Test authentication
            print("Reddit API client initialized successfully")
            return True
        except Exception as e: print(f"Error initializing Reddit API: {str(e)}"); return False

    def get_top_companies(self, num_companies=50, progress_callback=None, status_callback=None):
        """Fetch top N US companies from S&P 500 list."""
        # (Giữ nguyên logic, đã khá ổn định)
        if status_callback: status_callback(f"Fetching S&P 500 list ({num_companies} companies)...")
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            if table is None: raise ValueError("Could not find S&P 500 table.")
            companies_data = []
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    ticker = cols[0].text.strip().replace('.', '-')
                    name = cols[1].text.strip()
                    if ticker not in ['BF-B', 'BRK-B']: companies_data.append({'ticker': ticker, 'name': name})
            if not companies_data: raise ValueError("No companies extracted.")
            self.companies = companies_data[:num_companies]
            if progress_callback: progress_callback(1, 1)
            companies_df = pd.DataFrame(self.companies)
            save_path = os.path.join(self.data_dir, f'top_{len(self.companies)}_companies.csv')
            companies_df.to_csv(save_path, index=False)
            if status_callback: status_callback(f"✓ Selected {len(self.companies)} companies (saved: {os.path.basename(save_path)}).")
            return self.companies
        except Exception as e:
            error_msg = f"Error getting top companies: {e}"
            if status_callback: status_callback(f"Error: {error_msg}")
            else: print(f"Error: {error_msg}")
            self.companies = []; return []

    def collect_stock_data(self, start_date, end_date, progress_callback=None, status_callback=None):
        """Collect historical OHLCV data using yf.Ticker().history() and save raw."""
        # (Giữ nguyên logic, đã khá ổn định, lưu file raw)
        try: start_dt=pd.to_datetime(start_date); end_dt=pd.to_datetime(end_date)
        except ValueError: msg = "Error: Invalid date format."; status_callback(msg); return {}
        if not self.companies: msg = "Error: No companies set."; status_callback(msg); return {}

        tickers_to_fetch = [c['ticker'] for c in self.companies]
        num_tickers = len(tickers_to_fetch)
        if status_callback: status_callback(f"Collecting stock data for {num_tickers} tickers...")

        all_stock_data = {}
        company_map = {c['ticker']: c['name'] for c in self.companies}

        for i, ticker_symbol in enumerate(tickers_to_fetch):
            if status_callback: status_callback(f"Fetching {ticker_symbol} ({i+1}/{num_tickers})...")
            try:
                ticker_obj = yf.Ticker(ticker_symbol)
                end_date_yf = (end_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                stock_data = ticker_obj.history(start=start_date, end=end_date_yf, interval="1d", actions=True, auto_adjust=False)

                if not stock_data.empty:
                    stock_data.reset_index(inplace=True)
                    date_col = next((col for col in ['Date', 'Datetime'] if col in stock_data.columns), None)
                    if date_col:
                         stock_data.rename(columns={date_col: 'Date'}, inplace=True)
                         stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
                    else: continue
                    stock_data['Ticker'] = ticker_symbol
                    stock_data['Company'] = company_map.get(ticker_symbol, ticker_symbol)
                    stock_data.columns = stock_data.columns.str.strip()
                    cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Ticker', 'Company']
                    stock_data = stock_data[[col for col in cols_to_keep if col in stock_data.columns]]

                    # <<< CẢI TIẾN: Xử lý lỗi/dòng trống cơ bản TRƯỚC KHI LƯU RAW >>>
                    stock_data[['Open','High','Low','Close','Volume']] = stock_data[['Open','High','Low','Close','Volume']].apply(pd.to_numeric, errors='coerce')
                    stock_data.dropna(subset=['Open','High','Low','Close','Volume'], how='any', inplace=True) # Yêu cầu tất cả phải hợp lệ

                    if not stock_data.empty:
                        all_stock_data[ticker_symbol] = stock_data
                        # Save raw data
                        ticker_raw_path = os.path.join(self.raw_data_dir, f"{ticker_symbol}_raw_stock.csv") # Đổi tên file raw
                        try: stock_data.to_csv(ticker_raw_path, index=False)
                        except Exception as save_err: status_callback(f"Error saving raw {ticker_symbol}: {save_err}")
                    else: status_callback(f"Warning: No valid OHLCV data after cleaning for {ticker_symbol}")
                else: status_callback(f"Warning: No data returned for {ticker_symbol}")
            except Exception as e: status_callback(f"Error fetching {ticker_symbol}: {e}")
            if progress_callback: progress_callback(i + 1, num_tickers)
            time.sleep(0.1) # Giảm nhẹ delay

        num_collected = len(all_stock_data)
        msg = f"✓ Collected & saved raw stock data for {num_collected}/{num_tickers} companies."
        if status_callback: status_callback(msg)
        return all_stock_data

    def collect_reddit_sentiment(self, start_date, end_date, subreddits=['stocks', 'wallstreetbets', 'investing'],
                               progress_callback=None, status_callback=None):
        """Collect Reddit sentiment, save raw data per ticker."""
        # (Cải tiến error handling, logging, save raw)
        try: start_dt = pd.to_datetime(start_date).tz_localize('UTC'); end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        except ValueError: status_callback("Error: Invalid date format for Reddit."); return {}
        if not self.companies: status_callback("Error: No companies for sentiment."); return {}
        if not self.reddit or not self.sentiment_analyzer: status_callback("Reddit API or Analyzer not ready."); return {}

        status_callback(f"Collecting Reddit sentiment ({len(self.companies)} companies)...")
        sentiment_data_raw = {company['ticker']: [] for company in self.companies} # Store list of posts per ticker
        search_terms = {c['ticker']: [c['ticker'], f"${c['ticker']}", c['name']] for c in self.companies}

        total_units = len(subreddits) * len(self.companies); completed_units = 0
        for subreddit_name in subreddits:
            status_callback(f"  Subreddit: r/{subreddit_name}")
            try: subreddit = self.reddit.subreddit(subreddit_name)
            except Exception as sub_err: status_callback(f"  Error accessing r/{subreddit_name}: {sub_err}. Skipping."); completed_units += len(self.companies); continue

            for company in self.companies:
                ticker = company['ticker']; query = " OR ".join([f'"{t}"' if ' ' in t else t for t in search_terms[ticker]])
                # status_callback(f"    Searching {ticker}...") # Bớt log
                try:
                    submissions = list(subreddit.search(query, sort='new', limit=1000)) # Limit to prevent excessive time
                    # status_callback(f"    Found {len(submissions)} potential posts for {ticker}") # Bớt log
                    for sub in submissions:
                        post_date = datetime.datetime.fromtimestamp(sub.created_utc, tz=datetime.timezone.utc)
                        if start_dt <= post_date <= end_dt:
                            post_text = sub.title + ' ' + sub.selftext
                            sentiment = self.sentiment_analyzer.polarity_scores(post_text)
                            sentiment_data_raw[ticker].append({
                                'Date': post_date.strftime('%Y-%m-%d'), # Save date string
                                'Compound': sentiment['compound'], 'Positive': sentiment['pos'],
                                'Neutral': sentiment['neu'], 'Negative': sentiment['neg']
                            })
                except Exception as search_err: status_callback(f"    Error searching {ticker}: {search_err}")
                completed_units += 1
                if progress_callback: progress_callback(completed_units / total_units if total_units > 0 else 0)
                time.sleep(0.5) # Shorter sleep between companies per sub

        # Save raw sentiment data per ticker
        num_saved = 0
        for ticker, posts in sentiment_data_raw.items():
            if posts:
                df_raw_sentiment = pd.DataFrame(posts)
                save_path = os.path.join(self.raw_data_dir, f"{ticker}_reddit_raw.csv")
                try: df_raw_sentiment.to_csv(save_path, index=False); num_saved += 1
                except Exception as e: status_callback(f"Error saving raw Reddit for {ticker}: {e}")
        status_callback(f"✓ Saved raw Reddit data for {num_saved} companies.")
        if progress_callback: progress_callback(1.0) # Mark as complete
        # Return None, as processing now reads from saved file
        return None

    def collect_google_trends(self, start_date, end_date, progress_callback=None, status_callback=None):
        """Collect Google Trends data using pytrends and save raw."""
        # (Cải tiến error handling, save raw)
        try:
            start_dt = pd.to_datetime(start_date); end_dt = pd.to_datetime(end_date)
            timeframe = f"{start_dt.strftime('%Y-%m-%d')} {end_dt.strftime('%Y-%m-%d')}"
        except ValueError: status_callback("Error: Invalid date format for Trends."); return {}
        if not self.companies: status_callback("Error: No companies for Trends."); return {}

        status_callback(f"Collecting Google Trends ({len(self.companies)} companies)...")
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        num_companies = len(self.companies)
        chunk_size = 5
        name_to_ticker = {c['name']: c['ticker'] for c in self.companies}
        all_keywords = [c['name'] for c in self.companies]
        num_chunks = (num_companies + chunk_size - 1) // chunk_size
        num_saved = 0

        for i in range(0, num_companies, chunk_size):
            chunk_keywords = all_keywords[i:min(i + chunk_size, num_companies)]
            chunk_tickers = [name_to_ticker[name] for name in chunk_keywords]
            current_chunk_num = (i // chunk_size) + 1
            if i > 0: time.sleep(20) # Reduced sleep between chunks
            if status_callback: status_callback(f"  Trends Chunk {current_chunk_num}/{num_chunks} ({', '.join(chunk_tickers)})...")
            if progress_callback: progress_callback(current_chunk_num -1, num_chunks)

            try:
                pytrends.build_payload(chunk_keywords, cat=0, timeframe=timeframe, geo='', gprop='')
                interest_df = pytrends.interest_over_time()
                if not interest_df.empty and not interest_df['isPartial'].iloc[0]:
                    for name in chunk_keywords:
                        ticker = name_to_ticker[name]
                        if name in interest_df.columns:
                            daily_trends = interest_df[[name]].resample('D').ffill() # Resample to daily
                            daily_trends.rename(columns={name: 'Interest'}, inplace=True)
                            daily_trends.index = pd.to_datetime(daily_trends.index).tz_localize(None) # Naive datetime index
                            df_to_save = daily_trends.reset_index().rename(columns={'index': 'Date'})
                            save_path = os.path.join(self.raw_data_dir, f"{ticker}_trends_raw.csv")
                            try: df_to_save.to_csv(save_path, index=False); num_saved += 1
                            except Exception as e: status_callback(f"Error saving raw Trends for {ticker}: {e}")
                        # else: status_callback(f"    Warning: '{name}' not in results for {ticker}.") # Bớt log
                elif not interest_df.empty and interest_df['isPartial'].iloc[0]: status_callback(f"    Warning: Partial data received for chunk {current_chunk_num}. Skipping.")
                # else: status_callback(f"    No Trends data found for chunk {current_chunk_num}.") # Bớt log

            except Exception as e: status_callback(f"    Error fetching trends chunk {current_chunk_num}: {e}")

        status_callback(f"✓ Google Trends completed. Saved raw data for {num_saved}/{num_companies} companies.")
        if progress_callback: progress_callback(1, 1) # Pass two args for completion
        # Return None, as processing reads from saved file
        return None

    def collect_fred_data(self, start_date, end_date, series_id='FEDFUNDS', progress_callback=None, status_callback=None):
        """Collect FRED data and save raw."""
        # (Giữ nguyên logic, đã khá ổn định, lưu file raw)
        if status_callback: status_callback(f"Collecting FRED data ({series_id})...")
        try:
            start_dt = pd.to_datetime(start_date); end_dt = pd.to_datetime(end_date)
            fred_data = web.DataReader(series_id, 'fred', start_dt, end_dt)
            if fred_data.empty: status_callback(f"Warning: No FRED data for {series_id}."); return None
            fred_data_daily = fred_data.resample('D').ffill()
            fred_data_daily.index = fred_data_daily.index.tz_localize(None)
            save_path = os.path.join(self.raw_data_dir, f"{series_id}_raw.csv")
            fred_data_daily.reset_index().rename(columns={'index': 'Date'}).to_csv(save_path, index=False)
            if status_callback: status_callback(f"✓ Collected & saved raw FRED data ({series_id}).")
            if progress_callback: progress_callback(1, 1)
            return None # Return None, processing reads file
        except Exception as e:
            status_callback(f"Error collecting FRED data ({series_id}): {e}")
            if progress_callback: progress_callback(1, 1)
            return None

    def _calculate_basic_indicators(self, df, status_callback=None):
        """Calculate basic technical indicators without TA-Lib (Fallback)."""
        # (Giữ nguyên logic fallback, đã khá ổn định)
        ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns and not df.empty else 'Unknown'
        if status_callback: status_callback(f"[{ticker}] Calculating basic indicators (TA-Lib fallback)...")
        try:
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                else: df[col] = 0 # Assign 0 if missing
            df.fillna(method='ffill', inplace=True); df.fillna(method='bfill', inplace=True); df.fillna(0, inplace=True) # Robust fill

            df['SMA_5'] = df['Close'].rolling(5, min_periods=1).mean()
            df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            delta = df['Close'].diff(); gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean(); loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-6); df['RSI'] = 100 - (100 / (1 + rs)); df['RSI'].fillna(50, inplace=True)
            ema12 = df['Close'].ewm(span=12, adjust=False).mean(); ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26; df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean(); df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            df['BB_Middle'] = df['SMA_20']; std = df['Close'].rolling(20, min_periods=1).std().fillna(0)
            df['BB_Upper'] = df['BB_Middle'] + (std * 2); df['BB_Lower'] = df['BB_Middle'] - (std * 2)
            n = 14; hh = df['High'].rolling(n, min_periods=1).max(); ll = df['Low'].rolling(n, min_periods=1).min(); denom_k = (hh - ll).replace(0, 1e-6)
            df['SlowK'] = 100 * ((df['Close'] - ll) / denom_k); df['SlowD'] = df['SlowK'].rolling(3, min_periods=1).mean()
            df['ADX'] = 0.0 # Placeholder
            denom_ad = (df['High'] - df['Low']).replace(0, 1e-6); mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / denom_ad; mfv = mfm.fillna(0) * df['Volume']
            df['Chaikin_AD'] = mfv.cumsum()
            df['OBV'] = np.where(df['Close'].diff() > 0, df['Volume'], np.where(df['Close'].diff() < 0, -df['Volume'], 0)).cumsum().fillna(0)
            hl = df['High'] - df['Low']; hc = abs(df['High'] - df['Close'].shift()); lc = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1); df['ATR'] = tr.rolling(14, min_periods=1).mean()
            hh14 = df['High'].rolling(14, min_periods=1).max(); ll14 = df['Low'].rolling(14, min_periods=1).min(); denom_w = (hh14 - ll14).replace(0, 1e-6)
            df['Williams_R'] = -100 * ((hh14 - df['Close']) / denom_w)
            df['ROC'] = df['Close'].pct_change(10, fill_method=None) * 100
            tp = (df['High'] + df['Low'] + df['Close']) / 3; sma_tp = tp.rolling(20, min_periods=1).mean(); mad = abs(tp - sma_tp).rolling(20, min_periods=1).mean()
            df['CCI'] = (tp - sma_tp) / (0.015 * mad).replace(0, 1e-6)
            df['Close_Open_Ratio'] = (df['Close'] / df['Open'].replace(0, 1e-6)).replace([np.inf, -np.inf], np.nan)
            df['High_Low_Diff'] = df['High'] - df['Low']
            df['Close_Prev_Ratio'] = (df['Close'] / df['Close'].shift(1).replace(0, 1e-6)).replace([np.inf, -np.inf], np.nan)
        except Exception as e: status_callback(f"[{ticker}] Error in basic indicator calc: {e}")
        return df

    def process_data(self, stock_data, progress_callback=None, status_callback=None):
        """Process data for each stock: Load raw, merge, clean, engineer basic features, save processed."""
        processed_tickers = []
        final_feature_cols_list = [] # Store feature list from LAST successful ticker

        if not stock_data: status_callback("Error: No stock data provided to process."); return [], []
        valid_tickers = list(stock_data.keys()); total_tickers = len(valid_tickers)
        if total_tickers == 0: status_callback("Error: No valid tickers in stock data."); return [], []
        status_callback("Starting per-ticker data processing...")

        for i, ticker in enumerate(valid_tickers):
            if status_callback: status_callback(f"Processing {ticker} ({i+1}/{total_tickers})...")
            if progress_callback: progress_callback(i, total_tickers)

            # --- Load RAW Stock Data ---
            stock_raw_path = os.path.join(self.raw_data_dir, f"{ticker}_raw_stock.csv")
            if not os.path.exists(stock_raw_path): status_callback(f"[{ticker}] Raw stock file not found. Skipping."); continue
            try:
                df = pd.read_csv(stock_raw_path, parse_dates=['Date'])
                df['Date'] = pd.to_datetime(df['Date']).dt.normalize() # Normalize date, keep as column
            except Exception as e: status_callback(f"[{ticker}] Error loading raw stock data: {e}. Skipping."); continue

            # --- Load and Prepare RAW Reddit ---
            reddit_df_agg = pd.DataFrame(columns=['Date', 'Compound', 'Positive', 'Neutral', 'Negative', 'Count']) # Default empty
            reddit_raw_path = os.path.join(self.raw_data_dir, f"{ticker}_reddit_raw.csv")
            if os.path.exists(reddit_raw_path):
                try:
                    sentiment_df_raw = pd.read_csv(reddit_raw_path, parse_dates=['Date'])
                    sentiment_df_raw['Date'] = pd.to_datetime(sentiment_df_raw['Date']).dt.normalize() # Normalize date
                    # Aggregate sentiment per day
                    reddit_df_agg = sentiment_df_raw.groupby('Date').agg(
                        Compound=('Compound', 'mean'), Positive=('Positive', 'mean'),
                        Neutral=('Neutral', 'mean'), Negative=('Negative', 'mean'),
                        Count=('Date', 'size')
                    ).reset_index()
                except Exception as e: status_callback(f"[{ticker}] Error loading/aggregating Reddit: {e}.")

            # --- Load and Prepare RAW Trends ---
            trends_df = pd.DataFrame(columns=['Date', 'Interest']) # Default empty
            trends_raw_path = os.path.join(self.raw_data_dir, f"{ticker}_trends_raw.csv")
            if os.path.exists(trends_raw_path):
                try:
                    trends_df_raw = pd.read_csv(trends_raw_path, parse_dates=['date']) # lowercase 'date' from file
                    trends_df_raw['Date'] = pd.to_datetime(trends_df_raw['date']).dt.normalize() # Normalize and rename
                    trends_df = trends_df_raw[['Date', 'Interest']].copy()
                except Exception as e: status_callback(f"[{ticker}] Error loading Trends: {e}.")

            # --- Load and Prepare RAW FRED ---
            fred_df = pd.DataFrame(columns=['Date', 'FEDFUNDS']) # Default empty
            fred_series_id = 'FEDFUNDS'
            fred_raw_path = os.path.join(self.raw_data_dir, f"{fred_series_id}_raw.csv")
            if os.path.exists(fred_raw_path):
                try:
                    fred_df_raw = pd.read_csv(fred_raw_path, parse_dates=['DATE'])
                    fred_df_raw['Date'] = pd.to_datetime(fred_df_raw['DATE']).dt.normalize() # Normalize date
                    fred_df = fred_df_raw[['Date', fred_series_id]].copy()
                    fred_df.rename(columns={fred_series_id: 'FEDFUNDS'}, inplace=True) # Ensure consistent column name
                except Exception as e: status_callback(f"[{ticker}] Error loading FRED: {e}.")

            # --- Merge all dataframes ---
            try:
                # Merge Reddit
                if not reddit_df_agg.empty:
                    df = pd.merge(df, reddit_df_agg, on='Date', how='left')
                else: # Ensure columns exist if no reddit data
                    for col in ['Compound', 'Positive', 'Neutral', 'Negative', 'Count']: df[col] = 0

                # Merge Trends
                if not trends_df.empty:
                    df = pd.merge(df, trends_df, on='Date', how='left')
                else: # Ensure column exists if no trends data
                    df['Interest'] = 0.0 # Keep 0 default for Interest if missing

                # Merge FRED
                if not fred_df.empty:
                    df = pd.merge(df, fred_df, on='Date', how='left')
                else: # Ensure column exists if no FRED data
                    df['FEDFUNDS'] = np.nan # Use NaN as default for FRED

                # Set index AFTER all merges
                df.set_index('Date', inplace=True)
                df = df[~df.index.duplicated(keep='first')] # Handle duplicate dates if any from merges
                df = df.sort_index()
                # status_callback(f"[{ticker}] Merged all raw data sources.") # Bớt log

            except Exception as merge_err:
                status_callback(f"[{ticker}] Error during merge: {merge_err}. Skipping.")
                continue

            # --- Post-Merge Cleaning ---
            # Ensure essential columns are numeric
            essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in essential_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=essential_cols, inplace=True) # Drop if OHLCV are NaN
            if df.empty: status_callback(f"[{ticker}] Empty after essential col cleaning. Skipping."); continue

            # Fill NaNs robustly (Interpolate -> ffill -> bfill -> 0 for most, specific for FRED)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            # Define columns needing specific ffill/bfill (rates, trends)
            cols_to_ffill_bfill = [c for c in [fred_series_id, 'Interest'] if c in df.columns]
            # Define other numeric columns for interpolation + zero fill
            cols_to_interpolate_zero = [c for c in numeric_cols if c not in cols_to_ffill_bfill]

            # Apply ffill/bfill first to rates/trends
            if cols_to_ffill_bfill:
                df[cols_to_ffill_bfill] = df[cols_to_ffill_bfill].fillna(method='ffill').fillna(method='bfill')

            # Apply interpolation and zero fill to the rest
            if cols_to_interpolate_zero:
                df[cols_to_interpolate_zero] = df[cols_to_interpolate_zero].interpolate(method='linear', limit_direction='both', axis=0)
                df[cols_to_interpolate_zero] = df[cols_to_interpolate_zero].fillna(method='ffill').fillna(method='bfill').fillna(0)

            # --- Outlier Clipping (Z-score > 3 on OHLCV) ---
            z_thresh = 3
            for col in essential_cols:
                if df[col].std() > 1e-6: # Avoid clipping if std is near zero
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df[col] = np.where(z_scores > z_thresh,
                                      np.sign(df[col] - df[col].mean()) * z_thresh * df[col].std() + df[col].mean(),
                                      df[col])

            # --- Indicator Calculation ---
            # status_callback(f"[{ticker}] Calculating indicators...") # Bớt log
            try:
                import talib
                min_period = 50
                if len(df) < min_period:
                     status_callback(f"[{ticker}] Warning: Data length ({len(df)}) < {min_period}. Using basic indicators.")
                     df = self._calculate_basic_indicators(df, status_callback)
                else: # Use TA-Lib
                    op = df['Open'].values.astype(float); hi = df['High'].values.astype(float); lo = df['Low'].values.astype(float); cl = df['Close'].values.astype(float); vo = df['Volume'].values.astype(float)
                    # Calculate all indicators safely
                    df['SMA_5'] = talib.SMA(cl, 5); df['SMA_20'] = talib.SMA(cl, 20); df['SMA_50'] = talib.SMA(cl, 50)
                    df['EMA_5'] = talib.EMA(cl, 5); df['EMA_20'] = talib.EMA(cl, 20)
                    df['RSI'] = talib.RSI(cl, 14)
                    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(cl, 12, 26, 9)
                    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(cl, 20, 2, 2, 0)
                    df['SlowK'], df['SlowD'] = talib.STOCH(hi, lo, cl, 5, 3, 0, 3, 0)
                    df['ADX'] = talib.ADX(hi, lo, cl, 14)
                    df['Chaikin_AD'] = talib.AD(hi, lo, cl, vo)
                    df['OBV'] = talib.OBV(cl, vo)
                    df['ATR'] = talib.ATR(hi, lo, cl, 14)
                    df['Williams_R'] = talib.WILLR(hi, lo, cl, 14)
                    df['ROC'] = talib.ROC(cl, 10)
                    df['CCI'] = talib.CCI(hi, lo, cl, 20)
                    df['Close_Open_Ratio'] = (df['Close'] / df['Open'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                    df['High_Low_Diff'] = df['High'] - df['Low']
                    df['Close_Prev_Ratio'] = (df['Close'] / df['Close'].shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            except ImportError: df = self._calculate_basic_indicators(df, status_callback)
            except Exception as e: status_callback(f"[{ticker}] Error in indicator calc: {e}"); df = self._calculate_basic_indicators(df, status_callback)

            # --- Basic Feature Engineering (Lags, Vol, Time) ---
            lags = [1, 3, 5]
            for lag in lags:
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
                df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
                for col in ['Compound', 'Interest', fred_series_id]: # Lag sentiment, trends, macro if present
                    if col in df.columns: df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
            df['Volatility_20D'] = df['Close'].rolling(window=20, min_periods=5).std() # Require min 5 periods
            df['Day_Of_Week'] = df.index.dayofweek

            # --- Final Cleaning before Target ---
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True) # Final fill with 0

            # --- Create 1-Day Target for Processed File (N-day target created in model_training) ---
            if len(df) > 1:
                df['Close_Next'] = df['Close'].shift(-1)
                df['Price_Change'] = df['Close_Next'] - df['Close']
                df['Price_Increase'] = (df['Price_Change'] > 0).astype(int)
                df.dropna(subset=['Close_Next'], inplace=True) # Drop last row where target is NaN
            else: df['Price_Increase'] = 0 # Assign default if only 1 row

            if df.empty: status_callback(f"[{ticker}] Empty after final cleaning/target. Skipping save."); continue

            # --- Save Processed Data ---
            processed_file_path = os.path.join(self.processed_data_dir, f"{ticker}_processed_data.csv")
            df_to_save = df.reset_index()
            try:
                df_to_save.to_csv(processed_file_path, index=False)
                processed_tickers.append(ticker)
                final_feature_cols_list = [col for col in df_to_save.columns if col not in ['Date', 'Ticker', 'Company', 'Close_Next', 'Price_Change', 'Price_Increase']]
                # status_callback(f"[{ticker}] ✓ Saved processed data.") # Bớt log
            except Exception as e: status_callback(f"[{ticker}] Error saving processed data: {e}")

        if progress_callback: progress_callback(total_tickers, total_tickers)
        status_callback(f"✓ Data processing complete. Processed {len(processed_tickers)}/{total_tickers} tickers.")
        return processed_tickers, final_feature_cols_list # Return features from last success

# (main function for CLI execution - giữ nguyên logic)
def main():
    collector = DataCollector()
    start = '2022-01-01'; end = datetime.datetime.now().strftime('%Y-%m-%d'); num = 10
    def cli_progress(c, t): print(f"\rProgress: {c}/{t}", end='')
    def cli_status(m): print(f"\nStatus: {m}")
    companies = collector.get_top_companies(num_companies=num, status_callback=cli_status)
    if not companies: print("Failed to get companies."); return
    stock_data = collector.collect_stock_data(start_date=start, end_date=end, progress_callback=cli_progress, status_callback=cli_status)
    if not stock_data: print("Failed to collect stock data."); return
    # Add calls to collect Reddit/Trends/FRED raw data here if needed for CLI
    collector.collect_fred_data(start_date=start, end_date=end, status_callback=cli_status)
    # ... Add Reddit/Trends calls here ...
    processed_tickers, features = collector.process_data(stock_data, progress_callback=cli_progress, status_callback=cli_status)
    print(f"\nCLI Processing Complete. Processed {len(processed_tickers)} tickers.")
    if processed_tickers: print(f"Features: {features}")

if __name__ == "__main__":
    main()
