"""
Stock Price Prediction System - Data Collection Module
This module handles collecting stock data, Reddit sentiment, and Google Trends data
for the top 50 US companies by market cap, and performs data processing and feature engineering.
"""

import os
import pandas as pd
import yfinance as yf
import praw
import numpy as np
import requests
# Removed MinMaxScaler, StandardScaler is used in model_training
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import PolynomialFeatures
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import talib
import time
import datetime
import traceback # Import traceback
from pytrends.request import TrendReq # Import pytrends restored
from bs4 import BeautifulSoup
import pandas_datareader.data as web # Added for FRED data
# Removed Selenium imports
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from selenium.webdriver.chrome.options import Options as ChromeOptions
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import json # For parsing data from script tags
# import re # For extracting JSON from script
# from selenium.webdriver.common.by import By
# from webdriver_manager.chrome import ChromeDriverManager
import warnings
warnings.filterwarnings('ignore')

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models') # Define MODEL_DIR here as well

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class DataCollector:
    def __init__(self):
        # Removed hardcoded start/end dates
        self.companies = [] # This will be populated by get_top_companies or set externally
        self.reddit = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Add a requests session with headers - KEEPING for Wikipedia/other non-yf requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        })
        # Define data_dir attributes (using new paths)
        self.data_dir = DATA_DIR
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR


    def setup_reddit_api(self, client_id, client_secret, user_agent):
        """Initialize Reddit API client"""
        try:
            if not client_id or not client_secret or not user_agent:
                print("Error: Missing Reddit API credentials")
                return False

            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )

            # Test that credentials work by making a simple API call
            subreddit = self.reddit.subreddit("wallstreetbets")
            # Just access a property to force authentication
            _ = subreddit.display_name

            print("Reddit API client initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing Reddit API: {str(e)}")
            return False

    def get_top_companies(self, num_companies=50, progress_callback=None, status_callback=None):
        """Fetch top N US companies from the S&P 500 list (faster, skips market cap fetch)."""
        if status_callback:
            status_callback(f"Fetching S&P 500 list to select top {num_companies} companies...")
        else:
            print(f"Fetching S&P 500 list to select top {num_companies} companies...")

        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            # Use the session for the request
            response = self.session.get(url, timeout=15) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')

            table = soup.find('table', {'class': 'wikitable'})
            if table is None:
                 raise ValueError("Could not find the S&P 500 table on the Wikipedia page.")

            companies_data = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) >= 2:
                    ticker = cols[0].text.strip().replace('.', '-') # Replace dot with dash for yfinance compatibility
                    name = cols[1].text.strip()
                    # Skip tickers with known issues if necessary
                    if ticker not in ['BF-B', 'BRK-B']: # Skip problematic tickers like Berkshire Hathaway B
                         companies_data.append({'ticker': ticker, 'name': name})

            if not companies_data:
                 raise ValueError("No companies extracted from the Wikipedia table.")

            # Directly take the top N companies from the list (Wikipedia list is somewhat ordered)
            self.companies = companies_data[:num_companies] # Store the selected companies internally

            if progress_callback: # Simulate progress completion
                progress_callback(1, 1)

            # Save the selected companies list (useful for reference)
            companies_df = pd.DataFrame(self.companies)
            save_path = os.path.join(self.data_dir, f'top_{len(self.companies)}_companies.csv')
            companies_df.to_csv(save_path, index=False)

            if status_callback:
                status_callback(f"Successfully selected {len(self.companies)} companies from S&P 500 list (saved to {os.path.basename(save_path)}).")
            else:
                print(f"Successfully selected {len(self.companies)} companies from S&P 500 list.")

            return self.companies # Return the list of dicts

        except requests.exceptions.RequestException as e:
             error_msg = f"Network error fetching S&P 500 list: {e}"
             if status_callback: status_callback(f"Error: {error_msg}")
             else: print(f"Error: {error_msg}")
             self.companies = []
             return [] # Return empty list on error
        except Exception as e:
            error_msg = f"Error getting top companies: {e}"
            if status_callback:
                status_callback(f"Error: {error_msg}")
            else:
                print(f"Error: {error_msg}")
            traceback.print_exc()
            self.companies = [] # Ensure companies list is empty on error
            return [] # Return empty list on error

    def collect_stock_data(self, start_date, end_date, progress_callback=None, status_callback=None):
        """Collect historical OHLCV data using yf.Ticker().history() for robustness."""
        # Validate dates
        try:
            pd.to_datetime(start_date)
            pd.to_datetime(end_date)
        except ValueError:
            msg = "Error: Invalid start or end date format. Use YYYY-MM-DD."
            if status_callback: status_callback(msg)
            else: print(msg)
            return {}

        # Use the internal self.companies list
        if not self.companies:
             msg = "Error: No companies set for data collection (self.companies is empty)."
             if status_callback: status_callback(msg)
             else: print(msg)
             return {}

        tickers_to_fetch = [company['ticker'] for company in self.companies]
        if not tickers_to_fetch:
            msg = "Error: Company list provided but no valid tickers found."
            if status_callback: status_callback(msg)
            else: print(msg)
            return {}

        num_tickers = len(tickers_to_fetch)
        if status_callback:
            status_callback(f"Collecting historical stock data for {num_tickers} companies individually using Ticker.history()...")
        else:
            print(f"Collecting historical stock data for {num_tickers} companies individually using Ticker.history()...")

        all_stock_data = {}
        company_map = {c['ticker']: c['name'] for c in self.companies}
        successful_tickers = 0

        for i, ticker_symbol in enumerate(tickers_to_fetch):
            if status_callback:
                status_callback(f"Fetching data for {ticker_symbol} ({i+1}/{num_tickers})...")

            try:
                # Create Ticker object WITHOUT the custom session
                ticker_obj = yf.Ticker(ticker_symbol)

                # Fetch history using the history method
                # Add 1 day to end_date because yfinance end is exclusive
                try:
                    end_date_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                    end_date_yf = end_date_dt.strftime('%Y-%m-%d')
                except ValueError:
                    if status_callback: status_callback(f"[{ticker_symbol}] Error parsing end_date '{end_date}'. Using original.")
                    end_date_yf = end_date

                stock_data = ticker_obj.history(
                    start=start_date,
                    end=end_date_yf, # Use adjusted end date
                    interval="1d",
                    actions=True,    # Explicitly keep actions
                    auto_adjust=False # Set auto_adjust to False to potentially preserve start date
                )
                # Log the date range returned directly from yfinance
                if status_callback and not stock_data.empty:
                    raw_start = stock_data.index.min().strftime('%Y-%m-%d')
                    raw_end = stock_data.index.max().strftime('%Y-%m-%d')
                    status_callback(f"[{ticker_symbol}] yfinance raw data range: {raw_start} to {raw_end} ({len(stock_data)} rows)")
                if not stock_data.empty:
                    # Reset index to make Date a column
                    stock_data.reset_index(inplace=True)
                    # Ensure the column is named 'Date' and is datetime
                    if 'Date' not in stock_data.columns and 'index' in stock_data.columns:
                         stock_data.rename(columns={'index': 'Date'}, inplace=True)
                    elif 'Datetime' in stock_data.columns: # Sometimes history index is Datetime
                         stock_data.rename(columns={'Datetime': 'Date'}, inplace=True)

                    if 'Date' in stock_data.columns:
                        # Convert timezone-aware DatetimeIndex to naive datetime objects (YYYY-MM-DD)
                        # This avoids potential timezone issues later if mixing naive/aware
                        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
                    else:
                         if status_callback: status_callback(f"Warning: Could not find or convert Date column for {ticker_symbol}")
                         continue # Skip if no date column

                    # Add company identifier
                    stock_data['Ticker'] = ticker_symbol
                    stock_data['Company'] = company_map.get(ticker_symbol, ticker_symbol)
                    # Clean column names
                    stock_data.columns = stock_data.columns.str.strip()

                    # Select only standard OHLCV columns + Dividends/Stock Splits if needed
                    cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Ticker', 'Company']
                    stock_data = stock_data[[col for col in cols_to_keep if col in stock_data.columns]]


                    all_stock_data[ticker_symbol] = stock_data
                    successful_tickers += 1

                    if status_callback:
                        status_callback(f"✓ Successfully fetched {len(stock_data)} records for {ticker_symbol}")

                    # Save raw data
                    try:
                        ticker_raw_path = os.path.join(self.raw_data_dir, f"{ticker_symbol}_raw_data.csv")
                        stock_data.to_csv(ticker_raw_path, index=False)
                    except Exception as save_err:
                         if status_callback:
                             status_callback(f"Error saving raw data for {ticker_symbol}: {save_err}")
                else:
                    if status_callback:
                        status_callback(f"Warning: No data returned by Ticker.history() for {ticker_symbol}")

            except Exception as e:
                if status_callback:
                    # Log the specific error for this ticker
                    status_callback(f"Error fetching history for {ticker_symbol}: {e}")
                # traceback.print_exc() # Optionally print full traceback to console/log file

            if progress_callback:
                progress_callback(i + 1, num_tickers)

            time.sleep(0.2) # Slightly longer delay between individual requests

        num_collected = len(all_stock_data)
        msg = f"Successfully collected and saved raw stock data for {num_collected}/{num_tickers} requested companies."
        if status_callback: status_callback(msg)
        else: print(msg)
        if num_collected < num_tickers:
             missing_tickers = [t for t in tickers_to_fetch if t not in all_stock_data]
             if status_callback: status_callback(f"Tickers possibly missing data: {missing_tickers}")

        return all_stock_data

    def collect_reddit_sentiment(self, start_date, end_date, subreddits=['stocks', 'wallstreetbets', 'investing'],
                               progress_callback=None, status_callback=None):
        """Collect sentiment data from Reddit for the companies stored in self.companies"""
        # Validate dates
        try:
            # Convert to datetime and make timezone aware (UTC) for comparison
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        except ValueError:
            if status_callback:
                status_callback("Error: Invalid start or end date format for Reddit. Use YYYY-MM-DD.")
            else:
                print("Error: Invalid start or end date format for Reddit. Use YYYY-MM-DD.")
            return {}

        if not self.companies:
             if status_callback:
                 status_callback("Error: No companies set for sentiment collection.")
             else:
                 print("Error: No companies set for sentiment collection.")
             return {}

        if not self.reddit:
            if status_callback:
                status_callback("Reddit API client not initialized. Please run setup_reddit_api() first.")
            else:
                print("Reddit API client not initialized. Please run setup_reddit_api() first.")
            return {}

        if status_callback:
            status_callback(f"Collecting Reddit sentiment data for {len(self.companies)} companies...")
        else:
            print("Collecting Reddit sentiment data...")

        sentiment_data = {}

        # Generate list of search terms for each company
        search_terms = {}
        for company in self.companies:
            ticker = company['ticker']
            name = company['name']
            # Create search terms: ticker, $ticker, company name
            terms = [ticker, f"${ticker}", name]
            # Add variations if needed (e.g., 'Alphabet' for 'GOOGL')
            if ticker == 'GOOGL': terms.append('Alphabet')
            if ticker == 'META': terms.append('Facebook')
            search_terms[ticker] = terms

        # Determine years to search / Calculate total work units
        start_year = start_dt.year
        end_year = end_dt.year
        years_to_search = range(start_year, end_year + 1)
        # Calculate total work units based on subreddits and companies
        total_search_units = len(subreddits) * len(self.companies)
        completed_search_units = 0
        # --- Debugging: Log date range ---
        if status_callback:
            status_callback(f"[DEBUG] Reddit collection range: {start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        # --- End Debugging ---

        # Iterate through subreddits, then companies (Removed year loop)
        for subreddit_name in subreddits:
            if status_callback: status_callback(f"  Analyzing Subreddit: r/{subreddit_name}")
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
            except Exception as sub_err:
                if status_callback: status_callback(f"  Error accessing subreddit r/{subreddit_name}: {sub_err}. Skipping.")
                # Increment progress for skipped companies in this subreddit
                completed_search_units += len(self.companies)
                if progress_callback: progress_callback(completed_search_units / total_search_units if total_search_units > 0 else 0)
                continue # Skip to next subreddit

            for company in self.companies:
                ticker = company['ticker']
                company_search_terms = search_terms[ticker] # Get terms like ['AAPL', '$AAPL', 'Apple']
                # Construct a simple OR query for the company's terms
                # Escape quotes within names if necessary (though less common for company names)
                query_parts = [f'"{term}"' if ' ' in term else term for term in company_search_terms]
                query = " OR ".join(query_parts)
                # --- Debugging: Print exact search query and date range ---
                if status_callback:
                    status_callback(f"    Searching for '{ticker}' with query: '{query}' in r/{subreddit_name}...")
                    status_callback(f"    Target date range (UTC): {start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                # --- End Debugging ---

                try:
                    # Search for the specific company within the subreddit (removed time_filter)
                    # Note: PRAW search without time_filter defaults to 'all' but might be slow/limited by Reddit.
                    # Consider adding limit=1000 or similar if performance is an issue, but this might miss older posts.
                    submissions = list(subreddit.search(query, sort='new', limit=None)) # Fetch as many as possible
                    found_count = len(submissions)
                    if status_callback: status_callback(f"    PRAW Search returned {found_count} posts for '{ticker}' in r/{subreddit_name}.")
                    # --- Debugging: Log first/last post dates from PRAW search result ---
                    if submissions and status_callback:
                        first_post_dt_praw = datetime.datetime.fromtimestamp(submissions[-1].created_utc, tz=datetime.timezone.utc)
                        last_post_dt_praw = datetime.datetime.fromtimestamp(submissions[0].created_utc, tz=datetime.timezone.utc)
                        status_callback(f"    PRAW Result Date Range (UTC): {first_post_dt_praw.strftime('%Y-%m-%d %H:%M:%S %Z')} to {last_post_dt_praw.strftime('%Y-%m-%d %H:%M:%S %Z')} (from {found_count} posts)")
                    # --- End Debugging ---

                    processed_posts_for_company = 0
                    for i_sub, submission in enumerate(submissions): # Add index for logging
                        post_date = datetime.datetime.fromtimestamp(submission.created_utc, tz=datetime.timezone.utc)

                        # --- Debugging: Log post date before check ---
                        # Log only occasionally to avoid excessive output
                        # if i_sub % 100 == 0 and status_callback: # Reduce frequency if needed
                        #      status_callback(f"    Checking post {i_sub+1}/{found_count} with date: {post_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        # --- End Debugging ---

                        # Double-check if the post falls within the *exact* requested range (start_dt to end_dt)
                        # This is needed because time_filter='year' is less precise than the original timestamp query
                        in_range = (start_dt <= post_date <= end_dt)
                        if not in_range:
                            # --- Debugging: Log skipped post with reason ---
                            if i_sub < 5 and status_callback: # Log first few skipped posts for debugging
                                 skip_reason = "before start" if post_date < start_dt else "after end"
                                 status_callback(f"      -> Skipping post {i_sub+1} (Date {post_date.strftime('%Y-%m-%d %H:%M:%S %Z')} is {skip_reason} target range)")
                            # --- End Debugging ---
                            continue
                        # --- Debugging: Log first accepted post date ---
                        if processed_posts_for_company == 0 and status_callback:
                            status_callback(f"      -> First accepted post date: {post_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        # --- End Debugging ---

                        post_text = submission.title + ' ' + submission.selftext
                        sentiment = self.sentiment_analyzer.polarity_scores(post_text)
                        date_key = post_date.strftime('%Y-%m-%d')

                        # Store sentiment data for this specific ticker
                        if ticker not in sentiment_data:
                            sentiment_data[ticker] = {}
                        if date_key not in sentiment_data[ticker]:
                            sentiment_data[ticker][date_key] = {'compound': [], 'pos': [], 'neu': [], 'neg': [], 'count': 0}

                        sentiment_data[ticker][date_key]['compound'].append(sentiment['compound'])
                        sentiment_data[ticker][date_key]['pos'].append(sentiment['pos'])
                        sentiment_data[ticker][date_key]['neu'].append(sentiment['neu'])
                        sentiment_data[ticker][date_key]['neg'].append(sentiment['neg'])
                        sentiment_data[ticker][date_key]['count'] += 1
                        processed_posts_for_company += 1
                        # --- Debugging: Check if specific date is added ---
                        if date_key == '2022-04-05' and status_callback: # Example date
                            status_callback(f"[DEBUG] Added data for {ticker} on {date_key}. Current count: {sentiment_data[ticker][date_key]['count']}")
                        # --- End Debugging ---
                        # Add status update within the loop for long searches
                        if processed_posts_for_company % 100 == 0 and status_callback:
                            status_callback(f"    Processed {processed_posts_for_company} posts for '{ticker}' in r/{subreddit_name}...")

                    if status_callback and found_count > 0:
                         status_callback(f"    Processed {processed_posts_for_company} relevant posts for '{ticker}'.")

                except praw.exceptions.PRAWException as praw_err:
                     if status_callback: status_callback(f"    PRAW Error searching for {ticker} in r/{subreddit_name}: {praw_err}. Skipping company.")
                except Exception as search_err:
                     if status_callback: status_callback(f"    Error searching for {ticker} in r/{subreddit_name}: {search_err}. Skipping company.")
                     traceback.print_exc() # Print traceback for unexpected errors

                # Update progress after each company search attempt
                completed_search_units += 1
                if progress_callback:
                    # Calculate overall progress fraction (0.0 to 1.0)
                    overall_progress = completed_search_units / total_search_units if total_search_units > 0 else 0
                    progress_callback(overall_progress) # Pass the calculated fraction
                time.sleep(1) # Add a small delay between company searches within a subreddit
        # Removed year loop delay


        # --- Debugging: Check keys in sentiment_data before averaging ---
        if status_callback:
            if sentiment_data:
                status_callback(f"[DEBUG] Checking {len(sentiment_data)} tickers in sentiment_data BEFORE averaging...")
                for debug_ticker, date_data in sentiment_data.items():
                    try:
                        found_dates = sorted(date_data.keys())
                        if found_dates:
                            status_callback(f"[DEBUG]   Ticker {debug_ticker}: {len(found_dates)} dates from {found_dates[0]} to {found_dates[-1]}")
                        else:
                            status_callback(f"[DEBUG]   Ticker {debug_ticker}: Found in sentiment_data, but NO dates recorded.")
                    except Exception as debug_err:
                        status_callback(f"[DEBUG]   Error checking dates for ticker {debug_ticker}: {debug_err}")
            else:
                status_callback(f"[DEBUG] sentiment_data dictionary is empty BEFORE averaging.")
        # --- End Debugging ---


        # --- Debugging: Check keys in sentiment_data before averaging ---
        if status_callback and self.companies: # Check if companies list is not empty
             debug_ticker = self.companies[0]['ticker'] # Use the first ticker for debugging
             if debug_ticker in sentiment_data:
                 try:
                     found_dates = sorted(sentiment_data[debug_ticker].keys())
                     if found_dates:
                         status_callback(f"[DEBUG] Dates found for {debug_ticker} BEFORE averaging: {len(found_dates)} dates from {found_dates[0]} to {found_dates[-1]}")
                     else:
                         status_callback(f"[DEBUG] Ticker {debug_ticker} found in sentiment_data, but NO dates recorded.")
                 except Exception as debug_err:
                     status_callback(f"[DEBUG] Error checking sentiment_data keys for {debug_ticker}: {debug_err}")
             elif status_callback:
                  status_callback(f"[DEBUG] Ticker {debug_ticker} not found in sentiment_data before averaging.")
        # --- End Debugging ---

        # --- Debugging: Check keys in sentiment_data before averaging ---
        if status_callback and self.companies: # Check if companies list is not empty
             # Use the first ticker found in the collected data for debugging, or the first requested ticker
             debug_ticker = next(iter(sentiment_data.keys())) if sentiment_data else (self.companies[0]['ticker'] if self.companies else None)
             if debug_ticker and debug_ticker in sentiment_data:
                 try:
                     found_dates = sorted(sentiment_data[debug_ticker].keys())
                     if found_dates:
                         status_callback(f"[DEBUG] Dates found for {debug_ticker} BEFORE averaging: {len(found_dates)} dates from {found_dates[0]} to {found_dates[-1]}")
                     else:
                         status_callback(f"[DEBUG] Ticker {debug_ticker} found in sentiment_data, but NO dates recorded BEFORE averaging.")
                 except Exception as debug_err:
                     status_callback(f"[DEBUG] Error checking sentiment_data keys for {debug_ticker} BEFORE averaging: {debug_err}")
             elif debug_ticker:
                  status_callback(f"[DEBUG] Ticker {debug_ticker} not found in sentiment_data BEFORE averaging.")
             else:
                  status_callback(f"[DEBUG] No companies or sentiment data available to check BEFORE averaging.")
        # --- End Debugging ---

        # Calculate daily average sentiment
        if status_callback:
            status_callback("Processing collected sentiment data...")

        avg_sentiment_data = {}
        for ticker, dates in sentiment_data.items():
            avg_sentiment_data[ticker] = {}
            for date, values in dates.items():
                if values['count'] > 0: # Ensure count is positive before calculating mean
                    avg_sentiment_data[ticker][date] = {
                        'compound': np.mean(values['compound']),
                        'pos': np.mean(values['pos']),
                        'neu': np.mean(values['neu']),
                        'neg': np.mean(values['neg']),
                        'count': values['count']
                    }

        # Save individual raw sentiment files per ticker
        num_tickers_with_sentiment = 0
        for ticker, dates_data in avg_sentiment_data.items():
            if not dates_data: continue # Skip if no dates found for this ticker

            sentiment_rows = []
            for date, values in dates_data.items():
                sentiment_rows.append({
                    'Date': date,
                    # 'Ticker': ticker, # Ticker is implied by filename
                    'Compound': values['compound'],
                    'Positive': values['pos'],
                    'Neutral': values['neu'],
                    'Negative': values['neg'],
                    'Count': values['count']
                })

            if sentiment_rows:
                ticker_sentiment_df = pd.DataFrame(sentiment_rows)
                ticker_sentiment_df['Date'] = pd.to_datetime(ticker_sentiment_df['Date']) # Ensure Date is datetime
                ticker_sentiment_df.sort_values('Date', inplace=True) # Sort by date
                save_path = os.path.join(self.raw_data_dir, f"{ticker}_reddit_raw.csv")
                # --- Debugging: Check DataFrame before saving ---
                if status_callback:
                    status_callback(f"[{ticker}] Preparing to save {len(sentiment_rows)} rows to CSV.")
                    if not ticker_sentiment_df.empty:
                        try:
                            min_date_in_df = ticker_sentiment_df['Date'].min()
                            max_date_in_df = ticker_sentiment_df['Date'].max()
                            status_callback(f"[{ticker}] DataFrame date range before save: {min_date_in_df.strftime('%Y-%m-%d')} to {max_date_in_df.strftime('%Y-%m-%d')}")
                            status_callback(f"[{ticker}] DataFrame shape before save: {ticker_sentiment_df.shape}")
                        except Exception as debug_err:
                             status_callback(f"[{ticker}] Error getting debug info before save: {debug_err}")
                    else:
                        status_callback(f"[{ticker}] DataFrame is empty before save.")
                # --- End Debugging ---
                try:
                    ticker_sentiment_df.to_csv(save_path, index=False)
                    num_tickers_with_sentiment += 1
                    if status_callback: status_callback(f"[{ticker}] Saved raw Reddit sentiment to {os.path.basename(save_path)}")
                except Exception as save_err:
                    if status_callback: status_callback(f"[{ticker}] Error saving raw Reddit sentiment: {save_err}")
            elif status_callback:
                 status_callback(f"[{ticker}] No sentiment rows generated despite data presence (check logic).")


        msg = f"Finished collecting Reddit sentiment. Saved raw data for {num_tickers_with_sentiment} companies."
        if status_callback: status_callback(msg)
        else: print(msg)

        # Ensure we report 100% completion
        if progress_callback:
            progress_callback(1.0) # Pass 1.0 for 100% completion

        return avg_sentiment_data # Return dict format for merging

    # Removed _init_webdriver function

    def collect_google_trends(self, start_date, end_date, progress_callback=None, status_callback=None):
        """Collect Google Trends data for the companies stored in self.companies using pytrends"""
        # Reverted to pytrends implementation (without retry args)
        try:
            # Validate dates and format for pytrends
            try:
                start_dt_obj = pd.to_datetime(start_date)
                end_dt_obj = pd.to_datetime(end_date)
                # Format for pytrends: 'YYYY-MM-DD YYYY-MM-DD'
                timeframe = f"{start_dt_obj.strftime('%Y-%m-%d')} {end_dt_obj.strftime('%Y-%m-%d')}"
            except ValueError:
                if status_callback:
                    status_callback("Error: Invalid start or end date format for Google Trends. Use YYYY-MM-DD.")
                else:
                    print("Error: Invalid start or end date format for Google Trends. Use YYYY-MM-DD.")
                return {}

            if not self.companies:
                 if status_callback:
                     status_callback("Error: No companies set for Google Trends collection.")
                 else:
                     print("Error: No companies set for Google Trends collection.")
                 return {}

            if status_callback:
                status_callback(f"Collecting Google Trends data for {len(self.companies)} companies using pytrends...")

            # Initialize pytrends WITHOUT retry parameters due to urllib3 incompatibility
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25)) # Keep timeout (No change needed here, just confirming state)
            trends_data = {}
            num_companies = len(self.companies)
            chunk_size = 5 # pytrends allows up to 5 keywords per request

            # Create mapping from name back to ticker for processing results
            name_to_ticker_map = {c['name']: c['ticker'] for c in self.companies}
            all_keywords = [c['name'] for c in self.companies] # List of all company names (keywords)

            num_chunks = (num_companies + chunk_size - 1) // chunk_size

            for i in range(0, num_companies, chunk_size):
                chunk_keywords = all_keywords[i:min(i + chunk_size, num_companies)]
                chunk_tickers = [name_to_ticker_map[name] for name in chunk_keywords]
                current_chunk_num = (i // chunk_size) + 1

                # Add sleep *before* processing the next chunk
                if i > 0: # Don't sleep before the very first request
                    time.sleep(30) # Moderate delay between chunk requests

                if progress_callback:
                    # Update progress based on chunks processed
                    progress_callback(current_chunk_num -1, num_chunks) # Progress per chunk
                if status_callback:
                    status_callback(f"Fetching Google Trends chunk {current_chunk_num}/{num_chunks} ({', '.join(chunk_tickers)})...")

                try:
                    pytrends.build_payload(chunk_keywords, cat=0, timeframe=timeframe, geo='', gprop='')
                    interest_df = pytrends.interest_over_time()

                    # Process results for each keyword in the chunk
                    for name in chunk_keywords:
                        ticker = name_to_ticker_map[name]
                        if not interest_df.empty and name in interest_df.columns and not interest_df['isPartial'].iloc[0]:
                            daily_trends = interest_df[[name]].resample('D').ffill()
                            daily_trends.rename(columns={name: 'Interest'}, inplace=True)
                            trends_data[ticker] = daily_trends
                            if status_callback: status_callback(f"✓ Got {len(daily_trends)} daily trend points for {ticker} ('{name}')")
                        elif not interest_df.empty and interest_df['isPartial'].iloc[0]:
                            if status_callback: status_callback(f"Warning: Received partial data in chunk for {ticker} ('{name}'). Skipping.")
                            trends_data[ticker] = pd.DataFrame(columns=['Interest']) # Mark as skipped
                        elif name not in interest_df.columns:
                             if status_callback: status_callback(f"Warning: Keyword '{name}' not found in results for chunk. Skipping {ticker}.")
                             trends_data[ticker] = pd.DataFrame(columns=['Interest']) # Mark as skipped
                        else: # Empty dataframe or other issue
                            if status_callback: status_callback(f"No Google Trends data found for {ticker} ('{name}') in chunk.")
                            trends_data[ticker] = pd.DataFrame(columns=['Interest']) # Mark as skipped

                except requests.exceptions.Timeout:
                     if status_callback: status_callback(f"Timeout error fetching trends for chunk {current_chunk_num}. Marking tickers {chunk_tickers} as failed.")
                     for ticker in chunk_tickers: trends_data[ticker] = pd.DataFrame(columns=['Interest'])
                except Exception as e:
                    error_msg = str(e)
                    if "response code 429" in error_msg.lower() or "too many requests" in error_msg.lower():
                         if status_callback: status_callback(f"Rate limit hit for chunk {current_chunk_num}. Marking tickers {chunk_tickers} as failed and continuing.")
                         # No long sleep here, just mark as failed and continue
                         for ticker in chunk_tickers: trends_data[ticker] = pd.DataFrame(columns=['Interest']) # Mark as failed for this chunk
                    elif "response code 400" in error_msg.lower():
                         if status_callback: status_callback(f"Bad request (400) for chunk {current_chunk_num}. Keywords: {chunk_keywords}. Marking tickers {chunk_tickers} as failed.")
                         for ticker in chunk_tickers: trends_data[ticker] = pd.DataFrame(columns=['Interest'])
                    else:
                         if status_callback: status_callback(f"Error fetching Google Trends for chunk {current_chunk_num}: {error_msg}. Marking tickers {chunk_tickers} as failed.")
                         traceback.print_exc()
                         for ticker in chunk_tickers: trends_data[ticker] = pd.DataFrame(columns=['Interest'])

            if progress_callback:
                progress_callback(num_chunks, num_chunks) # Ensure 100% completion

            num_collected = sum(1 for df in trends_data.values() if df is not None and not df.empty) # Count successful fetches
            # Save individual raw trends files per ticker
            num_saved_trends = 0
            for ticker, df_trends in trends_data.items():
                if df_trends is not None and not df_trends.empty:
                    # Ensure index is Date and naive datetime
                    df_trends.index = pd.to_datetime(df_trends.index).tz_localize(None)
                    df_to_save = df_trends.reset_index().rename(columns={'index': 'Date'})
                    save_path = os.path.join(self.raw_data_dir, f"{ticker}_trends_raw.csv")
                    try:
                        df_to_save.to_csv(save_path, index=False)
                        num_saved_trends += 1
                        # Optional: status callback for individual save
                        # if status_callback: status_callback(f"[{ticker}] Saved raw Google Trends to {os.path.basename(save_path)}")
                    except Exception as save_err:
                        if status_callback: status_callback(f"[{ticker}] Error saving raw Google Trends: {save_err}")

            if status_callback:
                status_callback(f"Google Trends collection completed. Saved raw data for {num_saved_trends}/{num_companies} companies.")

            # Return the dictionary containing the collected dataframes
            return trends_data

        except Exception as e:
            if status_callback:
                status_callback(f"Critical Error in Google Trends collection: {str(e)}")
            traceback.print_exc()
            if progress_callback:
                progress_callback(1, 1) # Ensure progress shows completion even on error
            return {} # Return empty dict on critical error

            if status_callback:
                status_callback(f"Google Trends collection completed. Saved raw data for {num_saved_trends}/{num_companies} companies.")

            # Return the dictionary containing the collected dataframes
            return trends_data # Ensure the function returns the collected data
            return trends_data
            # return trends_data
            return # Indicate completion

        except Exception as e:
            if status_callback:
                status_callback(f"Critical Error in Google Trends collection: {str(e)}")
            traceback.print_exc()
            if progress_callback:
                progress_callback(1, 1)
            return {} # Return empty dict on critical error

    def collect_fred_data(self, start_date, end_date, series_id='FEDFUNDS', progress_callback=None, status_callback=None):
        """Collect economic data (e.g., Federal Funds Rate) from FRED."""
        if status_callback:
            status_callback(f"Collecting FRED data for series '{series_id}'...")
        else:
            print(f"Collecting FRED data for series '{series_id}'...")

        try:
            # Validate dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Fetch data from FRED
            fred_data = web.DataReader(series_id, 'fred', start_dt, end_dt)

            if fred_data.empty:
                if status_callback: status_callback(f"Warning: No data returned from FRED for {series_id}.")
                return None # Return None if no data

            # Resample to daily frequency using forward fill
            # FRED data might be monthly/weekly, so ffill propagates the last known value
            fred_data_daily = fred_data.resample('D').ffill()

            # Ensure index is naive datetime
            fred_data_daily.index = fred_data_daily.index.tz_localize(None)

            # Save raw data
            save_path = os.path.join(self.raw_data_dir, f"{series_id}_raw.csv")
            fred_data_daily.reset_index().rename(columns={'index': 'Date'}).to_csv(save_path, index=False)

            if status_callback:
                status_callback(f"✓ Successfully collected and saved raw FRED data ({series_id}) to {os.path.basename(save_path)}")
            if progress_callback:
                progress_callback(1, 1) # Indicate completion

            return fred_data_daily # Return the daily resampled DataFrame

        except requests.exceptions.RequestException as req_err:
             error_msg = f"Network error fetching FRED data ({series_id}): {req_err}"
             if status_callback: status_callback(f"Error: {error_msg}")
             else: print(f"Error: {error_msg}")
             if progress_callback: progress_callback(1, 1) # Indicate completion even on error
             return None
        except Exception as e:
            error_msg = f"Error collecting FRED data ({series_id}): {e}"
            if status_callback:
                status_callback(f"Error: {error_msg}")
            else:
                print(f"Error: {error_msg}")
            traceback.print_exc()
            if progress_callback:
                progress_callback(1, 1) # Indicate completion even on error
            return None

    def process_data(self, stock_data, # Removed reddit_data, google_trends args
                    progress_callback=None, status_callback=None):
        """
        Process collected data for each stock ticker individually by reading raw files
        and save the final processed data to separate files.

        Args:
            stock_data (dict): Dictionary of DataFrames containing stock data {ticker: df}.
                               Assumes df has 'Date' column.
            progress_callback (function): Callback for progress updates.
            status_callback (function): Callback for status updates.

        Returns:
            tuple: (list of successfully processed tickers, list of representative features)
               Returns empty lists if processing fails.
        """
        processed_tickers = []
        representative_features = [] # Will store features from the *last* processed ticker for now
        final_feature_cols = [] # Store feature columns determined during processing

        if status_callback:
            status_callback("Starting data processing per ticker...")

        # Make sure stock_data is not None and is a dictionary
        if stock_data is None or not isinstance(stock_data, dict) or not stock_data:
            if status_callback:
                status_callback("Error: Invalid or empty stock data provided for processing.")
            return [], [] # Return empty lists

        # Use tickers from the keys of the stock_data dictionary
        valid_tickers = list(stock_data.keys())
        total_tickers = len(valid_tickers)

        if total_tickers == 0:
            if status_callback:
                status_callback("Error: No stock data tickers found to process.")
            return [], []

        # Process stock data ticker by ticker
        for i, ticker in enumerate(valid_tickers):
            if status_callback:
                status_callback(f"Processing ticker {ticker} ({i+1}/{total_tickers})...")
            if progress_callback:
                 progress_callback(i, total_tickers) # Update overall progress

            # Get the DataFrame for the current ticker
            df = stock_data[ticker].copy() # Work on a copy

            try:
                # --- Start: Per-Ticker Processing Logic ---
                if status_callback: status_callback(f"[{ticker}] Processing started ({len(df)} raw rows).")

                # Make sure Date column is datetime type and set as index
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    if status_callback: status_callback(f"[{ticker}] Date index set.")
                else:
                     if status_callback: status_callback(f"[{ticker}] Warning: 'Date' column missing. Skipping.")
                     continue # Skip this ticker if no Date column

                # --- Early Cleaning: Ensure basic columns are numeric ---
                ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                valid_ticker_data = True
                for col in ohlcv_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        if status_callback: status_callback(f"[{ticker}] Warning: Critical column '{col}' missing. Skipping.")
                        valid_ticker_data = False
                        break # Stop checking columns for this ticker
                if not valid_ticker_data: continue # Skip to next ticker

                df.dropna(subset=ohlcv_cols, inplace=True) # Drop rows if critical values are missing after coercion
                if df.empty:
                    if status_callback: status_callback(f"[{ticker}] Warning: DataFrame empty after initial OHLCV cleaning. Skipping.")
                    continue
                if status_callback: status_callback(f"[{ticker}] Initial cleaning done ({len(df)} rows).")

                # --- Load and Merge Reddit Sentiment Data ---
                reddit_raw_path = os.path.join(self.raw_data_dir, f"{ticker}_reddit_raw.csv")
                sentiment_cols = ['Compound', 'Positive', 'Neutral', 'Negative', 'Count']
                if os.path.exists(reddit_raw_path):
                    try:
                        sentiment_df = pd.read_csv(reddit_raw_path, parse_dates=['Date'])
                        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None) # Ensure naive datetime

                        # Prepare stock df for merge
                        df_reset = df.reset_index()
                        df_reset['Date'] = pd.to_datetime(df_reset['Date']).dt.tz_localize(None)

                        # Merge
                        df_merged = pd.merge(df_reset, sentiment_df, on='Date', how='left')

                        # Fill NaNs *after* merge for sentiment columns only
                        for col in sentiment_cols:
                            if col in df_merged.columns:
                                df_merged[col] = df_merged[col].fillna(0)
                            else:
                                df_merged[col] = 0 # Add column if missing (shouldn't happen with left merge)

                        # Restore index
                        df = df_merged.set_index('Date')
                        if status_callback: status_callback(f"[{ticker}] Merged with raw Reddit data.")

                    except Exception as e:
                        if status_callback: status_callback(f"[{ticker}] Error loading/merging Reddit data from {os.path.basename(reddit_raw_path)}: {e}. Skipping sentiment.")
                        # Ensure sentiment columns exist with zeros if merge failed
                        for col in sentiment_cols: df[col] = 0
                else:
                    # File doesn't exist, add empty columns
                    if status_callback: status_callback(f"[{ticker}] No raw Reddit file found ({os.path.basename(reddit_raw_path)}). Skipping sentiment.")
                    for col in sentiment_cols: df[col] = 0

                # --- Load and Merge Google Trends Data ---
                trends_raw_path = os.path.join(self.raw_data_dir, f"{ticker}_trends_raw.csv")
                # Initialize Interest column in main df first to ensure it exists
                df['Interest'] = 0.0 # Use float

                if os.path.exists(trends_raw_path):
                    try:
                        # --- Robust Loading of Trends CSV ---
                        trends_df = pd.read_csv(trends_raw_path)
                        date_col_found = False
                        if 'Date' in trends_df.columns:
                            try:
                                trends_df['Date'] = pd.to_datetime(trends_df['Date'])
                                date_col_found = True
                                if status_callback: status_callback(f"[{ticker}] Found and parsed 'Date' column in trends file.")
                            except Exception as e:
                                if status_callback: status_callback(f"[{ticker}] Error parsing 'Date' column: {e}. Trying first column.")
                        elif trends_df.shape[1] > 0: # Check if there's at least one column
                            # Try parsing the first column as date if 'Date' column doesn't exist
                            first_col_name = trends_df.columns[0]
                            try:
                                trends_df[first_col_name] = pd.to_datetime(trends_df[first_col_name])
                                trends_df.rename(columns={first_col_name: 'Date'}, inplace=True)
                                date_col_found = True
                                if status_callback: status_callback(f"[{ticker}] Parsed first column ('{first_col_name}') as Date in trends file.")
                            except Exception as e:
                                if status_callback: status_callback(f"[{ticker}] Error parsing first column ('{first_col_name}') as Date: {e}. Cannot use trends data.")

                        if not date_col_found:
                             if status_callback: status_callback(f"[{ticker}] Could not find or parse a suitable Date column in {os.path.basename(trends_raw_path)}. Skipping trends merge.")
                             raise ValueError("Suitable Date column not found in trends CSV")

                        # --- Step 1 (cont.): Verify Raw Data Loading (Post Date Handling) ---
                        if status_callback:
                            status_callback(f"[{ticker}] Verified trends file. Columns: {trends_df.columns.tolist()}.")
                        if 'Interest' in trends_df.columns:
                            trends_df['Interest'] = pd.to_numeric(trends_df['Interest'], errors='coerce')
                            if status_callback: status_callback(f"[{ticker}] Raw 'Interest' (numeric) head:\n{trends_df['Interest'].head().to_string()}")
                        else:
                             if status_callback: status_callback(f"[{ticker}] Warning: 'Interest' column missing in raw trends file {os.path.basename(trends_raw_path)}. Skipping merge.")
                             raise ValueError("Interest column missing in raw trends CSV")

                        # --- Step 2: Standardize Date Columns Rigorously ---
                        trends_df['Date'] = trends_df['Date'].dt.tz_localize(None) # Ensure naive datetime

                        # Prepare stock df for merge (reset index if needed)
                        if isinstance(df.index, pd.DatetimeIndex):
                            df_reset = df.reset_index()
                        else: # If index is not Date
                            df_reset = df.copy() # Should already have 'Interest' column initialized
                            if 'Date' not in df_reset.columns:
                                 status_callback(f"[{ticker}] Critical Error: Cannot find Date index or column in stock data before trends merge.")
                                 raise ValueError("Date missing in stock data for merge")

                        df_reset['Date'] = pd.to_datetime(df_reset['Date']).dt.tz_localize(None) # Stock date to naive datetime

                        # --- Merge ---
                        # Drop the pre-initialized 'Interest' from df_reset before merge to avoid suffix issues
                        if 'Interest' in df_reset.columns:
                             df_reset = df_reset.drop(columns=['Interest'])
                        df_merged = pd.merge(df_reset, trends_df[['Date', 'Interest']], on='Date', how='left')

                        # --- Step 3: Check Post-Merge ---
                        if 'Interest' in df_merged.columns:
                            non_nan_interest = df_merged['Interest'].notna().sum()
                            if status_callback: status_callback(f"[{ticker}] Google Trends merge resulted in {non_nan_interest}/{len(df_merged)} non-NaN 'Interest' values.")

                            # --- NaN Handling for Interest ---
                            # Interpolate first, then fill remaining with 0
                            df_merged['Interest'] = df_merged['Interest'].interpolate(method='linear', limit_direction='both')
                            df_merged['Interest'] = df_merged['Interest'].fillna(0.0) # Use float 0.0
                            if status_callback: status_callback(f"[{ticker}] Applied interpolation and fillna(0) to 'Interest'. Final head:\n{df_merged['Interest'].head().to_string()}")
                        else:
                            if status_callback: status_callback(f"[{ticker}] 'Interest' column missing after merge. Initializing to 0.")
                            df_merged['Interest'] = 0.0

                        # Restore index
                        df = df_merged.set_index('Date')
                        if status_callback: status_callback(f"[{ticker}] Merged with raw Google Trends data and restored index.")

                    except Exception as e:
                        if status_callback: status_callback(f"[{ticker}] Error loading/merging Trends data from {os.path.basename(trends_raw_path)}: {e}. Setting 'Interest' to 0.")
                        # Ensure column exists with zeros even on error
                        if 'Interest' not in df.columns: df['Interest'] = 0.0
                        else: df['Interest'].fillna(0.0, inplace=True)
                else:
                     # File doesn't exist
                     if status_callback: status_callback(f"[{ticker}] No raw Trends file found ({os.path.basename(trends_raw_path)}). Setting 'Interest' to 0.")
                     df['Interest'] = 0.0 # Ensure column exists with zeros

                # --- Load and Merge FRED Data (e.g., FEDFUNDS) ---
                fred_series_id = 'FEDFUNDS'
                fred_raw_path = os.path.join(self.raw_data_dir, f"{fred_series_id}_raw.csv")
                # df[fred_series_id] = np.nan # DO NOT Initialize column here to avoid merge suffixes

                if os.path.exists(fred_raw_path):
                    try:
                        # Load without strict date parsing initially
                        fred_df = pd.read_csv(fred_raw_path)

                        # Find the date column robustly
                        date_col_name = None
                        if 'Date' in fred_df.columns:
                            date_col_name = 'Date'
                        elif 'index' in fred_df.columns: # Check for 'index' if 'Date' is missing
                            date_col_name = 'index'
                        elif fred_df.shape[1] > 0: # Fallback to the first column
                            date_col_name = fred_df.columns[0]

                        if date_col_name:
                            # Parse the identified date column
                            fred_df[date_col_name] = pd.to_datetime(fred_df[date_col_name])
                            # Rename to 'Date' if it's not already named 'Date'
                            if date_col_name != 'Date':
                                fred_df.rename(columns={date_col_name: 'Date'}, inplace=True)
                            if status_callback: status_callback(f"[{ticker}] Identified and parsed date column '{date_col_name}' as 'Date' in FRED file.")
                        else:
                            raise ValueError("Could not identify a suitable date column in FRED CSV.")

                        # Ensure 'Date' column is naive datetime and set as index
                        fred_df['Date'] = fred_df['Date'].dt.tz_localize(None)
                        fred_df.set_index('Date', inplace=True)

                        # Convert all column names to uppercase for case-insensitive matching
                        fred_df.columns = map(str.upper, fred_df.columns)
                        fred_series_id_upper = fred_series_id.upper() # Ensure our target ID is also uppercase

                        value_column_found = False
                        if fred_series_id_upper in fred_df.columns:
                            value_column_found = True
                            if status_callback: status_callback(f"[{ticker}] Found FRED value column '{fred_series_id_upper}' directly.")
                        else:
                            # Log error if specific column not found after uppercasing
                            if status_callback: status_callback(f"[{ticker}] Error: Column '{fred_series_id_upper}' not found in FRED data even after uppercasing. Columns: {fred_df.columns.tolist()}. Skipping merge.")

                        # Merge FRED data into the main DataFrame only if the value column was found
                        if value_column_found:
                            # Perform the merge, allowing suffixes if necessary (though unlikely now)
                            df = df.merge(fred_df[[fred_series_id_upper]], left_index=True, right_index=True, how='left', suffixes=('', '_FRED'))

                            # Check if the uppercase column exists (it should) and rename it to the original case
                            if fred_series_id_upper in df.columns:
                                df.rename(columns={fred_series_id_upper: fred_series_id}, inplace=True)
                                if status_callback: status_callback(f"[{ticker}] Renamed merged FRED column to '{fred_series_id}'.")
                            # Handle the unlikely case where merge added a suffix despite removing pre-initialization
                            elif f"{fred_series_id_upper}_FRED" in df.columns:
                                 if status_callback: status_callback(f"[{ticker}] Warning: Merge created suffix. Renaming '{fred_series_id_upper}_FRED' to '{fred_series_id}'.")
                                 df.rename(columns={f"{fred_series_id_upper}_FRED": fred_series_id}, inplace=True)
                            # else: # If the column is somehow still missing after a successful merge flag
                                 # if status_callback: status_callback(f"[{ticker}] Warning: FRED column '{fred_series_id}' missing after merge despite value_column_found=True. Check logic.")
                                 # df[fred_series_id] = np.nan # Ensure column exists

                            # Handle NaNs in FRED column after merge (ffill first, then bfill)
                            if fred_series_id in df.columns:
                                 df[fred_series_id] = df[fred_series_id].ffill().bfill()
                                 if status_callback: status_callback(f"[{ticker}] Merged and filled NaNs for FRED data ({fred_series_id}).")
                            else:
                                 # This warning should now be less likely
                                 if status_callback: status_callback(f"[{ticker}] Warning: Merged FRED column '{fred_series_id}' not found after merge/rename? Check logic.")
                                 df[fred_series_id] = np.nan # Ensure column exists
                        else: # Case where value_column_found is False
                             df[fred_series_id] = np.nan # Ensure column exists with NaN

                    except Exception as e:
                        if status_callback: status_callback(f"[{ticker}] Error loading/merging FRED data from {os.path.basename(fred_raw_path)}: {e}. Skipping FRED data.")
                        # Ensure column exists with NaN if merge failed or file error
                        if fred_series_id not in df.columns: df[fred_series_id] = np.nan
                else:
                    if status_callback: status_callback(f"[{ticker}] No raw FRED file found ({os.path.basename(fred_raw_path)}). Skipping FRED data.")
                    # Ensure column exists with NaN if file not found
                    if fred_series_id not in df.columns: df[fred_series_id] = np.nan


                # --- Intermediate Cleaning after ALL Merges ---
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                # 1. Interpolate linearly first to handle gaps better
                numeric_cols_for_interp = df.select_dtypes(include=np.number).columns
                if not df[numeric_cols_for_interp].empty:
                     try:
                         df[numeric_cols_for_interp] = df[numeric_cols_for_interp].interpolate(method='linear', limit_direction='both', axis=0)
                         if status_callback: status_callback(f"[{ticker}] Applied linear interpolation for NaNs.")
                     except Exception as interp_err:
                          if status_callback: status_callback(f"[{ticker}] Warning: Linear interpolation failed: {interp_err}. Proceeding with ffill/bfill.")

                # 2. Forward fill
                df.fillna(method='ffill', inplace=True)
                # 3. Backward fill for any remaining NaNs at the beginning
                df.fillna(method='bfill', inplace=True)
                # 4. Fill any remaining NaNs (e.g., if all values in a column were NaN) with 0
                df.fillna(0, inplace=True)

                if df.empty:
                    if status_callback: status_callback(f"[{ticker}] Warning: DataFrame empty after merge cleaning. Skipping.")
                    continue # This continue is correctly placed within the try block
                if status_callback: status_callback(f"[{ticker}] Merge cleaning done ({len(df)} rows).")
                # Removed duplicated status callback line

                # --- Outlier Detection and Clipping (Z-score method) ---
                if status_callback: status_callback(f"[{ticker}] Applying outlier detection (Z-score > 3)...")
                outlier_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] # Columns to check
                z_thresh = 3
                for col in outlier_cols:
                    if col in df.columns:
                        # Ensure column is numeric before calculating mean/std
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Drop rows where the column became NaN after coercion (should be rare after previous cleaning)
                        df.dropna(subset=[col], inplace=True)
                        if df.empty: # Check if df became empty after dropping NaNs
                             if status_callback: status_callback(f"[{ticker}] DataFrame empty after dropping NaNs in '{col}' during outlier check. Skipping further processing for this ticker.")
                             break # Exit the outlier loop for this ticker

                        col_mean = df[col].mean()
                        col_std = df[col].std()
                        if col_std > 0: # Avoid division by zero if std is 0
                            df[f'{col}_zscore'] = (df[col] - col_mean) / col_std
                            outliers = df[np.abs(df[f'{col}_zscore']) > z_thresh]
                            if not outliers.empty:
                                if status_callback: status_callback(f"[{ticker}] Detected {len(outliers)} outliers in '{col}'. Clipping...")
                                # Clip outliers: Replace with value at +/- z_thresh * std_dev
                                upper_limit = col_mean + z_thresh * col_std
                                lower_limit = col_mean - z_thresh * col_std
                                df[col] = np.clip(df[col], lower_limit, upper_limit)
                            # Remove the temporary z-score column
                            df.drop(columns=[f'{col}_zscore'], inplace=True)
                        elif status_callback:
                             status_callback(f"[{ticker}] Skipping outlier detection for '{col}' (standard deviation is zero).")
                    else:
                         if status_callback: status_callback(f"[{ticker}] Skipping outlier detection for '{col}' (column not found).")

                # Check if df became empty during the outlier loop
                if df.empty:
                    continue # This continue is correctly placed within the try block

                if status_callback: status_callback(f"[{ticker}] Outlier handling complete.")
                # --- End Outlier Detection ---

                # Calculate additional technical indicators
                if status_callback: status_callback(f"[{ticker}] Calculating indicators...")
                try:
                    # Import TA-Lib for technical indicators
                    import talib

                    # Ensure we have enough data for calculations
                    min_period = 50 # Find max period needed by indicators
                    if len(df) < min_period:
                        if status_callback: status_callback(f"[{ticker}] Warning: Not enough data ({len(df)} rows) for TA-Lib indicators (need {min_period}). Using basic.")
                        df = self._calculate_basic_indicators(df, status_callback)
                    else:
                        # Convert to numpy arrays for talib (ensure they are float)
                        open_prices = df['Open'].values.astype(float)
                        high_prices = df['High'].values.astype(float)
                        low_prices = df['Low'].values.astype(float)
                        close_prices = df['Close'].values.astype(float)
                        volume = df['Volume'].values.astype(float)

                        # --- Indicator Calculations (with individual try-except) ---
                        indicator_cols = [
                            'SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal',
                            'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SlowK', 'SlowD', 'ADX',
                            'Chaikin_AD', 'OBV', 'ATR', 'Williams_R', 'ROC', 'CCI'
                        ]
                        # Initialize columns to NaN
                        for col in indicator_cols: df[col] = np.nan

                        # Calculate indicators safely
                        try: df['SMA_5'] = talib.SMA(close_prices, timeperiod=5)
                        except Exception as e: status_callback(f"[{ticker}] Error SMA_5: {e}")
                        try: df['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
                        except Exception as e: status_callback(f"[{ticker}] Error SMA_20: {e}")
                        try: df['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
                        except Exception as e: status_callback(f"[{ticker}] Error SMA_50: {e}")
                        try: df['EMA_5'] = talib.EMA(close_prices, timeperiod=5)
                        except Exception as e: status_callback(f"[{ticker}] Error EMA_5: {e}")
                        try: df['EMA_20'] = talib.EMA(close_prices, timeperiod=20)
                        except Exception as e: status_callback(f"[{ticker}] Error EMA_20: {e}")
                        try: df['RSI'] = talib.RSI(close_prices, timeperiod=14)
                        except Exception as e: status_callback(f"[{ticker}] Error RSI: {e}")
                        try: macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9); df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd, macd_signal, macd_hist
                        except Exception as e: status_callback(f"[{ticker}] Error MACD: {e}")
                        try: upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0); df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = upper, middle, lower
                        except Exception as e: status_callback(f"[{ticker}] Error BBands: {e}")
                        try: slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0); df['SlowK'], df['SlowD'] = slowk, slowd
                        except Exception as e: status_callback(f"[{ticker}] Error Stoch: {e}")
                        try: df['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
                        except Exception as e: status_callback(f"[{ticker}] Error ADX: {e}")
                        try: df['Chaikin_AD'] = talib.AD(high_prices, low_prices, close_prices, volume)
                        except Exception as e: status_callback(f"[{ticker}] Error Chaikin AD: {e}")
                        try: df['OBV'] = talib.OBV(close_prices, volume)
                        except Exception as e: status_callback(f"[{ticker}] Error OBV: {e}")
                        try: df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                        except Exception as e: status_callback(f"[{ticker}] Error ATR: {e}")
                        try: df['Williams_R'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
                        except Exception as e: status_callback(f"[{ticker}] Error Williams %R: {e}")
                        try: df['ROC'] = talib.ROC(close_prices, timeperiod=10)
                        except Exception as e: status_callback(f"[{ticker}] Error ROC: {e}")
                        try: df['CCI'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=20)
                        except Exception as e: status_callback(f"[{ticker}] Error CCI: {e}")

                        # Ratio features (handle division by zero robustly)
                        df['Close_Open_Ratio'] = (df['Close'] / df['Open'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                        df['High_Low_Diff'] = df['High'] - df['Low']
                        df['Close_Prev_Ratio'] = (df['Close'] / df['Close'].shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                        if status_callback: status_callback(f"[{ticker}] TA-Lib Indicators calculated.")

                except ImportError as e:
                    if status_callback: status_callback(f"[{ticker}] TA-Lib not available: {str(e)}. Using basic calculations.")
                    df = self._calculate_basic_indicators(df, status_callback)
                except Exception as e:
                    if status_callback: status_callback(f"[{ticker}] Error calculating technical indicators: {str(e)}")
                    df = self._calculate_basic_indicators(df, status_callback) # Fallback

                # --- End: Per-Ticker Indicator Calculation ---

                # --- Start: Per-Ticker Feature Engineering & Cleaning ---
                if status_callback: status_callback(f"[{ticker}] Adding engineered features...")

                # Add Lagged Features
                lags = [1, 3, 5]
                for lag in lags:
                    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
                    df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
                    # Check if sentiment/trends columns exist before lagging
                    if 'Compound' in df.columns:
                        df[f'Compound_Lag_{lag}'] = df['Compound'].shift(lag)
                    if 'Interest' in df.columns:
                        df[f'Interest_Lag_{lag}'] = df['Interest'].shift(lag)
                    # Lag FRED data if it exists
                    if fred_series_id in df.columns:
                         df[f'{fred_series_id}_Lag_{lag}'] = df[fred_series_id].shift(lag)


                # Add Volatility Feature
                df['Volatility_20D'] = df['Close'].rolling(window=20, min_periods=1).std()

                # Add Time Feature
                df['Day_Of_Week'] = df.index.dayofweek # Monday=0, Sunday=6

                # ** NEW: Interaction Features **
                # Example interactions (add more as needed)
                if 'Volume' in df.columns and 'RSI' in df.columns:
                    df['Vol_x_RSI'] = df['Volume'] * df['RSI']
                if 'SMA_5' in df.columns and 'SMA_20' in df.columns and not df['SMA_20'].eq(0).any():
                     # Ensure SMA_20 is not zero before division
                     df['SMA5_SMA20_Ratio'] = df['SMA_5'] / df['SMA_20'].replace(0, np.nan)
                if 'Volatility_20D' in df.columns and 'Volume' in df.columns:
                    df['Vol_x_Volatility'] = df['Volume'] * df['Volatility_20D']
                if status_callback: status_callback(f"[{ticker}] Added interaction features.")

                # ** NEW: Polynomial Features (Example on Close and Volume) **
                # Select a subset of features for polynomial transformation to avoid excessive dimensionality
                poly_features_subset = ['Close', 'Volume']
                if all(f in df.columns for f in poly_features_subset):
                    try:
                        # Ensure subset columns are numeric and handle NaNs before poly transform
                        df[poly_features_subset] = df[poly_features_subset].apply(pd.to_numeric, errors='coerce').fillna(0)

                        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
                        poly_feats = poly.fit_transform(df[poly_features_subset])
                        # Get feature names (handle potential duplicates if needed)
                        poly_names = poly.get_feature_names_out(poly_features_subset)
                        poly_df = pd.DataFrame(poly_feats, index=df.index, columns=poly_names)

                        # Drop original columns used in poly features if desired (optional, but recommended if originals are now redundant)
                        # df = df.drop(columns=poly_features_subset)

                        # Merge polynomial features back (handle potential column name conflicts if originals weren't dropped)
                        df = pd.concat([df, poly_df.drop(columns=poly_features_subset, errors='ignore')], axis=1) # Drop originals from poly_df just in case
                        if status_callback: status_callback(f"[{ticker}] Added polynomial features for {poly_features_subset}.")
                    except Exception as e:
                        if status_callback: status_callback(f"[{ticker}] Warning: Could not create polynomial features: {e}")

                if status_callback: status_callback(f"[{ticker}] Finished feature engineering.")

                # --- Cleaning and Target Creation ---
                if status_callback: status_callback(f"[{ticker}] Final cleaning...")

                # Robustly fill NaNs/Infs again after indicator calculations and ratio features
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                # 1. Interpolate linearly first
                numeric_cols_for_interp = df.select_dtypes(include=np.number).columns
                if not df[numeric_cols_for_interp].empty:
                     try:
                         df[numeric_cols_for_interp] = df[numeric_cols_for_interp].interpolate(method='linear', limit_direction='both', axis=0)
                         if status_callback: status_callback(f"[{ticker}] Applied linear interpolation for NaNs.")
                     except Exception as interp_err:
                          if status_callback: status_callback(f"[{ticker}] Warning: Linear interpolation failed: {interp_err}. Proceeding with ffill/bfill.")
                # 2. Forward fill
                df.fillna(method='ffill', inplace=True)
                # 3. Backward fill for any remaining NaNs at the beginning
                df.fillna(method='bfill', inplace=True)
                # 4. Fill any remaining NaNs (e.g., if all values in a column were NaN) with 0
                df.fillna(0, inplace=True)

                if df.empty:
                    if status_callback: status_callback(f"[{ticker}] Warning: DataFrame empty after indicator cleaning. Skipping.")
                    continue # This continue is correctly placed within the try block
                if status_callback: status_callback(f"[{ticker}] Final cleaning done ({len(df)} rows).")

                # --- Normalization Removed ---
                # Normalization/Scaling is now handled in model_training.py

                # Create target columns
                if status_callback: status_callback(f"[{ticker}] Creating target columns...")
                if len(df) > 1:
                    df['Close_Next'] = df['Close'].shift(-1)
                    df['Price_Change'] = df['Close_Next'] - df['Close']
                    # Rename 'Trend' to 'Price_Increase' for consistency with model_training.py
                    df['Price_Increase'] = (df['Price_Change'] > 0).astype(int)
                    # Drop the last row where targets are NaN (Close_Next is NaN)
                    df.dropna(subset=['Close_Next', 'Price_Change', 'Price_Increase'], inplace=True)
                else:
                    if status_callback: status_callback(f"[{ticker}] Warning: Not enough data to create target columns.")
                    df['Close_Next'] = np.nan
                    df['Price_Change'] = np.nan
                    df['Price_Increase'] = np.nan

                if df.empty: # Check if empty after dropping NaN targets
                     if status_callback: status_callback(f"[{ticker}] Warning: DataFrame empty after target creation/NaN drop. Skipping save.")
                     continue # This continue is correctly placed within the try block
                if status_callback: status_callback(f"[{ticker}] Target columns created ({len(df)} rows).")

                # --- End: Per-Ticker Normalization and Target Creation ---

                # --- Step 4: Trace Column (Add logging before saving) ---
                if status_callback:
                    if 'Interest' in df.columns:
                        status_callback(f"[{ticker}] Before saving, 'Interest' column exists. Head:\n{df['Interest'].head().to_string()}. Sum: {df['Interest'].sum()}")
                    else:
                        status_callback(f"[{ticker}] Before saving, 'Interest' column is MISSING.")
                    # Also trace FRED column
                    if fred_series_id in df.columns:
                         status_callback(f"[{ticker}] Before saving, '{fred_series_id}' column exists. Head:\n{df[fred_series_id].head().to_string()}. Sum: {df[fred_series_id].sum()}")
                    else:
                         status_callback(f"[{ticker}] Before saving, '{fred_series_id}' column is MISSING.")


                # --- Step 4: Trace Column (Add logging before saving) --- # Removed duplicate block
                # if status_callback:
                #     if 'Interest' in df.columns:
                #         status_callback(f"[{ticker}] Before saving, 'Interest' column exists. Head:\n{df['Interest'].head().to_string()}. Sum: {df['Interest'].sum()}")
                #     else:
                #         status_callback(f"[{ticker}] Before saving, 'Interest' column is MISSING.")

                # Save processed data for this ticker
                if not df.empty:
                    processed_file_path = os.path.join(self.processed_data_dir, f"{ticker}_processed_data.csv")
                    # Add Ticker column before saving (ensure it's present)
                    if 'Ticker' not in df.columns: df['Ticker'] = ticker
                    # Reset index to save Date as column
                    df_to_save = df.reset_index()
                    # Ensure specific columns exist before trying to use them
                    # Exclude target and identifier columns from feature list
                    # Include the new FRED column in the feature list if present
                    feature_cols_present = [col for col in df_to_save.columns if col not in ['Date', 'Ticker', 'Company', 'Close_Next', 'Price_Change', 'Price_Increase']]


                    df_to_save.to_csv(processed_file_path, index=False)
                    processed_tickers.append(ticker)
                    if status_callback: status_callback(f"[{ticker}] ✓ Saved processed data to {os.path.basename(processed_file_path)}")

                    # Store feature columns from the last successfully processed ticker
                    final_feature_cols = feature_cols_present # Use columns actually present

                else:
                     if status_callback: status_callback(f"[{ticker}] Warning: No data left for processing after target creation.")

            # Correctly indented except block for the main try within the loop
            except Exception as e:
                if status_callback:
                    status_callback(f"[{ticker}] Error processing data: {str(e)}")
                # Print traceback for detailed debugging
                print(f"--- Traceback for error processing {ticker} ---")
                traceback.print_exc()
                print(f"--- End Traceback ---")

        # --- Post-Loop ---
        if progress_callback:
             progress_callback(total_tickers, total_tickers) # Ensure progress reaches 100%

        if status_callback:
            status_callback(f"Data processing complete. Successfully processed {len(processed_tickers)}/{total_tickers} tickers.")

        # Use features from the last successfully processed ticker as representative
        representative_features = final_feature_cols

        return processed_tickers, representative_features

    def _calculate_basic_indicators(self, df, status_callback=None):
        """Calculate basic technical indicators without TA-Lib"""
        if status_callback: status_callback(f"[{df['Ticker'].iloc[0] if 'Ticker' in df.columns else 'Unknown'}] Calculating basic indicators...")
        try:
            # Ensure Close column is numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True) # Drop rows if Close is NaN

            # 1. Moving Averages
            df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

            # 2. RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan) # Avoid division by zero
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'].fillna(50, inplace=True) # Fill initial NaNs with 50

            # 3. MACD
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

            # 4. Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

            # Ensure High/Low/Open/Volume are numeric for remaining indicators
            for col in ['High', 'Low', 'Open', 'Volume']:
                 if col in df.columns:
                     df[col] = pd.to_numeric(df[col], errors='coerce')
                 else:
                     df[col] = df['Close'] # Use Close as fallback if missing

            # 5. Stochastic Oscillator
            n = 14
            df['Highest_High'] = df['High'].rolling(n, min_periods=1).max()
            df['Lowest_Low'] = df['Low'].rolling(n, min_periods=1).min()
            denominator = (df['Highest_High'] - df['Lowest_Low']).replace(0, np.nan)
            df['SlowK'] = 100 * ((df['Close'] - df['Lowest_Low']) / denominator)
            df['SlowD'] = df['SlowK'].rolling(3, min_periods=1).mean()

            # 6-12. Additional indicators (simplified placeholders or basic calcs)
            df['ADX'] = np.nan # Placeholder

            # 7. Chaikin A/D
            denominator_ad = (df['High'] - df['Low']).replace(0, np.nan)
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / denominator_ad
            mfm = mfm.fillna(0) # Fill NaNs resulting from division by zero or identical High/Low
            mfv = mfm * pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            df['Chaikin_AD'] = mfv.cumsum()

            # 8. OBV
            df['OBV'] = np.nan  # Initialize column
            obv = 0
            close_prev = df['Close'].shift(1)
            volume_curr = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            delta_close = df['Close'].diff()
            df['OBV'] = np.where(delta_close > 0, volume_curr, np.where(delta_close < 0, -volume_curr, 0)).cumsum()
            df['OBV'].fillna(0, inplace=True) # Fill initial NaN

            # 9. ATR
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift())
            low_close_prev = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
            true_range = ranges.max(axis=1, skipna=False) # Don't skip NaNs initially
            df['ATR'] = true_range.rolling(14, min_periods=1).mean()

            # 10. Williams %R
            highest_high_14 = df['High'].rolling(14, min_periods=1).max()
            lowest_low_14 = df['Low'].rolling(14, min_periods=1).min()
            denominator_wr = (highest_high_14 - lowest_low_14).replace(0, np.nan)
            df['Williams_R'] = -100 * ((highest_high_14 - df['Close']) / denominator_wr)

            # 11. ROC
            df['ROC'] = df['Close'].pct_change(10, fill_method=None) * 100 # Use default fill_method

            # 12. CCI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20, min_periods=1).mean()
            mad = np.abs(typical_price - sma_tp).rolling(window=20, min_periods=1).mean()
            df['CCI'] = (typical_price - sma_tp) / (0.015 * mad).replace(0, np.nan) # Avoid division by zero

            # Feature ratios (handle division by zero)
            df['Close_Open_Ratio'] = (df['Close'] / df['Open'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            df['High_Low_Diff'] = df['High'] - df['Low']
            df['Close_Prev_Ratio'] = (df['Close'] / df['Close'].shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

            # Replace inf and NaN again after calculations
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

        except Exception as e:
            if status_callback:
                status_callback(f"[{df['Ticker'].iloc[0] if 'Ticker' in df.columns else 'Unknown'}] Error in basic indicator calculation: {str(e)}")
            # If all else fails, create placeholder columns with NaN
            for col in ['SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'RSI',
                       'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower',
                       'SlowK', 'SlowD', 'ADX', 'Chaikin_AD', 'OBV', 'ATR', 'Williams_R',
                       'ROC', 'CCI', 'Close_Open_Ratio', 'High_Low_Diff', 'Close_Prev_Ratio']:
                if col not in df.columns:
                    df[col] = np.nan

        return df

# Define main function for standalone execution
def main():
    # Example usage with parameters
    collector = DataCollector()
    start = '2022-01-01'
    end = datetime.datetime.now().strftime('%Y-%m-%d') # Use current date as end
    num = 10 # Example: Get top 10 companies

    # Define simple callbacks for CLI
    def cli_progress(current, total):
        print(f"\rProgress: {current}/{total}", end='')
    def cli_status(message):
        print(f"\nStatus: {message}")

    # Get top N companies
    companies = collector.get_top_companies(num_companies=num, status_callback=cli_status)
    if not companies:
         print("Failed to get company list. Exiting.")
         return

    # Collect stock data for the specified period
    stock_data = collector.collect_stock_data(start_date=start, end_date=end, progress_callback=cli_progress, status_callback=cli_status)
    if not stock_data:
         print("Failed to collect stock data. Exiting.")
         return

    # Setup Reddit API (replace with your credentials if needed)
    # reddit_api_setup_success = collector.setup_reddit_api('YOUR_CLIENT_ID', 'YOUR_CLIENT_SECRET', 'YOUR_USER_AGENT')
    # if reddit_api_setup_success:
    #     reddit_data = collector.collect_reddit_sentiment(start_date=start, end_date=end, progress_callback=cli_progress, status_callback=cli_status)
    # else:
    #     print("Skipping Reddit data collection due to API setup failure.")
    #     reddit_data = {}

    # For demo purposes, creating empty Reddit data
    reddit_data = {}

    # Collect Google Trends data for the specified period
    # Collect Google Trends data (Note: collect_google_trends now saves files directly and returns None)
    collector.collect_google_trends(start_date=start, end_date=end, progress_callback=cli_progress, status_callback=cli_status)

    # Collect FRED data
    collector.collect_fred_data(start_date=start, end_date=end, status_callback=cli_status) # Saves file, returns df (optional use)

    # Process data by reading raw files for each ticker in stock_data
    # process_data will now internally load the saved FRED data file
    processed_tickers, representative_features = collector.process_data(
        stock_data, # Pass only the stock data dictionary
        progress_callback=cli_progress,
        status_callback=cli_status
    )

    print(f"\nData collection and processing complete. Processed {len(processed_tickers)} tickers.")
    if processed_tickers:
        print(f"Processed data saved to {PROCESSED_DATA_DIR}")
        print(f"Features from last ticker ({processed_tickers[-1]}): {representative_features}")
    else:
        print("No tickers were processed successfully.")

    # Return value was missing, added it back
    return processed_tickers, representative_features

if __name__ == "__main__":
    main()

