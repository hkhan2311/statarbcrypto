## Statistical Arbitrage in Cryptocurrency
## Project Description:
##This project aims to research momentum and reversal strategies in cryptocurrency markets, 
# leveraging statistical arbitrage techniques. By analyzing price-volume inefficiencies, 
# it explores potential market opportunities.
#  The research includes studying different time horizons, applying activity indicators, 
# testing correlation-based mean reversion, and evaluating macroeconomic impacts.
#  The project involves gathering data, backtesting strategies, optimizing execution,
#  and assessing performance using risk-adjusted metrics. 
# Coins being considered: (BTC, ETH, AVAX, XRP, ADA, DOGE, SHIB, PEPE, SOL, LINK, UNI)
## Description end

## Phase 1 - Research and Data Collection (1 Year Data, 30-Day Visualization)
import requests
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt
import seaborn as sns
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

TARGET_COINS = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'avalanche-2': 'AVAX',
    'ripple': 'XRP',
    'cardano': 'ADA',
    'dogecoin': 'DOGE',
    'shiba-inu': 'SHIB',
    'pepe': 'PEPE',
    'solana': 'SOL',
    'chainlink': 'LINK',
    'uniswap': 'UNI'
}

COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
API_DELAY = 3  # seconds between requests
REQUEST_TIMEOUT = 30  # seconds
VISUALIZATION_DAYS = 30  # Days to show in visualizations

# DATABASE SETUP
def create_database():
    """Initialize database with proper schema"""
    if os.path.exists('crypto_arbitrage.db'):
        os.remove('crypto_arbitrage.db')
    
    conn = sqlite3.connect('crypto_arbitrage.db')
    cursor = conn.cursor()
    
    # Create coins table
    cursor.execute('''
    CREATE TABLE coins (
        coin_id TEXT PRIMARY KEY,
        symbol TEXT,
        name TEXT
    )''')
    
    # Create OHLCV table with all necessary fields
    cursor.execute('''
    CREATE TABLE ohlcv_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin_id TEXT,
        date DATE,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        market_cap REAL,
        FOREIGN KEY (coin_id) REFERENCES coins (coin_id),
        UNIQUE(coin_id, date)
    )''')
    
    conn.commit()
    conn.close()
    print("Database created successfully with OHLCV support")

# DATA COLLECTION
def setup_api_session():
    """Configure requests session with retry logic"""
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    return session

def fetch_ohlcv_data(session, coin_id, days):
    """Fetch OHLCV data from CoinGecko API"""
    try:
        # Fetch OHLC data
        ohlc_response = session.get(
            f"{COINGECKO_API_URL}/coins/{coin_id}/ohlc",
            params={'vs_currency': 'usd', 'days': days},
            timeout=REQUEST_TIMEOUT
        )
        time.sleep(API_DELAY)
        
        # Fetch market data (for volume and market cap)
        market_response = session.get(
            f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart",
            params={'vs_currency': 'usd', 'days': days},
            timeout=REQUEST_TIMEOUT
        )
        time.sleep(API_DELAY)
        
        if ohlc_response.status_code == 200 and market_response.status_code == 200:
            return {
                'ohlc': ohlc_response.json(),
                'volumes': market_response.json()['total_volumes'],
                'market_caps': market_response.json()['market_caps']
            }
        else:
            print(f"API error for {TARGET_COINS[coin_id]}: OHLC {ohlc_response.status_code}, Market {market_response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data for {TARGET_COINS[coin_id]}: {str(e)}")
        return None

def process_ohlcv_data(coin_id, raw_data):
    """Process raw API data into clean DataFrame"""
    if not raw_data:
        return None
    
    # Process OHLC data
    ohlc_df = pd.DataFrame(raw_data['ohlc'], columns=['timestamp', 'open', 'high', 'low', 'close'])
    ohlc_df['date'] = pd.to_datetime(ohlc_df['timestamp'], unit='ms').dt.date
    
    # Process volume data
    volume_df = pd.DataFrame(raw_data['volumes'], columns=['timestamp', 'volume'])
    volume_df['date'] = pd.to_datetime(volume_df['timestamp'], unit='ms').dt.date
    
    # Process market cap data
    market_cap_df = pd.DataFrame(raw_data['market_caps'], columns=['timestamp', 'market_cap'])
    market_cap_df['date'] = pd.to_datetime(market_cap_df['timestamp'], unit='ms').dt.date
    
    # Merge all data
    merged_df = ohlc_df.merge(volume_df, on='date', how='left') \
                      .merge(market_cap_df, on='date', how='left')
    
    # Add coin identifier
    merged_df['coin_id'] = coin_id
    
    return merged_df[['coin_id', 'date', 'open', 'high', 'low', 'close', 'volume', 'market_cap']]

def store_data(conn, df):
    """Store processed data in database"""
    if df is None or df.empty:
        return 0
        
    cursor = conn.cursor()
    records = 0
    
    for _, row in df.iterrows():
        try:
            cursor.execute('''
            INSERT OR IGNORE INTO ohlcv_daily 
            (coin_id, date, open, high, low, close, volume, market_cap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                row['coin_id'],
                row['date'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
                row['market_cap']
            ))
            records += 1
        except sqlite3.Error as e:
            print(f"Database error for {row['coin_id']} on {row['date']}: {str(e)}")
    
    return records

def collect_all_data(session):
    """Orchestrate the complete data collection process"""
    conn = sqlite3.connect('crypto_arbitrage.db')
    
    # First store coin metadata
    for coin_id, symbol in TARGET_COINS.items():
        conn.execute("INSERT OR IGNORE INTO coins VALUES (?, ?, ?)", 
                    (coin_id, symbol, coin_id.title()))
    
    # Then collect OHLCV data for each coin
    for coin_id, symbol in TARGET_COINS.items():
        print(f"\nProcessing {symbol}...")
        
        # Fetch 1 year of data for strategy development
        raw_data = fetch_ohlcv_data(session, coin_id, days=365)
        processed_data = process_ohlcv_data(coin_id, raw_data)
        
        if processed_data is not None:
            records_stored = store_data(conn, processed_data)
            print(f"Stored {records_stored} days of data for {symbol}")
            
            # Visualize last 30 days
            visualize_recent_data(conn, coin_id, symbol)
        else:
            print(f"Skipping {symbol} due to data issues")
    
    conn.commit()
    conn.close()

# VISUALIZATION (30 DAYS)
def visualize_recent_data(conn, coin_id, symbol):
    """Generate visualizations for the most recent 30 days"""
    # Get last 30 days data
    df = pd.read_sql(f'''
    SELECT date, open, high, low, close, volume 
    FROM ohlcv_daily 
    WHERE coin_id = '{coin_id}'
    ORDER BY date DESC 
    LIMIT {VISUALIZATION_DAYS}
    ''', conn, parse_dates=['date'])
    
    if df.empty:
        print(f"No data to visualize for {symbol}")
        return
    
    df = df.sort_values('date')  # Sort chronologically
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Price Line Chart
    ax1.plot(df['date'], df['close'], label='Closing Price', color='blue')
    ax1.set_title(f'{symbol} Price - Last {VISUALIZATION_DAYS} Days')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Candlestick Chart
    ohlc = df[['date', 'open', 'high', 'low', 'close']].copy()
    ohlc['date'] = ohlc['date'].map(mdates.date2num)
    candlestick_ohlc(ax2, ohlc.values, width=0.6, 
                    colorup='green', colordown='red', alpha=0.8)
    ax2.set_title(f'{symbol} Candlesticks - Last {VISUALIZATION_DAYS} Days')
    ax2.set_ylabel('Price (USD)')
    ax2.grid(True)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # 3. Volume Chart
    ax3.bar(df['date'], df['volume'], color='purple', alpha=0.5)
    ax3.set_title(f'{symbol} Trading Volume - Last {VISUALIZATION_DAYS} Days')
    ax3.set_ylabel('Volume (USD)')
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.tight_layout()
    plt.show()

# DATA VALIDATION
def validate_data():
    """Verify data quality and completeness"""
    print("\nValidating collected data...")
    
    conn = sqlite3.connect('crypto_arbitrage.db')
    
    # Check data completeness
    completeness = pd.read_sql('''
    SELECT 
        c.symbol,
        COUNT(o.date) as days_available,
        MIN(o.date) as start_date,
        MAX(o.date) as end_date
    FROM coins c
    LEFT JOIN ohlcv_daily o ON c.coin_id = o.coin_id
    GROUP BY c.symbol
    ORDER BY days_available DESC
    ''', conn)
    
    print("\nData Completeness:")
    print(completeness)
    
    # Check data quality
    quality = pd.read_sql('''
    SELECT
        c.symbol,
        SUM(CASE WHEN o.open <= 0 OR o.high <= 0 OR o.low <= 0 OR o.close <= 0 THEN 1 ELSE 0 END) as invalid_prices,
        SUM(CASE WHEN o.high < o.low THEN 1 ELSE 0 END) as high_low_errors,
        SUM(CASE WHEN o.volume <= 0 THEN 1 ELSE 0 END) as invalid_volumes
    FROM ohlcv_daily o
    JOIN coins c ON o.coin_id = c.coin_id
    GROUP BY c.symbol
    ''', conn)
    
    print("\nData Quality Issues:")
    print(quality)
    
    conn.close()

# MAIN EXECUTION
if __name__ == "__main__":
    print("Starting Phase 1: Data Collection")
    
    # Initialize database and API session
    create_database()
    session = setup_api_session()
    
    # Collect data and generate visualizations
    collect_all_data(session)
    
    # Validate the collected data
    validate_data()
    
    print("\nPhase 1 complete!")
    print("Collected 1 year of OHLCV data for strategy development")
    print(f"Visualized last {VISUALIZATION_DAYS} days for presentation")

#Phase 2 Backtesting (to be continued)
