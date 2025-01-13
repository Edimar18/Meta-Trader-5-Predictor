"""
MetaTrader 5 utility functions for data fetching and processing
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import config

def initialize_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return False
    return True

def get_historical_data(symbol, timeframe, start_pos, count):
    """Fetch historical data from MT5"""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, start_pos, count)
    if rates is None:
        print(f"Failed to get historical data: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataset"""
    # EMA calculations
    ema_short = EMAIndicator(close=df['close'], window=config.EMA_SHORT_PERIOD)
    ema_long = EMAIndicator(close=df['close'], window=config.EMA_LONG_PERIOD)
    df['ema_short'] = ema_short.ema_indicator()
    df['ema_long'] = ema_long.ema_indicator()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=config.BOLLINGER_PERIOD, window_dev=config.BOLLINGER_STD)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    
    # RSI
    rsi = RSIIndicator(close=df['close'], window=config.RSI_PERIOD)
    df['rsi'] = rsi.rsi()
    
    return df

def get_current_market_session():
    """Determine current market session based on UTC time"""
    current_hour = datetime.now(pytz.UTC).hour
    
    sessions = []
    for session, times in config.MARKET_SESSIONS.items():
        if times['start'] <= current_hour < times['end']:
            sessions.append(session)
    
    return sessions if sessions else ['No Major Session']

def prepare_live_data(symbol):
    """Prepare data for live prediction"""
    # Get enough historical data for features
    max_lookback = max(config.LOOKBACK_PERIODS.values())
    extra_periods = 50  # Extra periods for calculating indicators
    
    df = get_historical_data(symbol, mt5.TIMEFRAME_H1, 0, max_lookback + extra_periods)
    if df is None:
        return None
    
    # Calculate indicators
    df = calculate_technical_indicators(df)
    
    # Prepare feature sets
    features = {
        'ohlc': df[['open', 'high', 'low', 'close']].values[-config.LOOKBACK_PERIODS['OHLC']:],
        'ema_short': df['ema_short'].values[-config.LOOKBACK_PERIODS['EMA']:],
        'ema_long': df['ema_long'].values[-config.LOOKBACK_PERIODS['EMA']:],
        'bb_upper': df['bb_upper'].values[-config.LOOKBACK_PERIODS['BOLLINGER']:],
        'bb_middle': df['bb_middle'].values[-config.LOOKBACK_PERIODS['BOLLINGER']:],
        'bb_lower': df['bb_lower'].values[-config.LOOKBACK_PERIODS['BOLLINGER']:],
        'rsi': df['rsi'].values[-config.LOOKBACK_PERIODS['RSI']:],
    }
    
    return features, df.iloc[-2:]  # Return features and last 2 candles for visualization

def get_training_data(symbol, start_date):
    """Fetch and prepare training data"""
    # Calculate total required periods
    max_lookback = max(config.LOOKBACK_PERIODS.values())
    prediction_periods = 3  # Number of future candles to predict
    total_periods = (datetime.now() - start_date).days * 24  # Convert to hours
    
    # Fetch historical data
    df = get_historical_data(symbol, mt5.TIMEFRAME_H1, 0, total_periods + max_lookback)
    if df is None:
        return None, None
    
    # Calculate indicators
    df = calculate_technical_indicators(df)
    
    # Prepare training sequences
    X, y = [], []
    
    for i in range(max_lookback, len(df) - prediction_periods):
        # Input features
        features = []
        # Add OHLC
        features.extend(df[['open', 'high', 'low', 'close']].values[i-config.LOOKBACK_PERIODS['OHLC']:i].flatten())
        # Add indicators
        features.extend(df['ema_short'].values[i-config.LOOKBACK_PERIODS['EMA']:i])
        features.extend(df['ema_long'].values[i-config.LOOKBACK_PERIODS['EMA']:i])
        features.extend(df['bb_upper'].values[i-config.LOOKBACK_PERIODS['BOLLINGER']:i])
        features.extend(df['bb_middle'].values[i-config.LOOKBACK_PERIODS['BOLLINGER']:i])
        features.extend(df['bb_lower'].values[i-config.LOOKBACK_PERIODS['BOLLINGER']:i])
        features.extend(df['rsi'].values[i-config.LOOKBACK_PERIODS['RSI']:i])
        
        # Output labels (next 3 candles OHLC)
        labels = df[['open', 'high', 'low', 'close']].values[i:i+prediction_periods].flatten()
        
        X.append(features)
        y.append(labels)
    
    return np.array(X), np.array(y)

def cleanup_mt5():
    """Cleanup MT5 connection"""
    mt5.shutdown() 