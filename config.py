"""
Configuration settings for Meta Trader Predictor
"""

# MetaTrader 5 Settings
MT5_TIMEFRAME = "H1"  # 1-hour timeframe
MT5_SYMBOLS = ["EURUSD"]  # Add more symbols as needed
HISTORY_START_YEAR = 2020
HISTORY_START_MONTH = 1
HISTORY_START_DAY = 1

# Technical Indicators Parameters
EMA_SHORT_PERIOD = 9
EMA_LONG_PERIOD = 21
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
RSI_PERIOD = 14

# Feature Engineering
LOOKBACK_PERIODS = {
    "EMA": 25,
    "BOLLINGER": 25,
    "RSI": 25,
    "OHLC": 50
}

# Model Architecture
MODEL_PARAMS = {
    "lstm_units": [128, 64, 32],
    "dense_units": [64, 32],
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "validation_split": 0.2
}

# Training Settings
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
MODEL_CHECKPOINT_PATH = "models/best_model.h5"
TRAINING_DATA_PATH = "data/training_data.csv"

# Market Sessions (UTC)
MARKET_SESSIONS = {
    "Asian": {"start": 0, "end": 9},
    "European": {"start": 7, "end": 16},
    "US": {"start": 13, "end": 22}
}

# Display Settings
TERMINAL_COLORS = {
    "bullish": "green",
    "bearish": "red",
    "neutral": "white",
    "confidence": "yellow",
    "session": "blue"
}

UPDATE_FREQUENCY = 60  # seconds
CONFIDENCE_THRESHOLD = 0.7

# Trading Metrics
RISK_FREE_RATE = 0.02  # For Sharpe ratio calculation
DRAWDOWN_THRESHOLD = 0.1  # 10% maximum drawdown warning

# Paths
LOG_FILE = "logs/predictor.log"
CHART_SAVE_PATH = "charts/"

# Error Handling
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Debug Mode
DEBUG = False 