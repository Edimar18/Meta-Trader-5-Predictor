# Meta Trader Predictor - Technical Documentation

## System Architecture

### 1. Data Collection (mt5_util.py)
- Historical data fetching from MetaTrader 5 (2020 to present)
- Real-time data streaming for live predictions
- Market session detection (Asian, European, US)
- Technical indicator calculation:
  - EMA (short and long period)
  - Bollinger Bands (upper, middle, lower)
  - RSI
  - OHLC data processing

### 2. Data Preprocessing (training_data_maker.py)
- Feature engineering:
  - Past 25 H1 EMA (long period)
  - Past 25 H1 EMA (short period)
  - Past 25 H1 Bollinger Bands
  - Past 25 RSI
  - Past 50 H1 OHLC (200 features)
- Data cleaning and normalization
- Sequence generation with 1 H1 candle intervals
- Training/validation split using recent data approach

### 3. Model Architecture (ai_training.py)
#### Input Layer
- 275 features (25×2 EMA + 25×3 BB + 25 RSI + 50×4 OHLC)

#### Hidden Layers
1. LSTM/GRU Layer Stack
   - Temporal pattern recognition
   - Non-linear market behavior handling
   - Dropout for regularization

2. Dense Layer Stack
   - Feature combination
   - Pattern recognition
   - Activation: ReLU/LeakyReLU

#### Output Layer
- 12 nodes (3 candles × 4 OHLC values)
- Linear activation for price prediction

### 4. Training Process
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Validation strategy: Recent data validation
- Early stopping with patience
- Model checkpointing
- Performance metrics:
  - Prediction accuracy
  - Maximum drawdown
  - Sharpe ratio
  - Win/Loss ratio
  - Profit factor

### 5. Live Prediction System (main.py)
#### Data Pipeline
- Real-time MT5 data fetching
- Feature calculation
- Data normalization
- Prediction generation

#### Output Display
- Terminal Interface:
  - Color coding:
    - Green: Bullish predictions
    - Red: Bearish predictions
    - Yellow/White: Confidence levels
    - Blue: Market session indicators
  - Prediction confidence scores
  - Risk metrics
  - Market session information

- Graphical Interface:
  - Candlestick chart
  - Predicted candles overlay
  - Technical indicators
  - Confidence intervals

## Configuration (config.py)

### MT5 Settings
- Login credentials
- Server settings
- Timeframe configurations

### Model Parameters
- Sequence length
- Batch size
- Learning rate
- Layer configurations
- Dropout rates

### Technical Indicators
- EMA periods
- Bollinger Bands parameters
- RSI period

### Display Settings
- Update frequency
- Color schemes
- Chart parameters

## Error Handling
- MT5 connection issues
- Data validation
- Model prediction errors
- Real-time update failures

## Performance Optimization
- Batch processing
- Data caching
- Efficient feature calculation
- Memory management

## Future Improvements
- Additional technical indicators
- Economic calendar integration
- Market sentiment analysis
- Position sizing optimization
- Multi-timeframe analysis 