# Meta Trader Predictor

A professional AI-powered tool for predicting the next 3 H1 market candles using advanced machine learning techniques and technical indicators.

## Features

- Predicts next 3 H1 candles (Open, High, Low, Close)
- Real-time market data integration via MetaTrader 5
- Advanced technical indicators (EMA, Bollinger Bands, RSI)
- Professional terminal display with color-coded predictions
- Candlestick chart visualization
- Trading metrics (Sharpe ratio, Win/Loss ratio, Maximum drawdown)
- Market session awareness

## Requirements

- Python 3.8+
- MetaTrader 5 platform installed
- Required Python packages:
  - MetaTrader5
  - TensorFlow
  - Keras
  - pandas
  - numpy
  - matplotlib (for visualization)
  - colorama (for terminal colors)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MetaTraderPredictor.git
cd MetaTraderPredictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure MetaTrader 5 credentials in `config.py`

## Usage

1. Training the model:
```bash
python ai_training.py
```

2. Live prediction:
```bash
python main.py
```

## Project Structure

- `config.py`: Configuration settings and parameters
- `mt5_util.py`: MetaTrader 5 data fetching utilities
- `training_data_maker.py`: Data preparation and preprocessing
- `ai_training.py`: Model training and evaluation
- `main.py`: Live prediction interface
- `DOCUMENTARY.md`: Detailed technical documentation

## Model Architecture

- Stacked LSTM/GRU layers for temporal pattern recognition
- Specialized for non-linear market data
- Extensive feature engineering using technical indicators
- Validation on recent market data

## License

MIT License

## Disclaimer

This tool is for educational and research purposes only. Trading forex carries significant risks, and past performance does not guarantee future results. Always conduct your own analysis and risk assessment. 