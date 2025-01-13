"""
Main script for live market predictions
"""
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from colorama import init, Fore, Back, Style
import mplfinance as mpf
import mt5_util
import config

# Initialize colorama
init()

class MarketPredictor:
    def __init__(self):
        """Initialize the predictor"""
        # Load model with custom_objects=None to ensure proper loading of .keras format
        self.model = load_model(config.MODEL_CHECKPOINT_PATH, compile=True)
        data = np.load(config.TRAINING_DATA_PATH, allow_pickle=True)
        self.scaler_x = data['scaler_x'].item()
        self.scaler_y = data['scaler_y'].item()
        
        # Create charts directory
        os.makedirs(config.CHART_SAVE_PATH, exist_ok=True)
    
    def predict_next_candles(self, features):
        """Predict next 3 candles"""
        # Prepare features
        X = np.array([features])
        X_scaled = self.scaler_x.transform(X)
        X_reshaped = X_scaled.reshape((1, 1, -1))
        
        # Make prediction
        y_pred_scaled = self.model.predict(X_reshaped, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred[0]
    
    def calculate_confidence(self, features, prediction):
        """Calculate prediction confidence based on historical patterns"""
        # Simple confidence calculation based on feature similarity
        X = np.array([features])
        X_scaled = self.scaler_x.transform(X)
        
        # Calculate distance to training samples
        distances = np.linalg.norm(X_scaled - self.scaler_x.mean_, axis=1)
        confidence = 1 / (1 + distances)
        
        return confidence[0]
    
    def plot_predictions(self, last_candles, predictions):
        """Plot candlestick chart with predictions"""
        # Prepare data for plotting
        dates = pd.date_range(
            start=last_candles.index[-1],
            periods=5,  # 2 actual + 3 predicted
            freq='H'
        )
        
        # Create DataFrame with actual candles
        df_actual = last_candles.copy()
        
        # Create DataFrame with predicted candles
        df_pred = pd.DataFrame(
            predictions.reshape(3, 4),
            columns=['Open', 'High', 'Low', 'Close'],
            index=dates[-3:]
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual candles
        mpf.plot(
            df_actual,
            type='candle',
            style='charles',
            title='Market Prediction',
            ax=ax,
            volume=False
        )
        
        # Plot predicted candles (with transparency)
        mpf.plot(
            df_pred,
            type='candle',
            style='charles',
            alpha=0.7,
            ax=ax,
            volume=False
        )
        
        plt.savefig(os.path.join(config.CHART_SAVE_PATH, 'prediction.png'))
        plt.close()
    
    def display_prediction(self, current_time, session, confidence, last_price, predictions):
        """Display prediction in terminal"""
        print("\033[2J\033[H")  # Clear screen
        
        # Header
        print(f"{Back.BLUE}{Fore.WHITE} META TRADER PREDICTOR {Style.RESET_ALL}")
        print(f"{Fore.CYAN}Time: {current_time} UTC{Style.RESET_ALL}")
        print(f"Market Session: {Fore.BLUE}{session}{Style.RESET_ALL}")
        print(f"Current Price: {Fore.YELLOW}{last_price:.5f}{Style.RESET_ALL}")
        print("\n" + "="*50 + "\n")
        
        # Predictions for each candle
        for i in range(3):
            start_idx = i * 4
            candle = predictions[start_idx:start_idx+4]
            direction = "BULLISH" if candle[3] > candle[0] else "BEARISH"
            color = Fore.GREEN if direction == "BULLISH" else Fore.RED
            
            print(f"{Fore.CYAN}H{i+1} Prediction:{Style.RESET_ALL}")
            print(f"Direction: {color}{direction}{Style.RESET_ALL}")
            print(f"Open:  {Fore.WHITE}{candle[0]:.5f}{Style.RESET_ALL}")
            print(f"High:  {Fore.GREEN}{candle[1]:.5f}{Style.RESET_ALL}")
            print(f"Low:   {Fore.RED}{candle[2]:.5f}{Style.RESET_ALL}")
            print(f"Close: {color}{candle[3]:.5f}{Style.RESET_ALL}")
            print(f"Movement: {color}{(candle[3]-candle[0])*10000:.1f} pips{Style.RESET_ALL}")
            print("-"*30)
        
        # Confidence
        conf_color = Fore.GREEN if confidence > 0.7 else Fore.YELLOW if confidence > 0.5 else Fore.RED
        print(f"\nPrediction Confidence: {conf_color}{confidence:.1%}{Style.RESET_ALL}")
        
        print("\nChart saved as prediction.png")
        print("\n" + "="*50)
    
    def run(self):
        """Run live predictions"""
        print("Initializing MT5...")
        if not mt5_util.initialize_mt5():
            return
        
        try:
            while True:
                # Get current data
                features, last_candles = mt5_util.prepare_live_data(config.MT5_SYMBOLS[0])
                if features is None:
                    print("Failed to get market data")
                    time.sleep(config.RETRY_DELAY)
                    continue
                
                # Flatten and combine features
                flat_features = np.concatenate([
                    features['ohlc'].flatten(),
                    features['ema_short'],
                    features['ema_long'],
                    features['bb_upper'],
                    features['bb_middle'],
                    features['bb_lower'],
                    features['rsi']
                ])
                
                # Make prediction
                predictions = self.predict_next_candles(flat_features)
                confidence = self.calculate_confidence(flat_features, predictions)
                
                # Get current market session
                sessions = mt5_util.get_current_market_session()
                
                # Plot predictions
                self.plot_predictions(last_candles, predictions)
                
                # Display results
                self.display_prediction(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ", ".join(sessions),
                    confidence,
                    last_candles['close'].iloc[-1],
                    predictions
                )
                
                # Wait for next update
                time.sleep(config.UPDATE_FREQUENCY)
        
        finally:
            mt5_util.cleanup_mt5()

if __name__ == "__main__":
    predictor = MarketPredictor()
    predictor.run() 