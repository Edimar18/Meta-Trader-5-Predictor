"""
Training data preparation script
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import mt5_util
import config
from tqdm import tqdm

def create_training_data():
    """Create and save training data"""
    print("Initializing MT5...")
    if not mt5_util.initialize_mt5():
        return False
    
    try:
        # Set start date from config
        start_date = datetime(
            config.HISTORY_START_YEAR,
            config.HISTORY_START_MONTH,
            config.HISTORY_START_DAY
        )
        
        all_features = []
        all_labels = []
        
        # Process each symbol
        for symbol in tqdm(config.MT5_SYMBOLS, desc="Processing symbols"):
            print(f"\nFetching data for {symbol}...")
            
            # Get raw training data
            X, y = mt5_util.get_training_data(symbol, start_date)
            if X is None or y is None:
                print(f"Failed to get training data for {symbol}")
                continue
            
            all_features.append(X)
            all_labels.append(y)
        
        if not all_features:
            print("No training data collected")
            return False
        
        # Combine data from all symbols
        X = np.concatenate(all_features)
        y = np.concatenate(all_labels)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config.TRAINING_DATA_PATH), exist_ok=True)
        
        # Scale features
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Save training data and scalers
        np.savez(
            config.TRAINING_DATA_PATH,
            X=X_scaled,
            y=y_scaled,
            scaler_x=scaler_x,
            scaler_y=scaler_y
        )
        
        print(f"\nTraining data saved to {config.TRAINING_DATA_PATH}")
        print(f"Total samples: {len(X)}")
        print(f"Feature shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        
        return True
    
    finally:
        mt5_util.cleanup_mt5()

def load_training_data():
    """Load training data and scalers"""
    data = np.load(config.TRAINING_DATA_PATH, allow_pickle=True)
    return data['X'], data['y'], data['scaler_x'].item(), data['scaler_y'].item()

if __name__ == "__main__":
    create_training_data() 