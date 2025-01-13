"""
AI model training script
"""
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config
from training_data_maker import load_training_data

def create_model(input_shape):
    """Create the neural network model"""
    model = Sequential([
        # First LSTM layer
        LSTM(config.MODEL_PARAMS['lstm_units'][0],
             input_shape=input_shape,
             return_sequences=True),
        BatchNormalization(),
        Dropout(config.MODEL_PARAMS['dropout_rate']),
        
        # Second LSTM layer
        LSTM(config.MODEL_PARAMS['lstm_units'][1],
             return_sequences=True),
        BatchNormalization(),
        Dropout(config.MODEL_PARAMS['dropout_rate']),
        
        # GRU layer
        GRU(config.MODEL_PARAMS['lstm_units'][2]),
        BatchNormalization(),
        Dropout(config.MODEL_PARAMS['dropout_rate']),
        
        # Dense layers
        Dense(config.MODEL_PARAMS['dense_units'][0], activation='relu'),
        BatchNormalization(),
        Dropout(config.MODEL_PARAMS['dropout_rate']),
        
        Dense(config.MODEL_PARAMS['dense_units'][1], activation='relu'),
        BatchNormalization(),
        
        # Output layer (12 values: 3 candles × 4 OHLC values)
        Dense(12, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config.MODEL_PARAMS['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def calculate_trading_metrics(y_true, y_pred, scaler_y):
    """Calculate trading-specific metrics"""
    # Inverse transform predictions
    y_true = scaler_y.inverse_transform(y_true)
    y_pred = scaler_y.inverse_transform(y_pred)
    
    # Calculate metrics for each predicted candle
    metrics = {}
    for i in range(3):  # 3 predicted candles
        start_idx = i * 4
        true_candle = y_true[:, start_idx:start_idx+4]
        pred_candle = y_pred[:, start_idx:start_idx+4]
        
        # Direction accuracy
        true_direction = true_candle[:, -1] > true_candle[:, 0]  # Close > Open
        pred_direction = pred_candle[:, -1] > pred_candle[:, 0]
        direction_accuracy = np.mean(true_direction == pred_direction)
        
        # Price movement accuracy
        true_movement = true_candle[:, -1] - true_candle[:, 0]
        pred_movement = pred_candle[:, -1] - pred_candle[:, 0]
        movement_correlation = np.corrcoef(true_movement, pred_movement)[0, 1]
        
        metrics[f'candle_{i+1}'] = {
            'direction_accuracy': direction_accuracy,
            'movement_correlation': movement_correlation,
            'mse': mean_squared_error(true_candle, pred_candle),
            'mae': mean_absolute_error(true_candle, pred_candle)
        }
    
    return metrics

def train_model():
    """Train the AI model"""
    # Load training data
    X_train, y_train, scaler_x, scaler_y = load_training_data()
    
    # Split into training and validation
    split_idx = int(len(X_train) * (1 - config.MODEL_PARAMS['validation_split']))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    # Reshape input for LSTM [samples, timesteps, features]
    feature_size = X_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], 1, feature_size))
    X_val = X_val.reshape((X_val.shape[0], 1, feature_size))
    
    # Create model
    model = create_model((1, feature_size))
    print(model.summary())
    
    # Create directory for model checkpoints
    os.makedirs(os.path.dirname(config.MODEL_CHECKPOINT_PATH), exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            config.MODEL_CHECKPOINT_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.MODEL_PARAMS['epochs'],
        batch_size=config.MODEL_PARAMS['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Calculate predictions
    y_pred = model.predict(X_val)
    
    # Calculate trading metrics
    metrics = calculate_trading_metrics(y_val, y_pred, scaler_y)
    
    # Print metrics
    print("\nTrading Metrics:")
    for candle, m in metrics.items():
        print(f"\n{candle.upper()}:")
        print(f"Direction Accuracy: {m['direction_accuracy']:.2%}")
        print(f"Movement Correlation: {m['movement_correlation']:.4f}")
        print(f"MSE: {m['mse']:.6f}")
        print(f"MAE: {m['mae']:.6f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHART_SAVE_PATH, 'training_history.png'))
    plt.close()
    
    # Save final model explicitly in keras format
    model.save(config.MODEL_CHECKPOINT_PATH, save_format='keras')
    
    return model, scaler_x, scaler_y

if __name__ == "__main__":
    train_model() 