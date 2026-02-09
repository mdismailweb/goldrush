import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_and_save_models():
    """
    Train three separate models for predicting:
    1. Tomorrow's High price
    2. Tomorrow's Low price
    3. Tomorrow's Close price
    """
    print("Loading dataset...")
    df = pd.read_csv("gold_price_forecasting_dataset.csv")
    print(f"Dataset loaded: {len(df)} records")
    
    # Features (excluding date, adj close, and target variables)
    feature_cols = ["close", "low", "open", "volume", "ma_7", "ma_30", "ma_90",
                    "daily_return", "volatility_7", "volatility_30", "rsi", 
                    "macd", "macd_signal", "bb_upper", "bb_lower"]
    
    X = df[feature_cols]
    
    # Train three separate models
    models = {}
    targets = {
        'high': 'high',
        'low': 'low',
        'close': 'close'
    }
    
    for model_name, target_col in targets.items():
        print(f"\n{'='*60}")
        print(f"Training model for: {target_col.upper()}")
        print(f"{'='*60}")
        
        y = df[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train the model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"Training {model_name} model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"  Mean Squared Error: {mse:.2f}")
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  R² Score: {r2:.4f}")
        
        # Save the model
        model_filename = f'model_{model_name}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved as: {model_filename}")
        
        models[model_name] = model
    
    print(f"\n{'='*60}")
    print("All models trained and saved successfully!")
    print(f"{'='*60}")
    
    return models

if __name__ == "__main__":
    train_and_save_models()
