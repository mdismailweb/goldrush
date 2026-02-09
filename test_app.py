"""
Test script to demonstrate the stock predictor app
This will simulate user input with sample data
"""

from stock_predictor_app import StockPricePredictor
import pandas as pd

# Initialize predictor
predictor = StockPricePredictor()

# Load historical data
historical_data = pd.read_csv("gold_price_forecasting_dataset.csv")
print(f"✓ Historical data loaded: {len(historical_data)} records\n")

# Sample current day data (using the last day from the dataset as an example)
current_data = {
    'close': 4713.90,
    'open': 5376.40,
    'high': 5440.50,
    'low': 4700.00,
    'volume': 23709
}

print("="*60)
print("STOCK PRICE PREDICTION APP - TEST RUN")
print("="*60)
print("\nSample Current Day's Data:")
print(f"  Open:   {current_data['open']:.2f}")
print(f"  High:   {current_data['high']:.2f}")
print(f"  Low:    {current_data['low']:.2f}")
print(f"  Close:  {current_data['close']:.2f}")
print(f"  Volume: {current_data['volume']:,}")

# Make predictions
predictions = predictor.predict(current_data, historical_data)

# Display results
predictor.display_predictions(current_data, predictions)

print("\n✅ TEST COMPLETED SUCCESSFULLY!")
print("\nYou can now run 'python stock_predictor_app.py' to enter your own data.")
