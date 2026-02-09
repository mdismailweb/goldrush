import pickle
import numpy as np
import pandas as pd
from datetime import datetime

class StockPricePredictor:
    """
    Stock Price Prediction App
    Takes current day's OHLCV data and predicts tomorrow's high, low, and close prices
    """
    
    def __init__(self):
        """Load the trained models"""
        print("Loading trained models...")
        try:
            with open('model_high.pkl', 'rb') as f:
                self.model_high = pickle.load(f)
            with open('model_low.pkl', 'rb') as f:
                self.model_low = pickle.load(f)
            with open('model_close.pkl', 'rb') as f:
                self.model_close = pickle.load(f)
            print("✓ All models loaded successfully!\n")
        except FileNotFoundError as e:
            print(f"Error: Model files not found. Please run 'train_models.py' first.")
            raise e
    
    def calculate_technical_indicators(self, historical_data, current_data):
        """
        Calculate technical indicators based on historical and current data
        
        Parameters:
        - historical_data: DataFrame with past price data
        - current_data: dict with today's OHLCV data
        """
        # Combine historical and current data
        df = historical_data.copy()
        current_row = pd.DataFrame([current_data])
        df = pd.concat([df, current_row], ignore_index=True)
        
        # Calculate Moving Averages
        ma_7 = df['close'].tail(7).mean() if len(df) >= 7 else df['close'].mean()
        ma_30 = df['close'].tail(30).mean() if len(df) >= 30 else df['close'].mean()
        ma_90 = df['close'].tail(90).mean() if len(df) >= 90 else df['close'].mean()
        
        # Calculate Daily Return
        if len(df) >= 2:
            daily_return = (current_data['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']
        else:
            daily_return = 0.0
        
        # Calculate Volatility (standard deviation of returns)
        returns = df['close'].pct_change()
        volatility_7 = returns.tail(7).std() if len(df) >= 7 else 0.01
        volatility_30 = returns.tail(30).std() if len(df) >= 30 else 0.01
        
        # Calculate RSI
        rsi = self.calculate_rsi(df['close'])
        
        # Calculate MACD
        macd, macd_signal = self.calculate_macd(df['close'])
        
        # Calculate Bollinger Bands
        bb_upper, bb_lower = self.calculate_bollinger_bands(df['close'])
        
        indicators = {
            'close': current_data['close'],
            'low': current_data['low'],
            'open': current_data['open'],
            'volume': current_data['volume'],
            'ma_7': ma_7,
            'ma_30': ma_30,
            'ma_90': ma_90,
            'daily_return': daily_return,
            'volatility_7': volatility_7,
            'volatility_30': volatility_30,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower
        }
        
        return indicators
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0)
        loss = -deltas.where(deltas < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal line"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        return macd.iloc[-1], macd_signal.iloc[-1]
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        
        bb_upper = sma + (std * std_dev)
        bb_lower = sma - (std * std_dev)
        
        return bb_upper.iloc[-1], bb_lower.iloc[-1]
    
    def predict(self, current_data, historical_data=None):
        """
        Predict tomorrow's high, low, and close prices
        
        Parameters:
        - current_data: dict with keys 'close', 'open', 'high', 'low', 'volume'
        - historical_data: DataFrame with past price data (optional, for better indicator calculation)
        
        Returns:
        - dict with predicted high, low, and close prices
        """
        # If no historical data provided, use current data only
        if historical_data is None:
            historical_data = pd.DataFrame()
        
        # Calculate technical indicators
        print("\nCalculating technical indicators...")
        indicators = self.calculate_technical_indicators(historical_data, current_data)
        
        # Display calculated indicators
        print("\n" + "="*60)
        print("CALCULATED TECHNICAL INDICATORS")
        print("="*60)
        print(f"Moving Average (7-day):     {indicators['ma_7']:.2f}")
        print(f"Moving Average (30-day):    {indicators['ma_30']:.2f}")
        print(f"Moving Average (90-day):    {indicators['ma_90']:.2f}")
        print(f"Daily Return:               {indicators['daily_return']:.4f} ({indicators['daily_return']*100:.2f}%)")
        print(f"Volatility (7-day):         {indicators['volatility_7']:.4f}")
        print(f"Volatility (30-day):        {indicators['volatility_30']:.4f}")
        print(f"RSI:                        {indicators['rsi']:.2f}")
        print(f"MACD:                       {indicators['macd']:.2f}")
        print(f"MACD Signal:                {indicators['macd_signal']:.2f}")
        print(f"Bollinger Band (Upper):     {indicators['bb_upper']:.2f}")
        print(f"Bollinger Band (Lower):     {indicators['bb_lower']:.2f}")
        
        # Prepare features for prediction
        feature_cols = ["close", "low", "open", "volume", "ma_7", "ma_30", "ma_90",
                       "daily_return", "volatility_7", "volatility_30", "rsi", 
                       "macd", "macd_signal", "bb_upper", "bb_lower"]
        
        features = pd.DataFrame([indicators])[feature_cols]
        
        # Make predictions
        print("\n" + "="*60)
        print("MAKING PREDICTIONS...")
        print("="*60)
        
        pred_high = self.model_high.predict(features)[0]
        pred_low = self.model_low.predict(features)[0]
        pred_close = self.model_close.predict(features)[0]
        
        predictions = {
            'tomorrow_high': pred_high,
            'tomorrow_low': pred_low,
            'tomorrow_close': pred_close
        }
        
        return predictions
    
    def display_predictions(self, current_data, predictions):
        """Display the prediction results in a nice format"""
        print("\n" + "="*60)
        print("TOMORROW'S PRICE PREDICTIONS")
        print("="*60)
        print(f"\nCurrent Day's Data:")
        print(f"  Open:   {current_data['open']:.2f}")
        print(f"  High:   {current_data['high']:.2f}")
        print(f"  Low:    {current_data['low']:.2f}")
        print(f"  Close:  {current_data['close']:.2f}")
        print(f"  Volume: {current_data['volume']:,}")
        
        print(f"\nPredicted Tomorrow's Prices:")
        print(f"  High:   {predictions['tomorrow_high']:.2f}")
        print(f"  Low:    {predictions['tomorrow_low']:.2f}")
        print(f"  Close:  {predictions['tomorrow_close']:.2f}")
        
        # Calculate expected changes
        high_change = ((predictions['tomorrow_high'] - current_data['high']) / current_data['high']) * 100
        low_change = ((predictions['tomorrow_low'] - current_data['low']) / current_data['low']) * 100
        close_change = ((predictions['tomorrow_close'] - current_data['close']) / current_data['close']) * 100
        
        print(f"\nExpected Change:")
        print(f"  High:   {high_change:+.2f}%")
        print(f"  Low:    {low_change:+.2f}%")
        print(f"  Close:  {close_change:+.2f}%")
        print("="*60 + "\n")


def get_user_input():
    """Get current day's price data from user"""
    print("\n" + "="*60)
    print("STOCK PRICE PREDICTION APP")
    print("="*60)
    print("\nPlease enter today's price data:")
    print("-" * 60)
    
    try:
        close = float(input("Closing Price:  "))
        open_price = float(input("Opening Price:  "))
        high = float(input("High Price:     "))
        low = float(input("Low Price:      "))
        volume = int(input("Volume:         "))
        
        current_data = {
            'close': close,
            'open': open_price,
            'high': high,
            'low': low,
            'volume': volume
        }
        
        return current_data
    
    except ValueError:
        print("\nError: Please enter valid numeric values!")
        return None


def main():
    """Main application"""
    # Initialize predictor
    predictor = StockPricePredictor()
    
    # Load historical data (optional, for better indicator calculation)
    try:
        historical_data = pd.read_csv("gold_price_forecasting_dataset.csv")
        print(f"✓ Historical data loaded: {len(historical_data)} records\n")
    except FileNotFoundError:
        print("⚠ Warning: Historical data not found. Using current data only.\n")
        historical_data = None
    
    # Get user input
    current_data = get_user_input()
    
    if current_data is None:
        return
    
    # Make predictions
    predictions = predictor.predict(current_data, historical_data)
    
    # Display results
    predictor.display_predictions(current_data, predictions)


if __name__ == "__main__":
    main()
