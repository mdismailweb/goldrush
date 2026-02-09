# Stock Price Prediction App

This application uses machine learning (XGBoost) to predict tomorrow's stock prices based on today's data.

## Features

### User Input Required (Current Day):
- ✅ Closing Price
- ✅ Opening Price
- ✅ High Price
- ✅ Low Price
- ✅ Volume

### Automatically Calculated:
- ✅ Moving Averages (7, 30, 90 days)
- ✅ RSI (Relative Strength Index)
- ✅ MACD (Moving Average Convergence Divergence)
- ✅ Volatility (7-day and 30-day)
- ✅ Bollinger Bands (Upper and Lower)
- ✅ Daily Return

### Model Predictions:
- ✅ Tomorrow's **High Price**
- ✅ Tomorrow's **Low Price**
- ✅ Tomorrow's **Close Price**

## Files

1. `train_models.py` - Train the RandomForest models and save them as .pkl files
2. `stock_predictor_app.py` - Command-line application for making predictions
3. `app.py` - **Streamlit web UI** (recommended) - Beautiful web interface
4. `test_app.py` - Test script to verify functionality
5. `model_high.pkl` - Trained model for high price prediction
6. `model_low.pkl` - Trained model for low price prediction
7. `model_close.pkl` - Trained model for close price prediction

## How to Use

### 🌐 Option 1: Streamlit Web UI (Recommended)

**The easiest and most beautiful way to use the app!**

```bash
streamlit run app.py
```

This will:
- Open a beautiful web interface in your browser
- Provide interactive input forms for price data
- Display technical indicators in organized tabs
- Show predictions with visual charts
- Include real-time calculations

**Features:**
- 📱 Modern, responsive web interface
- 📊 Interactive Plotly charts
- 🎨 Color-coded metrics and indicators
- 📈 Visual price comparisons
- ⚡ Real-time predictions

### 💻 Option 2: Command Line Interface

```bash
python stock_predictor_app.py
```
- Load the gold price forecasting dataset
- Train three separate XGBoost models (high, low, close)
- Save the models as .pkl files
- Display model performance metrics

### Step 2: Run the Prediction App

```bash
python stock_predictor_app.py
```

This will:
- Load the trained models from .pkl files
- Prompt you to enter today's price data
- Automatically calculate all technical indicators
- Predict tomorrow's high, low, and close prices
- Display the results with expected percentage changes

## Example Usage

```
STOCK PRICE PREDICTION APP
============================================================

Please enter today's price data:
------------------------------------------------------------
Closing Price:  5318.40
Opening Price:  5415.70
High Price:     5586.20
Low Price:      5097.50
Volume:         23709

Calculating technical indicators...

============================================================
CALCULATED TECHNICAL INDICATORS
============================================================
Moving Average (7-day):     5070.91
Moving Average (30-day):    4607.14
Moving Average (90-day):    4241.72
Daily Return:               0.0032 (0.32%)
Volatility (7-day):         0.0142
Volatility (30-day):        0.0162
RSI:                        87.23
MACD:                       227.37
MACD Signal:                164.68
Bollinger Band (Upper):     5310.17
Bollinger Band (Lower):     4112.72

============================================================
MAKING PREDICTIONS...
============================================================

============================================================
TOMORROW'S PRICE PREDICTIONS
============================================================

Current Day's Data:
  Open:   5415.70
  High:   5586.20
  Low:    5097.50
  Close:  5318.40
  Volume: 23,709

Predicted Tomorrow's Prices:
  High:   5650.25
  Low:    5200.15
  Close:  5425.80

Expected Change:
  High:   +1.15%
  Low:    +2.01%
  Close:  +2.02%
============================================================
```

## Requirements

```bash
pip install pandas numpy scikit-learn streamlit plotly
```

## Technical Details

- **Algorithm**: XGBoost (Gradient Boosting)
- **Models**: 3 separate models (one for each prediction target)
- **Features**: 15 technical indicators + OHLCV data
- **Model Parameters**:
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
  - subsample: 0.8
  - colsample_bytree: 0.8

## Answer to Your Question

**Yes, it is absolutely possible!** This application:

1. ✅ Uses `.pkl` files to store trained ML models
2. ✅ Takes user input for current day's OHLCV data
3. ✅ Automatically calculates all technical indicators
4. ✅ Predicts tomorrow's **High**, **Low**, and **Close** prices

The app works by:
- Loading pre-trained models from `.pkl` files
- Using historical data (if available) to calculate accurate technical indicators
- Feeding the calculated indicators to the models
- Generating predictions for the next day's prices
