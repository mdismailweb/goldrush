import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #1f1f1f !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0e1117 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white !important;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card h2, .metric-card h3, .metric-card p {
        color: white !important;
    }
    .indicator-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .indicator-card h4 {
        color: #1f1f1f !important;
        margin-bottom: 10px;
    }
    .indicator-card p {
        color: #333333 !important;
        margin: 5px 0;
    }
    h1 {
        color: #667eea !important;
        font-weight: 700;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    /* Ensure good contrast for info boxes */
    .stAlert {
        background-color: #e8f4f8 !important;
        color: #0c5460 !important;
    }
    /* Tab styling for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        color: #1f1f1f !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

class StockPricePredictor:
    """Stock Price Prediction Engine"""
    
    def __init__(self):
        """Load the trained models"""
        self.models_loaded = False
        try:
            with open('model_high.pkl', 'rb') as f:
                self.model_high = pickle.load(f)
            with open('model_low.pkl', 'rb') as f:
                self.model_low = pickle.load(f)
            with open('model_close.pkl', 'rb') as f:
                self.model_close = pickle.load(f)
            self.models_loaded = True
        except FileNotFoundError:
            st.error("⚠️ Model files not found. Please run 'train_models.py' first.")
    
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
    
    def calculate_technical_indicators(self, historical_data, current_data):
        """Calculate all technical indicators"""
        df = historical_data.copy()
        current_row = pd.DataFrame([current_data])
        df = pd.concat([df, current_row], ignore_index=True)
        
        # Moving Averages
        ma_7 = df['close'].tail(7).mean() if len(df) >= 7 else df['close'].mean()
        ma_30 = df['close'].tail(30).mean() if len(df) >= 30 else df['close'].mean()
        ma_90 = df['close'].tail(90).mean() if len(df) >= 90 else df['close'].mean()
        
        # Daily Return
        if len(df) >= 2:
            daily_return = (current_data['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']
        else:
            daily_return = 0.0
        
        # Volatility
        returns = df['close'].pct_change()
        volatility_7 = returns.tail(7).std() if len(df) >= 7 else 0.01
        volatility_30 = returns.tail(30).std() if len(df) >= 30 else 0.01
        
        # RSI, MACD, Bollinger Bands
        rsi = self.calculate_rsi(df['close'])
        macd, macd_signal = self.calculate_macd(df['close'])
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
    
    def predict(self, current_data, historical_data):
        """Make predictions for tomorrow's prices"""
        if not self.models_loaded:
            return None
        
        indicators = self.calculate_technical_indicators(historical_data, current_data)
        
        feature_cols = ["close", "low", "open", "volume", "ma_7", "ma_30", "ma_90",
                       "daily_return", "volatility_7", "volatility_30", "rsi", 
                       "macd", "macd_signal", "bb_upper", "bb_lower"]
        
        features = pd.DataFrame([indicators])[feature_cols]
        
        pred_high = self.model_high.predict(features)[0]
        pred_low = self.model_low.predict(features)[0]
        pred_close = self.model_close.predict(features)[0]
        
        return {
            'indicators': indicators,
            'predictions': {
                'high': pred_high,
                'low': pred_low,
                'close': pred_close
            }
        }
    
    def evaluate_models(self, historical_data):
        """Evaluate model performance on test data"""
        if not self.models_loaded or len(historical_data) == 0:
            return None
        
        # Prepare features and targets
        feature_cols = ["close", "low", "open", "volume", "ma_7", "ma_30", "ma_90",
                       "daily_return", "volatility_7", "volatility_30", "rsi", 
                       "macd", "macd_signal", "bb_upper", "bb_lower"]
        
        X = historical_data[feature_cols]
        
        # Split data (same split as training)
        X_train, X_test, y_high_train, y_high_test = train_test_split(
            X, historical_data['high'], test_size=0.2, random_state=42
        )
        _, _, y_low_train, y_low_test = train_test_split(
            X, historical_data['low'], test_size=0.2, random_state=42
        )
        _, _, y_close_train, y_close_test = train_test_split(
            X, historical_data['close'], test_size=0.2, random_state=42
        )
        
        # Get predictions
        pred_high = self.model_high.predict(X_test)
        pred_low = self.model_low.predict(X_test)
        pred_close = self.model_close.predict(X_test)
        
        # Calculate metrics for each model
        metrics = {
            'high': {
                'mae': mean_absolute_error(y_high_test, pred_high),
                'mse': mean_squared_error(y_high_test, pred_high),
                'rmse': np.sqrt(mean_squared_error(y_high_test, pred_high)),
                'r2': r2_score(y_high_test, pred_high),
                'actual': y_high_test.values,
                'predicted': pred_high,
                'mape': np.mean(np.abs((y_high_test.values - pred_high) / y_high_test.values)) * 100
            },
            'low': {
                'mae': mean_absolute_error(y_low_test, pred_low),
                'mse': mean_squared_error(y_low_test, pred_low),
                'rmse': np.sqrt(mean_squared_error(y_low_test, pred_low)),
                'r2': r2_score(y_low_test, pred_low),
                'actual': y_low_test.values,
                'predicted': pred_low,
                'mape': np.mean(np.abs((y_low_test.values - pred_low) / y_low_test.values)) * 100
            },
            'close': {
                'mae': mean_absolute_error(y_close_test, pred_close),
                'mse': mean_squared_error(y_close_test, pred_close),
                'rmse': np.sqrt(mean_squared_error(y_close_test, pred_close)),
                'r2': r2_score(y_close_test, pred_close),
                'actual': y_close_test.values,
                'predicted': pred_close,
                'mape': np.mean(np.abs((y_close_test.values - pred_close) / y_close_test.values)) * 100
            }
        }
        
        return metrics

@st.cache_resource
def load_predictor():
    """Load predictor (cached)"""
    return StockPricePredictor()

@st.cache_data
def load_historical_data():
    """Load historical data (cached)"""
    try:
        df = pd.read_csv("gold_price_forecasting_dataset.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ Historical data not found. Predictions may be less accurate.")
        return pd.DataFrame()

def create_accuracy_chart(metrics, model_name):
    """Create scatter plot showing predicted vs actual values"""
    actual = metrics['actual']
    predicted = metrics['predicted']
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color='rgba(102, 126, 234, 0.6)',
            line=dict(width=1, color='white')
        )
    ))
    
    # Perfect prediction line (y=x)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=f'{model_name.title()} Price: Predicted vs Actual',
        xaxis_title='Actual Price',
        yaxis_title='Predicted Price',
        height=400,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_error_distribution(metrics, model_name):
    """Create histogram of prediction errors"""
    errors = metrics['actual'] - metrics['predicted']
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        name='Error Distribution',
        marker_color='rgba(118, 75, 162, 0.7)'
    ))
    
    fig.update_layout(
        title=f'{model_name.title()} Price: Error Distribution',
        xaxis_title='Prediction Error',
        yaxis_title='Frequency',
        height=350,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_price_chart(current_data, predictions):
    """Create interactive price chart"""
    fig = go.Figure()
    
    # Current prices
    fig.add_trace(go.Bar(
        name='Today',
        x=['Open', 'High', 'Low', 'Close'],
        y=[current_data['open'], current_data['high'], current_data['low'], current_data['close']],
        marker_color='rgba(102, 126, 234, 0.7)'
    ))
    
    # Predicted prices
    fig.add_trace(go.Bar(
        name='Tomorrow (Predicted)',
        x=['High', 'Low', 'Close'],
        y=[predictions['high'], predictions['low'], predictions['close']],
        marker_color='rgba(118, 75, 162, 0.7)'
    ))
    
    fig.update_layout(
        title='Price Comparison: Today vs Tomorrow (Predicted)',
        xaxis_title='Price Type',
        yaxis_title='Price',
        barmode='group',
        height=400,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.title("📈 Stock Price Predictor")
    st.markdown("### Predict Tomorrow's High, Low, and Close Prices using Machine Learning")
    st.markdown("---")
    
    # Load predictor and historical data
    predictor = load_predictor()
    historical_data = load_historical_data()
    
    if not predictor.models_loaded:
        st.error("🚫 Models not loaded. Please run 'train_models.py' to train the models first.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.info("""
        This app uses **Machine Learning** (RandomForest) to predict tomorrow's stock prices.
        
        **Features:**
        - 🎯 Predicts High, Low, Close prices
        - 📈 Calculates 11+ technical indicators
        - 🧠 Uses trained ML models (.pkl files)
        - 📊 Interactive visualizations
        """)
        
        st.markdown("---")
        st.header("📝 Instructions")
        st.markdown("""
        1. Enter today's price data
        2. Click **Predict Tomorrow's Prices**
        3. View technical indicators
        4. See predictions & changes
        """)
        
        if len(historical_data) > 0:
            st.markdown("---")
            st.success(f"✅ {len(historical_data)} historical records loaded")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Today's Price Data")
        
        # Input form
        with st.form(key='price_input_form'):
            close_price = st.number_input(
                "💰 Closing Price",
                min_value=0.0,
                value=5318.40,
                step=0.01,
                format="%.2f"
            )
            
            open_price = st.number_input(
                "🔓 Opening Price",
                min_value=0.0,
                value=5415.70,
                step=0.01,
                format="%.2f"
            )
            
            high_price = st.number_input(
                "⬆️ High Price",
                min_value=0.0,
                value=5586.20,
                step=0.01,
                format="%.2f"
            )
            
            low_price = st.number_input(
                "⬇️ Low Price",
                min_value=0.0,
                value=5097.50,
                step=0.01,
                format="%.2f"
            )
            
            volume = st.number_input(
                "📊 Volume",
                min_value=0,
                value=23709,
                step=1
            )
            
            submit_button = st.form_submit_button(label='🔮 Predict Tomorrow\'s Prices')
        
        if submit_button:
            current_data = {
                'close': close_price,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'volume': volume
            }
            
            with st.spinner('🧮 Calculating technical indicators and making predictions...'):
                result = predictor.predict(current_data, historical_data)
            
            if result:
                # Store results in session state
                st.session_state['result'] = result
                st.session_state['current_data'] = current_data
                st.success("✅ Predictions generated successfully!")
    
    with col2:
        st.subheader("📊 Current Summary")
        
        if 'current_data' in st.session_state:
            cd = st.session_state['current_data']
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Opening Price", f"${cd['open']:,.2f}")
                st.metric("High Price", f"${cd['high']:,.2f}")
            with metric_cols[1]:
                st.metric("Low Price", f"${cd['low']:,.2f}")
                st.metric("Closing Price", f"${cd['close']:,.2f}")
            
            st.metric("Volume", f"{cd['volume']:,}")
        else:
            st.info("👆 Enter today's price data and click 'Predict' to see results")
    
    # Display results
    if 'result' in st.session_state:
        st.markdown("---")
        
        result = st.session_state['result']
        indicators = result['indicators']
        predictions = result['predictions']
        current_data = st.session_state['current_data']
        
        # Technical Indicators
        st.subheader("🎯 Calculated Technical Indicators")
        
        tab1, tab2, tab3 = st.tabs(["📈 Moving Averages", "📊 Momentum", "📉 Volatility"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MA (7-day)", f"${indicators['ma_7']:,.2f}")
            with col2:
                st.metric("MA (30-day)", f"${indicators['ma_30']:,.2f}")
            with col3:
                st.metric("MA (90-day)", f"${indicators['ma_90']:,.2f}")
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_value = indicators['rsi']
                rsi_color = "🟢" if 30 <= rsi_value <= 70 else "🔴"
                st.metric(f"{rsi_color} RSI", f"{rsi_value:.2f}")
            with col2:
                st.metric("MACD", f"{indicators['macd']:.2f}")
            with col3:
                st.metric("MACD Signal", f"{indicators['macd_signal']:.2f}")
        
        with tab3:
            col1, col2, col3 = st.columns(3)
            with col1:
                daily_ret_pct = indicators['daily_return'] * 100
                st.metric("Daily Return", f"{daily_ret_pct:+.2f}%")
            with col2:
                st.metric("Volatility (7d)", f"{indicators['volatility_7']:.4f}")
            with col3:
                st.metric("Volatility (30d)", f"{indicators['volatility_30']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bollinger Upper", f"${indicators['bb_upper']:,.2f}")
            with col2:
                st.metric("Bollinger Lower", f"${indicators['bb_lower']:,.2f}")
        
        st.markdown("---")
        
        # Predictions
        st.subheader("🔮 Tomorrow's Price Predictions")
        
        pred_cols = st.columns(3)
        
        with pred_cols[0]:
            high_change = ((predictions['high'] - current_data['high']) / current_data['high']) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0;">⬆️ High Price</h3>
                <h2 style="margin:10px 0;">${predictions['high']:,.2f}</h2>
                <p style="margin:0; font-size:1.1em;">{high_change:+.2f}% vs today</p>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_cols[1]:
            low_change = ((predictions['low'] - current_data['low']) / current_data['low']) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0;">⬇️ Low Price</h3>
                <h2 style="margin:10px 0;">${predictions['low']:,.2f}</h2>
                <p style="margin:0; font-size:1.1em;">{low_change:+.2f}% vs today</p>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_cols[2]:
            close_change = ((predictions['close'] - current_data['close']) / current_data['close']) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0;">💰 Close Price</h3>
                <h2 style="margin:10px 0;">${predictions['close']:,.2f}</h2>
                <p style="margin:0; font-size:1.1em;">{close_change:+.2f}% vs today</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chart
        st.markdown("---")
        st.subheader("📊 Price Visualization")
        chart = create_price_chart(current_data, predictions)
        st.plotly_chart(chart, use_container_width=True)
        
        # Summary
        st.markdown("---")
        st.subheader("📋 Prediction Summary")
        
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        
        st.info(f"""
        **Prediction Date:** {tomorrow_date}
        
        Based on today's market data and historical patterns, the model predicts:
        - Expected trading range: **${predictions['low']:,.2f}** - **${predictions['high']:,.2f}**
        - Expected closing price: **${predictions['close']:,.2f}**
        - Overall market sentiment: **{'Bullish 📈' if close_change > 0 else 'Bearish 📉'}**
        """)
    
    # Performance Tab
    st.markdown("---")
    st.subheader("🎯 Model Performance & Accuracy")
    
    with st.expander("📊 View Model Performance Metrics", expanded=False):
        if len(historical_data) > 0:
            with st.spinner('Evaluating model performance on test data...'):
                performance_metrics = predictor.evaluate_models(historical_data)
            
            if performance_metrics:
                st.success("✅ Performance evaluation complete!")
                
                # Create tabs for each model
                perf_tabs = st.tabs(["⬆️ High Price Model", "⬇️ Low Price Model", "💰 Close Price Model"])
                
                for idx, (model_name, tab) in enumerate(zip(['high', 'low', 'close'], perf_tabs)):
                    with tab:
                        metrics = performance_metrics[model_name]
                        
                        # Display metrics
                        st.markdown(f"### {model_name.title()} Price Prediction Metrics")
                        
                        metric_cols = st.columns(5)
                        with metric_cols[0]:
                            st.metric("MAE", f"${metrics['mae']:,.2f}", help="Mean Absolute Error")
                        with metric_cols[1]:
                            st.metric("RMSE", f"${metrics['rmse']:,.2f}", help="Root Mean Squared Error")
                        with metric_cols[2]:
                            accuracy = (1 - metrics['mape']/100) * 100
                            st.metric("Accuracy", f"{accuracy:.2f}%", help="100% - MAPE")
                        with metric_cols[3]:
                            st.metric("R² Score", f"{metrics['r2']:.4f}", help="Coefficient of Determination")
                        with metric_cols[4]:
                            st.metric("MAPE", f"{metrics['mape']:.2f}%", help="Mean Absolute Percentage Error")
                        
                        st.markdown("---")
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            accuracy_chart = create_accuracy_chart(metrics, model_name)
                            st.plotly_chart(accuracy_chart, use_container_width=True)
                            
                            st.info(f"""
                            **Interpretation:** Points closer to the red line indicate better predictions. 
                            This model has an R² score of **{metrics['r2']:.4f}**, meaning it explains 
                            **{metrics['r2']*100:.2f}%** of the variance in {model_name} prices.
                            """)
                        
                        with col2:
                            error_chart = create_error_distribution(metrics, model_name)
                            st.plotly_chart(error_chart, use_container_width=True)
                            
                            st.info(f"""
                            **Error Summary:**
                            - Average error: **${metrics['mae']:,.2f}**
                            - Typical error range: **±${metrics['rmse']:,.2f}**
                            - Prediction accuracy: **{accuracy:.2f}%**
                            """)
                
                # Overall summary
                st.markdown("---")
                st.markdown("### 📈 Overall Model Summary")
                
                summary_cols = st.columns(3)
                for idx, model_name in enumerate(['high', 'low', 'close']):
                    with summary_cols[idx]:
                        metrics = performance_metrics[model_name]
                        accuracy = (1 - metrics['mape']/100) * 100
                        
                        st.markdown(f"""
                        <div class="indicator-card">
                            <h4>{model_name.title()} Price Model</h4>
                            <p><strong>Accuracy:</strong> {accuracy:.2f}%</p>
                            <p><strong>Avg Error:</strong> ${metrics['mae']:,.2f}</p>
                            <p><strong>R² Score:</strong> {metrics['r2']:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("📊 Historical data required for performance evaluation. Please ensure the dataset is loaded.")

if __name__ == "__main__":
    main()
