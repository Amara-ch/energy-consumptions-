"""
⚡ ENERGY CONSUMPTION FORECASTING
Real-time power usage prediction using ARIMA
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import urllib.request
import io
import warnings

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="⚡ Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF6B35;
        font-size: 2.5em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>⚡ Energy Consumption Forecasting</h1>", unsafe_allow_html=True)
st.markdown("Predict future power usage with Machine Learning")
st.markdown("---")

# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def load_energy_data():
    """Load energy data from GitHub"""
    url = "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/2-ARIMA/data/energy.csv"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
        energy = pd.read_csv(io.BytesIO(data), index_col=0, parse_dates=True)
        energy = energy[['load']]
        return energy, True
    except:
        # Create synthetic data
        dates = pd.date_range('2012-01-01', '2014-12-31', freq='H')
        np.random.seed(42)
        base_load = 3000
        daily = 500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
        weekly = 300 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*7))
        trend = np.linspace(0, 200, len(dates))
        noise = np.random.normal(0, 100, len(dates))
        
        load_values = base_load + daily + weekly + trend + noise
        energy = pd.DataFrame({'load': np.maximum(load_values, 1000)}, index=dates)
        return energy, False

# Load data
try:
    energy_data, is_real = load_energy_data()
    if is_real:
        st.success("✅ Real data loaded successfully!")
    else:
        st.warning("⚠️ Using synthetic data")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

forecast_hours = st.sidebar.slider("Hours to forecast", 1, 168, 24)
training_days = st.sidebar.slider("Training days", 7, 60, 30)

# ============================================
# TAB 1: OVERVIEW
# ============================================

tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 History", "🔮 Forecast"])

with tab1:
    st.subheader("Energy Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current = energy_data['load'].iloc[-1]
    avg = energy_data['load'].mean()
    max_val = energy_data['load'].max()
    min_val = energy_data['load'].min()
    
    col1.metric("Current Load", f"{current:.0f} MW")
    col2.metric("Average", f"{avg:.0f} MW")
    col3.metric("Peak", f"{max_val:.0f} MW")
    col4.metric("Min", f"{min_val:.0f} MW")
    
    st.markdown("---")
    
    # Recent data plot
    fig, ax = plt.subplots(figsize=(14, 5))
    recent = energy_data['load'].iloc[-7*24:]
    ax.plot(recent.index, recent.values, linewidth=2, color='#FF6B35')
    ax.fill_between(recent.index, recent.values, alpha=0.3, color='#FF6B35')
    ax.set_title('Energy Consumption - Last 7 Days', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Load (MW)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================
# TAB 2: HISTORY
# ============================================

with tab2:
    st.subheader("Historical Data Analysis")
    
    period = st.selectbox("Time Period", 
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Data"])
    
    if period == "Last 7 Days":
        data_plot = energy_data.iloc[-7*24:]
    elif period == "Last 30 Days":
        data_plot = energy_data.iloc[-30*24:]
    elif period == "Last 90 Days":
        data_plot = energy_data.iloc[-90*24:]
    else:
        data_plot = energy_data
    
    # Daily average
    fig, ax = plt.subplots(figsize=(14, 5))
    daily = data_plot.resample('D')['load'].mean()
    ax.plot(daily.index, daily.values, linewidth=2, marker='o', color='steelblue')
    ax.set_title('Daily Average Energy Consumption', fontsize=14, fontweight='bold')
    ax.set_ylabel('Load (MW)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Hourly pattern
    fig, ax = plt.subplots(figsize=(14, 5))
    hourly = data_plot.groupby(data_plot.index.hour)['load'].mean()
    ax.bar(hourly.index, hourly.values, color='#FF6B35', alpha=0.7)
    ax.set_title('Typical Hourly Pattern', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Load (MW)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)

# ============================================
# TAB 3: FORECAST
# ============================================

with tab3:
    st.subheader("🔮 Energy Forecast")
    
    if st.button("🚀 Generate Forecast", use_container_width=True):
        with st.spinner("Building ARIMA model..."):
            try:
                # Prepare data
                prices = energy_data['load'].values
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
                
                # Use last N days for training
                train_hours = training_days * 24
                train_data = scaled[-train_hours:]
                
                # Build ARIMA
                model = SARIMAX(
                    train_data,
                    order=(2, 1, 0),
                    seasonal_order=(1, 1, 0, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                
                # Forecast
                forecast_scaled = results.get_forecast(steps=forecast_hours).predicted_mean.values
                forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
                
                st.success("✅ Forecast complete!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Forecast", f"{forecast.mean():.0f} MW")
                col2.metric("Peak Forecast", f"{forecast.max():.0f} MW")
                col3.metric("Min Forecast", f"{forecast.min():.0f} MW")
                
                st.markdown("---")
                
                # Plot
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Historical
                recent_hist = energy_data['load'].iloc[-24*7:].values
                ax.plot(range(len(recent_hist)), recent_hist, label='Historical', 
                       linewidth=2, color='blue', alpha=0.7)
                
                # Forecast
                forecast_range = range(len(recent_hist), len(recent_hist) + len(forecast))
                ax.plot(forecast_range, forecast, label='Forecast', 
                       linewidth=2, color='red', marker='o', linestyle='--')
                
                ax.set_title(f'Energy Forecast ({forecast_hours} hours ahead)', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Time')
                ax.set_ylabel('Load (MW)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Table
                st.subheader("Forecast Details")
                forecast_dates = pd.date_range(
                    start=energy_data.index[-1] + timedelta(hours=1),
                    periods=forecast_hours,
                    freq='H'
                )
                forecast_df = pd.DataFrame({
                    'DateTime': forecast_dates,
                    'Predicted Load (MW)': forecast.round(2)
                })
                st.dataframe(forecast_df.head(24), use_container_width=True)
                
                # Download
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast",
                    data=csv,
                    file_name=f"energy_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Try reducing the forecast hours or training days")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray; font-size: 12px;'>
    ⚡ Energy Forecasting System | ARIMA Model
    </p>
</div>
""", unsafe_allow_html=True)
