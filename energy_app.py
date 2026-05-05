"""
⚡ ENERGY CONSUMPTION FORECASTING
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import urllib.request
import io
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="⚡ Energy Forecasting", page_icon="⚡", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF6B35;'>⚡ Energy Consumption Forecasting</h1>", 
            unsafe_allow_html=True)
st.markdown("Machine Learning Power Usage Prediction")
st.markdown("---")

@st.cache_data
def load_energy_data():
    url = "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/2-ARIMA/data/energy.csv"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
        energy = pd.read_csv(io.BytesIO(data), index_col=0, parse_dates=True)
        energy = energy[['load']]
        return energy, True
    except:
        dates = pd.date_range('2012-01-01', '2014-12-31', freq='h')
        np.random.seed(42)
        base = 3000
        daily = 500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
        weekly = 300 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*7))
        trend = np.linspace(0, 200, len(dates))
        noise = np.random.normal(0, 100, len(dates))
        
        load = base + daily + weekly + trend + noise
        energy = pd.DataFrame({'load': np.maximum(load, 1000)}, index=dates)
        return energy, False

try:
    energy_data, is_real = load_energy_data()
    if is_real:
        st.success("✅ Real data loaded!")
    else:
        st.warning("⚠️ Using synthetic data")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

st.sidebar.title("⚙️ Settings")
forecast_hours = st.sidebar.slider("Forecast hours", 1, 168, 24)
training_days = st.sidebar.slider("Training days", 7, 60, 30)

tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 History", "🔮 Forecast"])

with tab1:
    st.subheader("Energy Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current", f"{energy_data['load'].iloc[-1]:.0f} MW")
    col2.metric("Average", f"{energy_data['load'].mean():.0f} MW")
    col3.metric("Peak", f"{energy_data['load'].max():.0f} MW")
    col4.metric("Min", f"{energy_data['load'].min():.0f} MW")
    
    st.markdown("---")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    recent = energy_data['load'].iloc[-7*24:]
    ax.plot(recent.index, recent.values, linewidth=2, color='#FF6B35')
    ax.fill_between(recent.index, recent.values, alpha=0.3, color='#FF6B35')
    ax.set_title('Last 7 Days', fontsize=12, fontweight='bold')
    ax.set_ylabel('Load (MW)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Historical Analysis")
    
    period = st.selectbox("Period", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All"])
    
    if period == "Last 7 Days":
        data_plot = energy_data.iloc[-7*24:]
    elif period == "Last 30 Days":
        data_plot = energy_data.iloc[-30*24:]
    elif period == "Last 90 Days":
        data_plot = energy_data.iloc[-90*24:]
    else:
        data_plot = energy_data
    
    fig, ax = plt.subplots(figsize=(14, 5))
    daily = data_plot.resample('D')['load'].mean()
    ax.plot(daily.index, daily.values, linewidth=2, marker='o', color='steelblue')
    ax.set_title('Daily Average', fontsize=12, fontweight='bold')
    ax.set_ylabel('Load (MW)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    hourly = data_plot.groupby(data_plot.index.hour)['load'].mean()
    ax.bar(hourly.index, hourly.values, color='#FF6B35', alpha=0.7)
    ax.set_title('Hourly Pattern', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Avg Load (MW)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("🔮 Forecast")
    
    if st.button("Generate Forecast", use_container_width=True):
        with st.spinner("Building ARIMA model..."):
            try:
                # Prepare data
                prices = np.array(energy_data['load'].values)
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
                
                # Training data
                train_hours = training_days * 24
                train_data = scaled[-train_hours:]
                
                # Build model
                model = SARIMAX(
                    train_data,
                    order=(2, 1, 0),
                    seasonal_order=(1, 1, 0, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                
                # Forecast - FIX HERE
                forecast_obj = results.get_forecast(steps=forecast_hours)
                forecast_scaled = forecast_obj.predicted_mean
                
                # Handle both Series and array
                if hasattr(forecast_scaled, 'values'):
                    forecast_scaled = forecast_scaled.values
                else:
                    forecast_scaled = np.array(forecast_scaled)
                
                # Inverse transform
                forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
                
                st.success("✅ Forecast complete!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg", f"{np.mean(forecast):.0f} MW")
                col2.metric("Peak", f"{np.max(forecast):.0f} MW")
                col3.metric("Min", f"{np.min(forecast):.0f} MW")
                
                st.markdown("---")
                
                # Plot
                fig, ax = plt.subplots(figsize=(14, 6))
                recent_hist = np.array(energy_data['load'].iloc[-24*7:].values)
                ax.plot(range(len(recent_hist)), recent_hist, label='Historical', 
                       linewidth=2, color='blue', alpha=0.7)
                
                forecast_range = range(len(recent_hist), len(recent_hist) + len(forecast))
                ax.plot(forecast_range, forecast, label='Forecast', 
                       linewidth=2, color='red', marker='o', linestyle='--')
                
                ax.set_title(f'Energy Forecast ({forecast_hours}h ahead)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Load (MW)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Table
                st.subheader("Forecast Data")
                forecast_dates = pd.date_range(
                    start=energy_data.index[-1] + timedelta(hours=1),
                    periods=forecast_hours,
                    freq='h'
                )
                forecast_df = pd.DataFrame({
                    'DateTime': forecast_dates,
                    'Predicted Load (MW)': np.round(forecast, 2)
                })
                st.dataframe(forecast_df.head(24), use_container_width=True)
                
                # Download
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"energy_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try reducing forecast hours or training days")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>⚡ Energy Forecasting System | ARIMA/SARIMA</p>", 
            unsafe_allow_html=True)
