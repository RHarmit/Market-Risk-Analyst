import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import norm

# Function to fetch natural gas prices
def get_natural_gas_data(ticker='NG=F', start='2021-01-01', end='2025-01-01'):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            print("No data fetched. Please check the ticker or date range.")
        return data[['Adj Close']]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to calculate daily returns
def calculate_daily_returns(data):
    data = data.copy()
    data['Returns'] = data['Adj Close'].pct_change()
    return data.dropna()

# Function to compute portfolio P&L
def calculate_pnl(data, position_size=10000):
    data = data.copy()
    data['PnL'] = data['Returns'] * position_size
    return data

# Function to calculate Value at Risk (VaR)
def calculate_var(data, confidence_level=0.95):
    if 'Returns' not in data.columns or data.empty:
        return np.nan
    mean_return = np.mean(data['Returns'])
    std_dev = np.std(data['Returns'])
    var = norm.ppf(1 - confidence_level) * std_dev - mean_return
    return var

# Function to plot P&L trends
def plot_pnl(data):
    if 'PnL' not in data.columns or data.empty:
        return None
    fig = px.line(data, x=data.index, y='PnL', title='Portfolio P&L Over Time')
    return fig

# Main execution
def main():
    print('Natural Gas Market Risk Analysis')

    # Fetch Data
    data = get_natural_gas_data()
    if data.empty:
        return
    data = calculate_daily_returns(data)
    data = calculate_pnl(data)

    # Display Data
    print('Recent Natural Gas Price Data:')
    print(data.tail())

    # Plot P&L
    pnl_chart = plot_pnl(data)
    if pnl_chart:
        pnl_chart.show()

    # Compute and display VaR
    var_95 = calculate_var(data, confidence_level=0.95)
    print('Value at Risk (VaR)')
    if np.isnan(var_95):
        print("Insufficient data to compute VaR.")
    else:
        print(f'95% Confidence Level VaR: {var_95:.5f}')

if __name__ == '__main__':
    main()
