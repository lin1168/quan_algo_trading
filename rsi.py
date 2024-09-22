import pandas_ta as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import ticker for formatting

# Assuming cleaned_data is a pandas DataFrame with 'Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'
cleaned_data = pd.read_csv("cleaned_data.csv")

# Calculate the 2-day RSI using pandas_ta
cleaned_data['RSI_2'] = ta.rsi(cleaned_data['Adj Close'], length=2)

# Define buy/sell signals
cleaned_data['Signal'] = None
cleaned_data.loc[cleaned_data['RSI_2'] < 15, 'Signal'] = 'Buy'
cleaned_data.loc[cleaned_data['RSI_2'] > 85, 'Signal'] = 'Sell'

# View the data with the RSI and signals
print(cleaned_data[['Date', 'Adj Close', 'RSI_2', 'Signal']].head())

# Initialize variables for backtest
initial_cash = 100000  # Initial cash in the portfolio
cash = initial_cash
shares = 0  # Start with no shares
equity = initial_cash
equity_curve = []  # To store equity values over time
total_trades = 0
risk_percent = 0.02  # 2% risk per trade
stop_loss_percent = 0.02  # Assume 2% stop loss below the entry price
buy_signals = cleaned_data[cleaned_data['Signal'] == 'Buy'].index
sell_signals = cleaned_data[cleaned_data['Signal'] == 'Sell'].index
started = False  # To track if the strategy has started with a buy

# Loop through the data to execute trades
for i, row in cleaned_data.iterrows():
    # Ensure we start with a Buy signal
    if not started and i in buy_signals:
        # Calculate position size based on 2% risk
        position_risk = equity * risk_percent
        stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
        risk_per_share = row['Adj Close'] - stop_loss
        shares_to_buy = int(position_risk // risk_per_share)  # Ensure shares are an integer
        
        if cash >= shares_to_buy * row['Adj Close']:
            shares += shares_to_buy
            cash -= shares_to_buy * row['Adj Close']  # Update cash after buying
            total_trades += 1
            started = True  # Mark the strategy as started after the first buy
    elif started:
        # Buy signal: buy more shares if the buy signal is triggered
        if i in buy_signals and shares == 0:
            # Recalculate position size based on current equity and 2% risk
            position_risk = equity * risk_percent
            stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
            risk_per_share = row['Adj Close'] - stop_loss
            shares_to_buy = int(position_risk // risk_per_share)  # Ensure shares are an integer
            
            if cash >= shares_to_buy * row['Adj Close']:
                shares += shares_to_buy
                cash -= shares_to_buy * row['Adj Close']  # Update cash after buying
                total_trades += 1
        # Sell signal: sell all shares if triggered
        elif i in sell_signals and shares > 0:
            cash += shares * row['Adj Close']  # Sell all shares
            shares = 0  # Reset the shares to zero after selling
            total_trades += 1
    
    # Calculate current equity (cash + value of shares)
    equity = cash + shares * row['Adj Close']
    equity_curve.append(equity)

# Convert equity_curve to pandas Series
cleaned_data['Equity'] = equity_curve

# Calculate performance metrics
final_equity = equity_curve[-1]
total_return = (final_equity - initial_cash) / initial_cash
cagr = (final_equity / initial_cash) ** (1 / (len(cleaned_data) / 252)) - 1

# Calculate max drawdown
running_max = np.maximum.accumulate(equity_curve)
drawdown = (equity_curve - running_max) / running_max
max_drawdown = drawdown.min()

# Calculate time in the market
time_in_market = len(cleaned_data[(cleaned_data['Signal'] == 'Buy') | (cleaned_data['Signal'] == 'Sell')]) / len(cleaned_data)

# Sharpe ratio (assuming 0 risk-free rate)
daily_returns = pd.Series(equity_curve).pct_change().dropna()
sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

# Display performance metrics
metrics = {
    'Final Equity': final_equity,
    'Sharpe Ratio': sharpe_ratio,
    'Max Drawdown (%)': max_drawdown * 100,
    'Total Trades': total_trades,
    'Total Return (%)': total_return * 100,
    'CAGR (%)': cagr * 100,
    'Time in Market (%)': time_in_market * 100
}

# Print metrics
for key, value in metrics.items():
    print(f"{key}: {value:.2f}")

# Plot equity curve with y-axis formatted to show large numbers
plt.figure(figsize=(10, 6))
plt.plot(cleaned_data['Date'], cleaned_data['Equity'], label='Equity Curve')

# Format y-axis to show in dollar amounts
ax = plt.gca()  # Get the current axis
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))  # Format as dollars

plt.xlabel('Date')
plt.ylabel('Equity (in dollars)')
plt.title('Equity Curve')
plt.legend()
plt.show()

