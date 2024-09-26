import pandas_ta as ta  # Technical analysis indicators
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting library
import matplotlib.ticker as ticker  # Formatting for tickers on axes


# Assuming cleaned_data is a pandas DataFrame with 'Date',
# 'Open', 'High', 'Low', 'Adj Close', 'Volume'
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
initial_cash = 100_000  # Initial cash in the portfolio
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
    """
    Execute the RSI-based trading strategy by iterating through the dataset.

    Parameters:
    -----------
    None (uses the dataset's rows).

    Returns:
    --------
    None (directly updates `equity_curve` and trading decisions).

    Results:
    --------
    - Buy shares when RSI(2) falls below 15 (buy signal).
    - Sell shares when RSI(2) rises above 85 (sell signal).
    - Track equity value over time using a risk management strategy.
    """
    # Ensure we start with a Buy signal
    if not started and i in buy_signals:
        # Calculate position size based on 2% risk
        position_risk = equity * risk_percent
        stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
        risk_per_share = row['Adj Close'] - stop_loss
        shares_to_buy = int(position_risk // risk_per_share)  # Ensure integer

        if cash >= shares_to_buy * row['Adj Close']:
            shares += shares_to_buy
            cash -= shares_to_buy * row['Adj Close']  
            # Update cash after buying
            total_trades += 1
            started = True  # Mark the strategy as started after the first buy

    elif started:
        # Buy signal: buy more shares if the buy signal is triggered
        if i in buy_signals and shares == 0:
            # Recalculate position size based on current equity and 2% risk
            position_risk = equity * risk_percent
            stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
            risk_per_share = row['Adj Close'] - stop_loss
            shares_to_buy = int(position_risk // risk_per_share)
   
            if cash >= shares_to_buy * row['Adj Close']:
                shares += shares_to_buy
                cash -= shares_to_buy * row['Adj Close']
                total_trades += 1

        # Sell signal: sell all shares if triggered
        elif i in sell_signals and shares > 0:
            cash += shares * row['Adj Close']  # Sell all shares
            shares = 0  # Reset shares to zero after selling
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
time_in_market = len(
    cleaned_data[(cleaned_data['Signal'] == 'Buy') |
                 (cleaned_data['Signal'] == 'Sell')]
) / len(cleaned_data)

# Sharpe ratio (assuming 0 risk-free rate)
daily_returns = pd.Series(equity_curve).pct_change().dropna()
sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

# Calculate the buy-and-hold strategy's equity curve
initial_price = cleaned_data['Adj Close'].iloc[0]  # The price on the first day
cleaned_data['Buy_and_Hold'] = initial_cash * (
    cleaned_data['Adj Close'] / initial_price)

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

# Buy/sell signal data for plotting
buy_signals_plot = cleaned_data[cleaned_data['Signal'] == 'Buy']
sell_signals_plot = cleaned_data[cleaned_data['Signal'] == 'Sell']

# Plot all three charts (Price, RSI, and Equity Curves)
# with the Buy-and-Hold strategy
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
# 3 rows, 1 column

# 1. Price chart with buy and sell signals
axs[0].plot(cleaned_data['Date'],
            cleaned_data['Adj Close'],
            label='SPY Price',
            color='blue')
axs[0].scatter(buy_signals_plot['Date'],
               buy_signals_plot['Adj Close'],
               marker='^', color='green',
               label='Buy Signal',
               alpha=1)
axs[0].scatter(sell_signals_plot['Date'], 
               sell_signals_plot['Adj Close'],
               marker='v',
               color='red',
               label='Sell Signal',
               alpha=1)
axs[0].set_ylabel('Price (Adj Close)')
axs[0].set_title('Price Chart with Buy and Sell Signals')
axs[0].legend()

# 2. RSI chart (without buy and sell signals)
axs[1].plot(cleaned_data['Date'],
            cleaned_data['RSI_2'],
            label='RSI(2)',
            color='blue')
axs[1].axhline(15, color='green', linestyle='--', alpha=0.7)
axs[1].axhline(85, color='red', linestyle='--', alpha=0.7)
axs[1].set_ylabel('RSI(2)')
axs[1].set_title('RSI(2)')
axs[1].legend()

# 3. Equity curve comparison: RSI strategy vs. buy-and-hold
axs[2].plot(cleaned_data['Date'],
            cleaned_data['Equity'],
            label='RSI Strategy Equity Curve',
            color='blue')
axs[2].plot(cleaned_data['Date'],
            cleaned_data['Buy_and_Hold'],
            label='Buy-and-Hold Strategy',
            color='orange')

# Format y-axis as dollar amounts
axs[2].yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
axs[2].set_ylabel('Equity (in dollars)')
axs[2].set_title('Equity Curve Comparison: RSI Strategy vs Buy-and-Hold SPY')
axs[2].legend()

# Set common labels
axs[2].set_xlabel('Date')

# Auto-adjust layout
plt.tight_layout()

# Save the figure as a PDF (ensure this happens before plt.show())
plt.savefig('reproduced_strategy_backtest.pdf', format='pdf')

# Display the plot (plt.show() comes after savefig)
plt.show()
