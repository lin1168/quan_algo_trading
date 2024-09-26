import pandas_ta as ta  # For technical analysis indicators like RSI
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
from itertools import product  # For generating hyperparameter combinations
import seaborn as sns  # For creating visualizations like heatmaps


# Load the cleaned data
cleaned_data = pd.read_csv("cleaned_data.csv")


# Function to backtest the strategy with given hyperparameters
def backtest_strategy(
        rsi_length, rsi_buy_threshold, rsi_sell_threshold, stop_loss_percent, take_profit_percent, risk_percent):
    """
    Description:
    -------------
    Backtest a trading strategy based on the 2-day RSI indicator and additional hyperparameters such as stop loss, 
    take profit, and risk allocation per trade.
    
    Parameters:
    ----------
    rsi_length : int
        The period used to calculate the RSI (Relative Strength Index) indicator.
    rsi_buy_threshold : float
        The RSI value below which a buy signal is triggered.
    rsi_sell_threshold : float
        The RSI value above which a sell signal is triggered.
    stop_loss_percent : float
        The percentage drop from the entry price at which the stop-loss will be triggered (e.g., 0.02 for 2%).
    take_profit_percent : float
        The percentage gain from the entry price at which the take-profit will be triggered (e.g., 0.10 for 10%).
    risk_percent : float
        The percentage of equity allocated to risk per trade (e.g., 0.02 for 2%).

    Returns:
    -------
    final_equity : float
        The final equity (total value) at the end of the backtest.
    sharpe_ratio : float
        The Sharpe Ratio of the strategy based on daily returns.
    max_drawdown : float
        The maximum drawdown experienced during the backtest, expressed as a percentage.
    total_trades : int
        The total number of trades executed during the backtest.
    total_return : float
        The total return of the strategy over the backtest period as a percentage.
    cagr : float
        The Compound Annual Growth Rate (CAGR) of the strategy over the backtest period.
    time_in_market : float
        The percentage of time the strategy was in the market (i.e., holding a position).
    
    Notes:
    -----
    - This function assumes that the cleaned_data DataFrame contains the necessary columns such as 'Adj Close' and 
      that it has been preprocessed to include the RSI indicator.
    - A position is entered only if there is enough cash to buy the calculated number of shares, and it is exited when 
      either a sell signal is triggered or the price reaches the take-profit level.
    - The equity curve is updated daily based on the closing price and the current position (if any).
    - Daily returns are used to calculate the Sharpe Ratio.
    
    Example:
    --------
    final_equity, sharpe, max_dd, trades, total_ret, cagr, time_in_mkt = backtest_strategy(
        rsi_length=2, 
        rsi_buy_threshold=10, 
        rsi_sell_threshold=90, 
        stop_loss_percent=0.02, 
        take_profit_percent=0.10, 
        risk_percent=0.02
    )
    """
    cleaned_data['RSI'] = ta.rsi(cleaned_data['Adj Close'], length=rsi_length)

    # Define buy/sell signals
    cleaned_data['Signal'] = None
    cleaned_data.loc[cleaned_data['RSI'] < rsi_buy_threshold, 'Signal'] = 'Buy'
    cleaned_data.loc[cleaned_data['RSI'] > rsi_sell_threshold, 'Signal'] = 'Sell'

    initial_cash = 100000
    cash = initial_cash
    shares = 0
    equity = initial_cash
    equity_curve = []
    total_trades = 0

    buy_signals = cleaned_data[cleaned_data['Signal'] == 'Buy'].index
    sell_signals = cleaned_data[cleaned_data['Signal'] == 'Sell'].index
    started = False
    purchase_price = 0

    for i, row in cleaned_data.iterrows():
        if not started and i in buy_signals:
            position_risk = equity * risk_percent
            stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
            risk_per_share = row['Adj Close'] - stop_loss
            shares_to_buy = int(position_risk // risk_per_share)

            if cash >= shares_to_buy * row['Adj Close']:
                shares += shares_to_buy
                cash -= shares_to_buy * row['Adj Close']
                total_trades += 1
                purchase_price = row['Adj Close']
                started = True
        elif started:
            if i in buy_signals and shares == 0:
                position_risk = equity * risk_percent
                stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
                risk_per_share = row['Adj Close'] - stop_loss
                shares_to_buy = int(position_risk // risk_per_share)

                if cash >= shares_to_buy * row['Adj Close']:
                    shares += shares_to_buy
                    cash -= shares_to_buy * row['Adj Close']
                    total_trades += 1
                    purchase_price = row['Adj Close']

            elif i in sell_signals and shares > 0:
                cash += shares * row['Adj Close']
                shares = 0
                total_trades += 1
            elif shares > 0 and row['Adj Close'] >= purchase_price * (1 + take_profit_percent):
                cash += shares * row['Adj Close']
                shares = 0
                total_trades += 1

        equity = cash + shares * row['Adj Close']
        equity_curve.append(equity)

    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_cash) / initial_cash
    cagr = (final_equity / initial_cash) ** (1 / (len(cleaned_data) / 252)) - 1

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    time_in_market = len(
        cleaned_data[(cleaned_data['Signal'] == 'Buy') | (cleaned_data['Signal'] == 'Sell')]
    ) / len(cleaned_data)

    daily_returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

    return final_equity, sharpe_ratio, max_drawdown, total_trades, total_return, cagr, time_in_market


# Define hyperparameter ranges for optimization
rsi_lengths = [2, 3, 5]
rsi_buy_thresholds = [10, 15, 20]
rsi_sell_thresholds = [80, 85, 90]
stop_loss_percents = [0.02, 0.03]
take_profit_percents = [0.05, 0.10]
risk_percents = [0.01, 0.02]

# Store results for analysis
results = []

# Perform grid search
for rsi_length, rsi_buy_threshold, rsi_sell_threshold, stop_loss_percent, take_profit_percent, risk_percent in product(
        rsi_lengths, rsi_buy_thresholds, rsi_sell_thresholds, stop_loss_percents, take_profit_percents, risk_percents):
  
    final_equity, sharpe_ratio, max_drawdown, total_trades, total_return, cagr, time_in_market = backtest_strategy(
        rsi_length, rsi_buy_threshold, rsi_sell_threshold, stop_loss_percent, take_profit_percent, risk_percent)

    results.append({
        'RSI Length': rsi_length,
        'RSI Buy Threshold': rsi_buy_threshold,
        'RSI Sell Threshold': rsi_sell_threshold,
        'Stop Loss (%)': stop_loss_percent,
        'Take Profit (%)': take_profit_percent,
        'Risk (%)': risk_percent,
        'Final Equity': final_equity,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Total Trades': total_trades,
        'Total Return (%)': total_return * 100,
        'CAGR (%)': cagr * 100,
        'Time in Market (%)': time_in_market * 100
    })

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Find the best parameters based on final equity or Sharpe ratio
best_result = results_df.loc[results_df['Final Equity'].idxmax()]

results_df.to_csv('optimization_performance_date.csv')

# Print the best result
print("Best Strategy Parameters:")
for key, value in best_result.items():
    print(f"{key}: {value:.2f}")


# Function to calculate the buy-and-hold strategy's equity curve
def buy_and_hold(cleaned_data, initial_cash=100000):
    """
    Description:
    -------------
    Calculate the equity curve for a simple buy-and-hold strategy based on the adjusted closing price.

    Parameters:
    ----------
    cleaned_data : pd.DataFrame
        A DataFrame containing the historical price data. It must include an 'Adj Close' column representing the 
        adjusted closing prices.
    initial_cash : float, optional
        The initial cash amount used to simulate the buy-and-hold strategy. Default is $100,000.

    Returns:
    -------
    pd.Series
        A Pandas Series representing the equity curve of the buy-and-hold strategy, starting with the initial cash 
        and scaling based on the price performance.
    
    Example:
    --------
    cleaned_data['Buy_Hold_Equity'] = buy_and_hold(cleaned_data, initial_cash=100000)
    """
    cleaned_data['Buy_Hold_Equity'] = cleaned_data['Adj Close'] / cleaned_data['Adj Close'].iloc[0] * initial_cash
    return cleaned_data['Buy_Hold_Equity']


# Add the equity curve for the optimized strategy
def plot_equity_curve(cleaned_data, best_params, initial_cash=100000):
    """
    Description:
    ------------
    Plot the equity curve for both the optimized trading strategy and the buy-and-hold strategy.

    Parameters:
    ----------
    cleaned_data : pd.DataFrame
        A DataFrame containing historical price data including 'Adj Close' prices, buy/sell signals, and other indicators.
    best_params : dict
        A dictionary of the best hyperparameters for the optimized trading strategy.
    initial_cash : float, optional
        The starting capital for both the optimized strategy and the buy-and-hold strategy. Default is $100,000.

    Returns:
    -------
    None
        The function adds two new columns to `cleaned_data`:
        - 'Optimized_Equity': The equity curve of the optimized strategy.
        - 'Buy_Hold_Equity': The equity curve of the buy-and-hold strategy.
    """
    rsi_length = 2
    rsi_buy_threshold = 20
    rsi_sell_threshold = 90
    stop_loss_percent = 0.02
    take_profit_percent = 0.1
    risk_percent = 0.02

    # Backtest the optimized strategy
    final_equity, sharpe_ratio, max_drawdown, total_trades, total_return, cagr, time_in_market = backtest_strategy(
        rsi_length, rsi_buy_threshold, rsi_sell_threshold, stop_loss_percent, take_profit_percent, risk_percent)

    # Generate equity curve for the optimized strategy
    optimized_equity_curve = cleaned_data['Adj Close'].copy()  # Placeholder, the real curve comes from the backtest
    cash = initial_cash
    shares = 0
    equity = initial_cash
    equity_curve = []
    started = False
    purchase_price = 0

    buy_signals = cleaned_data[cleaned_data['Signal'] == 'Buy'].index
    sell_signals = cleaned_data[cleaned_data['Signal'] == 'Sell'].index

    for i, row in cleaned_data.iterrows():
        if not started and i in buy_signals:
            position_risk = equity * risk_percent
            stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
            risk_per_share = row['Adj Close'] - stop_loss
            shares_to_buy = int(position_risk // risk_per_share)

            if cash >= shares_to_buy * row['Adj Close']:
                shares += shares_to_buy
                cash -= shares_to_buy * row['Adj Close']
                purchase_price = row['Adj Close']
                started = True
        elif started:
            if i in buy_signals and shares == 0:
                position_risk = equity * risk_percent
                stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
                risk_per_share = row['Adj Close'] - stop_loss
                shares_to_buy = int(position_risk // risk_per_share)

                if cash >= shares_to_buy * row['Adj Close']:
                    shares += shares_to_buy
                    cash -= shares_to_buy * row['Adj Close']

            elif i in sell_signals and shares > 0:
                cash += shares * row['Adj Close']
                shares = 0
            elif shares > 0 and row['Adj Close'] >= purchase_price * (1 + take_profit_percent):
                cash += shares * row['Adj Close']
                shares = 0

        equity = cash + shares * row['Adj Close']
        equity_curve.append(equity)

    # Convert the equity curve to a DataFrame for easier plotting
    cleaned_data['Optimized_Equity'] = equity_curve

    # Calculate buy-and-hold equity curve
    cleaned_data['Buy_Hold_Equity'] = buy_and_hold(cleaned_data)


# Convert 'Date' column to datetime if necessary
cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])

# Use the best parameters found
plot_equity_curve(cleaned_data, best_result)


# Function to generate heatmaps based on final equity for different hyperparameters
def generate_heatmap(results_df, varying_param, title):
    """
    Description:
    --------------
    Generate a heatmap to visualize final equity for different combinations of hyperparameters.

    Parameters:
    ----------
    results_df : pd.DataFrame
        A DataFrame containing the backtest results.
    varying_param : list or tuple
        A list or tuple containing two strings representing the column names of the hyperparameters.
    title : str
        The title of the heatmap.

    Returns:
    -------
    None
        The function displays a heatmap of final equity values based on the provided hyperparameters.
    """
    # Use pivot_table instead of pivot to create a matrix for the heatmap
    heatmap_data = results_df.pivot_table(
        index=varying_param[0], columns=varying_param[1], values='Final Equity', aggfunc='mean')

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", cbar=True)
    plt.title(f'Final Equity Heatmap: {title}')
    plt.show()

# Optimized hyperparameters (from your previous specification):
optimized_rsi_length = 2
optimized_rsi_buy_threshold = 20
optimized_rsi_sell_threshold = 90
optimized_stop_loss_percent = 0.02
optimized_take_profit_percent = 0.1
optimized_risk_percent = 0.02


# 1. Vary RSI Length and Buy Threshold while holding other parameters constant
filtered_df_1 = results_df[
    (results_df['RSI Sell Threshold'] == optimized_rsi_sell_threshold) &
    (results_df['Stop Loss (%)'] == optimized_stop_loss_percent) &
    (results_df['Take Profit (%)'] == optimized_take_profit_percent) &
    (results_df['Risk (%)'] == optimized_risk_percent)
]
generate_heatmap(filtered_df_1, ['RSI Length', 'RSI Buy Threshold'], 'RSI Length vs RSI Buy Threshold')

# 2. Vary Stop Loss and Take Profit while holding other parameters constant
filtered_df_2 = results_df[
    (results_df['RSI Length'] == optimized_rsi_length) &
    (results_df['RSI Buy Threshold'] == optimized_rsi_buy_threshold) &
    (results_df['RSI Sell Threshold'] == optimized_rsi_sell_threshold) &
    (results_df['Risk (%)'] == optimized_risk_percent)
]
generate_heatmap(filtered_df_2, ['Stop Loss (%)', 'Take Profit (%)'], 'Stop Loss vs Take Profit')

# 3. Vary RSI Sell Threshold and Risk Percent while holding other parameters constant
filtered_df_3 = results_df[
    (results_df['RSI Length'] == optimized_rsi_length) &
    (results_df['RSI Buy Threshold'] == optimized_rsi_buy_threshold) &
    (results_df['Stop Loss (%)'] == optimized_stop_loss_percent) &
    (results_df['Take Profit (%)'] == optimized_take_profit_percent)
]
generate_heatmap(filtered_df_3, ['RSI Sell Threshold', 'Risk (%)'], 'RSI Sell Threshold vs Risk (%)')
