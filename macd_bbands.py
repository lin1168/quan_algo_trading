import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# TRADE SIGNALS 
# BUY WHEN: The MACD histogram crosses below the lower Bollinger Band
# SELL WHEN: The MACD histogram crosses above the upper Bollinger Band

# Load data
train_data = pd.read_csv('train_data.csv', index_col='Date', parse_dates=True)
test_data = pd.read_csv('test_data.csv', index_col='Date', parse_dates=True)

# Calculate MACD and MACD histogram
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['Short_EMA'] - data['Long_EMA']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    return data

# Calculate Bollinger Bands for MACD histogram
def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['MACD_Histogram'].rolling(window=window).mean()
    rolling_std = data['MACD_Histogram'].rolling(window=window).std()
    
    data['Bollinger_Upper'] = rolling_mean + (rolling_std * num_std)
    data['Bollinger_Lower'] = rolling_mean - (rolling_std * num_std)
    return data

# Generate buy/sell signals based on MACD and Bollinger Bands
def generate_signals(data):
    data['Signal'] = 0
    # Buy signal: MACD histogram crosses below the lower Bollinger Band
    data.loc[data['MACD_Histogram'] < data['Bollinger_Lower'], 'Signal'] = 1
    # Sell signal: MACD histogram crosses above the upper Bollinger Band
    data.loc[data['MACD_Histogram'] > data['Bollinger_Upper'], 'Signal'] = -1
    return data

# Backtest strategy with total trades and returns
def backtest_strategy(data, initial_balance=10000):
    balance = initial_balance
    position = 0
    total_trades = 0
    equity_curve = []
    trades = []
    entry_price = None
    
    for i, row in data.iterrows():
        if row['Signal'] == 1 and position == 0:  # Buy
            entry_price = row['Close']
            position = balance / row['Close']
            balance = 0
            total_trades += 1  # Count the trade
        elif row['Signal'] == -1 and position > 0:  # Sell
            balance = position * row['Close']
            trades.append(row['Close'] - entry_price)  # Track trade outcome
            position = 0
            entry_price = None
        
        total_balance = balance + (position * row['Close'])
        equity_curve.append(total_balance)
    
    data['Equity'] = equity_curve
    total_returns = (equity_curve[-1] - initial_balance) / initial_balance
    return data, trades, total_trades, total_returns

# Calculate performance metrics: Sharpe ratio and Max Drawdown
def calculate_performance_metrics(data):
    data['Returns'] = data['Equity'].pct_change()
    sharpe_ratio = np.sqrt(252) * data['Returns'].mean() / data['Returns'].std()
    drawdown = data['Equity'] / data['Equity'].cummax() - 1
    max_drawdown = drawdown.min()
    return sharpe_ratio, max_drawdown

# Calculate expectancy and expected profit
def calculate_expectancy(trades):
    winning_trades = np.array([trade for trade in trades if trade > 0])
    losing_trades = np.array([trade for trade in trades if trade < 0])
    
    p_win = len(winning_trades) / len(trades) if len(trades) > 0 else 0
    p_loss = len(losing_trades) / len(trades) if len(trades) > 0 else 0
    
    avg_profit = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    
    expectancy = (avg_profit * p_win) + (avg_loss * p_loss)
    expected_profit = expectancy * len(trades)
    
    return expectancy, expected_profit

# Plot MACD histogram with Bollinger Bands and equity curve with signals, save the plot
def plot_equity_curve(data, title, filename=None):
    fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot the closing price and trade signals
    ax[0].plot(data.index, data['Close'], label='Close Price')
    ax[0].scatter(data.index[data['Signal'] == 1], data['Close'][data['Signal'] == 1], label='Buy Signal', marker='^', color='g')
    ax[0].scatter(data.index[data['Signal'] == -1], data['Close'][data['Signal'] == -1], label='Sell Signal', marker='v', color='r')
    ax[0].set_title(f'{title} - Price and Trade Signals')
    ax[0].legend()

    # Plot the MACD histogram with Bollinger Bands and buy/sell signals
    ax[1].bar(data.index, data['MACD_Histogram'], label='MACD Histogram', color='gray')
    ax[1].plot(data.index, data['Bollinger_Upper'], label='Bollinger Upper', linestyle='--', color='blue')
    ax[1].plot(data.index, data['Bollinger_Lower'], label='Bollinger Lower', linestyle='--', color='red')
    ax[1].scatter(data.index[data['Signal'] == 1], data['MACD_Histogram'][data['Signal'] == 1], label='Buy Signal', marker='^', color='g', s=100)
    ax[1].scatter(data.index[data['Signal'] == -1], data['MACD_Histogram'][data['Signal'] == -1], label='Sell Signal', marker='v', color='r', s=100)
    ax[1].set_title('MACD Histogram with Bollinger Bands and Trade Signals')
    ax[1].legend()

    # Plot the equity curve with buy/sell signals
    ax[2].plot(data.index, data['Equity'], label='Equity Curve', color='green')
    ax[2].scatter(data.index[data['Signal'] == 1], data['Equity'][data['Signal'] == 1], label='Buy Signal', marker='^', color='g', s=100)
    ax[2].scatter(data.index[data['Signal'] == -1], data['Equity'][data['Signal'] == -1], label='Sell Signal', marker='v', color='r', s=100)
    ax[2].set_title('Equity Curve with Trade Signals')
    ax[2].legend()
    
    plt.suptitle(title)  # Set the overall title for the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to include the overall title
    
    if filename:
        plt.savefig(filename)  # Save the plot if a filename is provided

    plt.show()

# Function to format the output into a table
def print_results(train_sharpe, train_max_drawdown, train_total_trades, train_total_returns, train_expectancy, train_expected_profit, 
                  test_sharpe, test_max_drawdown, test_total_trades, test_total_returns, test_expectancy, test_expected_profit):
    
    results = {
        'Metric': ['Sharpe Ratio', 'Max Drawdown', 'Total Trades', 'Total Returns', 'Expectancy', 'Expected Profit'],
        'Training Data': [train_sharpe, train_max_drawdown, train_total_trades, train_total_returns, train_expectancy, train_expected_profit],
        'Testing Data': [test_sharpe, test_max_drawdown, test_total_trades, test_total_returns, test_expectancy, test_expected_profit]
    }

    df = pd.DataFrame(results)
    print(df)

# Apply MACD, Bollinger Bands, signal generation, and backtest for training data
train_data = calculate_macd(train_data)
train_data = calculate_bollinger_bands(train_data)
train_data = generate_signals(train_data)
train_data, train_trades, train_total_trades, train_total_returns = backtest_strategy(train_data)

# Apply MACD, Bollinger Bands, signal generation, and backtest for testing data
test_data = calculate_macd(test_data)
test_data = calculate_bollinger_bands(test_data)
test_data = generate_signals(test_data)
test_data, test_trades, test_total_trades, test_total_returns = backtest_strategy(test_data)

# Calculate performance metrics for both train and test data
train_sharpe, train_max_drawdown = calculate_performance_metrics(train_data)
test_sharpe, test_max_drawdown = calculate_performance_metrics(test_data)

# Calculate expectancy and expected profit for both train and test data
train_expectancy, train_expected_profit = calculate_expectancy(train_trades)
test_expectancy, test_expected_profit = calculate_expectancy(test_trades)

# Print the results in table format
print_results(train_sharpe, train_max_drawdown, train_total_trades, train_total_returns, train_expectancy, train_expected_profit,
              test_sharpe, test_max_drawdown, test_total_trades, test_total_returns, test_expectancy, test_expected_profit)

# Call the plotting functions with titles and filenames for saving
plot_equity_curve(train_data, 'Training Data - MACD Bollinger Band Strategy', filename='training_macd_bb_backtest_plot.png')
plot_equity_curve(test_data, 'Testing Data - MACD Bollinger Band Strategy', filename='testing_macd_bb_backtest_plot.png')