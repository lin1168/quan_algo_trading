import pandas_ta as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns

# Load the cleaned data
cleaned_data = pd.read_csv("cleaned_data.csv")

# Function to backtest the strategy with given hyperparameters
def backtest_strategy(rsi_length, rsi_buy_threshold, rsi_sell_threshold, stop_loss_percent, take_profit_percent, risk_percent):
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

    time_in_market = len(cleaned_data[(cleaned_data['Signal'] == 'Buy') | (cleaned_data['Signal'] == 'Sell')]) / len(cleaned_data)

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

# Print the best result
print("Best Strategy Parameters:")
for key, value in best_result.items():
    print(f"{key}: {value:.2f}")

# Optionally, plot the performance of different strategies
results_df.sort_values('Final Equity', ascending=False).plot(x='Total Trades', y='Final Equity', kind='scatter')
plt.title('Performance of Different Strategies')
plt.xlabel('Total Trades')
plt.ylabel('Final Equity')
plt.show()

pivot_sharpe = results_df.pivot_table(values='Sharpe Ratio', 
                                       index='RSI Length', 
                                       columns=['RSI Buy Threshold', 'RSI Sell Threshold'])
sns.heatmap(pivot_sharpe, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Sharpe Ratio Heatmap')
plt.xlabel('RSI Buy and Sell Thresholds')
plt.ylabel('RSI Length')
plt.show()

# # Prepare data for plotting
# pivot_df = results_df.pivot_table(values='Final Equity', 
#                                    index='RSI Length', 
#                                    columns='Risk (%)')

# # Plotting
# plt.figure(figsize=(12, 6))
# for risk_percent in pivot_df.columns:
#     plt.plot(pivot_df.index, pivot_df[risk_percent], label=f'Risk: {risk_percent:.0%}')

# plt.title('Final Equity vs. RSI Length by Risk Percentage')
# plt.xlabel('RSI Length')
# plt.ylabel('Final Equity')
# plt.xticks(pivot_df.index)
# plt.legend(title='Risk Percentage')
# plt.grid()
# plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select relevant metrics for PCA
metrics = results_df[['Final Equity', 'Sharpe Ratio', 'Max Drawdown (%)', 'Total Return (%)', 'CAGR (%)', 'Time in Market (%)']]

# Standardize the metrics
scaler = StandardScaler()
metrics_scaled = scaler.fit_transform(metrics)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(metrics_scaled)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Final Equity'] = results_df['Final Equity']

# Plotting
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Final Equity'], cmap='viridis', edgecolor='k')
plt.colorbar(scatter, label='Final Equity')
plt.title('PCA of Strategy Metrics')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()