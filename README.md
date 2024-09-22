# **Trading Strategy Backtesting Project**

## **Project Overview**

This project focuses on backtesting a well-known technical trading strategy, the RSI (Relative Strength Index), using historical SPY data. The primary goal is to evaluate the strategy's performance across various time periods. The project is divided into several key parts: the first objective is to replicate the strategy's results as outlined in [this resource](https://www.quantifiedstrategies.com/rsi-trading-strategy/). Once replication is achieved, the second phase involves optimizing the strategy by adjusting and fine-tuning different hyperparameters. Additionally, emphasis is placed on developing a robust data pipeline to preprocess and split the data for training and testing purposes.

## **Objectives**

1.  **Backtest RSI Strategy**: Implement and evaluate a trading strategy based on the 2-day RSI.
    -   **Buy Signal**: When the RSI crosses below 15.
    -   **Sell Signal**: When the RSI crosses above 85.
        **Position Sizing**: 2% Risk Based
        **Stop Loss**


## **Data Pipeline**

To ensure high-quality data for backtesting, a data pipeline was developed with the following steps: 1. **Data Collection**: Historical Forex data, including Open, High, Low, Close, and Volume fields, was gathered. 2. **Data Preprocessing**: Data was cleaned to remove missing values and handle any inconsistencies. 3. **Feature Engineering**: New features such as MACD, RSI, and Bollinger Bands were calculated. 

## **Backtesting Strategies**

### **RSI Strategy**

-   Buy when the 2-day RSI crosses below 15.
-   Sell when the RSI crosses above 85.

This strategy is designed to identify extreme market conditions and capitalize on oversold or overbought states.

## **Implementation Details**

-   **Framework**: The backtesting is implemented using the **Backtrader** library.
-   **Performance Metrics**: The strategies are evaluated using key performance indicators such as **Sharpe Ratio**, **Return**, and **Maximum Drawdown**.
-   **Data Slicing**: Both in-sample and out-of-sample testing were performed to ensure the strategies are robust.

## **Results**

Here are the performance metrics of the backtested strategies:

### **RSI Strategy**

| Metric          |  Original    | Replicated |
|-----------------|--------------|------------|
| Initial Equity  |  100,000     |  100,000   |
| Final Equity    |  861,655     |  861.433.64|
| Sharpe Ratio    |  NA          |  0.58      |
| Max Drawdown    |  -33%        |  -36.31%   |
| Total Trades    |  NA          |  658       |
| Total Returns   |  761.66%     |  761.43%   |  
| CAGR            |  8.3%        |  8.05%     |
| Time in Market  |  42.%        |  39.02     |

## **Project Structure**

The project is organized into the following structure:

- **data/**
  - `cleaned_data.csv`: SPY historical dataset
- **strategies/**
- `rsi.py`: RSI-based trading strategy script
- `yahoo_finance_pipeline.py`: Script for retrieving data from yfinance
- `README.md`: Project documentation
