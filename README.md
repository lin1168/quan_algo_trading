# **Forex Trading Strategy Backtesting Project**

## **Project Overview**

This project explores backtesting two prominent technical trading strategies — **RSI (Relative Strength Index)** and **MACD (Moving Average Convergence Divergence) combined with Bollinger Bands (BBands)**. The goal is to evaluate their effectiveness using historical Forex data and understand how well these strategies perform across different time periods. The project also emphasizes the creation of a robust data pipeline to preprocess and split the data for training and testing.

## **Objectives**

1.  **Backtest RSI Strategy**: Implement and evaluate a trading strategy based on the 2-day RSI.
    -   **Buy Signal**: When the RSI crosses below 10.
    -   **Sell Signal**: When the RSI crosses above 80.
2.  **Backtest MACD + Bollinger Bands Strategy**: Apply a more advanced strategy combining MACD with Bollinger Bands.
    -   **Buy Signal**: When the MACD crosses above the signal line and the price touches or falls below the lower Bollinger Band.
    -   **Sell Signal**: When the MACD crosses below the signal line and the price touches or rises above the upper Bollinger Band.

## **Data Pipeline**

To ensure high-quality data for backtesting, a data pipeline was developed with the following steps: 1. **Data Collection**: Historical Forex data, including Open, High, Low, Close, and Volume fields, was gathered. 2. **Data Preprocessing**: Data was cleaned to remove missing values and handle any inconsistencies. 3. **Feature Engineering**: New features such as MACD, RSI, and Bollinger Bands were calculated. 4. **Slicing**: Data was split into training and testing sets to ensure robust model evaluation.

## **Backtesting Strategies**

### **RSI Strategy**

-   Buy when the 2-day RSI crosses below 10.
-   Sell when the RSI crosses above 80.

This strategy is designed to identify extreme market conditions and capitalize on oversold or overbought states.

### **MACD + Bollinger Bands Strategy**

This strategy seeks to combine trend-following and volatility-based indicators for more effective trade signals. - **Buy**: MACD crosses above the signal line near the lower Bollinger Band. - **Sell**: MACD crosses below the signal line near the upper Bollinger Band.

## **Implementation Details**

-   **Framework**: The backtesting is implemented using the **Backtrader** library.
-   **Performance Metrics**: The strategies are evaluated using key performance indicators such as **Sharpe Ratio**, **Return**, and **Maximum Drawdown**.
-   **Data Slicing**: Both in-sample and out-of-sample testing were performed to ensure the strategies are robust.

## **Results**

Here are the performance metrics of the backtested strategies:

### **RSI Strategy**

| Metric          | Training Data | Testing Data |
|-----------------|---------------|--------------|
| Sharpe Ratio    | 0.461454      | 0.346087     |
| Max Drawdown    | -0.266943     | -0.340116    |
| Total Trades    | 470.0         | 153.0        |
| Total Returns   | 1.796679      | 0.290230     |
| Expectancy      | 0.337615      | 0.864182     |
| Expected Profit | 158.678917    | 132.219910   |

### **MACD + Bollinger Bands Strategy**

| Metric          | Training Data | Testing Data |
|-----------------|---------------|--------------|
| Sharpe Ratio    | 0.210435      | 0.233050     |
| Max Drawdown    | -0.535703     | -0.341047    |
| Total Trades    | 45.0          | 17.0         |
| Total Returns   | 0.450522      | 0.161250     |
| Expectancy      | 1.829472      | 2.625626     |
| Expected Profit | 82.326248     | 42.010010    |

## **Project Structure**

The project is organized into the following structure:

- **data/**
  - `train_data.csv`: Training dataset
  - `test_data.csv`: Testing dataset
- **strategies/**
  - `rsi_strategy.py`: RSI-based trading strategy script
  - `macd_bb_strategy.py`: MACD + Bollinger Bands strategy script
- `backtest.py`: Main script for running the backtests
- `README.md`: Project documentation
