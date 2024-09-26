# **Trading Strategy Backtesting Project**

## **Project Overview**

This project focuses on backtesting a well-known technical trading strategy, the RSI (Relative Strength Index), using historical data for SPY (S&P 500 ETF).

The project is divided into 2 major parts: the first objective is to replicate the strategy's results as outlined in [this resource](https://www.quantifiedstrategies.com/rsi-trading-strategy/). Once replication is achieved, the second phase involves optimizing the strategy by adjusting and adding different hyperparameters within the optimization universe, including parameters such as RSI period, buy/sell thresholds, stop loss, take profit, and risk allocation. Additionally, emphasis is placed on developing a robust data pipeline to preprocess and split the data for training and testing purposes.



- **Data/**
  - `full_data.csv`: SPY historical datase (start='1993-01-01', end='2023-12-31')
  - `cleaned_data.csv`: SPY historical dataset (start='1993-01-01', end='2020-11-30')

- **Data Pipeline/**
  - `yahoo_finance_pipeline.py`: Script for retrieving data from yfinance

- **Reproduced strategy/**
  - `reproduced_rsi.py`: Script for RSI strategy and backtest based on [this resource](https://www.quantifiedstrategies.com/rsi-trading-strategy/)
  - `reproduced_strategy_backtest.pdf`: Backtest Equity, Price, RSI plot in pdf 

- **Optimization/**
  - `optimization_universe_rsi.py` = Script for optimizing hyperparameters
  - `optimization_performance_date.csv` = Data for all possible performance metrics result

- **Optimized Strategy/**
  - `optimization_universe_rsi.py` = Script for the optimized strategy and backtest
