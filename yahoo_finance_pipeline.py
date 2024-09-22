import yfinance as yf
import pandas as pd 

# Define the currency pair and the time period
ticker = 'SPY'
data = yf.download(ticker, start='1993-01-01', end='2023-12-31', interval='1d')

# Inspecting the Data
print("Data Information:")
print(data.info())  # Get a summary of the data
print("\nStatistical Details:")
print(data.describe())  # Get statistical details
print("\nFirst Few Rows:")
print(data.head())  # Print the first few rows

# Handling Missing Data
print("\nMissing Values in Original Data:")
print(data.isnull().sum()) # Check for missing values
data_cleaned = data.dropna() # Drop rows with missing values

# Confirm the number of rows before and after cleaning
print("\nRow Count Before Cleaning:")
print(data.count())
print("\nRow Count After Cleaning:")
print(data_cleaned.count())

# Reset index to make Date a column (if Date is currently the index)
data_cleaned = data_cleaned.reset_index()

# Ensure Date is in datetime format
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])

# Set Date as index for time series analysis (if needed)
data_cleaned = data_cleaned.set_index('Date')

print("\nData cleaning complete. Cleaned data saved to 'cleaned_forex_data.csv'.")

# Define training and testing periods
# train_end = '2022-1-1'
date_end = '2020-11-30'

# Slice the data into training and testing sets
cleaned_data = data_cleaned.loc[:date_end]

# Check the slices
print("Cleaned Data:")
print(cleaned_data.head())

# Save train and test data
cleaned_data.to_csv('cleaned_data.csv')

cleaned_data.head()
