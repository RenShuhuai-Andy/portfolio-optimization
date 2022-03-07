import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from utils import plotting
import matplotlib.pyplot as plt
import numpy as np

# Read in price data
# df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")
df = pd.read_csv("data/2021Q1_Top 50_of_Mutual_Funds_20160819_to_20200115.csv", index_col=0)

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)


# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)

# raw_weights = ef.max_sharpe()
# raw_weights = ef.min_volatility()
raw_weights = ef.max_quadratic_utility()

cleaned_weights = ef.clean_weights()


ef.save_weights_to_file("results/weights_mean_variance_max_qu.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

# max sharpe
# Expected annual return: 80.3%
# Annual volatility: 26.4%
# Sharpe Ratio: 2.97

# min volatility
# Expected annual return: 19.2%
# Annual volatility: 15.4%
# Sharpe Ratio: 1.12

# max quadratic utility
# Expected annual return: 148.0%
# Annual volatility: 60.4%
# Sharpe Ratio: 2.42
