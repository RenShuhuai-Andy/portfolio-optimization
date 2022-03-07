import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from utils import plotting
import matplotlib.pyplot as plt
import numpy as np

# Read in price data
df = pd.read_csv("data/2021Q1_Top 50_of_Mutual_Funds_20160819_to_20200115.csv", index_col=0)

# Calculate expected returns and sample covariance
mu = expected_returns.capm_return(df)
# S = risk_models.sample_cov(df)
S = risk_models.semicovariance(df)
# S = risk_models.CovarianceShrinkage(df).ledoit_wolf()

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

ef.save_weights_to_file("results/weights_CAPM.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

# Expected annual return: 37.2%
# Annual volatility: 14.6%
# Sharpe Ratio: 2.41