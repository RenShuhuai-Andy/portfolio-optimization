import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
# Test Function for evaluating the portolio learned
import argparse


def calc_stock_yield(weights, df, tickers, start_date, end_date):
    print(start_date, end_date)
    init_price = df.loc[start_date, tickers].values
    final_price = df.loc[end_date, tickers].values
    weights = np.array(weights)
    earning_rate = (final_price - init_price) / init_price
    weighted_earning_rate = np.dot(weights, earning_rate)
    return weighted_earning_rate * 100


def calc_volatility(weights, df):
    ln_ratio = (df.pct_change() + 1).applymap(np.log)
    ln_ratio_std = ln_ratio.std()
    ln_ratio_std *= np.sqrt(252)
    weights = np.array(weights)
    ln_ratio_std = np.array(ln_ratio_std.values)
    volatility = np.dot(weights, ln_ratio_std)
    return volatility * 100

# def calc_volatility(weights, df, tickers):
#     lg_ratio = (df.pct_change() + 1).applymap(np.log)
#     lg_ratio_std = lg_ratio.std()
#     lg_ratio_std *= 252
#     volatility = 0
#     for w, t in zip(weights, tickers):
#         volatility += w * lg_ratio_std[t]
#     return volatility

# def calc_drawdown(weights, df, tickers, end_date):
#     init = (df.loc[end_date, tickers] * weights).sum()
#     hold = (df * weights).sum(axis=1)
#     hold = (hold - init) / init
#     return -hold.min() * 100


def calc_drawdown(weights, df, tickers):
    weights = np.array(weights)
    max_drawdown_list = []
    for t in tickers:
        cur_history_max = []
        cur_max = 0
        for i in df[t].values:
            cur_max = max(cur_max, i)
            cur_history_max.append(cur_max)
        cur_raw_price = np.array(df[t].values)
        cur_history_max = np.array(cur_history_max)
        max_drawdown = np.max((cur_history_max - cur_raw_price) / cur_history_max)
        max_drawdown_list.append(max_drawdown)
    max_drawdown_list = np.array(max_drawdown_list)
    weighted_max_drawdown = np.dot(weights, max_drawdown_list)
    return weighted_max_drawdown * 100


def evaluate(decision, start_year=2020, end_year=2021, start_month=1, end_month=1, start_day=15, end_day=15):
    """decision: portfolio dict {ticker_number: weight }"""

    # Get Ticker data for evaluation
    tickers = list(decision.keys())
    weights = np.array([decision[k] for k in tickers])
    start = datetime.datetime(start_year, start_month, start_day)
    end = datetime.datetime(end_year, end_month, end_day)
    df = pd.DataFrame([web.DataReader(ticker, 'yahoo', start, end)['Adj Close'] for ticker in tickers]).T
    df.columns = tickers

    # Sharp Ratio
    mean_returns = df.pct_change().mean()
    cov = df.pct_change().cov()
    _, _, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf=0)
    sharpe_ratio = round(sharpe_ratio, 2)
    print('sharpe_ratio: {}'.format(sharpe_ratio))

    #  Yield
    stock_yield = calc_stock_yield(weights, df, tickers, start_date=f"{start_year}-{start_month}-{start_day}",
                                   end_date=f"{end_year}-{end_month}-{end_day}")
    stock_yield = round(stock_yield, 2)
    print('stock_yield: {}%'.format(stock_yield))

    # Voliatity
    volatility = calc_volatility(weights, df)  # , tickers)
    volatility = round(volatility, 2)
    print('volatility: {}%'.format(volatility))

    # max drawdown
    drawdown = calc_drawdown(weights, df, tickers)  # , end_date=f"{end_year}-{end_month}-{end_day}")
    drawdown = round(drawdown, 2)
    print('drawdown: {}%'.format(drawdown))

    return {
        'sharpe_ratio': sharpe_ratio,
        'stock_yield': stock_yield,
        'volatility': volatility,
        'drawdown': drawdown
    }


def calc_portfolio_perf(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision_file', type=str, help='decision file of portfolio')
    args = parser.parse_args()

    file1 = open('data/2021Q1_Top 50_of_Mutual_Funds_list.txt', 'r', encoding='utf-8')
    Lines = file1.readlines()

    count = 0
    tickers = []
    names = []

    # Strips the newline character
    for line in Lines:
        if line[0] == '#':
            continue

        count += 1
        ticker, name = line.split()
        tickers.append(ticker)
        names.append(name)
    name_ticker_map = dict(zip(names, tickers))

    df = pd.read_csv(args.decision_file, header=None)
    decision = df.to_dict(orient='list')
    decision = {name_ticker_map[decision[0][i]]: decision[1][i] for i in range(len(decision[0]))}
    evaluate(decision)


# Uniformity
# sharpe_ratio: 2.87
# 2020-1-15 2021-1-15
# stock_yield: 102.95%
# volatility: 44.74%
# drawdown: 27.87%

# Mean-Variance max sharpe
# sharpe_ratio: 1.97
# 2020-1-15 2021-1-15
# stock_yield: 73.92%
# volatility: 46.77%
# drawdown: 30.98%

# Mean-Variance min volatility
# sharpe_ratio: 2.56
# stock_yield: 71.23%
# volatility: 37.28%
# drawdown: 23.47%

# Mean-Variance max quadratic utility
# sharpe_ratio: 0.58
# stock_yield: 17.59%
# volatility: 59.95%
# drawdown: 44.57%

# NN
# sharpe_ratio: 2.55
# stock_yield: 92.06%
# volatility: 41.73%
# drawdown: 23.93%

# CAPM
# sharpe_ratio: 3.11
# stock_yield: 118.41%
# volatility: 39.5%
# drawdown: 21.46%