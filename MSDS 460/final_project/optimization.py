"""Execute complex portfolio optimization tasks."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB
import fix_yahoo_finance as yf


def calc_return(stock_payload, symbol):
    """Calculate daily stock returns given a pandas DataFrame of price data."""
    # Use difference in natural log as return
    stock_payload['Return'] = np.log(stock_payload['Adj Close'])
    stock_return = stock_payload[['Return']].diff().dropna()
    # Rename column as its symbol
    stock_return.columns = [symbol]
    return stock_return


def calc_port_returns(symbols, start_date, end_date):
    """Combine daily returns from all possible stocks in the portfolio.

    Symbol of RISK-FREE will be ignored for return data.

    Args:
        symbols (list): strings of each stock symbol e.g. 'AAPL'.
        start_date (str): first day of returns to select YYYY-MM-DD
        end_date (str): last day of returns to select YYYY-MM-DD
    Returns:
        a pandas dataframe of daily returns with date as the index.
    """
    try:
        symbols.remove('RISK-FREE')
    except ValueError:
        pass
    for i, symbol in enumerate(symbols):
        # Download daily stock price data
        stock_payload = yf.download(symbol, start_date, end_date,
                                    progress=False).dropna()
        # First symbol will be base with subsequent joining
        if i == 0:
            base = calc_return(stock_payload, symbol)
        else:
            base = base.join(calc_return(stock_payload, symbol))
    return base


def optimize_portfolio(portfolio, hist_data, cost=0, risk_free=None,
                       growth=None, verbose=False):
    """Optimize a given stock portfolio to minimize risk.

    This model assumes no short salesself.
    Risk free asset has assumed symbol: 'RISK-FREE'.

    Args:
        portfolio (dict): stock symbol and balance key-value pairs.
        hist_data (DataFrame): historical stock returns
        cost (float): % cost of transactions (default=0)
        risk_free (float): risk free rate (default no risk free assset)
        growth (float): minimum daily return required (default=None)
        verbose (bool): whether to print results (default=False)
    Returns:
        a dictionary of the new portfolio and risk statistics.

    """
    # Initialize model
    model = Model('portfolio')

    # Decision Variables
    # Variables for amount to invest in each stock (continuous with lb=0)
    xvars = pd.Series(model.addVars(hist_data.columns.tolist(), name='x'),
                      index=hist_data.columns.tolist())
    # Variables for amount to buy and sell of each stock
    bvars = pd.Series(model.addVars(hist_data.columns.tolist(), name='b'),
                      index=hist_data.columns.tolist())
    svars = pd.Series(model.addVars(hist_data.columns.tolist(), name='s'),
                      index=hist_data.columns.tolist())

    # Objective - Minimize portfolio risk
    portfolio_risk = hist_data.cov().dot(xvars).dot(xvars)
    model.setObjective(portfolio_risk, GRB.MINIMIZE)

    # Calculated Variables
    # Budget is sum of current holdings
    budg = sum(portfolio.values())
    # Amount in risk free investments is the budget less commission and stocks
    amt_risk_free = budg - cost*(bvars.sum()+svars.sum()) - xvars.sum()

    # Constraints
    # Enforce being at or under current holdings
    model.addConstr(xvars.sum() + cost*(bvars.sum()+svars.sum()) <= budg,
                    "budget")
    # Enforce that transactions balance and no short sales
    for i, stock in enumerate(hist_data.columns.tolist()):
        stock_budget = portfolio[hist_data.columns.tolist()[i]]
        model.addConstr(xvars[i] - bvars[i] + svars[i] == stock_budget,
                        f"budget {stock}")
        model.addConstr(svars[i] <= stock_budget, f"no short {stock}")
    # Optionally require no risk free assets
    portfolio_return = np.dot(hist_data.mean(), xvars)
    if risk_free is not None:
        portfolio_return += risk_free*amt_risk_free
    else:
        model.addConstr(xvars.sum() == (budg - cost*(bvars.sum()+svars.sum())),
                        "no rf")
    # Optionally require a minimum return
    if growth is not None:
        model.addConstr(portfolio_return == growth*budg, "return")
    model.update()

    # Run the model
    model.setParam('OutputFlag', 0)
    model.optimize()

    # Save xvars to new portfolio
    new_portfolio = {}
    for i, stock in enumerate(hist_data.columns.tolist()):
        if verbose:
            print(f"{stock}: {xvars[i].x:0.2f}, "
                  f"Return: {100*hist_data.mean()[i]:0.4f}%, "
                  f"Buy: {bvars[i].x:0.2f}, "
                  f"Sell: {svars[i].x:0.2f}")
        new_portfolio[stock] = np.round(xvars[i].x, 2)
    minrisk_volatility = np.sqrt(portfolio_risk.getValue()) / budg
    minrisk_return = portfolio_return.getValue() / budg

    try:
        amt_rf_change = amt_risk_free.getValue() - portfolio['RISK-FREE']
    except KeyError:
        amt_rf_change = amt_risk_free.getValue()
    if verbose:
        if amt_rf_change == 0:
            print(f"RISK-FREE: {amt_risk_free.getValue():0.2f}, "
                  f"Buy: {0:0.2f}, "
                  f"Sell: {0:0.2f}")
        elif amt_rf_change < 0:
            print(f"RISK-FREE: {amt_risk_free.getValue():0.2f}, "
                  f"Buy: {0:0.2f}, "
                  f"Sell: {-amt_rf_change:0.2f}")
        else:
            print(f"RISK-FREE: {amt_risk_free.getValue():0.2f}, "
                  f"Buy: {amt_rf_change:0.2f}, "
                  f"Sell: {0:0.2f}")
    new_portfolio['RISK-FREE'] = np.round(amt_risk_free.getValue(), 2)
    if verbose:
        print(f"Minimum Daily Volatility = {100*minrisk_volatility:0.4f}%")
        print(f"Expected Daily Return = {100*minrisk_return:0.4f}%")

    # Save model statistics for output
    model_stats = {'stock_return': hist_data.mean(),
                   'stock_volatility': hist_data.std(),
                   'minrisk_volatility': minrisk_volatility,
                   'minrisk_return': minrisk_return}

    return new_portfolio, model_stats


def calc_current_stats(portfolio, hist_data, risk_free):
    """Calculate current portfolio stats."""
    stock_amt = {k: portfolio[k] for k in hist_data.columns.tolist()}.values()
    budg = sum(portfolio.values())

    amt_risk_free = budg - sum(stock_amt)
    portfolio_return = np.dot(hist_data.mean(), list(stock_amt))
    portfolio_return += risk_free*amt_risk_free
    portfolio_risk = hist_data.cov().dot(list(stock_amt)).dot(list(stock_amt))
    return np.sqrt(portfolio_risk) / budg, portfolio_return / budg


def efficient_frontier(portfolio, hist_data, risk_free=0):
    """Plot efficient frontier."""
    non_rf_port = {i: portfolio[i] for i in portfolio if i != 'RISK-FREE'}
    _starting_portfolio, starting_stats = optimize_portfolio(non_rf_port,
                                                             hist_data)
    # Plot optimal risky portfolio
    curr_risk, curr_return = calc_current_stats(portfolio, hist_data,
                                                risk_free)
    # Solve for efficient frontier by varying target return
    eff_frontier = pd.Series()
    # Save optimally risky portfolio
    max_sharpe = 0
    for target in np.linspace(starting_stats['stock_return'].min(),
                              starting_stats['stock_return'].max(), 500):
        _loop_port, loop_stats = optimize_portfolio(non_rf_port, hist_data,
                                                    growth=target)
        eff_frontier.loc[loop_stats['minrisk_volatility']] = target
        risk_prem = loop_stats['minrisk_return'] - risk_free
        if risk_prem / loop_stats['minrisk_volatility'] > max_sharpe:
            optimal_risk = loop_stats['minrisk_volatility']
            optimal_return = loop_stats['minrisk_return']
            max_sharpe = risk_prem / loop_stats['minrisk_volatility']

    # Solve for capital allocation line
    ca_line = pd.Series()
    for risk_level in np.linspace(0, optimal_risk, 500):
        slope = (optimal_return - risk_free) / optimal_risk
        ca_line.loc[risk_level] = risk_free + risk_level*slope

    # Plot volatility versus expected return for individual stocks
    ax_rr = plt.gca()

    # Plot efficient frontier
    eff_frontier.plot(color='Black', label='Efficient Frontier', ax=ax_rr,
                      zorder=2)

    # Plot capital allocation line
    ca_line.plot(color='Coral', label='Capital Allocation Line', ax=ax_rr,
                 zorder=1)

    ax_rr.scatter(x=starting_stats['stock_volatility'],
                  y=starting_stats['stock_return'],
                  color='dodgerblue', zorder=3)
    for i, stock in enumerate(hist_data.columns.tolist()):
        ax_rr.annotate(stock,
                       (starting_stats['stock_volatility'][i] + 0.0004,
                        starting_stats['stock_return'][i] - 0.00005))

    # Plot volatility versus expected return for minimum risk portfolio
    ax_rr.scatter(x=starting_stats['minrisk_volatility'],
                  y=starting_stats['minrisk_return'], color='Black', zorder=4)
    ax_rr.annotate('Min\nRisk\nPortfolio',
                   (starting_stats['minrisk_volatility'] - 0.0005,
                    starting_stats['minrisk_return']),
                   horizontalalignment='right')

    # Plot optimal risky portfolio
    ax_rr.scatter(x=optimal_risk, y=optimal_return, color='Coral', zorder=5)
    ax_rr.annotate('Optimal\nRisky\nPortfolio',
                   (optimal_risk + 0.0001, optimal_return + 0.00005),
                   horizontalalignment='right')

    ax_rr.scatter(x=curr_risk, y=curr_return, color='Red', zorder=6)
    ax_rr.annotate('Current\nPortfolio',
                   (curr_risk + 0.0001, curr_return + 0.00005),
                   horizontalalignment='right')

    # Format and display the final plot
    ax_rr.axis([0, 0.04, -0.0005, 0.002])
    ax_rr.set_xlabel('Standard Deviation')
    ax_rr.set_ylabel('Expected Return')
    ax_rr.legend()
    ax_rr.grid()
    plt.show()


if __name__ == '__main__':
    # Starting stock portfolio
    HOLDINGS = {'AAPL': 1000, 'MSFT': 2000, 'INTC': 1000, 'AMZN': 0,
                'GOOGL': 3000, 'NFLX': 3000}
    # Download historical returns data
    STOCK_DATA = calc_port_returns(list(HOLDINGS.keys()), '2018-01-01',
                                   '2018-12-31')
    # Complex model example
    NEW_PORTFOLIO, MODEL_STATS = optimize_portfolio(HOLDINGS, STOCK_DATA,
                                                    cost=0.01,
                                                    risk_free=0.00007,
                                                    growth=0.0004,
                                                    verbose=True)
    # Example efficient frontier
    efficient_frontier(NEW_PORTFOLIO, STOCK_DATA, risk_free=0.00007)
