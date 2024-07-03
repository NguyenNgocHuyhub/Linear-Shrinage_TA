"""
Name: Minimum Volatility (Shrinkage Constant Correlation Model - SCCM)
"""
import numpy as np
import pandas as pd
from zipline.api import record, symbol, order
from sklearn.model_selection import train_test_split, KFold
from numpy import linalg as LA

from xquant.api import (
    eod_cancel,
    pip_install,
    get_universe,
    rebalance_portfolio,
    schedule_rebalance,
    short_nonexist_assets,
    history,
    is_end_date,
)

import subprocess

subprocess.run("pip uninstall -y PyPortfolioOpt", shell=True)
# subprocess.run(["pip", "uninstall", "PyPortfolioOpt"], shell=True)
import os

# os.system("pip uninstall PyPortfolioOpt")
pip_install("PyPortfolioOpt")

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# risk_models.fix_nonpositive_semidefinite(cov, fix_method="spectral")
# min(np.linalg.eigvalsh(cov))
# np.linalg.eigvalsh(risk_models.fix_nonpositive_semidefinite(cov, fix_method="spectral") )
# np.linalg.eigvalsh(risk_models.fix_nonpositive_semidefinite(cov, fix_method="diag") )

WINDOW = 2
WEEKLY_DATA = True
WEEKLY_REBALANCE = True
USE_SMA_GOLDEN = True
USE_EMA_GOLDEN = False
METHODS = [
    "sample",
    "identity",
    "single_index",
    "constant_correlation",
    "shrinkage_identity",
    "shrinkage_single_index",
    "shrinkage_constant_correlation",
    "nonlinear_shrinkage",
    "generic_ccm",
    "shrinkage_generic_ccm",
    "SCM + IM",
    "SCM + SIM",
    "SCM + CCM",
    "IM + SIM",
    "IM + CCM",
    "SIM + CCM",
    "sample + IM + SIM",
    "sample + IM + CCM",
    "sample + SIM + CCM",
    "RMSE: sample + IM + SIM",
    "RMSE: sample + IM + CCM",
    "RMSE: sample + SIM + CCM",
]
METHOD = "sample"
TOP_N = 10000


def initialize(context):
    context.frequency = 250
    context.window_length = WINDOW * context.frequency
    context.weights = dict()
    context.shrinkage = None
    context.cov = None


def handle_data(context, data):
    if WEEKLY_REBALANCE:
        if not schedule_rebalance(context, data, date_rule="week_end"):
            if len(context._miss_shares_df):
                _df = context._miss_shares_df.loc[context.datetime.strftime("%Y-%m-%d")]
                _df = _df[(_df != 0) & (_df.notnull())]
                for asset in _df.index:
                    order(asset, _df.loc[asset])
            return None

    # get the universe
    universe = get_universe(context, data)
    df = data.history(universe, "close", context.window_length, "1d")
    record(Len_Original_Universe=len(universe))

    # only consider a ticker with at least 1-year data
    df = df.dropna(axis=1, thresh=250)
    universe = df.columns
    record(Len_Universe=len(universe))

    # drop penny 5k
    raw_df = data.history(universe, "raw_close", 1, "1d")
    raw_df = raw_df.T
    raw_df = raw_df[raw_df > 5].dropna()
    universe = raw_df.index.tolist()
    df = df[universe]
    record(Len_Penny=len(universe))

    if TOP_N < len(universe):
        # market capital
        shares = data.history(universe, "shares_outstanding", context.window_length, "1d")
        market_cap = (shares * df).mean().sort_values()
        universe = market_cap.tail(TOP_N).index
        df = df[universe]

    if USE_SMA_GOLDEN:
        sma_df = df.tail(50).mean() > df.tail(200).mean()
        record(Len_SMA=sma_df.sum())
        record(Ratio_SMA=sma_df.sum() / len(sma_df))

        universe = sma_df[sma_df].index
        df = df[universe]

    if USE_EMA_GOLDEN:
        ema_50 = df.tail(50).ewm(span=50, adjust=False, ignore_na=False).mean()
        ema_200 = df.tail(200).ewm(span=200, adjust=False, ignore_na=False).mean()

        ema_df = ema_50.iloc[-1] > ema_200.iloc[-1]
        universe = ema_df[ema_df].index
        df = df[universe]

    if WEEKLY_DATA:
        context.frequency = 52

        # last of week
        df["yearweek"] = df.index.map(lambda x: x.strftime("%Y%W"))
        df = df.groupby("yearweek").tail(1).drop("yearweek", axis=1)

    #
    # drop unchanged assets
    #
    df = df.loc[:, df.apply(pd.Series.nunique) > 1]
    universe = df.columns

    # shrinkage
    if 0 in df.shape:
        return None

    if METHOD == "equally_weighted":
        weights = {asset: 1.0 / len(universe) for asset in universe}
        rebalance_portfolio(context, data, weights)
        return None

    if METHOD == "sample":
        cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
    elif METHOD == "identity":
        cov, _ = shrinkage_identity(x=df, shrink=1, frequency=context.frequency)
    elif METHOD == "single_index":
        cov, _ = shrinkage_single_index(x=df, shrink=1, frequency=context.frequency)
    elif METHOD == "constant_correlation":
        cov, _ = shrinkage_constant_correlation(x=df, shrink=1, frequency=context.frequency)
    elif METHOD == "shrinkage_identity":
        cov, context.shrinkage = shrinkage_identity(x=df, frequency=context.frequency)
    elif METHOD == "shrinkage_single_index":
        cov, context.shrinkage = shrinkage_single_index(x=df, frequency=context.frequency)
    elif METHOD == "shrinkage_constant_correlation":
        cov, context.shrinkage = shrinkage_constant_correlation(x=df, frequency=context.frequency)
    elif METHOD == "nonlinear_shrinkage":
        cov = nonlinear_shrinkage(df, frequency=context.frequency)
        # cov = cov.as_matrix()
    elif METHOD == "generic_ccm":
        returns = df.pct_change()
        diag = returns.var().mean()
        sample = returns.cov()

        A = sample.as_matrix()
        ij = np.mean(A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1))

        cov = np.full((df.shape[1], df.shape[1]), ij)
        np.fill_diagonal(cov, diag)

    elif METHOD == "SCM + IM":
        sample_cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
        im_cov, _ = shrinkage_identity(x=df, shrink=1, frequency=context.frequency)
        cov = (sample_cov + im_cov) / 2

    elif METHOD == "SCM + SIM":
        sample_cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
        sim_cov, _ = shrinkage_single_index(x=df, shrink=1, frequency=context.frequency)
        cov = (sample_cov + sim_cov) / 2

    elif METHOD == "SCM + CCM":
        sample_cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
        ccm_cov, _ = shrinkage_constant_correlation(x=df, shrink=1, frequency=context.frequency)
        cov = (sample_cov + ccm_cov) / 2

    elif METHOD == "IM + SIM":
        im_cov, _ = shrinkage_identity(x=df, shrink=1, frequency=context.frequency)
        sim_cov, _ = shrinkage_single_index(x=df, shrink=1, frequency=context.frequency)
        cov = (im_cov + sim_cov) / 2

    elif METHOD == "IM + CCM":
        im_cov, _ = shrinkage_identity(x=df, shrink=1, frequency=context.frequency)
        ccm_cov, _ = shrinkage_constant_correlation(x=df, shrink=1, frequency=context.frequency)
        cov = (im_cov + ccm_cov) / 2

    elif METHOD == "SIM + CCM":
        sim_cov, _ = shrinkage_single_index(x=df, shrink=1, frequency=context.frequency)
        ccm_cov, _ = shrinkage_constant_correlation(x=df, shrink=1, frequency=context.frequency)
        cov = (sim_cov + ccm_cov) / 2

    elif METHOD == "sample + IM + SIM":
        sample_cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
        im_cov, _ = shrinkage_identity(x=df, shrink=1, frequency=context.frequency)
        sim_cov, _ = shrinkage_single_index(x=df, shrink=1, frequency=context.frequency)
        cov = (sample_cov + im_cov + sim_cov) / 3

    elif METHOD == "sample + IM + CCM":
        sample_cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
        im_cov, _ = shrinkage_identity(x=df, shrink=1, frequency=context.frequency)
        ccm_cov, _ = shrinkage_constant_correlation(x=df, shrink=1, frequency=context.frequency)
        cov = (sample_cov + im_cov + ccm_cov) / 3

    elif METHOD == "sample + SIM + CCM":
        sample_cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
        sim_cov, _ = shrinkage_single_index(x=df, shrink=1, frequency=context.frequency)
        ccm_cov, _ = shrinkage_constant_correlation(x=df, shrink=1, frequency=context.frequency)
        cov = (sample_cov + sim_cov + ccm_cov) / 3

    elif METHOD == "RMSE: sample + IM + SIM":
        train_df = df.iloc[: -int(0.4 * len(df))]
        val_df = df.iloc[-int(0.4 * len(df)) :]

        rmse_df = pd.DataFrame()
        val_sample_cov = risk_models.sample_cov(prices=val_df, frequency=context.frequency)
        for alpha in np.arange(0, 1 + 0.05, 0.05):
            for beta in np.arange(0, 1 + 0.05, 0.05):
                if not (0 <= 1 - alpha - beta <= 1):
                    continue
                #
                _scm_cov = risk_models.sample_cov(prices=train_df, frequency=context.frequency)
                _im_cov, _ = shrinkage_identity(x=train_df, shrink=1, frequency=context.frequency)
                _sim_cov, _ = shrinkage_single_index(x=train_df, shrink=1, frequency=context.frequency)
                _cov = (1 - alpha - beta) * _scm_cov + alpha * _im_cov + beta + _sim_cov
                rmse = LA.norm(_cov - val_sample_cov)
                rmse_df = rmse_df._append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "rmse": rmse,
                    },
                    ignore_index=True,
                )

        best_row = rmse_df.iloc[rmse_df["rmse"].argmin()]
        alpha = best_row["alpha"]
        beta = best_row["beta"]
        record(cv_alpha=alpha)
        record(cv_beta=beta)
        record(cv_rmse=best_row["rmse"])
        record(cv_max_rmse=rmse_df["rmse"].max())

        scm_cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
        im_cov, _ = shrinkage_identity(x=df, shrink=1, frequency=context.frequency)
        sim_cov, _ = shrinkage_single_index(x=df, shrink=1, frequency=context.frequency)
        cov = (1 - alpha - beta) * scm_cov + alpha * im_cov + beta + sim_cov

    elif METHOD == "RMSE: sample + IM + CCM":
        train_df = df.iloc[: -int(0.4 * len(df))]
        val_df = df.iloc[-int(0.4 * len(df)) :]

        rmse_df = pd.DataFrame()
        val_sample_cov = risk_models.sample_cov(prices=val_df, frequency=context.frequency)
        for alpha in np.arange(0, 1 + 0.05, 0.05):
            for beta in np.arange(0, 1 + 0.05, 0.05):
                if not (0 <= 1 - alpha - beta <= 1):
                    continue
                #
                _scm_cov = risk_models.sample_cov(prices=train_df, frequency=context.frequency)
                _im_cov, _ = shrinkage_identity(x=train_df, shrink=1, frequency=context.frequency)
                _ccm_cov, _ = shrinkage_constant_correlation(x=train_df, shrink=1, frequency=context.frequency)
                _cov = (1 - alpha - beta) * _scm_cov + alpha * _im_cov + beta + _ccm_cov
                rmse = LA.norm(_cov - val_sample_cov)
                rmse_df = rmse_df._append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "rmse": rmse,
                    },
                    ignore_index=True,
                )

        best_row = rmse_df.iloc[rmse_df["rmse"].argmin()]
        alpha = best_row["alpha"]
        beta = best_row["beta"]
        record(cv_alpha=alpha)
        record(cv_beta=beta)
        record(cv_rmse=best_row["rmse"])
        record(cv_max_rmse=rmse_df["rmse"].max())

        scm_cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
        im_cov, _ = shrinkage_identity(x=df, shrink=1, frequency=context.frequency)
        ccm_cov, _ = shrinkage_constant_correlation(x=df, shrink=1, frequency=context.frequency)
        cov = (1 - alpha - beta) * scm_cov + alpha * im_cov + beta + ccm_cov

    elif METHOD == "RMSE: sample + SIM + CCM":
        train_df = df.iloc[: -int(0.4 * len(df))]
        val_df = df.iloc[-int(0.4 * len(df)) :]

        rmse_df = pd.DataFrame()
        val_sample_cov = risk_models.sample_cov(prices=val_df, frequency=context.frequency)
        for alpha in np.arange(0, 1 + 0.01, 0.01):
            for beta in np.arange(0, 1 + 0.01, 0.01):
                if 1 - alpha - beta < 0 or 1 - alpha - beta > 1:
                    continue
                #
                _scm_cov = risk_models.sample_cov(prices=train_df, frequency=context.frequency)
                _sim_cov, _ = shrinkage_single_index(x=train_df, shrink=1, frequency=context.frequency)
                _ccm_cov, _ = shrinkage_constant_correlation(x=train_df, shrink=1, frequency=context.frequency)
                _cov = alpha * _scm_cov + beta * _sim_cov + (1 - alpha - beta) * _ccm_cov
                rmse = LA.norm(_cov - val_sample_cov)
                rmse_df = rmse_df._append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "rmse": rmse,
                    },
                    ignore_index=True,
                )

        best_row = rmse_df.iloc[rmse_df["rmse"].argmin()]
        alpha = best_row["alpha"]
        beta = best_row["beta"]
        record(cv_alpha=alpha)
        record(cv_beta=beta)
        record(cv_rmse=best_row["rmse"])
        record(cv_max_rmse=rmse_df["rmse"].max())

        scm_cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
        sim_cov, _ = shrinkage_single_index(x=df, shrink=1, frequency=context.frequency)
        ccm_cov, _ = shrinkage_constant_correlation(x=df, shrink=1, frequency=context.frequency)
        cov = alpha * scm_cov + beta * sim_cov + (1 - alpha - beta) * ccm_cov

    else:
        raise ValueError("Covariance estimator not found")

    cov = pd.DataFrame(cov, index=df.columns, columns=df.columns)
    if context.cov is not None:
        idx = list(set(cov.index).intersection(context.cov.index))
        rmse = np.sqrt((context.cov.loc[idx, idx] - cov.loc[idx, idx]).pow(2).sum().sum() / len(idx) / 2)
        record(RMSE=rmse)
    context.cov = cov

    record(p_div_n=df.shape[1] / len(df))
    if context.shrinkage is not None:
        record(Shrinkage=context.shrinkage)

    # portfolio optimization - minimum variance
    if df.shape[1] > 10:
        max_weights = pd.Series(0.1, index=universe)

        weight_bounds = tuple([(0, max_weight) for max_weight in max_weights])
        try:
            import time

            s = time.time()

            # weights = min_volatility(cov, weight_bounds)

            mu = expected_returns.mean_historical_return(df)
            ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds, solver_options={"warm_start": False})
            _ = ef.min_volatility()
            weights = ef.clean_weights()

            weights = pd.Series(weights, index=universe)
            weights = weights[weights > 1e-6]

            e = time.time()
            print("min_volatility in", round(e - s, 2))
        except Exception as e:
            print("Failed optimiazation, skip date {}: {}".format(context.datetime, e))
            return None

        record(Total_Weight=weights.sum())
        context.weights = weights.to_dict()
    else:
        context.weights = {k: v for k, v in context.weights.items() if k in universe}
    record(Len_Portfolio=len(context.weights))
    record(Len_Before_Optimization=len(universe))

    # rebalance portfolio
    if WEEKLY_REBALANCE:
        if not schedule_rebalance(context, data, date_rule="week_end"):
            return None
    rebalance_portfolio(context, data, context.weights)


def shrinkage_identity(x, shrink=None, frequency=252):
    """
    Shrinkage to identity matrix
    """
    if shrink is None:
        cs = risk_models.CovarianceShrinkage(prices=x, frequency=frequency)
        cov = cs.ledoit_wolf()
        shrinkage = cs.delta
        return cov, shrinkage

    else:
        x = x.pct_change().dropna(how="all")

        # de-mean returns
        t, n = np.shape(x)
        meanx = x.mean(axis=0)
        x = x - np.tile(meanx, (t, 1))

        # compute sample covariance matrix
        # sample = (1.0 / t) * np.dot(x.T, x)
        sample = x.cov().values  # .as_matrix()
        diag = np.diag(sample)
        cov = np.zeros((n, n), float)
        np.fill_diagonal(cov, diag)

        shrinkage = shrink
        prior = cov
        # compute the estimator
        sigma = shrinkage * prior + (1 - shrinkage) * sample
        sigma = sigma * frequency
        return sigma, shrinkage


def shrinkage_single_index(x, shrink=None, frequency=252):
    """
    This estimator is a weighted average of the sample
    covariance matrix and a "prior" or "shrinkage target".
    Here, the prior is given by a one-factor model.
    The factor is equal to the cross-sectional average
    of all the random variables.

    The notation follows Ledoit and Wolf (2003), version: 04/2014

    NOTE: use (pairwise) covariance on raw returns

    Parameters
    ----------
    x : T x N stock returns
    shrink : given shrinkage intensity factor if none, code calculates

    Returns
    -------
    tuple : np.ndarray which contains the shrunk covariance matrix
          : float shrinkage intensity factor

    """
    x = x.pct_change().dropna(how="all")

    # de-mean returns
    t, n = np.shape(x)
    meanx = x.mean(axis=0)
    x = x - np.tile(meanx, (t, 1))
    xmkt = x.mean(axis=1)

    # # compute sample covariance matrix
    # sample = np.cov(np.append(x, xmkt, axis=1), rowvar=False) * (t - 1) / t
    sample = pd.concat([x, xmkt], axis=1).cov() * (t - 1) / t

    sample = sample.values  # as_matrix()
    covmkt = sample[0:n, n].reshape(n, 1)
    varmkt = sample[n, n]
    sample = sample[:n, :n]
    # NOTE: use product operator instead of divide
    prior = np.dot(covmkt, covmkt.T) * varmkt  # / varmkt
    prior[np.eye(n) == 1] = np.diag(sample)

    if shrink == 1:
        return prior, 1

    # NOTE: here we have to fillna since we have no assumption
    x = x.values  # as_matrix()
    x = np.nan_to_num(x)

    # compute shrinkage parameters
    if shrink is None:
        c = np.linalg.norm(sample - prior, "fro") ** 2
        y = x**2
        p = 1 / t * np.sum(np.dot(y.T, y)) - np.sum(sample**2)

        # r is divided into diagonal
        # and off-diagonal terms, and the off-diagonal term
        # is itself divided into smaller terms
        rdiag = 1 / t * np.sum(y**2) - sum(np.diag(sample) ** 2)
        z = x * np.tile(xmkt.to_numpy().reshape(-1, 1), (n,))
        v1 = 1 / t * np.dot(y.T, z) - np.tile(covmkt, (n,)) * sample
        roff1 = np.sum(v1 * np.tile(covmkt, (n,)).T) / varmkt - np.sum(np.diag(v1) * covmkt.T) / varmkt
        v3 = 1 / t * np.dot(z.T, z) - varmkt * sample
        roff3 = (
            np.sum(v3 * np.dot(covmkt, covmkt.T)) / varmkt**2
            - np.sum(np.diag(v3).reshape(-1, 1) * covmkt**2) / varmkt**2
        )
        roff = 2 * roff1 - roff3
        r = rdiag + roff

        # compute shrinkage constant
        k = (p - r) / c
        shrinkage = max(0, min(1, k / t))
    else:
        # use specified constant
        shrinkage = shrink

    # compute the estimator
    sigma = shrinkage * prior + (1 - shrinkage) * sample
    sigma = sigma * frequency
    return sigma, shrinkage


def shrinkage_constant_correlation(x, shrink=None, frequency=252):
    """
    Shrinks towards constant correlation matrix
    if shrink is specified, then this constant is used for shrinkage

    The notation follows Ledoit and Wolf (2003, 2004) version 04/2014

    NOTE: use (pairwise) covariance on raw returns
    NOTE: shrink as float to return default behavior, as list to return
        different covariance of different shrinkage intensity

    Parameters
    ----------
    x : T x N stock returns
    shrink : given shrinkage intensity factor if none, code calculates

    Returns
    -------
    tuple : np.ndarray which contains the shrunk covariance matrix
          : float shrinkage intensity factor

    """
    x = x.pct_change().dropna(how="all")

    # de-mean returns
    t, n = np.shape(x)
    meanx = x.mean(axis=0)
    x = x - np.tile(meanx, (t, 1))

    # compute sample covariance matrix
    # sample = (1.0 / t) * np.dot(x.T, x)
    sample = x.cov().values  # as_matrix()

    # NOTE: here we have to fillna since we have no assumption
    x = x.values  # as_matrix()
    x = np.nan_to_num(x)

    # compute prior
    var = np.diag(sample).reshape(-1, 1)
    sqrtvar = np.sqrt(var)
    _var = np.tile(var, (n,))
    _sqrtvar = np.tile(sqrtvar, (n,))
    r_bar = (sum(sum(sample / (_sqrtvar * _sqrtvar.T))) - n) / (n * (n - 1))
    prior = r_bar * (_sqrtvar * _sqrtvar.T)
    prior[np.eye(n) == 1] = var.reshape(-1)

    # compute shrinkage parameters and constant
    if shrink is None:
        # what we call pi-hat
        y = x**2.0
        phi_mat = np.dot(y.T, y) / t - 2 * np.dot(x.T, x) * sample / t + sample**2
        phi = np.sum(phi_mat)

        # what we call rho-hat
        term1 = np.dot((x**3).T, x) / t
        help_ = np.dot(x.T, x) / t
        help_diag = np.diag(help_)
        term2 = np.tile(help_diag, (n, 1)).T * sample
        term3 = help_ * _var
        term4 = _var * sample
        theta_mat = term1 - term2 - term3 + term4
        theta_mat[np.eye(n) == 1] = np.zeros(n)
        rho = sum(np.diag(phi_mat)) + r_bar * np.sum(np.dot((1.0 / sqrtvar), sqrtvar.T) * theta_mat)

        # what we call gamma-hat
        gamma = np.linalg.norm(sample - prior, "fro") ** 2

        # compute shrinkage constant
        kappa = (phi - rho) / gamma
        shrinkage = max(0.0, min(1.0, kappa / t))
    else:
        # use specified constant
        shrinkage = shrink

    # compute the estimator
    sigma = shrinkage * prior + (1 - shrinkage) * sample
    sigma = sigma * frequency
    return sigma, shrinkage


import numpy as np
import pandas as pd
import scipy.optimize as sco


def volatility(weights, cov, gamma=0):
    portfolio_variance = np.dot(weights.T, np.dot(cov, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    if gamma != 0:
        l2_reg = gamma * (weights**2).sum()
        portfolio_volatility += l2_reg
    return portfolio_volatility


def min_volatility(cov, weight_bounds):
    # try valid initial guess
    initial_guess = np.array([v[1] for v in weight_bounds])
    initial_guess = initial_guess / sum(initial_guess)

    args = (cov, 0)
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

    result = sco.minimize(
        volatility,
        x0=initial_guess,
        args=args,
        method="SLSQP",
        bounds=weight_bounds,
        constraints=constraints,
    )

    if result["success"]:
        record(Optimized_Volatility=result["fun"])
        weights = result["x"]
        return weights


# np.linalg.eig(cov).eigenvalues


# risk_models._is_positive_semidefinite(cov)
# cov2 = risk_models.fix_nonpositive_semidefinite(cov)
# risk_models._is_positive_semidefinite(cov2)
# ef = EfficientFrontier(mu, cov2, weight_bounds=weight_bounds, verbose=True)
# raw_weights = ef.min_volatility()
# cleaned_weights = ef.clean_weights()
# cleaned_weights = [cleaned_weights[v] for v in df.columns]
# volatility(np.array(cleaned_weights), S2)


# S = df.pct_change().cov() * 52
# S = risk_models.sample_cov(prices=df, frequency=context.frequency)
# risk_models._is_positive_semidefinite(S)
# S = risk_models.fix_nonpositive_semidefinite(S)
# risk_models._is_positive_semidefinite(S)
# np.linalg.eigvalsh(S)
# [v for v in np.linalg.eigvalsh(S) if v < 0]
# min(np.linalg.eigvalsh(S))
# ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds, verbose=True, solver_options={"warm_start": False})
# raw_weights = ef.min_volatility()
# cleaned_weights = ef.clean_weights()
# cleaned_weights = [cleaned_weights[v] for v in df.columns]
# volatility(np.array(cleaned_weights), S)


# volatility(np.array(result), cov)
# volatility(np.array(result2), S2)


# S2 = risk_models.CovarianceShrinkage(df, frequency=52).ledoit_wolf(shrinkage_target='constant_correlation')

# s = time.time()
# ef = EfficientFrontier(mu, S2, weight_bounds=(0, 0.1), verbose=True)
# raw_weights = ef.min_volatility()
# time.time() - s

# c = []
# for i in range(100):
#     s = time.time()
#     ef2 = EfficientFrontier(mu, S2, weight_bounds=(0, 0.1), verbose=True)#, solver='SCS')
#     raw_weights = ef2.min_volatility()
#     t = time.time() - s
#     c.append(t)

# ecos [1.66806960105896, 1.5436642169952393, 1.6268398761749268, 1.5489182472229004, 1.5244731903076172, 1.1395714282989502, 1.2674388885498047, 1.5627336502075195, 1.5855915546417236, 1.6894886493682861, 1.5990519523620605, 1.6115527153015137, 1.5790319442749023, 1.5859105587005615, 1.5751848220825195, 1.4984185695648193, 1.437441110610962, 1.6762008666992188, 1.755901575088501, 1.3320837020874023, 1.5480258464813232, 1.5719177722930908, 1.1923110485076904, 1.5491082668304443, 1.4547033309936523, 1.5923287868499756, 1.4518074989318848, 1.147801399230957, 1.623157262802124, 1.3722460269927979, 1.6763298511505127, 1.4787564277648926, 1.5864710807800293, 1.6948177814483643, 1.8304271697998047, 1.5048553943634033, 1.5976760387420654, 1.5146074295043945, 1.5170156955718994, 1.3016180992126465, 1.385638952255249, 1.460688591003418, 1.4099667072296143, 1.553251028060913, 1.5459167957305908, 1.5660371780395508, 1.6894457340240479, 1.3445994853973389, 1.4404380321502686, 1.5988125801086426, 1.5180270671844482, 1.5558371543884277, 1.7049577236175537, 1.4206440448760986, 1.319570779800415, 1.516643762588501, 1.4048118591308594, 1.5998046398162842, 1.7534914016723633, 1.7920973300933838, 1.4595191478729248, 1.527315378189087, 1.3953471183776855, 1.482844352722168, 1.535182237625122, 1.7415661811828613, 1.6381840705871582, 1.2795586585998535, 1.2674281597137451, 1.5140297412872314, 1.497765064239502, 1.0934054851531982, 0.9838018417358398, 1.4632606506347656, 1.6043038368225098, 1.6783907413482666, 1.55790114402771, 1.7048017978668213, 1.2229712009429932, 1.6315710544586182, 1.6215033531188965, 1.564063549041748, 1.6981725692749023, 1.6726484298706055, 1.6822459697723389, 1.7734181880950928, 1.563169002532959, 1.5011491775512695, 1.5775256156921387, 1.558655023574829, 1.4508013725280762, 1.6029481887817383, 1.6371989250183105, 1.6117076873779297, 1.658733606338501, 1.51800537109375, 1.7160718441009521, 1.6458101272583008, 1.5694856643676758, 1.6253509521484375]
# ecos >>> np.mean(a)
# 1.5312204265594482
# >>> np.std(a)
# 0.15486512097198166

# scs [1.131347417831421, 1.2625656127929688, 1.4081833362579346, 1.2772035598754883, 1.3847463130950928, 1.133415937423706, 1.2711081504821777, 1.2864644527435303, 0.8920352458953857, 1.4349267482757568, 1.49749755859375, 1.275552749633789, 1.3151023387908936, 1.0014288425445557, 1.2365078926086426, 1.4256765842437744, 1.4002389907836914, 1.2965638637542725, 1.200850248336792, 1.3547353744506836, 1.3785157203674316, 1.4888336658477783, 1.308271884918213, 1.2021691799163818, 1.4700496196746826, 1.2799465656280518, 1.1403050422668457, 1.2989988327026367, 1.371901512145996, 1.3717098236083984, 1.3091528415679932, 1.3687448501586914, 1.2949283123016357, 1.2759079933166504, 1.3476054668426514, 1.4948742389678955, 1.4151382446289062, 1.5049476623535156, 1.3273060321807861, 1.4400649070739746, 1.5522093772888184, 1.0506939888000488, 1.3352599143981934, 1.5256016254425049, 1.4126479625701904, 1.428229808807373, 1.345080852508545, 1.5086913108825684, 1.4546804428100586, 1.3778469562530518, 1.3481063842773438, 1.4377639293670654, 1.519777536392212, 1.3170218467712402, 1.2920119762420654, 1.5690898895263672, 1.4315249919891357, 1.4878203868865967, 1.3230860233306885, 1.501896858215332, 1.4681425094604492, 1.403008222579956, 1.2396044731140137, 1.0949690341949463, 1.480903148651123, 1.2528424263000488, 1.2894673347473145, 1.293750286102295, 1.2867980003356934, 1.2944586277008057, 1.4440045356750488, 1.3983845710754395, 1.3655564785003662, 1.2897560596466064, 1.4040772914886475, 1.3929615020751953, 1.418121099472046, 1.3921713829040527, 1.3580126762390137, 1.4558484554290771, 1.1112382411956787, 1.1265912055969238, 1.4871242046356201, 1.5391700267791748, 1.4591329097747803, 1.3342750072479248, 1.2719855308532715, 1.4889941215515137, 1.2998991012573242, 1.3188152313232422, 1.4573204517364502, 1.4922595024108887, 1.3498115539550781, 1.6290264129638672, 1.562471866607666, 1.3936562538146973, 1.3768551349639893, 1.285088300704956, 1.4539122581481934, 1.2888073921203613]
# scs 1.3554383850097655
# >>> np.std(b)
# 0.12685903145285762

# cleaned_weights = ef.clean_weights()
# cleaned_weights = [cleaned_weights[v] for v in df.columns]
# volatility(np.array(cleaned_weights), S2)

# 0.19416651082407305 - 0.1941665107227627

# 0.19416651082407305 - 0.1941665107227627


import numpy as np
from numpy import matlib as ml

EPS = 1e-8


def shrink_cov(data, k=None):
    """Shrink covarince matrix using non-linear shrinkage as described in
    Ledoit and Wolf 2018 http://www.econ.uzh.ch/static/wp/econwp264.pdf .
    The code uses an analytic formula which was previously not available
    and is thus much faster because there is no optimization necessary. The code can
    also handle the high-dimensional setting with p>n .
    Args:
        data (`numpy.ndarray`): Data matrix with each observation in rows of the matrix,
            i.e. an n-by-p matrix with n observations and p dimensional variables.
        k (int, optional): If this parameter k is None,
             the algorithm demeans the data by default, and then adjusts
             the effective sample size accordingly by subtracting one.
            If the user inputs k = 0, then no demeaning takes place and
             the effective sample size remains n.
             If the user inputs k >= 1, then it signifies that the data X
             has already been demeaned or otherwise pre-processed; for example,
             the data might constitute OLS residuals based on a linear regression
             model with k regressors. No further demeaning takes place then,
             but the effective sample size is adjusted accordingly by subtracting k.
    Returns:
        `numpy.ndarray`: Shrunk covariance matrix
    """
    shape = data.shape
    assert len(shape) == 2, "input must be a 2d array"
    n, p = shape
    if k is None:
        data = data - np.mean(data, axis=0)
        k = 1
    n = n - k  # effective sample size
    assert n >= 12, "sample size n must be >= 12"
    sample_cov = pd.DataFrame(data).cov()  # np.dot(data.T, data) / n
    # % extract sample eigenvalues sorted in ascending order and eigenvectors
    lam, u = np.linalg.eigh(sample_cov)
    if any(lam[: np.maximum(0, p - n)] < EPS):
        print(">>> p, n, eigenvalues:", p, n, lam[: np.maximum(0, p - n)])
        print(">>> WARNING: We are singular")
    # compute analytical nonlinear shrinkage kernel formula
    lam = lam[np.maximum(0, p - n) :]
    if any(lam / sum(lam) < EPS):
        print(">>> WARNING: They are singular")
        raise ValueError("Matrix is singular")
    L = ml.repmat(lam.T, np.minimum(p, n), 1).T
    h = np.power(n, -1 / 3.0)
    # % Equation(4.9)
    H = h * L.T
    x = (L - L.T) / H
    ftilde = (3 / 4.0 / np.sqrt(5)) * np.mean(np.maximum(1 - x**2.0 / 5.0, 0) / H, 1)
    # % Equation(4.7)
    Hftemp = (-3 / 10 / np.pi) * x + (3 / 4.0 / np.sqrt(5) / np.pi) * (1 - x**2.0 / 5.0) * np.log(
        np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x))
    )
    # % Equation(4.8)
    Hftemp[np.abs(x) == np.sqrt(5)] = (-3 / 10 / np.pi) * x[np.abs(x) == np.sqrt(5)]
    Hftilde = np.mean(Hftemp / H, 1)
    if p <= n:
        dtilde = lam / ((np.pi * (p / n) * lam * ftilde) ** 2 + (1 - (p / n) - np.pi * (p / n) * lam * Hftilde) ** 2)
    # % Equation(4.3)
    else:
        Hftilde0 = (
            (1 / np.pi)
            * (
                3 / 10.0 / h**2
                + 3
                / 4.0
                / np.sqrt(5)
                / h
                * (1 - 1 / 5.0 / h**2)
                * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))
            )
            * np.mean(1 / lam)
        )
        # % Equation(C.8)
        dtilde0 = 1 / (np.pi * (p - n) / n * Hftilde0)
        # % Equation(C.5)
        dtilde1 = lam / (np.pi**2 * lam**2.0 * (ftilde**2 + Hftilde**2))
        # % Eq. (C.4)
        dtilde = np.concatenate([dtilde0 * np.ones((p - n)), dtilde1])
    #
    sigmatilde = np.dot(np.dot(u, np.diag(dtilde)), u.T)
    return sigmatilde


def prial(sample, sigma_hat, sigma):
    """Percentage Relative Improvement in Average Loss
    Args:
        sample (`numpy.ndarray`): Sample covariance
        sigma_hat (`numpy.ndarray`): Estimated Covariance
        sigma (`numpy.ndarray`): True Covariance
    Returns:
        float: Percentage improvement (between 0,1)
    """
    num = loss_mv(sample, sigma) - loss_mv(sigma_hat, sigma)
    denom = loss_mv(sample, sigma) - loss_mv(fsopt(sample, sigma), sigma)
    return num / float(denom)


def fsopt(sample, sigma):
    lam, u = np.linalg.eigh(sample)
    d_start = np.einsum("ji, jk, ki -> i", u, sigma, u)
    ud = np.dot(u, np.diag(d_start))

    return np.dot(ud, u.T)


def loss_mv(sigma_hat, sigma):
    n, p = sigma.shape
    omega_hat = np.linalg.inv(sigma_hat)
    num = np.trace(np.dot(np.dot(omega_hat, sigma), omega_hat)) / p
    denom = (np.trace(omega_hat) / p) ** 2
    alpha = np.trace(np.linalg.inv(sigma)) / p
    return num / denom - alpha


def loss_fr(sigma_hat, sigma):
    n, p = sigma.shape
    delta = sigma_hat - sigma
    # return np.trace(np.dot(delta.T, delta)) / p
    return np.trace(delta**2) / p


def nonlinear_shrinkage(x, frequency=250):
    """
        Debug singular matrix

    import itertools
    from zipline.api import symbol
    for i in range(len(cov)):
        assets = cov.index[:i+1]
        if symbol('HCM') in assets:
            assets = assets.drop([symbol('HCM')])
        if symbol('SHI') in assets:
            assets = assets.drop([symbol('SHI')])
        if symbol('DRC') in assets:
            assets = assets.drop([symbol('SHI')])
        matrix = cov.loc[assets, assets]
        try:
            _ = np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
            # print(assets)
        except np.linalg.LinAlgError:
            print(assets)
            break

    """
    x = x.pct_change().dropna(how="all")
    cov = shrink_cov(data=x)
    #
    matrix = cov
    np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
    #
    cov = cov * frequency
    return cov

if __name__ == '__main__':
    from xquant_backtest.backtest import XQuantAlgorithm

    algo = XQuantAlgorithm(
        backtest_id='tmp6l_lngg3',
        start='2013-01-01',
        end='2023-12-31',
        capital_base=1000000,
        available_capital=None,
        universe='HOSE',
        manual_edits=[],
        total_weight=1.0,
        is_production_mode=False,
        do_limit_capital=False,
        do_visualization=True,
        do_debug=True,
        do_backup=True,
    )
    algo.run_backtest()
