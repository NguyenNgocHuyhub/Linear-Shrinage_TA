"""
Name: Minimum Volatility (Shrinkage Constant Correlation Model - SCCM)
Author: tuanta
"""
import code

import numpy as np
import pandas as pd
from zipline.api import order, record, symbol

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

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

WINDOW = 2
WEEKLY_DATA = True
WEEKLY_REBALANCE = True
TOP_N = 10000
COV_METHODS = [
    "sample",
    "CCC",
    "DCC",
    "constant_correlation",  # CCM
]
COV_METHOD = "sample"

pip_install("arch==6.2.0")


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

    if COV_METHOD == "sample":
        cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
    elif COV_METHOD == "CCC":
        cov = ccc(prices=df, frequency=context.frequency)
    elif COV_METHOD == "SCCC":
        cov = sccc(prices=df, frequency=context.frequency)
    elif COV_METHOD == "DCC":
        returns_matrix = df.pct_change().iloc[1:].fillna(0)
        cov = covregpy_dcc(returns_matrix, p=1, q=1, days=1, print_correlation=False, rescale=True)
    elif COV_METHOD == "constant_correlation":
        cov, _ = shrinkage_constant_correlation(x=df, shrink=1, frequency=context.frequency)

    # calculate RMSE
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
        try:
            weights = min_volatility(cov)

            weights = pd.Series(weights, index=universe)
            weights = weights[weights > 1e-6]
        except Exception as e:
            print("Failed optimiazation, skip date {}: {}".format(context.datetime, e))
            import traceback

            print(traceback.print_exc())
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

    # calculate Diversification ratio
    w = pd.Series([0] * len(cov), index=cov.index)
    w.loc[weights.index] = weights
    w_vol = np.dot(np.sqrt(np.diag(cov)), w.T)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    diversification_ratio = w_vol / port_vol
    record(Diversification_ratio=diversification_ratio)


def ccc(prices, frequency):
    df = prices

    # ref: https://github.com/annkon22/Finance_python/blob/main/CCC-GARCH%20model%20for%20multivariate%20volatility%20forecasting.ipynb
    returns = df.pct_change().iloc[1:]

    # define lists for storing objects
    coeffs = []
    cond_vol = []
    std_resids = []
    models = []

    # estimate the univariate garch models
    for asset in returns.columns:
        model = arch_model(returns[asset].fillna(0), mean="Constant", vol="GARCH", p=1, o=0, q=1).fit(
            update_freq=0, disp="off"
        )
        coeffs.append(model.params)
        cond_vol.append(model.conditional_volatility)
        std_resids.append(model.resid / model.conditional_volatility)
        models.append(model)

    # store the results in df
    coeffs_df = pd.DataFrame(coeffs, index=returns.columns)
    cond_vol_df = pd.DataFrame(cond_vol, index=returns.columns).transpose()
    std_resids_df = pd.DataFrame(std_resids, index=returns.columns).transpose()

    # calculate the constant conditional correlation matrix (CCC) R:
    R = std_resids_df.transpose().dot(std_resids_df).div(len(std_resids_df))

    # calculate one step ahead forecastof the conditional covariance matrix
    diag = []
    N = len(df.columns)
    D = np.zeros((N, N))

    for model in models:
        diag.append(model.forecast(horizon=1).variance.values[-1][0])

    diag = np.sqrt(np.array(diag))
    np.fill_diagonal(D, diag)

    H = np.matmul(np.matmul(D, R.values), D)
    cov = pd.DataFrame(H, index=df.columns, columns=df.columns)
    cov = cov * frequency
    return cov


def sccc(prices, frequency: int):
    S = risk_models.sample_cov(prices=prices, frequency=frequency)
    ccc_cov = ccc(prices=prices, frequency=frequency)
    cov = (S + ccc_cov) / 2
    return cov


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
    # print(portfolio_volatility)
    return portfolio_volatility


def min_volatility(cov, weight_bounds=None):
    # try valid initial guess
    if weight_bounds is None:
        weight_bounds = [(0, 1)] * len(cov)
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
        tol=1e-4,
    )
    record(Optimized_Volatility=result["fun"])
    record(Optimization_iteration=result.nfev)
    record(Optimization_success=result.success)
    if result["message"] == "Singular matrix E in LSQ subproblem":
        print(result)
        return None
    if not result["success"]:
        print(result)
        # code.interact(local=locals())
        return None

    return result["x"]


"""
DCC
"""

#     ________
#     \      /
#      \    /
#       \  /
#        \/

# Document Strings Publication

# Reference work:

# Bauwens, L., Laurent, S. and Rombouts, J.,
# Multivariate GARCH Models: A Survey. Journal of Applied
# Econometrics, 2006, 21, 79–109.

# Log-likelihood reference work - equations referenced throughout:

# Engle, R., Dynamic Conditional Correlation: A Simple Class of
# Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models.
# Journal of Business & Economic Statistics, 2002, 20, 339–350.

# This package created as the following Python package did not work for our purposes:
# mgarch 0.2.0 - available at https://pypi.org/project/mgarch/

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize


def covregpy_dcc(returns_matrix, p=3, q=3, days=10, print_correlation=False, rescale=False):
    """
    Dynamic Conditional Correlation - Multivariate Generalized Autoregressive Conditional Heteroskedasticity model.

    Parameters
    ----------
    returns_matrix : real ndarray
        Matrix of asset returns.

    p : positive integer
        GARCH(p, q) model 'p' value used in calculation of uni-variate variance used in model.

    q : positive integer
        GARCH(p, q) model 'q' value used in calculation of uni-variate variance used in model.

    days : positive integer
        Number of days ahead to forecast covariance.

    print_correlation : boolean
        Debugging to gauge if correlation structure is reasonable.

    rescale : boolean
        Automatically rescale data to help if encountering convergence problems - see:
        https://arch.readthedocs.io/en/latest/univariate/introduction.html

    Returns
    -------
    forecasted_covariance : real ndarray
        Forecasted covariance.

    Notes
    -----

    """
    if not isinstance(
        returns_matrix,
        (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])), type(pd.DataFrame(np.asarray([[1.0, 2.0], [3.0, 4.0]])))),
    ):
        raise TypeError("Returns must be of type np.ndarray and pd.Dataframe.")
    if pd.isnull(np.asarray(returns_matrix)).any():
        raise TypeError("Returns must not contain nans.")
    if np.array(returns_matrix).dtype != np.array([[1.0, 1.0], [1.0, 1.0]]).dtype:
        raise TypeError("Returns must only contain floats.")
    if (not isinstance(p, int)) or (p <= 0):
        raise ValueError("'p' must be a positive integer.")
    if (not isinstance(q, int)) or (q <= 0):
        raise ValueError("'q' must be a positive integer.")
    if (not isinstance(days, int)) or (days <= 0):
        raise ValueError("'days' must be a positive integer.")
    if not isinstance(print_correlation, bool):
        raise TypeError("'print_correlation' must be boolean.")
    if not isinstance(rescale, bool):
        raise TypeError("'rescale' must be boolean.")

    # convert Dataframe
    returns_matrix = np.asarray(returns_matrix)

    # # flip matrix to be consistent
    # if np.shape(returns_matrix)[0] < np.shape(returns_matrix)[1]:
    #     returns_matrix = returns_matrix.T

    # initialised modelled variance matrix
    modelled_variance = np.zeros_like(returns_matrix)

    # iteratively calculate modelled variance using univariate GARCH model
    for stock in range(np.shape(returns_matrix)[1]):
        model = arch_model(returns_matrix[:, stock], mean="Zero", vol="GARCH", p=p, q=q, rescale=rescale)
        model_fit = model.fit(disp="off")
        # if rescale:
        #     modelled_variance[:, stock] = model_fit.conditional_volatility / model.scale
        # else:
        #     modelled_variance[:, stock] = model_fit.conditional_volatility
        modelled_variance[:, stock] = model_fit.conditional_volatility

    # optimise alpha & beta parameters to be used in page 90 equation (40)
    params = minimize(
        dcc_loglike,
        (0.01, 0.94),
        args=(returns_matrix, modelled_variance),
        bounds=((1e-6, 1), (1e-6, 1)),
        method="Nelder-Mead",
    )

    # list of optimisation methods available

    # Nelder-Mead, Powell, CG, BFGS, Newton-CG,
    # L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr,
    # dogleg, trust-ncg, trust-exact, trust-krylov

    a = params.x[0]
    b = params.x[1]

    t = np.shape(returns_matrix)[0]  # time interval
    q_bar = np.cov(returns_matrix.T)  # base (unconditional) covariance

    # setup matrices
    q_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - Equation (40)
    h_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 89 - Equation (35)
    dts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 88 - Equation (32)
    dts_inv = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # used in calculation of u_t
    u_t = np.zeros((np.shape(q_bar)[0], t))  # page 89 - defined between Equation (37) and (38)
    qts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - defined within Equation (39)
    r_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - Equation (39)

    # initialise q_t
    # unsure of initialisation
    q_t[0] = np.zeros_like(np.matmul(returns_matrix[0, :].reshape(-1, 1) / 2, returns_matrix[0, :].reshape(1, -1) / 2))

    for var in range(t):
        # page 88 - Equation (32)
        dts[var] = np.diag(modelled_variance[int(var - 1), :])  # modelled variance - page 89 - Equation (34)
        # page 89 - defined between Equation (37) and (38)
        try:
            dts_inv[var] = np.linalg.inv(dts[var])
        except:
            dts_inv[var] = np.linalg.pinv(dts[var])
        u_t[:, var] = np.matmul(dts_inv[var], returns_matrix[var, :].reshape(-1, 1))[:, 0]

    for var in range(1, t):
        # page 90 - Equation (40)
        q_t[var] = (
            (1 - a - b) * q_bar
            + a * np.matmul(u_t[:, int(var - 1)].reshape(-1, 1), u_t[:, int(var - 1)].reshape(1, -1))
            + b * q_t[int(var - 1)]
        )

        # page 90 - defined within Equation (39)
        try:
            qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))
        except:
            qts[var] = np.linalg.pinv(np.sqrt(np.diag(np.diag(q_t[var]))))

        # page 90 - Equation (39)
        r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))

        # page 89 - Equation (35)
        h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

    # Brownian uncertainty with  variance increasing linearly with square root of time
    forecasted_covariance = h_t[-1] * np.sqrt(days)

    if print_correlation:
        corr = np.zeros_like(forecasted_covariance)
        for i in range(np.shape(corr)[0]):
            for j in range(np.shape(corr)[1]):
                corr[i, j] = forecasted_covariance[i, j] / (
                    np.sqrt(forecasted_covariance[i, i]) * np.sqrt(forecasted_covariance[j, j])
                )
        print(corr)

    return forecasted_covariance


def dcc_loglike(params, returns_matrix, modelled_variance):
    """
    Dynamic Conditional Correlation loglikelihood function to optimise.

    Parameters
    ----------
    params : real ndarray
        Real array of shape (2, ) with two parameters to optimise.

    returns_matrix : real ndarray
        Matrix of asset returns.

    modelled_variance : real ndarray
        Uni-variate modelled variance to be used in optimisation.

    Returns
    -------
    loglike : float
        Loglikelihood function to be minimised.

    Notes
    -----

    """
    if pd.isnull(np.asarray(params)).any():
        raise ValueError("Parameters must not contain nans.")
    if np.array(params).dtype != np.array([[1.0, 1.0], [1.0, 1.0]]).dtype:
        raise ValueError("Parameters must only contain floats.")
    if pd.isnull(np.asarray(returns_matrix)).any():
        raise ValueError("Returns must not contain nans.")
    if np.array(returns_matrix).dtype != np.array([[1.0, 1.0], [1.0, 1.0]]).dtype:
        raise ValueError("Returns must only contain floats.")
    if pd.isnull(np.asarray(modelled_variance)).any():
        raise ValueError("Covariance must not contain nans.")
    if np.array(modelled_variance).dtype != np.array([[1.0, 1.0], [1.0, 1.0]]).dtype:
        raise ValueError("Covariance must only contain floats.")

    t = np.shape(returns_matrix)[0]  # time interval
    q_bar = np.cov(returns_matrix.T)  # base (unconditional) covariance

    # setup matrices
    q_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - Equation (40)
    h_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 89 - Equation (35)
    dts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 88 - Equation (32)
    dts_inv = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # used in calculation of u_t
    u_t = np.zeros((np.shape(q_bar)[0], t))  # page 89 - defined between Equation (37) and (38)
    qts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - defined within Equation (39)
    r_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - Equation (39)

    # initialise q_t
    # unsure of initialisation
    q_t[0] = np.zeros_like(np.matmul(returns_matrix[0, :].reshape(-1, 1) / 2, returns_matrix[0, :].reshape(1, -1) / 2))

    for var in range(t):
        # page 88 - Equation (32)
        dts[var] = np.diag(modelled_variance[int(var - 1), :])  # modelled variance - page 89 - Equation (34)
        # page 89 - defined between Equation (37) and (38)
        try:
            dts_inv[var] = np.linalg.inv(dts[var])
        except:
            dts_inv[var] = np.linalg.pinv(dts[var])
        u_t[:, var] = np.matmul(dts_inv[var], returns_matrix[var, :].reshape(-1, 1))[:, 0]

    # initialise log-likehood value
    loglike = 0

    for var in range(1, t):
        # page 90 - Equation (40)
        q_t[var] = (
            (1 - params[0] - params[1]) * q_bar
            + params[0] * np.matmul(u_t[:, int(var - 1)].reshape(-1, 1), u_t[:, int(var - 1)].reshape(1, -1))
            + params[1] * q_t[int(var - 1)]
        )

        # page 90 - defined within Equation (39)
        try:
            qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))
        except:
            qts[var] = np.linalg.pinv(np.sqrt(np.diag(np.diag(q_t[var]))))

        # page 90 - Equation (39)
        r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))

        # page 89 - Equation (35)
        h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

        # likelihood function from reference work page 11 - Equation 26
        try:
            loglike -= (
                np.shape(q_bar)[0] * np.log(2 * np.pi)
                + 2 * np.log(np.linalg.det(dts[var]))
                + np.log(np.linalg.det(r_t[var]))
                + np.matmul(u_t[:, var].reshape(1, -1), np.matmul(np.linalg.inv(r_t[var]), u_t[:, var].reshape(-1, 1)))[
                    0
                ][0]
            )
        except:
            loglike -= (
                np.shape(q_bar)[0] * np.log(2 * np.pi)
                + 2 * np.log(np.linalg.det(dts[var]))
                + np.log(np.linalg.det(r_t[var]))
                + np.matmul(
                    u_t[:, var].reshape(1, -1), np.matmul(np.linalg.pinv(r_t[var]), u_t[:, var].reshape(-1, 1))
                )[0][0]
            )

    return loglike

if __name__ == '__main__':
    from xquant_backtest.backtest import XQuantAlgorithm

    algo = XQuantAlgorithm(
        backtest_id='tmp11a0h02b',
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
