import numpy as np
from services.estimators import *
from services.optimization import *


# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class


def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        print(len(returns))
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use
        
    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q, _, _ = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x

class FF3_MVO:
    def __init__(self, NumObs=36):
        self.NumObs = NumObs

    def execute_strategy(self, periodReturns, factorReturns):
        returns = periodReturns.iloc[-self.NumObs:, :]
        factRet = factorReturns.iloc[-self.NumObs:, :]
        mu, Q, _, _ = FF3(returns, factRet)
        x = MVO(mu, Q)
        return x

class Lasso_MVO:
    def __init__(self, NumObs=60, lam=0.01):
        self.NumObs = NumObs
        self.lam = lam

    def execute_strategy(self, periodReturns, factorReturns):
        returns = periodReturns.iloc[-self.NumObs:, :]
        factRet = factorReturns.iloc[-self.NumObs:, :]
        mu, Q, _, _ = LASSO(returns, factRet, lam=self.lam)
        x = MVO(mu, Q)
        return x

class BSS_MVO:
    def __init__(self, NumObs=36, K=3):
        self.NumObs = NumObs
        self.K = K

    def execute_strategy(self, periodReturns, factorReturns):
        returns = periodReturns.iloc[-self.NumObs:, :]
        factRet = factorReturns.iloc[-self.NumObs:, :]
        mu, Q, _, _ = BSS(returns, factRet, K=self.K)
        x = MVO(mu, Q)
        return x

class Ensemble_MVO:
    """
    Combines multiple factor model estimates (OLS, FF3, Lasso, BSS) to form an ensemble expected return
    and runs MVO on the combined estimate.
    """

    def __init__(self, NumObs=36, model_weights=[0.4,0.1,0.4,0.1]):
        self.NumObs = NumObs
        self.model_weights = model_weights  # e.g., [0.25, 0.25, 0.25, 0.25] or custom

    def execute_strategy(self, periodReturns, factorReturns):
        """
        periodReturns: DataFrame of asset excess returns
        factorReturns: DataFrame of factor returns (no RF)
        """
        returns = periodReturns.iloc[-self.NumObs:, :]
        factRet = factorReturns.iloc[-self.NumObs:, :]

        mu_ols, Q_ols, _ , _ = OLS(returns, factRet)
        mu_ff3, Q_ff3, _ , _ = FF3(returns, factRet)
        mu_lasso, Q_lasso, _ , _  = LASSO(returns, factRet, lam=0.0001)  # or use best_lambda
        mu_bss, Q_bss, _ , _ = BSS(returns, factRet, K=3)

        mu_list = [mu_ols, mu_ff3, mu_lasso, mu_bss]
        Q_list = [Q_ols, Q_ff3, Q_lasso, Q_bss]

        # Default to equal weights
        if self.model_weights is None:
            weights = np.array([1/len(mu_list)] * len(mu_list)).reshape(1, 1, -1)
        else:
            weights = np.array(self.model_weights).reshape(1, 1, -1)

        mu_stack = np.stack(mu_list, axis=2)  # shape (n, 1, 4)
        mu_ensemble = np.sum(mu_stack * weights, axis=2)  # (n, 1)

        Q_stack = np.stack(Q_list, axis=2)
        Q_ensemble = np.mean(Q_stack, axis=2)  # equal-weighted Q

        x = MVO(mu_ensemble, Q_ensemble)
        return x
