import numpy as np
import pandas as pd
import cvxpy as cp 
import matplotlib.pyplot as plt


def OLS(returns, factRet):
    """
    OLS factor model.
    Returns:
        mu      : (n,1) expected returns
        Q       : (n,n) covariance matrix
        R2      : (n,1) in-sample R² for each asset
        R2_adj  : (n,1) adjusted R² for each asset
    """
    T, p = factRet.shape
    n = returns.shape[1]

    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)  # shape (T, p+1)
    B = np.linalg.solve(X.T @ X, X.T @ returns)  # shape (p+1, n)

    a = B[0, :]
    V = B[1:, :]

    Y_hat = X @ B  # shape (T, n)
    residuals = returns.values - Y_hat
    sigma_ep = 1 / (T - p - 1) * np.sum(residuals ** 2, axis=0)
    D = np.diag(sigma_ep)

    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D
    Q = (Q + Q.T) / 2  # ensure symmetry

    # Compute R² and Adjusted R²
    ss_res = np.sum(residuals ** 2, axis=0)
    ss_tot = np.sum((returns.values - returns.values.mean(axis=0)) ** 2, axis=0)
    R2 = 1 - ss_res / ss_tot

    k = p  # number of predictors (exclude intercept)
    R2_adj = 1 - (ss_res / (T - k - 1)) / (ss_tot / (T - 1))

    return mu, Q, R2.reshape((n, 1)), R2_adj.reshape((n, 1))

def FF3(returns, factRet):
    """
    Fama-French 3-Factor model.
    Uses only the first 3 columns of the factor returns.
    Returns:
        mu       : (n,1) expected returns
        Q        : (n,n) covariance matrix
        R2       : (n,1) in-sample R²
        R2_adj   : (n,1) adjusted R²
    """
    T, n = returns.shape
    X_ff3 = factRet.iloc[:, :3].values  # use first 3 factors
    p = X_ff3.shape[1]

    X = np.concatenate([np.ones((T, 1)), X_ff3], axis=1)  # intercept + 3 factors
    B = np.linalg.solve(X.T @ X, X.T @ returns)  # shape (p+1, n)

    a = B[0, :]
    V = B[1:, :]

    Y_hat = X @ B
    residuals = returns.values - Y_hat
    sigma_ep = 1 / (T - p - 1) * np.sum(residuals ** 2, axis=0)
    D = np.diag(sigma_ep)

    f_bar = np.expand_dims(X_ff3.mean(axis=0), 1)
    F = np.cov(X_ff3, rowvar=False)
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D
    Q = (Q + Q.T) / 2

    # R² and Adjusted R²
    ss_res = np.sum(residuals ** 2, axis=0)
    ss_tot = np.sum((returns.values - returns.values.mean(axis=0)) ** 2, axis=0)
    R2 = 1 - ss_res / ss_tot
    k = p  # 3 factors (no intercept counted)
    R2_adj = 1 - (ss_res / (T - k - 1)) / (ss_tot / (T - 1))

    return mu, Q, R2.reshape((n, 1)), R2_adj.reshape((n, 1))


def cross_validate_lasso(X, y, lam_values, k=4):
    """
    Perform deterministic K-fold cross-validation to select the best lambda for Lasso.

    Inputs:
        X           : NumPy array or DataFrame (T x p), factor matrix (already normalized)
        y           : Pandas Series (T,), excess return for one asset
        lam_values  : list/array of candidate lambda values
        k           : number of CV folds (default 4)

    Returns:
        best_lambda : lambda value with lowest average validation error
    """
    # Ensure X is a NumPy array
    if isinstance(X, pd.DataFrame):
        X = X.values

    T = len(y)
    fold_size = T // k
    indices = np.arange(T)  # No shuffling for deterministic split

    avg_val_errors = []

    for lam in lam_values:
        val_errors = []

        for j in range(k):
            val_idx = indices[j * fold_size : (j + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y.iloc[train_idx].values
            y_val = y.iloc[val_idx].values

            beta = cp.Variable(X.shape[1])
            objective = cp.Minimize(cp.sum_squares(X_train @ beta - y_train) + lam * cp.norm1(beta))
            problem = cp.Problem(objective)
            problem.solve(solver=cp.OSQP)

            y_pred = X_val @ beta.value
            val_error = np.mean((y_val - y_pred) ** 2)
            val_errors.append(val_error)

        avg_error = np.mean(val_errors)
        avg_val_errors.append(avg_error)

    # Find best lambda (smallest CV error)
    min_index = np.argmin(avg_val_errors)
    best_lambda = lam_values[min_index]

    return best_lambda

def LASSO(returns, factRet, lam):
    """
    Lasso factor regression.
    Returns:
        mu       : (n,1) expected excess returns
        Q        : (n,n) covariance matrix
        R2       : (n,1) in-sample R²
        R2_adj   : (n,1) adjusted R²
    """
    T, n = returns.shape
    p = factRet.shape[1]

    # Design matrix with intercept
    X = np.column_stack([np.ones(T), factRet.values])  # shape (T, p+1)

    # Containers
    B_hat = np.zeros((p + 1, n))
    residuals = np.zeros((T, n))
    R2 = np.zeros((n, 1))
    R2_adj = np.zeros((n, 1))

    for j in range(n):
        y = returns.iloc[:, j].values

        beta = cp.Variable(p + 1)
        objective = cp.Minimize(cp.sum_squares(X @ beta - y) + lam * cp.norm1(beta))
        prob = cp.Problem(objective)
        prob.solve(solver=cp.OSQP)

        b = beta.value
        B_hat[:, j] = b
        y_hat = X @ b
        res = y - y_hat
        residuals[:, j] = res

        # R² and adjusted R²
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        R2[j, 0] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Count non-zero coefficients excluding intercept
        k = np.sum(np.abs(b[1:]) > 1e-6)
        if ss_tot > 0 and T - k - 1 > 0:
            R2_adj[j, 0] = 1 - (ss_res / (T - k - 1)) / (ss_tot / (T - 1))
        else:
            R2_adj[j, 0] = np.nan

    # Compute mu and Q
    a = B_hat[0, :]
    V = B_hat[1:, :]

    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar

    sigma_ep = np.var(residuals, axis=0)
    Q = V.T @ F @ V + np.diag(sigma_ep)
    Q = (Q + Q.T) / 2

    return mu, Q, R2, R2_adj


def BSS(returns, factorRet, K):
    """
    Best Subset Selection (BSS) via MIQP.

    Inputs:
        returns   : (T x n) DataFrame of excess returns
        factorRet : (T x p) DataFrame of factor returns (excluding RF)
        K         : max number of predictors allowed (excluding intercept)

    Returns:
        mu      : (n, 1) expected excess returns
        Q       : (n, n) covariance matrix
        R2      : (n, 1) in-sample R²
        R2_adj  : (n, 1) in-sample adjusted R²
    """
    T, n = returns.shape
    p = factorRet.shape[1]
    X = np.column_stack([np.ones(T), factorRet.values])  # T x (p+1)

    B_all = np.zeros((n, p + 1))
    residuals = np.zeros((T, n))
    R2 = np.zeros((n, 1))
    R2_adj = np.zeros((n, 1))

    for i in range(n):
        r_i = returns.iloc[:, i].values
        B = cp.Variable(p + 1)
        y = cp.Variable(p + 1, boolean=True)
        M = 5.0  # Big-M constant

        constraints = [
            B <=  M * y,
            B >= -M * y,
            cp.sum(y[1:]) <= K  # exclude intercept
        ]

        obj = cp.Minimize(cp.sum_squares(r_i - X @ B))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GUROBI)

        b_hat = B.value
        B_all[i, :] = b_hat
        y_hat = X @ b_hat
        res = r_i - y_hat
        residuals[:, i] = res

        # Compute R² and adjusted R²
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((r_i - np.mean(r_i)) ** 2)
        R2[i, 0] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Count number of nonzero coefficients excluding intercept
        k = np.sum(np.abs(b_hat[1:]) > 1e-6)
        if ss_tot > 0 and T - k - 1 > 0:
            R2_adj[i, 0] = 1 - (ss_res / (T - k - 1)) / (ss_tot / (T - 1))
        else:
            R2_adj[i, 0] = np.nan

    # Compute mu
    mean_factors = factorRet.mean(axis=0).values  # (p,)
    mu = B_all[:, 0] + B_all[:, 1:] @ mean_factors
    mu = mu.reshape((n, 1))

    # Compute Q
    cov_factors = np.cov(factorRet.values, rowvar=False)
    B_factors = B_all[:, 1:]
    Q_factor = B_factors @ cov_factors @ B_factors.T
    var_resid = np.var(residuals, axis=0)
    Q = Q_factor + np.diag(var_resid)
    Q = (Q + Q.T) / 2

    return mu, Q, R2, R2_adj

if __name__ == "__main__":
    # Load prices and factor returns
    prices = pd.read_csv("MMF1921_AssetPrices_3.csv", index_col=0, parse_dates=True)
    factorRet = pd.read_csv("MMF1921_FactorReturns_3.csv", index_col=0, parse_dates=True)

    # Compute monthly returns from prices
    assetRet = prices.pct_change().dropna()

    # Compute excess returns: subtract RF (risk-free rate)
    RF = factorRet['RF']  # Usually RF is in percent
    excessRet = assetRet.sub(RF, axis=0)  # auto-aligns on index

    # Drop RF column from factors before using them in OLS
    factors = factorRet.drop(columns=['RF'])

    # Align time indices
    common_idx = excessRet.index.intersection(factors.index)
    excessRet = excessRet.loc[common_idx]
    factors = factors.loc[common_idx]

    # Run OLS, FF3
    mu_ols, Q_ols, R2_ols, R2adj_ols = OLS(excessRet, factorRet)
    mu_ff3, Q_ff3, R2_ff3, R2adj_ff3 = FF3(excessRet, factorRet)
    mu_lasso, Q_lasso, R2_lasso, R2adj_lasso = LASSO(excessRet, factorRet, lam=0.01)
    mu_bss, Q_bss, R2_bss, R2adj_bss = BSS(excessRet, factorRet, K=3)

    ''' 
    # Run lambda 
    # Step 1: Choose lambda grid
    lam_grid = np.logspace(-3, 0, 10)
    y = excessRet.iloc[:, 0]  # This is a Series of shape (T,)
    X = np.column_stack([np.ones(len(factors)), ((factors - factors.mean()) / factors.std()).values])
    # Step 2: Run cross-validation once across all assets
    best_lam = cross_validate_lasso(X, y, lam_grid)
    '''

    # LASSO result
    print("\n--- LASSO Results ---")
    #print("Best lambda:", best_lam)
    print("mu shape:", mu_lasso.shape)
    print("Average R²:", np.mean(R2_lasso))

    # Print output summaries
    print("\n--- OLS Results ---")
    print("mu shape:", mu_ols.shape)
    print("Q shape:", Q_ols.shape)
    print("R2 shape:", R2_ols.shape)
    print("Average R²:", np.mean(R2_ols))
    
    print("\n--- Fama-French 3-Factor Results ---")
    print("mu shape:", mu_ff3.shape)
    print("Q shape:", Q_ff3.shape)
    print("R2 shape:", R2_ff3.shape)
    print("Average R²:", np.mean(R2_ff3))

    K = 3  # or try 2, 4, etc.
    print("mu shape:", mu_bss.shape)          # should be (33, 1)
    print("Q shape:", Q_bss.shape)            # should be (33, 33)
    print("R² shape:", R2_bss.shape)          # should be (33, 1)
    print("Average R²:", np.mean(R2_bss))     # summary of fit

