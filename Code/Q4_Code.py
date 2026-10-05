import pandas as pd
import numpy as np
from io import StringIO
from ipca import InstrumentedPCA

# Question 4: Rolling-window evaluation

WINDOW = 240

# Characteristics for IPCA
# Same construction as Question 3

def load_block(filepath, title=None):
    """Reads one table from a Ken French file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    search_from = 0
    if title is not None:
        search_from = next(i for i, line in enumerate(lines) if title in line)

    start_idx = next(
        i for i in range(search_from, len(lines))
        if lines[i].startswith(',')
    )

    end_idx = next(
        (i for i in range(start_idx + 1, len(lines))
         if lines[i].strip() == ''),
        len(lines)
    )

    block = ''.join(lines[start_idx:end_idx])
    df = pd.read_csv(StringIO(block))

    df = df.rename(columns={df.columns[0]: 'date'})
    df.columns = [c.strip() for c in df.columns]
    df['date'] = df['date'].astype(str).str.strip()

    df = df[df['date'].str.match(r'^\d{6}$')]
    df['date'] = df['date'].astype(int)
    df = df.set_index('date').astype(float)

    return df.mask(df <= -99.99)


def rank_transform(df):
    """Ranks portfolios each month and scales ranks to [-0.5, 0.5]."""
    ranks = df.rank(axis=1)
    n = df.notna().sum(axis=1)
    return ranks.sub(1).div(n - 1, axis=0) - 0.5


PORT_FILE = 'Data/portfolios_25_size_mom.csv'

# Load long history so lagged characteristics are available
returns = load_block(PORT_FILE)
market_cap = load_block(PORT_FILE, 'Average Market Cap')
prior_vw = load_block(
    PORT_FILE,
    'Value-Weighted Average of Prior Returns'
)

ff3 = load_block('Data/factor_ff5.csv')[['Mkt-RF', 'RF']]

portfolio_names = returns.columns.tolist()

ff3 = ff3.reindex(returns.index)

excess_full = returns.sub(ff3['RF'], axis=0)


# Construct the same six characteristics as Q3
chars = {}

# 1. Size
chars['size'] = np.log(market_cap)

# 2. Momentum: months t-12 to t-2
chars['mom'] = prior_vw

# 3. Short-term reversal: month t-1
chars['st_rev'] = returns.shift(1)

# 4. Long-term reversal: months t-60 to t-13
chars['lt_rev'] = returns.shift(13).rolling(48).sum()

# 5. Volatility: previous 12 months
chars['vol'] = returns.shift(1).rolling(12).std()

# 6. Market beta: previous 36 months
past_ex = excess_full.shift(1)
past_mkt = ff3['Mkt-RF'].shift(1)

beta = pd.DataFrame(
    index=returns.index,
    columns=portfolio_names,
    dtype=float
)

for c in portfolio_names:
    beta[c] = (
        past_ex[c].rolling(36).cov(past_mkt)
        / past_mkt.rolling(36).var()
    )

chars['beta'] = beta


# Rank-transform over our assignment sample
for name in chars:
    chars[name] = rank_transform(
        chars[name].loc[197208:202409, portfolio_names]
    )

# Load the same excess returns used in Q3
data = pd.read_csv('Data/merged_clean.csv', index_col='date')

portfolio_cols = [c for c in data.columns if c.endswith('_excess')]
names = [c.replace('_excess', '') for c in portfolio_cols]

R = data[portfolio_cols].values

T, N = R.shape

print("T =", T)
print("N =", N)
print("Rolling window =", WINDOW)
print("Number of evaluation months =", T - WINDOW)

print("\nFirst sample month:", data.index[0])
print("Last sample month:", data.index[-1])

print("\nFirst estimation window:")
print(data.index[0], "to", data.index[WINDOW - 1])

print("First evaluation month:")
print(data.index[WINDOW])

print("\nSecond estimation window:")
print(data.index[1], "to", data.index[WINDOW])

print("Second evaluation month:")
print(data.index[WINDOW + 1])

print("\nFinal estimation window:")
print(data.index[T - WINDOW - 1], "to", data.index[T - 2])

print("Final evaluation month:")
print(data.index[T - 1])

# STEP 4.1: PCA
# reusing Q3 
def pca(R_window, K):
    """Estimate PCA using only the returns in the current rolling window."""
    
    second_moment = R_window.T @ R_window / R_window.shape[0]

    eigvals, eigvecs = np.linalg.eigh(second_moment)

    order = np.argsort(eigvals)[::-1]

    B = eigvecs[:, order[:K]]

    # Factors inside the 240-month estimation window
    F = R_window @ B

    # Sign normalization
    signs = np.where(F.mean(axis=0) < 0, -1.0, 1.0)

    B = B * signs
    F = F * signs

    return B, F


# Test PCA K=1 on FIRST rolling window only

R_train = R[:WINDOW]       # months 1,...,240
r_next = R[WINDOW]         # month 241

B, F = pca(R_train, K=1)

print("\n--- First-window PCA K=1 ---")
print("R_train shape:", R_train.shape)
print("r_next shape:", r_next.shape)
print("B shape:", B.shape)
print("F shape:", F.shape)

# Check that PCA loadings are orthonormal
print("B'B:")
print(B.T @ B)

# Mean factor over the 240 estimation months
lambda_hat = F.mean(axis=0)

print("lambda_hat:", lambda_hat)

# Factor realization in the NEW month
f_next = B.T @ r_next

print("f_next:", f_next)

# First-window PCA K=1: errors for month t+1

# Total-fit residual
fitted_total_next = B @ f_next
eta_next = r_next - fitted_total_next

# Predicted pricing error
fitted_pred_next = B @ lambda_hat
alpha_next = r_next - fitted_pred_next

print("\n--- First-window PCA K=1 errors ---")

print("fitted_total_next shape:", fitted_total_next.shape)
print("eta_next shape:", eta_next.shape)

print("fitted_pred_next shape:", fitted_pred_next.shape)
print("alpha_next shape:", alpha_next.shape)

print("\nFirst 5 actual returns:")
print(r_next[:5])

print("\nFirst 5 total fitted returns:")
print(fitted_total_next[:5])

print("\nFirst 5 eta:")
print(eta_next[:5])

print("\nFirst 5 predicted returns:")
print(fitted_pred_next[:5])

print("\nFirst 5 alpha:")
print(alpha_next[:5])


# Rolling PCA K=1

K = 1

eta_pca1 = []
alpha_pca1 = []

for test_idx in range(WINDOW, T):

    # Previous 240 months only
    R_train = R[test_idx - WINDOW:test_idx]

    # Next month, not used in estimation
    r_next = R[test_idx]

    # Estimate PCA using the 240-month window
    B, F = pca(R_train, K)

    # Mean factor realization over those 240 months
    lambda_hat = F.mean(axis=0)

    # Realized factor in the next month
    f_next = B.T @ r_next

    # Total-fit residual
    eta_next = r_next - B @ f_next

    # Predicted pricing error
    alpha_next = r_next - B @ lambda_hat

    eta_pca1.append(eta_next)
    alpha_pca1.append(alpha_next)


eta_pca1 = np.array(eta_pca1)
alpha_pca1 = np.array(alpha_pca1)

print("\n--- Rolling PCA K=1 ---")
print("eta shape:", eta_pca1.shape)
print("alpha shape:", alpha_pca1.shape)

print("\nFirst 5 average eta:")
print(eta_pca1.mean(axis=0)[:5])

print("\nFirst 5 average alpha:")
print(alpha_pca1.mean(axis=0)[:5])


# Out-of-sample R2: PCA K=1

# Returns for the 386 evaluation months only
R_oos = R[WINDOW:]

SST_oos = np.sum(R_oos ** 2)

SSE_total_pca1 = np.sum(eta_pca1 ** 2)
SSE_pred_pca1 = np.sum(alpha_pca1 ** 2)

r2_total_pca1 = 1 - SSE_total_pca1 / SST_oos
r2_pred_pca1 = 1 - SSE_pred_pca1 / SST_oos

print("\n--- PCA K=1 out-of-sample R2 ---")
print("R_oos shape:", R_oos.shape)
print("SST:", SST_oos)
print("Total-fit SSE:", SSE_total_pca1)
print("Predictive SSE:", SSE_pred_pca1)
print("Total R2 (%):", 100 * r2_total_pca1)
print("Predictive R2 (%):", 100 * r2_pred_pca1)

#making function
def rolling_pca(R, K, window=240):
    """
    Rolling PCA evaluation.

    For each month t+1:
    - estimate PCA using the previous 240 months
    - construct the realized factor at t+1
    - calculate total-fit residual eta
    - calculate predicted pricing error alpha
    """

    T, N = R.shape

    eta_list = []
    alpha_list = []
    lambda_list = []

    for test_idx in range(window, T):

        # Previous 240 months
        R_train = R[test_idx - window:test_idx]

        # Next month's returns
        r_next = R[test_idx]

        # Estimate PCA
        B, F = pca(R_train, K)

        # Mean factors over estimation window
        lambda_hat = F.mean(axis=0)
        # step 4.3: store factor means
        lambda_list.append(lambda_hat)

        # Realized factors in month t+1
        f_next = B.T @ r_next

        # Total-fit residual
        eta_next = r_next - B @ f_next

        # Predicted pricing error
        alpha_next = r_next - B @ lambda_hat

        eta_list.append(eta_next)
        alpha_list.append(alpha_next)

    return (
    np.array(eta_list),
    np.array(alpha_list),
    np.array(lambda_list)
)

# using functon for K3
eta_pca3, alpha_pca3, lambdas_pca3 = rolling_pca(
    R, K=3, window=WINDOW
)

print("\n--- Rolling PCA K=3 ---")
print("eta shape:", eta_pca3.shape)
print("alpha shape:", alpha_pca3.shape)

print("\nFirst 5 average eta:")
print(eta_pca3.mean(axis=0)[:5])

print("\nFirst 5 average alpha:")
print(alpha_pca3.mean(axis=0)[:5])

# step 4.3: store PCA K=1 factor means for all rolling windows
lambdas_pca1 = []

for test_idx in range(WINDOW, T):

    R_train = R[test_idx - WINDOW:test_idx]

    B, F = pca(R_train, K=1)

    lambda_hat = F.mean(axis=0)

    lambdas_pca1.append(lambda_hat)

lambdas_pca1 = np.array(lambdas_pca1)


print("\n--- STEP 3: PCA factor means ---")
print("Lambda PCA 1 shape:", lambdas_pca1.shape)
print("Lambda PCA 3 shape:", lambdas_pca3.shape)

# calculating its R-squared 
SSE_total_pca3 = np.sum(eta_pca3 ** 2)
SSE_pred_pca3 = np.sum(alpha_pca3 ** 2)

r2_total_pca3 = 1 - SSE_total_pca3 / SST_oos
r2_pred_pca3 = 1 - SSE_pred_pca3 / SST_oos

print("\n--- PCA K=3 out-of-sample R2 ---")
print("Total-fit SSE:", SSE_total_pca3)
print("Predictive SSE:", SSE_pred_pca3)
print("Total R2 (%):", 100 * r2_total_pca3)
print("Predictive R2 (%):", 100 * r2_pred_pca3)


# STEP 4.2: IPCA

# Same 6 characteristics as Q3, plus a constant
char_cols = list(chars.keys()) + ['const']
L = len(char_cols)

# Put characteristics into one array:
# months x portfolios x characteristics
Z = np.stack(
    [chars[c].loc[data.index, names].values for c in chars]
    + [np.ones((T, N))],
    axis=2
)

print("\n--- Q4 IPCA characteristic check ---")
print("Z shape:", Z.shape)
print("T =", T)
print("N =", N)
print("L =", L)
print("Characteristics:", char_cols)
print("Missing values:", np.isnan(Z).sum())


# # Function: estimate IPCA in ONE 240-month window
# def ipca_window(R_window, Z_window, dates, K):
#     """Estimate IPCA in one 240-month window."""

#     W = R_window.shape[0]

#     # Convert returns to long format
#     ret_long = pd.DataFrame(
#         R_window,
#         index=dates,
#         columns=range(N)
#     ).stack()

#     ret_long.index.names = ['date', 'portfolio']
#     ret_long = ret_long.swaplevel().sort_index()

#     # Convert characteristics to long format
#     char_long = pd.DataFrame(
#         Z_window.reshape(W * N, L),
#         index=pd.MultiIndex.from_product(
#             [dates, range(N)],
#             names=['date', 'portfolio']
#         ),
#         columns=char_cols
#     ).swaplevel().sort_index()

#     # Estimate IPCA
#     model = InstrumentedPCA(
#         n_factors=K,
#         intercept=False,
#         iter_tol=1e-8
#     )

#     model.fit(
#         X=char_long,
#         y=ret_long,
#         data_type='panel'
#     )

#     # Get estimated Gamma and factors
#     Gamma, factors = model.get_factors()

#     # Package gives factors as K x months.
#     # We want months x K, like in Q3.
#     F = factors.T

#     return Gamma, F


# # Check IPCA on the FIRST 240-month window
# R_train = R[:WINDOW]
# Z_train = Z[:WINDOW]
# train_dates = data.index[:WINDOW]

# print("\n--- First-window setup ---")
# print("R_train shape:", R_train.shape)
# print("Z_train shape:", Z_train.shape)


# # Test K = 1
# Gamma1_test, F1_test = ipca_window(
#     R_train,
#     Z_train,
#     train_dates,
#     K=1
# )

# print("\n--- First-window IPCA K=1 ---")
# print("Gamma shape:", Gamma1_test.shape)
# print("F shape:", F1_test.shape)

# print("\nGamma K=1:")
# print(
#     pd.DataFrame(
#         Gamma1_test,
#         index=char_cols,
#         columns=['Factor 1']
#     )
# )


# # Test K = 3
# Gamma3_test, F3_test = ipca_window(
#     R_train,
#     Z_train,
#     train_dates,
#     K=3
# )

# print("\n--- First-window IPCA K=3 ---")
# print("Gamma shape:", Gamma3_test.shape)
# print("F shape:", F3_test.shape)


# # Test K = 5
# Gamma5_test, F5_test = ipca_window(
#     R_train,
#     Z_train,
#     train_dates,
#     K=5
# )

# print("\n--- First-window IPCA K=5 ---")
# print("Gamma shape:", Gamma5_test.shape)
# print("F shape:", F5_test.shape)


# # Rolling IPCA
# # Save everything needed so IPCA never has to be estimated again

# def rolling_ipca_save(R, Z, dates, K, window=240):

#     gammas = []
#     lambdas = []

#     print(f"\n--- Starting rolling IPCA K={K} ---")

#     for test_idx in range(window, len(R)):

#         # Previous 240 months
#         R_train = R[test_idx-window:test_idx]
#         Z_train = Z[test_idx-window:test_idx]
#         train_dates = dates[test_idx-window:test_idx]

#         # Estimate IPCA
#         Gamma, F = ipca_window(
#             R_train,
#             Z_train,
#             train_dates,
#             K=K
#         )

#         # STEP 2: store Gamma
#         gammas.append(Gamma)

#         # STEP 3: store mean factor from this window
#         lambdas.append(F.mean(axis=0))

#         # Progress
#         window_number = test_idx - window + 1

#         if window_number % 25 == 0 or window_number == 1:
#             print(
#                 f"K={K}: finished window "
#                 f"{window_number}/{len(R)-window}"
#             )

#     return np.array(gammas), np.array(lambdas)


# # Run rolling IPCA

# gammas_ipca1, lambdas_ipca1 = rolling_ipca_save(
#     R, Z, data.index, K=1, window=WINDOW
# )

# gammas_ipca3, lambdas_ipca3 = rolling_ipca_save(
#     R, Z, data.index, K=3, window=WINDOW
# )

# gammas_ipca5, lambdas_ipca5 = rolling_ipca_save(
#     R, Z, data.index, K=5, window=WINDOW
# )


# # SAVE RESULTS
# np.save("Data/gammas_ipca1.npy", gammas_ipca1)
# np.save("Data/gammas_ipca3.npy", gammas_ipca3)
# np.save("Data/gammas_ipca5.npy", gammas_ipca5)

# np.save("Data/lambdas_ipca1.npy", lambdas_ipca1)
# np.save("Data/lambdas_ipca3.npy", lambdas_ipca3)
# np.save("Data/lambdas_ipca5.npy", lambdas_ipca5)


# print("\n========== ROLLING IPCA SAVED ==========")

# print("Gamma IPCA 1:", gammas_ipca1.shape)
# print("Gamma IPCA 3:", gammas_ipca3.shape)
# print("Gamma IPCA 5:", gammas_ipca5.shape)

# print("Lambda IPCA 1:", lambdas_ipca1.shape)
# print("Lambda IPCA 3:", lambdas_ipca3.shape)
# print("Lambda IPCA 5:", lambdas_ipca5.shape)

# print("\nAll rolling IPCA results saved to Data/.")
print("We do NOT need to estimate IPCA again.")

# LOAD SAVED ROLLING IPCA RESULTS
gammas_ipca1 = np.load("Data/gammas_ipca1.npy")
gammas_ipca3 = np.load("Data/gammas_ipca3.npy")
gammas_ipca5 = np.load("Data/gammas_ipca5.npy")

lambdas_ipca1 = np.load("Data/lambdas_ipca1.npy")
lambdas_ipca3 = np.load("Data/lambdas_ipca3.npy")
lambdas_ipca5 = np.load("Data/lambdas_ipca5.npy")

print("\n========== SAVED IPCA RESULTS LOADED ==========")
print("Gamma IPCA 1:", gammas_ipca1.shape)
print("Gamma IPCA 3:", gammas_ipca3.shape)
print("Gamma IPCA 5:", gammas_ipca5.shape)
print("Lambda IPCA 1:", lambdas_ipca1.shape)
print("Lambda IPCA 3:", lambdas_ipca3.shape)
print("Lambda IPCA 5:", lambdas_ipca5.shape)


# step 4.4: IPCA factor realization
# First rolling window only - K=1 test

# First out-of-sample month
test_idx = WINDOW

# Return realized in the new month
r_next = R[test_idx]

# Characteristics available at the beginning of that month
Z_next = Z[test_idx]

# Gamma estimated using the previous 240 months
Gamma = gammas_ipca1[0]

# Predicted IPCA loadings for the new month
B_next = Z_next @ Gamma

# Factor realization in the new month
f_next_ipca1 = (
    np.linalg.inv(B_next.T @ B_next)
    @ B_next.T
    @ r_next
)

print("\n--- STEP 4: First-window IPCA K=1 ---")
print("Evaluation month:", data.index[test_idx])
print("r_next shape:", r_next.shape)
print("Z_next shape:", Z_next.shape)
print("Gamma shape:", Gamma.shape)
print("B_next shape:", B_next.shape)
print("f_next shape:", f_next_ipca1.shape)
print("f_next:", f_next_ipca1)

# First-window IPCA K=3 and K=5 tests

# ---------- K = 3 ----------
Gamma3 = gammas_ipca3[0]

B_next3 = Z_next @ Gamma3

f_next_ipca3 = (
    np.linalg.inv(B_next3.T @ B_next3)
    @ B_next3.T
    @ r_next
)

print("\n--- STEP 4: First-window IPCA K=3 ---")
print("Gamma shape:", Gamma3.shape)
print("B_next shape:", B_next3.shape)
print("f_next shape:", f_next_ipca3.shape)
print("f_next:", f_next_ipca3)


# ---------- K = 5 ----------
Gamma5 = gammas_ipca5[0]

B_next5 = Z_next @ Gamma5

f_next_ipca5 = (
    np.linalg.inv(B_next5.T @ B_next5)
    @ B_next5.T
    @ r_next
)

print("\n--- STEP 4: First-window IPCA K=5 ---")
print("Gamma shape:", Gamma5.shape)
print("B_next shape:", B_next5.shape)
print("f_next shape:", f_next_ipca5.shape)
print("f_next:", f_next_ipca5)

# STEP 4: IPCA factor realizations for all evaluation months

def rolling_ipca_factors(R, Z, gammas, window=240):

    factors_next = []

    for j, test_idx in enumerate(range(window, R.shape[0])):

        # Return in the new evaluation month
        r_next = R[test_idx]

        # Characteristics available at beginning of that month
        Z_next = Z[test_idx]

        # Gamma estimated from the preceding 240-month window
        Gamma = gammas[j]

        # Predicted loadings
        B_next = Z_next @ Gamma

        # Factor realization
        f_next = (
            np.linalg.inv(B_next.T @ B_next)
            @ B_next.T
            @ r_next
        )

        factors_next.append(f_next)

    return np.array(factors_next)


factors_ipca1_oos = rolling_ipca_factors(
    R, Z, gammas_ipca1, window=WINDOW
)

factors_ipca3_oos = rolling_ipca_factors(
    R, Z, gammas_ipca3, window=WINDOW
)

factors_ipca5_oos = rolling_ipca_factors(
    R, Z, gammas_ipca5, window=WINDOW
)


print("\n--- STEP 4: All IPCA factor realizations ---")
print("IPCA 1 factors shape:", factors_ipca1_oos.shape)
print("IPCA 3 factors shape:", factors_ipca3_oos.shape)
print("IPCA 5 factors shape:", factors_ipca5_oos.shape)

print("\nFirst IPCA 1 factor:", factors_ipca1_oos[0])
print("First IPCA 3 factors:", factors_ipca3_oos[0])
print("First IPCA 5 factors:", factors_ipca5_oos[0])

# step 4.5: First-window IPCA K=1 test

# First evaluation month = 199208
test_idx = WINDOW

# Actual return in evaluation month
r_next = R[test_idx]

# Characteristics available for this evaluation return
Z_next = Z[test_idx]

# Gamma estimated from the previous 240 months
Gamma = gammas_ipca1[0]

# Predicted loadings
B_next = Z_next @ Gamma

# Factor realization from Step 4
f_next = factors_ipca1_oos[0]

# Factor mean from Step 3
lambda_hat = lambdas_ipca1[0]


# ----- Total-fit residual eta -----

fitted_total_next = B_next @ f_next

eta_next_ipca1 = (
    r_next - fitted_total_next
)


# ----- Predicted pricing error alpha -----

fitted_pred_next = B_next @ lambda_hat

alpha_next_ipca1 = (
    r_next - fitted_pred_next
)


print("\n--- STEP 5: First-window IPCA K=1 ---")
print("Evaluation month:", data.index[test_idx])

print("\nShapes:")
print("r_next:", r_next.shape)
print("B_next:", B_next.shape)
print("f_next:", f_next.shape)
print("lambda_hat:", lambda_hat.shape)
print("eta:", eta_next_ipca1.shape)
print("alpha:", alpha_next_ipca1.shape)

print("\nFirst 5 actual returns:")
print(r_next[:5])

print("\nFirst 5 total fitted returns:")
print(fitted_total_next[:5])

print("\nFirst 5 eta:")
print(eta_next_ipca1[:5])

print("\nFirst 5 predicted returns:")
print(fitted_pred_next[:5])

print("\nFirst 5 alpha:")
print(alpha_next_ipca1[:5])

# First-window IPCA K=3 and K=5 tests
# ---------- K = 3 ----------

Gamma3 = gammas_ipca3[0]
B_next3 = Z_next @ Gamma3

f_next3 = factors_ipca3_oos[0]
lambda_hat3 = lambdas_ipca3[0]

# Total-fit residual
fitted_total_next3 = B_next3 @ f_next3
eta_next_ipca3 = r_next - fitted_total_next3

# Predicted pricing error
fitted_pred_next3 = B_next3 @ lambda_hat3
alpha_next_ipca3 = r_next - fitted_pred_next3

print("\n--- STEP 5: First-window IPCA K=3 ---")
print("B_next shape:", B_next3.shape)
print("f_next shape:", f_next3.shape)
print("lambda_hat shape:", lambda_hat3.shape)
print("eta shape:", eta_next_ipca3.shape)
print("alpha shape:", alpha_next_ipca3.shape)

print("\nFirst 5 eta:")
print(eta_next_ipca3[:5])

print("\nFirst 5 alpha:")
print(alpha_next_ipca3[:5])


# ---------- K = 5 ----------

Gamma5 = gammas_ipca5[0]
B_next5 = Z_next @ Gamma5

f_next5 = factors_ipca5_oos[0]
lambda_hat5 = lambdas_ipca5[0]

# Total-fit residual
fitted_total_next5 = B_next5 @ f_next5
eta_next_ipca5 = r_next - fitted_total_next5

# Predicted pricing error
fitted_pred_next5 = B_next5 @ lambda_hat5
alpha_next_ipca5 = r_next - fitted_pred_next5

print("\n--- STEP 5: First-window IPCA K=5 ---")
print("B_next shape:", B_next5.shape)
print("f_next shape:", f_next5.shape)
print("lambda_hat shape:", lambda_hat5.shape)
print("eta shape:", eta_next_ipca5.shape)
print("alpha shape:", alpha_next_ipca5.shape)

print("\nFirst 5 eta:")
print(eta_next_ipca5[:5])

print("\nFirst 5 alpha:")
print(alpha_next_ipca5[:5])


# IPCA errors for all evaluation months

def rolling_ipca_errors(R, Z, gammas, lambdas, factors_oos, window=240):

    eta_list = []
    alpha_list = []

    for j, test_idx in enumerate(range(window, R.shape[0])):

        # Actual return in evaluation month
        r_next = R[test_idx]

        # Characteristics for evaluation month
        Z_next = Z[test_idx]

        # Gamma estimated from previous 240 months
        Gamma = gammas[j]

        # Predicted loadings
        B_next = Z_next @ Gamma

        # Factor realization from Step 4
        f_next = factors_oos[j]

        # Mean factor from Step 3
        lambda_hat = lambdas[j]

        # Total-fit residual
        eta_next = r_next - B_next @ f_next

        # Predicted pricing error
        alpha_next = r_next - B_next @ lambda_hat

        eta_list.append(eta_next)
        alpha_list.append(alpha_next)

    return np.array(eta_list), np.array(alpha_list)


# IPCA K=1
eta_ipca1, alpha_ipca1 = rolling_ipca_errors(
    R, Z,
    gammas_ipca1,
    lambdas_ipca1,
    factors_ipca1_oos,
    window=WINDOW
)

# IPCA K=3
eta_ipca3, alpha_ipca3 = rolling_ipca_errors(
    R, Z,
    gammas_ipca3,
    lambdas_ipca3,
    factors_ipca3_oos,
    window=WINDOW
)

# IPCA K=5
eta_ipca5, alpha_ipca5 = rolling_ipca_errors(
    R, Z,
    gammas_ipca5,
    lambdas_ipca5,
    factors_ipca5_oos,
    window=WINDOW
)


print("\n--- STEP 5: All IPCA errors ---")

print("IPCA 1 eta:", eta_ipca1.shape)
print("IPCA 1 alpha:", alpha_ipca1.shape)

print("IPCA 3 eta:", eta_ipca3.shape)
print("IPCA 3 alpha:", alpha_ipca3.shape)

print("IPCA 5 eta:", eta_ipca5.shape)
print("IPCA 5 alpha:", alpha_ipca5.shape)

print("\nFirst 5 eta IPCA 1:")
print(eta_ipca1[0, :5])

print("\nFirst 5 alpha IPCA 1:")
print(alpha_ipca1[0, :5])

print("\nFirst 5 eta IPCA 3:")
print(eta_ipca3[0, :5])

print("\nFirst 5 alpha IPCA 3:")
print(alpha_ipca3[0, :5])

print("\nFirst 5 eta IPCA 5:")
print(eta_ipca5[0, :5])

print("\nFirst 5 alpha IPCA 5:")
print(alpha_ipca5[0, :5])

# stpe 4.6 is already calculated during the implementations
# Average eta and alpha for PCA and IPCA

avg_eta_pca1 = eta_pca1.mean(axis=0)
avg_alpha_pca1 = alpha_pca1.mean(axis=0)

avg_eta_pca3 = eta_pca3.mean(axis=0)
avg_alpha_pca3 = alpha_pca3.mean(axis=0)

avg_eta_ipca1 = eta_ipca1.mean(axis=0)
avg_alpha_ipca1 = alpha_ipca1.mean(axis=0)

avg_eta_ipca3 = eta_ipca3.mean(axis=0)
avg_alpha_ipca3 = alpha_ipca3.mean(axis=0)

avg_eta_ipca5 = eta_ipca5.mean(axis=0)
avg_alpha_ipca5 = alpha_ipca5.mean(axis=0)


print("\n--- Average eta and alpha ---")

print("PCA 1:", avg_eta_pca1.shape, avg_alpha_pca1.shape)
print("PCA 3:", avg_eta_pca3.shape, avg_alpha_pca3.shape)

print("IPCA 1:", avg_eta_ipca1.shape, avg_alpha_ipca1.shape)
print("IPCA 3:", avg_eta_ipca3.shape, avg_alpha_ipca3.shape)
print("IPCA 5:", avg_eta_ipca5.shape, avg_alpha_ipca5.shape)

print("\nFirst 5 average eta IPCA 1:")
print(avg_eta_ipca1[:5])

print("\nFirst 5 average alpha IPCA 1:")
print(avg_alpha_ipca1[:5])

# IPCA out-of-sample R-squared

# IPCA K=1
SSE_total_ipca1 = np.sum(eta_ipca1 ** 2)
SSE_pred_ipca1 = np.sum(alpha_ipca1 ** 2)

r2_total_ipca1 = 1 - SSE_total_ipca1 / SST_oos
r2_pred_ipca1 = 1 - SSE_pred_ipca1 / SST_oos


# IPCA K=3
SSE_total_ipca3 = np.sum(eta_ipca3 ** 2)
SSE_pred_ipca3 = np.sum(alpha_ipca3 ** 2)

r2_total_ipca3 = 1 - SSE_total_ipca3 / SST_oos
r2_pred_ipca3 = 1 - SSE_pred_ipca3 / SST_oos


# IPCA K=5
SSE_total_ipca5 = np.sum(eta_ipca5 ** 2)
SSE_pred_ipca5 = np.sum(alpha_ipca5 ** 2)

r2_total_ipca5 = 1 - SSE_total_ipca5 / SST_oos
r2_pred_ipca5 = 1 - SSE_pred_ipca5 / SST_oos


print("\n--- IPCA out-of-sample R2 ---")

print("IPCA 1 Total R2 (%):", 100 * r2_total_ipca1)
print("IPCA 1 Predictive R2 (%):", 100 * r2_pred_ipca1)

print("IPCA 3 Total R2 (%):", 100 * r2_total_ipca3)
print("IPCA 3 Predictive R2 (%):", 100 * r2_pred_ipca3)

print("IPCA 5 Total R2 (%):", 100 * r2_total_ipca5)
print("IPCA 5 Predictive R2 (%):", 100 * r2_pred_ipca5)


# CAPM rolling evaluation: prepare market factor

# Market excess return over the assignment sample
market = ff3.loc[data.index, 'Mkt-RF'].values

print("\n--- CAPM factor check ---")
print("Market shape:", market.shape)

print("First month:", data.index[0])
print("First market return:", market[0])

print("Last month:", data.index[-1])
print("Last market return:", market[-1])

print("Missing values:", np.isnan(market).sum())


# CAPM: first 240-month regression only

# First 240 months
R_train = R[:WINDOW]

# Market factor over the same 240 months
F_train = market[:WINDOW].reshape(-1, 1)

# Estimate CAPM factor loadings:
# B = (F'F)^(-1) F'R
B_capm = (
    np.linalg.inv(F_train.T @ F_train)
    @ F_train.T
    @ R_train
).T

print("\n--- First-window CAPM regression ---")

print("R_train shape:", R_train.shape)
print("F_train shape:", F_train.shape)
print("B_capm shape:", B_capm.shape)

print("\nFirst 5 CAPM betas:")
print(B_capm[:5, 0])

# CAPM: Step 3 for first window
# Mean market factor over the 240 estimation months

lambda_capm = F_train.mean(axis=0)

print("\n--- First-window CAPM factor mean ---")
print("lambda_capm shape:", lambda_capm.shape)
print("lambda_capm:", lambda_capm)


# CAPM: Step 5 for first evaluation month only

# First evaluation month
test_idx = WINDOW

# Actual portfolio returns in 199208
r_next = R[test_idx]

# Actual market factor in 199208
f_next_capm = np.array([market[test_idx]])

# ----- Total-fit residual eta -----

fitted_total_capm = B_capm @ f_next_capm

eta_next_capm = r_next - fitted_total_capm


# ----- Predicted pricing error alpha -----

fitted_pred_capm = B_capm @ lambda_capm

alpha_next_capm = r_next - fitted_pred_capm


print("\n--- CAPM Step 5: first evaluation month ---")
print("Evaluation month:", data.index[test_idx])
print("Actual market factor:", f_next_capm)

print("\nShapes:")
print("B_capm:", B_capm.shape)
print("eta:", eta_next_capm.shape)
print("alpha:", alpha_next_capm.shape)

print("\nFirst 5 actual returns:")
print(r_next[:5])

print("\nFirst 5 total fitted returns:")
print(fitted_total_capm[:5])

print("\nFirst 5 eta:")
print(eta_next_capm[:5])

print("\nFirst 5 predicted returns:")
print(fitted_pred_capm[:5])

print("\nFirst 5 alpha:")
print(alpha_next_capm[:5])


# CAPM: rolling evaluation for all 386 months

def rolling_capm(R, market, window=240):

    eta_list = []
    alpha_list = []

    for test_idx in range(window, R.shape[0]):

        # Previous 240 months
        R_train = R[test_idx - window:test_idx]

        F_train = market[
            test_idx - window:test_idx
        ].reshape(-1, 1)

        # Estimate CAPM betas
        B = (
            np.linalg.inv(F_train.T @ F_train)
            @ F_train.T
            @ R_train
        ).T

        # Step 3: mean market factor
        lambda_hat = F_train.mean(axis=0)

        # Actual market factor in evaluation month
        f_next = np.array([market[test_idx]])

        # Step 5: total-fit residual
        eta_next = (
            R[test_idx]
            - B @ f_next
        )

        # Step 5: predicted pricing error
        alpha_next = (
            R[test_idx]
            - B @ lambda_hat
        )

        eta_list.append(eta_next)
        alpha_list.append(alpha_next)

    return np.array(eta_list), np.array(alpha_list)


eta_capm, alpha_capm = rolling_capm(
    R, market, window=WINDOW
)


print("\n--- Rolling CAPM ---")
print("eta shape:", eta_capm.shape)
print("alpha shape:", alpha_capm.shape)

print("\nFirst 5 eta from first month:")
print(eta_capm[0, :5])

print("\nFirst 5 alpha from first month:")
print(alpha_capm[0, :5])


# CAPM: averages and out-of-sample R-squared

# Average errors for each of the 25 portfolios
avg_eta_capm = eta_capm.mean(axis=0)
avg_alpha_capm = alpha_capm.mean(axis=0)

# R-squared
SSE_total_capm = np.sum(eta_capm ** 2)
SSE_pred_capm = np.sum(alpha_capm ** 2)

r2_total_capm = 1 - SSE_total_capm / SST_oos
r2_pred_capm = 1 - SSE_pred_capm / SST_oos


print("\n--- CAPM results ---")

print("Average eta shape:", avg_eta_capm.shape)
print("Average alpha shape:", avg_alpha_capm.shape)

print("\nFirst 5 average eta:")
print(avg_eta_capm[:5])

print("\nFirst 5 average alpha:")
print(avg_alpha_capm[:5])

print("\nCAPM Total R2 (%):", 100 * r2_total_capm)
print("CAPM Predictive R2 (%):", 100 * r2_pred_capm)

print("\n--- Available columns ---")
print("ff3 columns:")
print(ff3.columns.tolist())

print("\ndata columns:")
print(data.columns.tolist())


# 3-factor model: prepare factors

factors_3f = data[['Mkt-RF', 'SMB', 'Mom']].values

print("\n--- 3-factor check ---")
print("Factor shape:", factors_3f.shape)

print("\nFirst month:", data.index[0])
print("First factor values [Mkt-RF, SMB, Mom]:")
print(factors_3f[0])

print("\nLast month:", data.index[-1])
print("Last factor values [Mkt-RF, SMB, Mom]:")
print(factors_3f[-1])

print("\nMissing values:", np.isnan(factors_3f).sum())


# 3-factor model: first 240-month regression only

# First 240 months of portfolio returns
R_train = R[:WINDOW]

# First 240 months of the three factors
F_train_3f = factors_3f[:WINDOW]

# Estimate factor loadings:
# B = (F'F)^(-1) F'R
B_3f = (
    np.linalg.inv(F_train_3f.T @ F_train_3f)
    @ F_train_3f.T
    @ R_train
).T



print("\n--- First-window 3-factor regression ---")

print("R_train shape:", R_train.shape)
print("F_train_3f shape:", F_train_3f.shape)
print("B_3f shape:", B_3f.shape)

print("\nFirst portfolio loadings [Mkt-RF, SMB, Mom]:")
print(B_3f[0])


# 3-factor model: Step 3 for first window

lambda_3f = F_train_3f.mean(axis=0)

print("\n--- First-window 3-factor means ---")
print("lambda_3f shape:", lambda_3f.shape)

print("Factor means [Mkt-RF, SMB, Mom]:")
print(lambda_3f)


# 3-factor model: Step 5 for first evaluation month only

test_idx = WINDOW

# Actual portfolio returns in 199208
r_next = R[test_idx]

# Actual three factors in 199208
f_next_3f = factors_3f[test_idx]


# ----- Total-fit residual eta -----

fitted_total_3f = B_3f @ f_next_3f

eta_next_3f = r_next - fitted_total_3f


# ----- Predicted pricing error alpha -----

fitted_pred_3f = B_3f @ lambda_3f

alpha_next_3f = r_next - fitted_pred_3f


print("\n--- 3-factor Step 5: first evaluation month ---")
print("Evaluation month:", data.index[test_idx])

print("\nActual factors [Mkt-RF, SMB, Mom]:")
print(f_next_3f)

print("\nShapes:")
print("B_3f:", B_3f.shape)
print("eta:", eta_next_3f.shape)
print("alpha:", alpha_next_3f.shape)

print("\nFirst 5 actual returns:")
print(r_next[:5])

print("\nFirst 5 total fitted returns:")
print(fitted_total_3f[:5])

print("\nFirst 5 eta:")
print(eta_next_3f[:5])

print("\nFirst 5 predicted returns:")
print(fitted_pred_3f[:5])

print("\nFirst 5 alpha:")
print(alpha_next_3f[:5])


# 3-factor model: rolling evaluation for all 386 months

def rolling_3factor(R, factors_3f, window=240):

    eta_list = []
    alpha_list = []

    for test_idx in range(window, R.shape[0]):

        # Previous 240 months
        R_train = R[test_idx - window:test_idx]

        F_train = factors_3f[
            test_idx - window:test_idx
        ]

        # Estimate 3-factor loadings
        B = (
            np.linalg.inv(F_train.T @ F_train)
            @ F_train.T
            @ R_train
        ).T

        # Step 3: factor means
        lambda_hat = F_train.mean(axis=0)

        # Actual factors in evaluation month
        f_next = factors_3f[test_idx]

        # Step 5: total-fit residual
        eta_next = (
            R[test_idx]
            - B @ f_next
        )

        # Step 5: predicted pricing error
        alpha_next = (
            R[test_idx]
            - B @ lambda_hat
        )

        eta_list.append(eta_next)
        alpha_list.append(alpha_next)

    return np.array(eta_list), np.array(alpha_list)


eta_3f, alpha_3f = rolling_3factor(
    R, factors_3f, window=WINDOW
)


print("\n--- Rolling 3-factor model ---")
print("eta shape:", eta_3f.shape)
print("alpha shape:", alpha_3f.shape)

print("\nFirst 5 eta from first month:")
print(eta_3f[0, :5])

print("\nFirst 5 alpha from first month:")
print(alpha_3f[0, :5])


# 3-factor model: averages and out-of-sample R-squared

# Average errors for each of the 25 portfolios
avg_eta_3f = eta_3f.mean(axis=0)
avg_alpha_3f = alpha_3f.mean(axis=0)

# R-squared
SSE_total_3f = np.sum(eta_3f ** 2)
SSE_pred_3f = np.sum(alpha_3f ** 2)

r2_total_3f = 1 - SSE_total_3f / SST_oos
r2_pred_3f = 1 - SSE_pred_3f / SST_oos


print("\n--- 3-factor results ---")

print("Average eta shape:", avg_eta_3f.shape)
print("Average alpha shape:", avg_alpha_3f.shape)

print("\nFirst 5 average eta:")
print(avg_eta_3f[:5])

print("\nFirst 5 average alpha:")
print(avg_alpha_3f[:5])

print("\n3-factor Total R2 (%):", 100 * r2_total_3f)
print("3-factor Predictive R2 (%):", 100 * r2_pred_3f)


# Q4: Final 25-portfolio error table

q4_table = pd.DataFrame({
    'CAPM eta': avg_eta_capm,
    'CAPM alpha': avg_alpha_capm,

    '3F eta': avg_eta_3f,
    '3F alpha': avg_alpha_3f,

    'PCA1 eta': avg_eta_pca1,
    'PCA1 alpha': avg_alpha_pca1,

    'PCA3 eta': avg_eta_pca3,
    'PCA3 alpha': avg_alpha_pca3,

    'IPCA1 eta': avg_eta_ipca1,
    'IPCA1 alpha': avg_alpha_ipca1,

    'IPCA3 eta': avg_eta_ipca3,
    'IPCA3 alpha': avg_alpha_ipca3,

    'IPCA5 eta': avg_eta_ipca5,
    'IPCA5 alpha': avg_alpha_ipca5,
}, index=names)

# Assignment asks returns/errors in % with two decimals
q4_table = q4_table.round(2)

print("\n--- Q4: Average errors by portfolio ---")
print(q4_table.to_string())


# Q4: Final R-squared summary

q4_r2 = pd.DataFrame({
    'Total R2 (%)': [
        100 * r2_total_capm,
        100 * r2_total_3f,
        100 * r2_total_pca1,
        100 * r2_total_pca3,
        100 * r2_total_ipca1,
        100 * r2_total_ipca3,
        100 * r2_total_ipca5
    ],
    'Predictive R2 (%)': [
        100 * r2_pred_capm,
        100 * r2_pred_3f,
        100 * r2_pred_pca1,
        100 * r2_pred_pca3,
        100 * r2_pred_ipca1,
        100 * r2_pred_ipca3,
        100 * r2_pred_ipca5
    ]
}, index=[
    'CAPM',
    '3-Factor',
    'PCA 1',
    'PCA 3',
    'IPCA 1',
    'IPCA 3',
    'IPCA 5'
])

q4_r2 = q4_r2.round(2)

print("\n--- Q4: Out-of-sample R-squared ---")
print(q4_r2)