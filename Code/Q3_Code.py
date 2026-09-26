import pandas as pd
import numpy as np
from io import StringIO

# Question 3: characteristics, PCA and IPCA
# Requires Data/merged_clean.csv, so run Data_Prep.py first.

# Part 1: characteristics for IPCA

# Timing convention (assignment Section 4):
# a characteristic in row t is known at the START of month t,
# the return in row t is realised at the END of month t.
# So characteristics from the French file need no lag, but characteristics
# we build from returns may only use returns from rows t-1 and earlier.


def load_block(filepath, title=None):
    """
    Reads one block from a Ken French CSV.
    title=None returns the first block (value-weighted monthly returns).
    Otherwise returns the block that follows the line containing `title`.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Start searching at the block title (or at the top of the file)
    search_from = 0
    if title is not None:
        search_from = next(i for i, line in enumerate(lines) if title in line)

    # Header row: first line starting with a comma after the title
    start_idx = next(i for i in range(search_from, len(lines)) if lines[i].startswith(','))

    # Block ends at the first blank line after the header
    end_idx = next(
        (i for i in range(start_idx + 1, len(lines)) if lines[i].strip() == ''),
        len(lines)
    )

    block = ''.join(lines[start_idx:end_idx])
    df = pd.read_csv(StringIO(block))

    df = df.rename(columns={df.columns[0]: 'date'})
    df.columns = [c.strip() for c in df.columns]
    df['date'] = df['date'].astype(str).str.strip()

    # Keep only monthly rows (YYYYMM)
    df = df[df['date'].str.match(r'^\d{6}$')]
    df['date'] = df['date'].astype(int)
    df = df.set_index('date').astype(float)

    # Missing data are coded as -99.99 or -999
    return df.mask(df <= -99.99)


def rank_transform(df):
    """Cross-sectional rank per month, mapped to [-0.5, 0.5] (Kelly et al., 2019)."""
    ranks = df.rank(axis=1)
    n = df.notna().sum(axis=1)
    return ranks.sub(1).div(n - 1, axis=0) - 0.5


PORT_FILE = 'Data/portfolios_25_size_mom.csv'

# Load the full history (1927 onwards) so rolling windows are complete in 1972
returns = load_block(PORT_FILE)
market_cap = load_block(PORT_FILE, 'Average Market Cap')
prior_vw = load_block(PORT_FILE, 'Value-Weighted Average of Prior Returns')
ff3 = load_block('Data/factor_ff3.csv')[['Mkt-RF', 'RF']]

names = returns.columns.tolist()
ff3 = ff3.reindex(returns.index)
excess_full = returns.sub(ff3['RF'], axis=0)

# Characteristics from the French file (already known at the start of month t)
chars = {}
chars['size'] = np.log(market_cap)          # log average market cap
chars['mom'] = prior_vw                     # value-weighted prior (2-12) return

# Characteristics built from past portfolio returns (rows t-1 and earlier only)
chars['st_rev'] = returns.shift(1)                          # last month's return
chars['lt_rev'] = returns.shift(13).rolling(48).sum()       # sum of returns t-60 to t-13
chars['vol'] = returns.shift(1).rolling(12).std()           # volatility, past 12 months

# Rolling market beta, past 36 months of excess returns
past_ex = excess_full.shift(1)
past_mkt = ff3['Mkt-RF'].shift(1)
beta = pd.DataFrame(index=returns.index, columns=names, dtype=float)
for c in names:
    beta[c] = past_ex[c].rolling(36).cov(past_mkt) / past_mkt.rolling(36).var()
chars['beta'] = beta

# Trim to the sample period and rank-transform each characteristic
for name in chars:
    chars[name] = rank_transform(chars[name].loc[197208:202409, names])


# Part 2: PCA and IPCA factors (in-sample)

# Load excess returns
data = pd.read_csv('Data/merged_clean.csv', index_col='date')
portfolio_cols = [c for c in data.columns if c.endswith('_excess')]
names = [c.replace('_excess', '') for c in portfolio_cols]

R = data[portfolio_cols].values              # T x N excess returns (%)
T, N = R.shape

# Characteristics in a T x N x L array, same date and portfolio order as R
char_cols = list(chars.keys()) + ['const']
L = len(char_cols)

Z = np.stack(
    [chars[c].loc[data.index, names].values for c in chars] + [np.ones((T, N))],
    axis=2
)

# Check that no two characteristics are (close to) perfectly correlated
print("Pooled correlations between characteristics:")
print(pd.DataFrame(Z[:, :, :-1].reshape(T * N, L - 1), columns=char_cols[:-1]).corr().round(2))

print("T =", T, " N =", N, " L =", L)
print("Characteristics:", char_cols)


# PCA

def pca(R, K):
    """
    PCA on the uncentered second-moment matrix R'R/T.
    This minimises the same objective as IPCA, sum of (r - B f)^2.
    Returns orthonormal loadings B (N x K) and factors F (T x K).
    """
    second_moment = R.T @ R / R.shape[0]
    eigvals, eigvecs = np.linalg.eigh(second_moment)
    order = np.argsort(eigvals)[::-1]
    B = eigvecs[:, order[:K]]

    # Sign convention: factors with a positive mean
    F = R @ B
    signs = np.where(F.mean(axis=0) < 0, -1.0, 1.0)
    return B * signs, F * signs


# IPCA with Gamma_alpha = 0, using Seth Pruitt's ipca package

from ipca import InstrumentedPCA

# The package needs a long panel indexed by (portfolio id, date),
# with integer portfolio ids
ret_long = pd.DataFrame(R, index=data.index, columns=range(N)).stack()
ret_long.index.names = ['date', 'portfolio']
ret_long = ret_long.swaplevel().sort_index()

char_long = pd.DataFrame(
    Z.reshape(T * N, L),
    index=pd.MultiIndex.from_product([data.index, range(N)], names=['date', 'portfolio']),
    columns=char_cols
).swaplevel().sort_index()


def ipca(K):
    """Returns Gamma_beta (L x K) and factors F (T x K)."""
    model = InstrumentedPCA(n_factors=K, intercept=False, iter_tol=1e-8)
    model = model.fit(X=char_long, y=ret_long, data_type='panel')
    Gamma, factors = model.get_factors()
    return Gamma, factors.T


# Evaluation: total R2, predictive R2 and pricing errors (Kelly et al., eq. 15-16)

def evaluate(R, fitted_total, fitted_pred):
    SST = np.sum(R ** 2)
    r2_total = 1 - np.sum((R - fitted_total) ** 2) / SST
    r2_pred = 1 - np.sum((R - fitted_pred) ** 2) / SST
    alpha = (R - fitted_pred).mean(axis=0)              # average of r_it - beta_it' lambda
    return alpha, r2_total, r2_pred


alphas = {}
r2 = {}
gammas = {}

# PCA with 1 and 3 factors
for K in [1, 3]:
    B, F = pca(R, K)
    lam = F.mean(axis=0)
    fitted_total = F @ B.T
    fitted_pred = np.tile(B @ lam, (T, 1))
    alpha, r2_total, r2_pred = evaluate(R, fitted_total, fitted_pred)

    alphas[f'PCA {K}'] = alpha
    r2[f'PCA {K}'] = (r2_total, r2_pred)

# IPCA with 1, 3 and 5 factors
for K in [1, 3, 5]:
    Gamma, F = ipca(K)
    lam = F.mean(axis=0)
    betas = np.einsum('tnl,lk->tnk', Z, Gamma)          # beta_it = z_it' Gamma
    fitted_total = np.einsum('tnk,tk->tn', betas, F)
    fitted_pred = np.einsum('tnk,k->tn', betas, lam)
    alpha, r2_total, r2_pred = evaluate(R, fitted_total, fitted_pred)

    alphas[f'IPCA {K}'] = alpha
    r2[f'IPCA {K}'] = (r2_total, r2_pred)
    gammas[f'IPCA {K}'] = pd.DataFrame(Gamma, index=char_cols,
                                       columns=[f'f{k + 1}' for k in range(K)])


# Results table
alpha_table = pd.DataFrame(alphas, index=names)

summary = pd.DataFrame({
    model: {
        'Total R2 (%)': 100 * r2[model][0],
        'Predictive R2 (%)': 100 * r2[model][1],
        'Mean |alpha| (%)': np.abs(alpha_table[model]).mean(),
    }
    for model in alpha_table.columns
})

print("\nPricing errors alpha_i (% per month):")
print(alpha_table.round(2).to_string())

print("\nModel comparison:")
print(summary.round(2).to_string())

# Gamma for the 1-factor IPCA: which characteristics drive the loading
print("\nGamma_beta, IPCA 1 factor:")
print(gammas['IPCA 1'].round(3).to_string())

# Save for the report
results = pd.concat([alpha_table, summary])
results.round(2).to_csv('Code/q3_results.csv')
print("\nSaved results to Code/q3_results.csv")