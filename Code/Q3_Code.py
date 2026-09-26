import pandas as pd
import numpy as np
from io import StringIO
from ipca import InstrumentedPCA

# Question 3: PCA and IPCA
# Run Data_Prep.py first (this script needs Data/merged_clean.csv)

# Part 1: Characteristics

# Timing: a characteristic in month t is known at the start of the month,
# the return in month t is known at the end of the month.
# So characteristics we build ourselves may only use returns from earlier months.

def load_block(filepath, title=None):
    """Reads one table from a Ken French file (the first table if no title is given)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the title of the table we want
    search_from = 0
    if title is not None:
        search_from = next(i for i, line in enumerate(lines) if title in line)

    # The header row starts with a comma
    start_idx = next(i for i in range(search_from, len(lines)) if lines[i].startswith(','))

    # The table ends at the next blank line
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

    # Missing values are coded as -99.99 or -999
    return df.mask(df <= -99.99)


def rank_transform(df):
    """Ranks the 25 portfolios each month and scales the ranks to [-0.5, 0.5]."""
    ranks = df.rank(axis=1)
    n = df.notna().sum(axis=1)
    return ranks.sub(1).div(n - 1, axis=0) - 0.5


PORT_FILE = 'Data/portfolios_25_size_mom.csv'

# Load all data from 1927, so the rolling windows are full by 1972
returns = load_block(PORT_FILE)
market_cap = load_block(PORT_FILE, 'Average Market Cap')
prior_vw = load_block(PORT_FILE, 'Value-Weighted Average of Prior Returns')
ff3 = load_block('Data/factor_ff3.csv')[['Mkt-RF', 'RF']]

names = returns.columns.tolist()
ff3 = ff3.reindex(returns.index)
excess_full = returns.sub(ff3['RF'], axis=0)

# Characteristics from the French file
chars = {}
chars['size'] = np.log(market_cap)          # log of average market cap
chars['mom'] = prior_vw                     # average past return (months t-12 to t-2)

# Characteristics we build from past returns
chars['st_rev'] = returns.shift(1)                          # last month's return
chars['lt_rev'] = returns.shift(13).rolling(48).sum()       # sum of returns, months t-60 to t-13
chars['vol'] = returns.shift(1).rolling(12).std()           # volatility over the last 12 months

# Market beta over the last 36 months
past_ex = excess_full.shift(1)
past_mkt = ff3['Mkt-RF'].shift(1)
beta = pd.DataFrame(index=returns.index, columns=names, dtype=float)
for c in names:
    beta[c] = past_ex[c].rolling(36).cov(past_mkt) / past_mkt.rolling(36).var()
chars['beta'] = beta

# Keep our sample period and rank each characteristic
for name in chars:
    chars[name] = rank_transform(chars[name].loc[197208:202409, names])


# Part 2: PCA and IPCA

# Load excess returns
data = pd.read_csv('Data/merged_clean.csv', index_col='date')
portfolio_cols = [c for c in data.columns if c.endswith('_excess')]
names = [c.replace('_excess', '') for c in portfolio_cols]

R = data[portfolio_cols].values              # returns: months x portfolios
T, N = R.shape

# Characteristics: months x portfolios x characteristics, plus a constant
char_cols = list(chars.keys()) + ['const']
L = len(char_cols)

Z = np.stack(
    [chars[c].loc[data.index, names].values for c in chars] + [np.ones((T, N))],
    axis=2
)

# Check that no two characteristics are almost the same
print("Correlations between characteristics:")
print(pd.DataFrame(Z[:, :, :-1].reshape(T * N, L - 1), columns=char_cols[:-1]).corr().round(2))

print("T =", T, " N =", N, " L =", L)
print("Characteristics:", char_cols)


def pca(R, K):
    """PCA on the returns. Returns loadings B (N x K) and factors F (T x K)."""
    second_moment = R.T @ R / R.shape[0]
    eigvals, eigvecs = np.linalg.eigh(second_moment)
    order = np.argsort(eigvals)[::-1]
    B = eigvecs[:, order[:K]]

    # Flip signs so each factor has a positive mean
    F = R @ B
    signs = np.where(F.mean(axis=0) < 0, -1.0, 1.0)
    return B * signs, F * signs


# The ipca package needs the data in long format with number ids for the portfolios
ret_long = pd.DataFrame(R, index=data.index, columns=range(N)).stack()
ret_long.index.names = ['date', 'portfolio']
ret_long = ret_long.swaplevel().sort_index()

char_long = pd.DataFrame(
    Z.reshape(T * N, L),
    index=pd.MultiIndex.from_product([data.index, range(N)], names=['date', 'portfolio']),
    columns=char_cols
).swaplevel().sort_index()


def ipca(K):
    """IPCA with Gamma_alpha = 0. Returns Gamma (L x K) and factors F (T x K)."""
    model = InstrumentedPCA(n_factors=K, intercept=False, iter_tol=1e-8)
    model = model.fit(X=char_long, y=ret_long, data_type='panel')
    Gamma, factors = model.get_factors()
    return Gamma, factors.T


def evaluate(R, fitted_total, fitted_pred):
    """Total R2, predictive R2 and pricing errors, as in Kelly et al. (2019)."""
    # We compute R2 ourselves: the package's own R2 uses a different formula
    SST = np.sum(R ** 2)
    r2_total = 1 - np.sum((R - fitted_total) ** 2) / SST
    r2_pred = 1 - np.sum((R - fitted_pred) ** 2) / SST
    alpha = (R - fitted_pred).mean(axis=0)      # average pricing error per portfolio
    return alpha, r2_total, r2_pred


alphas = {}
r2 = {}
gammas = {}

# PCA with 1 and 3 factors
for K in [1, 3]:
    B, F = pca(R, K)
    lam = F.mean(axis=0)                         # factor risk premia
    fitted_total = F @ B.T
    fitted_pred = np.tile(B @ lam, (T, 1))
    alpha, r2_total, r2_pred = evaluate(R, fitted_total, fitted_pred)

    alphas[f'PCA {K}'] = alpha
    r2[f'PCA {K}'] = (r2_total, r2_pred)

# IPCA with 1, 3 and 5 factors
for K in [1, 3, 5]:
    Gamma, F = ipca(K)
    lam = F.mean(axis=0)                         # factor risk premia
    betas = np.einsum('tnl,lk->tnk', Z, Gamma)  # loadings: characteristics x Gamma
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

# Which characteristics matter most for the 1-factor IPCA
print("\nGamma, IPCA 1 factor:")
print(gammas['IPCA 1'].round(3).to_string())

# Save the results for the report
results = pd.concat([alpha_table, summary])
results.round(2).to_csv('Code/q3_results.csv')
print("\nSaved results to Code/q3_results.csv")