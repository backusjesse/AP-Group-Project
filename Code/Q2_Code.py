import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import f

data = pd.read_csv('Data/merged_clean.csv', index_col='date')
portfolio_cols = [c for c in data.columns if c.endswith('_excess')]
print(portfolio_cols)

# Select the first portfolio
portfolio = portfolio_cols[0]
y = data[portfolio]

print("\nSelected portfolio:")
print(portfolio)
print("\nFirst five excess returns:")
print(y.head())

X = data[['Mkt-RF']]

print("\nMarket excess returns:")
print(X.head())

X = sm.add_constant(X)

print("\nRegression X matrix:")
print(X.head())

model = sm.OLS(y, X).fit()
print(model.summary())

# CAPM fitted values
fitted = model.fittedvalues

# CAPM residuals
residuals = model.resid

# Compare actual, predicted and residual returns
check = pd.DataFrame({
    'Actual': y,
    'Predicted': fitted,
    'Residual': residuals
})

print("\nActual vs predicted returns:")
print(check.head())

# Store results for all 25 CAPM regressions
capm_results = []

for portfolio in portfolio_cols:

    # Dependent variable: portfolio excess return
    y_i = data[portfolio]

    # Estimate CAPM regression
    model_i = sm.OLS(y_i, X).fit()

    # Save alpha and its standard error
    capm_results.append({
        'Portfolio': portfolio,
        'Alpha': model_i.params['const'],
        'Alpha_SE': model_i.bse['const']
    })

# Convert results to a DataFrame
capm_table = pd.DataFrame(capm_results)

print("\nCAPM alphas and standard errors:")
print(capm_table)

# Three-factor explanatory variables
X_3f = data[['Mkt-RF', 'SMB', 'Mom']]

# Add intercept for alpha
X_3f = sm.add_constant(X_3f)

print("\nThree-factor X matrix:")
print(X_3f.head())

# Estimate three-factor model for SMALL LoPRIOR
model_3f = sm.OLS(y, X_3f).fit()

print(model_3f.summary())

# Store results for all 25 three-factor regressions
three_factor_results = []

for portfolio in portfolio_cols:

    # Dependent variable
    y_i = data[portfolio]

    # Estimate three-factor regression
    model_i = sm.OLS(y_i, X_3f).fit()

    # Save alpha and its standard error
    three_factor_results.append({
        'Portfolio': portfolio,
        'Alpha': model_i.params['const'],
        'Alpha_SE': model_i.bse['const']
    })

# Convert to DataFrame
three_factor_table = pd.DataFrame(three_factor_results)

print("\nThree-factor alphas and standard errors:")
print(three_factor_table)

comparison = pd.DataFrame({
    'Portfolio': portfolio_cols,
    'CAPM Alpha': capm_table['Alpha'],
    'CAPM SE': capm_table['Alpha_SE'],
    '3F Alpha': three_factor_table['Alpha'],
    '3F SE': three_factor_table['Alpha_SE']
})

print("\nCAPM vs Three-Factor Model:")
print(comparison.to_string(index=False))

# Store CAPM residuals for all 25 portfolios
capm_residuals = pd.DataFrame(index=data.index)

for portfolio in portfolio_cols:

    y_i = data[portfolio]

    model_i = sm.OLS(y_i, X).fit()

    capm_residuals[portfolio] = model_i.resid


print("\nCAPM residual matrix:")
print(capm_residuals.head())

print("\nShape:")
print(capm_residuals.shape)

# CAPM residual covariance matrix
Sigma_e_capm = capm_residuals.cov()

print("\nCAPM residual covariance matrix:")
print(Sigma_e_capm)

print("\nShape:")
print(Sigma_e_capm.shape)

# CAPM alpha vector
alpha_capm = capm_table['Alpha'].values

print("\nCAPM alpha vector:")
print(alpha_capm)

print("\nShape:")
print(alpha_capm.shape)

# Dimensions
T = len(data)
N = len(portfolio_cols)

# Residual matrix: T x N
E_capm = capm_residuals.values

# Residual covariance matrix for GRS
Sigma_e_capm_grs = (E_capm.T @ E_capm) / T

print("T =", T)
print("N =", N)
print("Shape of Sigma_e:", Sigma_e_capm_grs.shape)

alpha_term_capm = (
    alpha_capm
    @ np.linalg.inv(Sigma_e_capm_grs)
    @ alpha_capm
)

print("\nCAPM alpha quadratic form:")
print(alpha_term_capm)

# CAPM factor
F_capm = data[['Mkt-RF']].values

# Mean factor return
mean_f_capm = F_capm.mean(axis=0)

print("\nMean CAPM factor return:")
print(mean_f_capm)

print("Shape:")
print(mean_f_capm.shape)

# Demean the CAPM factor
F_capm_demeaned = F_capm - mean_f_capm

# Factor covariance matrix using denominator T
Sigma_f_capm = (F_capm_demeaned.T @ F_capm_demeaned) / T

print("\nCAPM factor covariance:")
print(Sigma_f_capm)

print("Shape:")
print(Sigma_f_capm.shape)

market_std = np.sqrt(Sigma_f_capm[0, 0])

print("\nMarket standard deviation:")
print(market_std)

# Market Sharpe ratio
sharpe_market = mean_f_capm[0] / market_std

print("\nMarket Sharpe ratio:")
print(sharpe_market)

# Factor component of the CAPM GRS statistic
factor_term_capm = (
    1
    + mean_f_capm
    @ np.linalg.inv(Sigma_f_capm)
    @ mean_f_capm
)

print("\nCAPM factor term:")
print(factor_term_capm)

print("\nCheck using 1 + Sharpe ratio squared:")
print(1 + sharpe_market**2)

K_capm = 1
scaling_capm = (T - N - K_capm) / N

grs_capm = (
    scaling_capm
    * alpha_term_capm
    / factor_term_capm
)

print("Scaling term:", scaling_capm)
print("CAPM GRS statistic:", grs_capm)

# Degrees of freedom
df1_capm = N
df2_capm = T - N - K_capm

# GRS p-value
pvalue_capm = f.sf(grs_capm, df1_capm, df2_capm)

print("\nCAPM GRS results:")
print("GRS statistic:", grs_capm)
print("Degrees of freedom:", df1_capm, df2_capm)
print("p-value:", pvalue_capm)

# Repeat for all three factors now 
# Store three-factor residuals for all 25 portfolios
three_factor_residuals = pd.DataFrame(index=data.index)

for portfolio in portfolio_cols:

    y_i = data[portfolio]

    model_i = sm.OLS(y_i, X_3f).fit()

    three_factor_residuals[portfolio] = model_i.resid


print("\nThree-factor residual matrix:")
print(three_factor_residuals.head())

print("\nShape:")
print(three_factor_residuals.shape)

# Three-factor residual matrix
E_3f = three_factor_residuals.values

# Residual covariance matrix for GRS
Sigma_e_3f_grs = (E_3f.T @ E_3f) / T

print("\nThree-factor residual covariance matrix shape:")
print(Sigma_e_3f_grs.shape)

alpha_3f = three_factor_table['Alpha'].values

print("\nThree-factor alpha vector:")
print(alpha_3f)

print("\nShape:")
print(alpha_3f.shape)

alpha_term_3f = (
    alpha_3f
    @ np.linalg.inv(Sigma_e_3f_grs)
    @ alpha_3f
)

print("\nThree-factor alpha quadratic form:")
print(alpha_term_3f)

# Factor Matrix
F_3f = data[['Mkt-RF', 'SMB', 'Mom']].values

K_3f = 3

print("\nThree-factor matrix shape:")
print(F_3f.shape)

# Three factor means
mean_f_3f = F_3f.mean(axis=0)

print("\nThree-factor mean returns:")
print(mean_f_3f)

# Factor covariance matrix
F_3f_demeaned = F_3f - mean_f_3f

Sigma_f_3f = (
    F_3f_demeaned.T @ F_3f_demeaned
) / T

print("\nThree-factor covariance matrix:")
print(Sigma_f_3f)

print("\nShape:")
print(Sigma_f_3f.shape)

# Squared maximum Sharpe ratio of the three factors
sharpe_3f_squared = (
    mean_f_3f
    @ np.linalg.inv(Sigma_f_3f)
    @ mean_f_3f
)

# Maximum Sharpe ratio
sharpe_3f = np.sqrt(sharpe_3f_squared)

print("\nThree-factor squared Sharpe ratio:")
print(sharpe_3f_squared)

print("\nThree-factor efficient portfolio Sharpe ratio:")
print(sharpe_3f)

factor_term_3f = (
    1
    + mean_f_3f
    @ np.linalg.inv(Sigma_f_3f)
    @ mean_f_3f
)

print("\nThree-factor GRS factor term:")
print(factor_term_3f)

print("\nCheck using 1 + Sharpe ratio squared:")
print(1 + sharpe_3f**2)

# scaling them
scaling_3f = (T - N - K_3f) / N

print("\nThree-factor GRS scaling term:")
print(scaling_3f)

# GRS statistics 
grs_3f = (
    scaling_3f
    * alpha_term_3f
    / factor_term_3f
)

print("\nThree-factor GRS statistic:")
print(grs_3f)

# p-value
df1_3f = N
df2_3f = T - N - K_3f

pvalue_3f = f.sf(grs_3f, df1_3f, df2_3f)

print("\nThree-factor GRS results:")
print("GRS statistic:", grs_3f)
print("Degrees of freedom:", df1_3f, df2_3f)
print("p-value:", pvalue_3f)

# Store CAPM market betas
beta_capm = []

for portfolio in portfolio_cols:

    y_i = data[portfolio]
    model_i = sm.OLS(y_i, X).fit()

    beta_capm.append(model_i.params['Mkt-RF'])

beta_capm = np.array(beta_capm)

print("\nCAPM betas:")
print(beta_capm)

print("\nShape:")
print(beta_capm.shape)

# Actual excess returns: T x N
R = data[portfolio_cols].values

print("Return matrix shape:", R.shape)

# Market factor: T x 1
F_capm = data[['Mkt-RF']].values

# CAPM fitted factor component: T x N
Rhat_capm = F_capm @ beta_capm.reshape(1, -1)

print("CAPM fitted return matrix shape:", Rhat_capm.shape)

print("\nFirst 5 fitted returns for SMALL LoPRIOR:")
print(Rhat_capm[:5, 0])

# CAPM total-fit errors
total_errors_capm = R - Rhat_capm

print("\nCAPM total-fit error matrix shape:")
print(total_errors_capm.shape)

print("\nFirst 5 total-fit errors for SMALL LoPRIOR:")
print(total_errors_capm[:5, 0])

# Sum of squared CAPM total-fit errors
SSE_total_capm = np.sum(total_errors_capm ** 2)

# Sum of squared actual excess returns
SST_total = np.sum(R ** 2)

print("\nCAPM total-fit SSE:")
print(SSE_total_capm)

print("\nTotal return sum of squares:")
print(SST_total)

R2_total_capm = 1 - SSE_total_capm / SST_total

print("\nCAPM Total R-squared:")
print(R2_total_capm)

print("CAPM Total R-squared (%):")
print(R2_total_capm * 100)

# Store three-factor betas
beta_3f = []

for portfolio in portfolio_cols:

    y_i = data[portfolio]
    model_i = sm.OLS(y_i, X_3f).fit()

    beta_3f.append([
        model_i.params['Mkt-RF'],
        model_i.params['SMB'],
        model_i.params['Mom']
    ])

beta_3f = np.array(beta_3f)

print("\nThree-factor beta matrix:")
print(beta_3f)

print("\nShape:")
print(beta_3f.shape)

# Three-factor fitted returns
Rhat_3f = F_3f @ beta_3f.T

print("\nThree-factor fitted return matrix shape:")
print(Rhat_3f.shape)

print("\nFirst 5 fitted returns for SMALL LoPRIOR:")
print(Rhat_3f[:5, 0])

# Three-factor total-fit errors
total_errors_3f = R - Rhat_3f

print("\nThree-factor total-fit error matrix shape:")
print(total_errors_3f.shape)

print("\nFirst 5 total-fit errors for SMALL LoPRIOR:")
print(total_errors_3f[:5, 0])

# Sum of squared three-factor total-fit errors
SSE_total_3f = np.sum(total_errors_3f ** 2)

print("\nThree-factor total-fit SSE:")
print(SSE_total_3f)

print("\nTotal return sum of squares:")
print(SST_total)

R2_total_3f = 1 - SSE_total_3f / SST_total

print("\nThree-factor Total R-squared:")
print(R2_total_3f)

print("Three-factor Total R-squared (%):")
print(R2_total_3f * 100)

# CAPM predicted expected returns for each portfolio
expected_capm = beta_capm * mean_f_capm[0]

print("\nCAPM predicted expected returns:")
print(expected_capm)

print("\nShape:")
print(expected_capm.shape)

# CAPM predictive errors
predictive_errors_capm = R - expected_capm

print("\nCAPM predictive error matrix shape:")
print(predictive_errors_capm.shape)

print("\nFirst 5 predictive errors for SMALL LoPRIOR:")
print(predictive_errors_capm[:5, 0])

# CAPM predictive sum of squared errors
SSE_pred_capm = np.sum(predictive_errors_capm ** 2)

print("\nCAPM predictive SSE:")
print(SSE_pred_capm)

print("\nTotal return sum of squares:")
print(SST_total)

R2_pred_capm = 1 - SSE_pred_capm / SST_total

print("\nCAPM Predictive R-squared:")
print(R2_pred_capm)

print("CAPM Predictive R-squared (%):")
print(R2_pred_capm * 100)

# Three-factor predicted expected returns
expected_3f = beta_3f @ mean_f_3f

print("\nThree-factor predicted expected returns:")
print(expected_3f)

print("\nShape:")
print(expected_3f.shape)

# Three-factor predictive errors
predictive_errors_3f = R - expected_3f

print("\nThree-factor predictive error matrix shape:")
print(predictive_errors_3f.shape)

print("\nFirst 5 predictive errors for SMALL LoPRIOR:")
print(predictive_errors_3f[:5, 0])

# Three-factor predictive sum of squared errors
SSE_pred_3f = np.sum(predictive_errors_3f ** 2)

print("\nThree-factor predictive SSE:")
print(SSE_pred_3f)

print("\nTotal return sum of squares:")
print(SST_total)

R2_pred_3f = 1 - SSE_pred_3f / SST_total

print("\nThree-factor Predictive R-squared:")

print(R2_pred_3f)

print("Three-factor Predictive R-squared (%):")

print(R2_pred_3f * 100)

print("\n" + "=" * 60)
print("QUESTION 2: MODEL COMPARISON")
print("=" * 60)

print(f"{'Measure':<25}{'CAPM':>15}{'3-Factor':>15}")
print("-" * 55)

print(f"{'GRS statistic':<25}{grs_capm:>15.3f}{grs_3f:>15.3f}")
print(f"{'GRS p-value':<25}{pvalue_capm:>15.3e}{pvalue_3f:>15.3e}")
print(f"{'Total R-squared':<25}{R2_total_capm:>15.4f}{R2_total_3f:>15.4f}")
print(f"{'Predictive R-squared':<25}{R2_pred_capm:>15.4f}{R2_pred_3f:>15.4f}")
print(f"{'Sharpe ratio':<25}{sharpe_market:>15.4f}{sharpe_3f:>15.4f}")