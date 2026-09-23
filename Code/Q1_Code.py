import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the prepped data
data = pd.read_csv('Data/merged_clean.csv', index_col='date')

portfolio_cols = [c for c in data.columns if c.endswith('_excess')]
factor_cols = ['Mkt-RF', 'SMB', 'Mom']

# Excess mean return and covariance
mu_e_port = data[portfolio_cols].mean().values        # 25x1
Sigma_port = data[portfolio_cols].cov().values        # 25x25

mu_e_fac = data[factor_cols].mean().values            # 3x1
Sigma_fac = data[factor_cols].cov().values            # 3x3

# Shift to return space with fixed rf
rf_fixed = 0.25  # % per month
mu_port = mu_e_port + rf_fixed
mu_fac = mu_e_fac + rf_fixed

n_port = len(portfolio_cols)
n_fac = len(factor_cols)
ones_port = np.ones(n_port)
ones_fac = np.ones(n_fac)


def efficient_frontier(mu, Sigma, ones, target_max, n_points=100):
    """No-riskless-asset frontier: returns (target_returns, min_std_devs)."""
    Sigma_inv = np.linalg.inv(Sigma)
    A = ones @ Sigma_inv @ ones
    B = ones @ Sigma_inv @ mu
    C = mu @ Sigma_inv @ mu
    D = A * C - B ** 2

    targets = np.linspace(mu.min(), target_max, n_points)
    variances = (A * targets ** 2 - 2 * B * targets + C) / D
    stds = np.sqrt(np.maximum(variances, 0))
    return targets, stds


def tangency_portfolio(mu_excess, Sigma):
    """Tangency portfolio weights, using EXCESS returns (mu_excess)."""
    Sigma_inv = np.linalg.inv(Sigma)
    w = Sigma_inv @ mu_excess
    w = w / (np.ones(len(mu_excess)) @ Sigma_inv @ mu_excess)
    exp_excess_return = w @ mu_excess
    variance = w @ Sigma @ w
    std = np.sqrt(variance)
    sharpe = exp_excess_return / std
    return w, exp_excess_return, std, sharpe


w_tan_port, exret_tan_port, std_tan_port, sharpe_tan_port = tangency_portfolio(mu_e_port, Sigma_port)
w_tan_fac, exret_tan_fac, std_tan_fac, sharpe_tan_fac = tangency_portfolio(mu_e_fac, Sigma_fac)

mean_tan_port = rf_fixed + exret_tan_port
mean_tan_fac = rf_fixed + exret_tan_fac

# Global minimum-variance portfolio, 25 portfolios
Sigma_inv_port = np.linalg.inv(Sigma_port)
w_gmv = Sigma_inv_port @ ones_port
w_gmv = w_gmv / (ones_port @ Sigma_inv_port @ ones_port)
mean_gmv = rf_fixed + w_gmv @ mu_e_port
std_gmv = np.sqrt(w_gmv @ Sigma_port @ w_gmv)
sharpe_gmv = (w_gmv @ mu_e_port) / std_gmv

# No-riskless frontier, 25 portfolios — now mean_tan_port exists
targets_port, stds_port = efficient_frontier(mu_port, Sigma_port, ones_port, target_max=mean_tan_port * 1.6)

# Capital allocation lines (with riskless asset)
cal_std_range = np.linspace(0, max(std_tan_port, std_tan_fac) * 1.3, 50)
cal_port = rf_fixed + sharpe_tan_port * cal_std_range
cal_fac = rf_fixed + sharpe_tan_fac * cal_std_range

# Plot
plt.figure(figsize=(9, 6))
plt.plot(stds_port, targets_port, label='Efficient frontier (25 portfolios, no riskless)', color='steelblue')
plt.plot(cal_std_range, cal_port, label='CAL: riskless + 25 portfolios (tangency)', color='steelblue', linestyle='--')
plt.plot(cal_std_range, cal_fac, label='CAL: riskless + Mkt, SMB, Mom (tangency)', color='darkorange', linestyle='--')

plt.scatter([std_tan_port], [mean_tan_port], color='steelblue', zorder=5, s=60, marker='*')
plt.scatter([std_tan_fac], [mean_tan_fac], color='darkorange', zorder=5, s=60, marker='*')
plt.scatter([std_gmv], [mean_gmv], color='green', zorder=5, s=60, marker='o', label='GMV (25 portfolios)')

plt.xlabel('Volatility (std dev, % per month)')
plt.ylabel('Mean return (% per month)')
plt.title('Efficient Frontiers: 25 Size-Momentum Portfolios vs. 3 Factor Model')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Code/q1_frontier.png', dpi=150)
plt.show()

# Tangency portfolio comparison table
print("\n Tangency Portfolio Comparison")
print(f"{'':30s}{'25 Portfolios':>18s}{'3 Factors':>18s}")
print(f"{'Expected return (%)':30s}{mean_tan_port:18.2f}{mean_tan_fac:18.2f}")
print(f"{'Volatility (%)':30s}{std_tan_port:18.2f}{std_tan_fac:18.2f}")
print(f"{'Sharpe ratio':30s}{sharpe_tan_port:18.3f}{sharpe_tan_fac:18.3f}")

print("\n GMV Portfolio (25 portfolios):")
print(f"Mean return (%): {mean_gmv:.2f}")
print(f"Volatility (%): {std_gmv:.2f}")
print(f"Sharpe ratio: {sharpe_gmv:.3f}")

print("\n Tangency Weight Ranges:")
print(f"25-portfolio tangency weights: min {w_tan_port.min():.2f}, max {w_tan_port.max():.2f}")
print(f"3-factor tangency weights: min {w_tan_fac.min():.2f}, max {w_tan_fac.max():.2f}")