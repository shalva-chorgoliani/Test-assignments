import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#%% Getting the data

# QQQ ETF used as a NASDAQ100 proxy
nq = yf.download(
    'QQQ', 
    start="2019-01-01",
    interval = '1mo',
    auto_adjust=True
)["Open"].squeeze()

# GEL USD exchange rate
fx = pd.read_csv("USD_GEL Historical Data.csv")
fx['Date'] = pd.to_datetime(fx['Date'], format='%m/%d/%Y')
fx.set_index('Date', inplace=True)
fx = fx['Price']

# Tbilisi Interbank Interest Rate
tibr = pd.read_csv("tibr.csv")
tibr['Date'] = pd.to_datetime(tibr['Date'], format='%m/%d/%Y')
tibr.set_index('Date', inplace=True)
tibr = tibr['TIBR1M'].dropna()
tibr = (
    tibr[~tibr.index.duplicated(keep='last')]
    .asfreq('D')
    .ffill()
    .resample('MS')
    .first()
).sort_index(ascending = False)

# Secured Overnight Financing Rate
sofr = pd.read_excel("SOFR.xlsx")
sofr['Effective Date'] = pd.to_datetime(sofr['Effective Date'])
sofr.set_index('Effective Date', inplace=True)
sofr = sofr['Rate (%)'].dropna()
sofr = (
    sofr[~sofr.index.duplicated(keep='last')]
    .asfreq('D')
    .ffill()
    .resample('MS')
    .first()
).sort_index(ascending=False)

# Combine all series into one dataframe
df = pd.DataFrame({
    'qqq_price': nq,
    'fx_spot': fx,
    'tibr': tibr,
    'sofr': sofr
}).sort_index().dropna()

df['tibr'] = df['tibr'] / 100
df['sofr'] = df['sofr'] / 100

# NASDAQ100 retuerns
df['nq_return'] = np.log(df['qqq_price']).diff()

# Hedge carry (monthly)
df['carry'] = (df['tibr'] - df['sofr']) / 12

df = df.dropna()


#%% Useful functions

# Estimating the parameters of Brownian Motion with Normal errors using Maximum likelihood estimation

def neg_loglik_normal(params, data, dt): # Defining the log likelihood function for normal distribution
    """
    Negative log-likelihood for normal homoskedastic errors
    Δln S_t = μ Δt + σ sqrt(Δt) ε_t, ε_t ~ N(0,1)
    """
    mu, sigma = params
    if sigma <= 0:
        return 1e12  # forbid negative volatility
    
    resid = data - mu*dt
    ll = 0.5 * np.sum(np.log(2*np.pi) + np.log(sigma**2 * dt) + resid**2 / (sigma**2 * dt))
    return ll

def fit_normal_mle(log_returns, dt):
    """
    log_returns : 1D array-like of Δ ln S_t
    dt : time step 
    """
    data = np.asarray(log_returns).astype(float)
    data = data[~np.isnan(data)]
    
    # Initializing parameters
    mu0 = np.mean(data)/dt
    sigma0 = np.std(data)/np.sqrt(dt)
    
    res = minimize(lambda x: neg_loglik_normal(x, data, dt),
                   x0=[mu0, sigma0],
                   bounds=[(None,None), (1e-12,None)],
                   method="L-BFGS-B")
    
    return {
        "mu": res.x[0],
        "sigma": res.x[1],
        "neg_loglik": res.fun,
        "success": res.success,
        "message": res.message,
        "optim_result": res
    }

# Estimating Ornstein-Uhlenbeck Process using Maximum Likelihoodd estimation

def neg_loglik_ornstein_uhlenbeck(params, data, dt):
    """
    Negative log-likelihood for Ornstein-Uhlenbeck process
    Δr_t = κ(θ - r_t)Δt + σ sqrt(Δt) ε_t, ε_t ~ N(0,1)
    """
    kappa, theta, sigma = params
    if kappa <= 0 or sigma <= 0:
        return 1e12  # forbid negative parameters
    
    r_t = data[:-1]  # r_t at time t
    delta_r = np.diff(data)  # Δr_t = r_{t+1} - r_t
    
    # Expected change: κ(θ - r_t)Δt
    expected_delta = kappa * (theta - r_t) * dt
    resid = delta_r - expected_delta
    
    ll = 0.5 * np.sum(np.log(2*np.pi) + np.log(sigma**2 * dt) + resid**2 / (sigma**2 * dt))
    return ll

def fit_ou_mle(rate_series, dt=1/12):
    """
    rate_series : 1D array-like of interest rates r_t
    dt : time step (default monthly = 1/12)
    """
    data = np.asarray(rate_series).astype(float)
    data = data[~np.isnan(data)]
    
    # Initial parameter guesses
    delta_r = np.diff(data)
    theta0 = np.mean(data)
    kappa0 = 0.1  # reasonable initial mean reversion
    sigma0 = np.std(delta_r) / np.sqrt(dt)
    
    res = minimize(lambda x: neg_loglik_ornstein_uhlenbeck(x, data, dt),
                   x0=[kappa0, theta0, sigma0],
                   bounds=[(1e-12, None), (None, None), (1e-12, None)],
                   method="L-BFGS-B")
    
    return {
        "kappa": res.x[0],      # mean reversion speed
        "theta": res.x[1],      # long-term mean  
        "sigma": res.x[2],      # volatility
        "neg_loglik": res.fun,
        "success": res.success,
        "message": res.message,
        "optim_result": res
    }

#%% Parameter Estimations

nq_estimates = fit_normal_mle(df['nq_return'], dt=1/12)
carry_estimates = fit_ou_mle(df['carry'], dt=1/12)

#%% Monte carlo simulation 

n_years = 2
n_months = n_years * 12
n_paths = 100000  # Number of scenarios
dt = 1/12 # Monthly time step
starting_nav = 100
annual_fee = 0.012 # assume it includes the forward transaction fees as well
monthly_fee = annual_fee / 12

mu_nq = nq_estimates['mu']
sigma_nq = nq_estimates['sigma']
kappa_c = carry_estimates['kappa']
theta_c = carry_estimates['theta']
sigma_c = carry_estimates['sigma']

# Initial values from the last available data point
last_carry = df['carry'].iloc[-1]

# Initialize arrays to store results
nq_log_returns = np.zeros((n_months, n_paths)) # Rows = Time (Months), Columns = Scenarios
carry_paths = np.zeros((n_months + 1, n_paths))
carry_paths[0, :] = last_carry

# Random shocks for both processes (no correlation assumed)
Z_nq = np.random.normal(0, 1, (n_months, n_paths))
Z_carry = np.random.normal(0, 1, (n_months, n_paths))

for t in range(n_months):
    
    # 1. Simulate NASDAQ Log Returns (GBM)
    nq_log_returns[t, :] = (mu_nq - 0.5 * sigma_nq**2) * dt + sigma_nq * np.sqrt(dt) * Z_nq[t, :]
    
    # 2. Simulate Carry (OU Process)
    carry_paths[t+1, :] = carry_paths[t, :] + \
                          kappa_c * (theta_c - carry_paths[t, :]) * dt + \
                          sigma_c * np.sqrt(dt) * Z_carry[t, :]

#%% Calculating Fund Returns

# Monthly Stock Return + Carry - Fee
stock_returns = np.exp(nq_log_returns) - 1
monthly_fund_returns = stock_returns + carry_paths[:-1, :] - monthly_fee

# Calculate Cumulative NAV

nav_paths = starting_nav * np.cumprod(1 + monthly_fund_returns, axis=0)
nav_paths = np.vstack([np.full(n_paths, starting_nav), nav_paths])

#%% Analysis and Visualisation 

# Calculate Statistics
mean_nav = np.mean(nav_paths, axis=1)
p95 = np.percentile(nav_paths, 95, axis=1)
p5 = np.percentile(nav_paths, 5, axis=1)

# Visualization
plt.figure(figsize=(12, 6))
time_axis = np.arange(0, n_months + 1)

plt.plot(time_axis, nav_paths[:, :50], color='lightgray', alpha=0.3)
plt.plot(time_axis, mean_nav, color='blue', linewidth=2, label='Expected NAV (Mean)')
plt.fill_between(time_axis, p5, p95, color='blue', alpha=0.1, label='90% Confidence Interval')

plt.title(f'2-Year Forward Simulation: NASDAQ-100 GEL Hedged Fund')
plt.xlabel('Months')
plt.ylabel('NAV (GEL)')
plt.legend(['Scenarios', 'Expected NAV', '90% CI'], loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

# Print Final Statistics
print(f"Expected NAV after {n_years} years: {mean_nav[-1]:.2f} GEL")
print(f"Worst 5% Outcome: {p5[-1]:.2f} GEL")
print(f"Annualized Expected Return: {((mean_nav[-1]/100)**(1/n_years) - 1)*100:.2f}%")
