import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
sse_returns = pd.read_csv('data/sse_returns.csv')['return']
eustoxx_returns = pd.read_csv('data/eustoxx_returns.csv')['return']
ftse_returns = pd.read_csv('data/ftse_returns.csv')['return']
nikkei_returns = pd.read_csv('data/nikkei_returns.csv')['return']
sp500_returns = pd.read_csv('data/sp500_returns.csv')['return']

def fitter(data):
    mu = data.mean()
    residuals = []
    for r in data:
        residuals.append(r - mu)
    def goal(params):
        a, b, w = params
        variance = [data.var()]
        for t in range(1, len(residuals)):
            daily_variance = w + a * residuals[t - 1] ** 2 + b * variance[t - 1]
            variance.append(daily_variance)
        
        f = 0
        for t in range(len(residuals)):
            daily_term = (math.log(2 * math.pi) + math.log(variance[t]) + residuals[t] ** 2 / variance[t])
            f += -0.5 * daily_term
        return -f
    bounds = [(0, None), (0, None), (1e-8, None)]
    initials = [0.10, 0.80, 1.00  * data.var()]
    result = minimize(goal,x0=initials, bounds=bounds)

    return result


results = pd.DataFrame(columns=[
    'alpha', 'beta', 'omega', 'alpha+beta',
    'nll', 'success'
])

sse_result = fitter(sse_returns)
a, b, w = sse_result.x
results.loc['SSE'] = [
    a, b, w, a + b,
    sse_result.fun, sse_result.success
]

eustoxx_result = fitter(eustoxx_returns)
a, b, w = eustoxx_result.x
results.loc['EUSTOXX'] = [
    a, b, w, a + b,
    eustoxx_result.fun, eustoxx_result.success
]

ftse_result = fitter(ftse_returns)
a, b, w = ftse_result.x
results.loc['FTSE'] = [
    a, b, w, a + b,
    ftse_result.fun, ftse_result.success
]

nikkei_result = fitter(nikkei_returns)
a, b, w = nikkei_result.x
results.loc['NIKKEI'] = [
    a, b, w, a + b,
    nikkei_result.fun, nikkei_result.success
]

sp500_result = fitter(sp500_returns)
a,b,w =sp500_result.x
results.loc['S&P'] = [
    a, b, w, a + b,
    sp500_result.fun, sp500_result.success
]

print(results)
results.to_csv('results/garch_results_7.csv', index_label='index')