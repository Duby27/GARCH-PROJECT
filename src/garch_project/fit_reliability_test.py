from fit_garch import fitter
import pandas as pd

initial_points = pd.read_csv('initial_points.csv')

sse = pd.read_csv('data/sse_returns.csv')['return']
eustoxx = pd.read_csv('data/eustoxx_returns.csv')['return']
ftse = pd.read_csv('data/ftse_returns.csv')['return']
nikkei = pd.read_csv('data/nikkei_returns.csv')['return']
sp500 = pd.read_csv('data/sp500_returns.csv')['return']

indeces = [sse, eustoxx, ftse, nikkei, sp500]
names = ['sse', 'eustoxx', 'ftse', 'nikkei', 'sp500']
results_list = []
for i, data in enumerate(indeces):
    for j in initial_points.index:
        initial = initial_points.iloc[j]
        initial_list = [initial['alpha'], initial['beta'], initial['omega_scale']]
        result = fitter(data, initial_list)
        a, b, w = result.x
        results_list.append({
            'index' : names[i],
            'alpha_0': initial_list[0],
            'beta_0': initial_list[1],
            'omega_scale': initial_list[2],
            'omega_0': initial_list[2] * data.var(),
            'alpha' : a,
            'beta' : b,
            'alpha+beta': a + b,
            'omega' : w,
            'NLL' : result.fun,
            'grad_alpha': result.jac[0],
            'grad_beta': result.jac[1],
            'grad_omega': result.jac[2],
            'success': result.success,
            'message': result.message,
            'nit': result.nit
        })

results = pd.DataFrame(results_list)
results.to_csv('results/stage1/reliability_results.csv', index=False)