from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]

def student_t_nll(params, epsilon, initial_variance):
    alpha, beta, omega, nu = params

    variance = np.empty(len(epsilon))
    variance[0] = initial_variance
    for day in range(1, len(epsilon)):
        variance[day] = (omega + alpha * epsilon[day - 1] ** 2
                         + beta * variance[day - 1])

    scale = np.sqrt(variance * (nu - 2) / nu)
    daily_log_likelihood = student_t.logpdf(epsilon, df=nu, scale=scale)
    return -daily_log_likelihood.sum()

def main():
    normal_fits = pd.read_csv(
        ROOT / 'results/stage1/final_params.csv', index_col='index'
    )
    rows = []

    for name in normal_fits.index:
        returns = pd.read_csv(ROOT / 'data' / f'{name}_returns.csv')['return']
        epsilon = (returns - returns.mean()).to_numpy()
        initial_variance = returns.var()
        normal = normal_fits.loc[name]

        best_nll = np.inf
        for nu_start in [4, 8, 20]:
            start = [normal['alpha'], normal['beta'], normal['omega'], nu_start]

            fit = minimize(
                student_t_nll, start, args=(epsilon, initial_variance),
                method='L-BFGS-B',
                bounds=[(0, None), (0, 1), (1e-8, None), (2.0001, None)],
                options={'ftol': 1e-13, 'maxiter': 1000},
            )

            if fit.success and fit.fun < best_nll:
                best_nll = fit.fun
                best_params = fit.x

        if not np.isfinite(best_nll):
            raise RuntimeError(f'No Student-t fit converged for {name}')

        alpha, beta, omega, nu = best_params
        rows.append([name, 'Normal', len(returns), normal['alpha'], normal['beta'],
                     normal['omega'], np.nan, normal['NLL'], 4])
        rows.append([name, 'Student-t', len(returns), alpha, beta, omega, nu,
                     best_nll, 5])

    columns = ['index', 'model', 'n', 'alpha', 'beta', 'omega', 'nu', 'NLL', 'k']
    results = pd.DataFrame(rows, columns=columns)
    results['persistence'] = results['alpha'] + results['beta']
    results['AIC'] = 2 * results['NLL'] + 2 * results['k']
    results['BIC'] = 2 * results['NLL'] + results['k'] * np.log(results['n'])

    output = ROOT / 'results/stage4'
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / 'model_comparison.csv', index=False)


if __name__ == '__main__':
    main()