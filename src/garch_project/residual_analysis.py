from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[2]
INDICES = ['sse', 'eustoxx', 'ftse', 'nikkei', 'sp500']
MAX_LAG = 20
LB_LAGS = [10, 20]

def sample_acf(values, max_lag):
    centered = np.asarray(values, dtype=float) - np.mean(values)
    denominator = np.sum(centered ** 2)

    correlations = []
    for lag in range(1, max_lag + 1):
        numerator = np.sum(centered[lag:] * centered[:-lag])
        correlations.append(numerator / denominator)
    return np.array(correlations)


def main():
    final_params = pd.read_csv(
        ROOT / 'results/stage1/final_params.csv', index_col='index'
    )
    output = ROOT / 'results/stage3'
    output.mkdir(parents=True, exist_ok=True)
    lb_rows = []
    lags = np.arange(1, MAX_LAG + 1)

    for name in INDICES:
        data = pd.read_csv(
            ROOT / 'data' / f'{name}_returns.csv', parse_dates=['Date']
        )
        dates = data['Date']
        returns = data['return']
        n = len(returns)
        params = final_params.loc[name]
        alpha, beta, omega = params[['alpha', 'beta', 'omega']].to_numpy(dtype=float)
        mu = returns.mean()
        epsilon = (returns - mu).to_numpy(dtype=float)
        h = np.empty(n)
        h[0] = returns.var()
        for t in range(1, n):
            h[t] = omega + alpha * epsilon[t - 1] ** 2 + beta * h[t - 1]
        sigma = np.sqrt(h)
        z = epsilon / sigma
        limit = 1.05 * np.max(np.abs(z))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(dates, z, linewidth=0.6)
        axes[0].axhline(0, color='black', linewidth=0.8)
        axes[0].set_ylim(-limit, limit)
        axes[0].set_title('Standardized residuals over time')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('z')
        axes[1].hist(z, bins=np.linspace(-limit, limit, 51), density=True,
                     edgecolor='white')
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].set_xlim(-limit, limit)
        axes[1].set_title('Standardized residual distribution')
        axes[1].set_xlabel('z')
        axes[1].set_ylabel('Density')
        fig.suptitle(f'{name.upper()} | In-sample standardized residuals')
        fig.tight_layout()
        fig.savefig(output / f'{name}_standardized_residuals.png', dpi=300)
        plt.close(fig)

        series = {
            'epsilon_squared': epsilon ** 2,
            'z': z,
            'z_squared': z ** 2,
        }
        titles = ['Raw squared residuals', 'Standardized residuals',
                  'Squared standardized residuals']
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, sharey=True)
        band = 1.96 / np.sqrt(n)
        scale = max(0.1, band)

        for ax, (series_name, values), title in zip(axes, series.items(), titles):
            correlations = sample_acf(values, MAX_LAG)
            scale = max(scale, np.max(np.abs(correlations)))
            ax.stem(lags, correlations, basefmt=' ', linefmt='C0-', markerfmt='C0o')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.axhspan(-band, band, color='C0', alpha=0.12,
                       label='95% iid reference')
            ax.set_title(title)
            ax.set_ylabel('Sample ACF')
            ax.legend(loc='upper right', fontsize=8)

            for m in LB_LAGS:
                included_lags = np.arange(1, m + 1)
                q = n * (n + 2) * np.sum(
                    correlations[:m] ** 2 / (n - included_lags)
                )
                lb_rows.append({
                    'index': name, 'series': series_name, 'lag': m,
                    'Q': q, 'nominal_p_value': chi2.sf(q, df=m),
                })

        axes[0].set_ylim(-1.1 * scale, 1.1 * scale)
        axes[-1].set_xlabel('Lag (trading observations)')
        axes[-1].set_xticks(np.arange(0, MAX_LAG + 1, 5))
        axes[-1].set_xlim(0.5, MAX_LAG + 0.5)
        fig.suptitle(f'{name.upper()} | ACF comparison')
        fig.tight_layout()
        fig.savefig(output / f'{name}_acf_comparison.png', dpi=300)
        plt.close(fig)

    pd.DataFrame(lb_rows).to_csv(output / 'ljung_box_results.csv', index=False)


if __name__ == '__main__':
    main()
