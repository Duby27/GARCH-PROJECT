import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

params = pd.read_csv(
    'results/stage1/final_params.csv',
    index_col='index'
)

indices = {
    'sse': 'SSE',
    'eustoxx': 'EURO STOXX 50',
    'ftse': 'FTSE 100',
    'nikkei': 'Nikkei 225',
    'sp500': 'S&P 500'
}

from pathlib import Path
Path('results/stage2').mkdir(parents=True, exist_ok=True)

for name, title in indices.items():
    # Read returns and fitted parameters
    data = pd.read_csv(
        f'data/{name}_returns.csv',
        index_col='Date',
        parse_dates=True
    )
    returns = data['return']

    alpha = params.loc[name, 'alpha']
    beta = params.loc[name, 'beta']
    omega = params.loc[name, 'omega']

    
    rolling_std = returns.rolling(window=20).std()

    
    residuals = returns - returns.mean()
    variance = [returns.var()]

    for t in range(1, len(returns)):
        daily_variance = (
            omega
            + alpha * residuals.iloc[t - 1] ** 2
            + beta * variance[t - 1]
        )
        variance.append(daily_variance)

    garch_volatility = pd.Series(
        np.sqrt(variance),
        index=returns.index
    )

    
    plt.figure(figsize=(12, 5))

    plt.plot(
        rolling_std,
        label='20-day rolling standard deviation',
        linewidth=1.2
    )
    plt.plot(
        garch_volatility,
        label='GARCH(1,1) conditional volatility',
        linewidth=1,
        alpha=0.8
    )

    plt.title(f'{title}: rolling std vs GARCH volatility')
    plt.xlabel('Date')
    plt.ylabel('Daily volatility (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f'results/stage2/{name}_volatility.png', dpi=300)
    plt.show()