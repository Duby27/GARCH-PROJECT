from pathlib import Path

import numpy as np
import pandas as pd


# Locate the project root from this script's location
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'stage1'

PARAMETERS = ['alpha', 'beta', 'omega']


def negative_log_likelihood(params, residuals, initial_variance):
    alpha, beta, omega = params

    variance = np.empty(len(residuals))
    variance[0] = initial_variance

    for t in range(1, len(residuals)):
        variance[t] = (
            omega
            + alpha * residuals[t - 1] ** 2
            + beta * variance[t - 1]
        )

    return 0.5 * np.sum(
        np.log(2 * np.pi)
        + np.log(variance)
        + residuals ** 2 / variance
    )


def numerical_hessian(function, params, steps):
    size = len(params)
    hessian = np.zeros((size, size))
    centre = function(params)

    for i in range(size):
        shift_i = np.zeros(size)
        shift_i[i] = steps[i]

        # Second derivative with respect to parameter i
        hessian[i, i] = (
            function(params + shift_i)
            - 2 * centre
            + function(params - shift_i)
        ) / steps[i] ** 2

        # Mixed second derivatives
        for j in range(i + 1, size):
            shift_j = np.zeros(size)
            shift_j[j] = steps[j]

            value = (
                function(params + shift_i + shift_j)
                - function(params + shift_i - shift_j)
                - function(params - shift_i + shift_j)
                + function(params - shift_i - shift_j)
            ) / (4 * steps[i] * steps[j])

            hessian[i, j] = value
            hessian[j, i] = value

    return hessian


def covariance_and_se(hessian):
    if not np.all(np.isfinite(hessian)):
        raise ValueError('Hessian contains non-finite values.')

    if np.any(np.linalg.eigvalsh(hessian) <= 0):
        raise ValueError('Hessian is not positive definite.')

    covariance = np.linalg.inv(hessian)
    standard_errors = np.sqrt(np.diag(covariance))

    return covariance, standard_errors


def main():
    fitted = pd.read_csv(
        RESULTS / 'final_params.csv',
        index_col='index'
    )

    rows = []

    for name in fitted.index:
        returns = pd.read_csv(
            ROOT / 'data' / f'{name}_returns.csv'
        )['return']

        if len(returns) < 2 or not np.all(np.isfinite(returns)):
            raise ValueError(f'{name}: invalid returns data.')

        params = fitted.loc[name, PARAMETERS].to_numpy(dtype=float)

        # Keep the same mean and initial variance as the original fit
        residuals = (returns - returns.mean()).to_numpy()
        initial_variance = returns.var()

        def objective(candidate):
            return negative_log_likelihood(
                candidate, residuals, initial_variance
            )

        # Check that this objective reproduces the saved NLL
        nll = objective(params)

        if not np.isclose(
            nll, fitted.loc[name, 'NLL'], rtol=1e-8, atol=1e-5
        ):
            raise ValueError(f'{name}: NLL does not match Stage 1.')

        # Parameter-specific finite-difference steps
        steps = 1e-4 * np.maximum(np.abs(params), 0.01)

        if np.any(params - steps <= 0):
            raise ValueError(f'{name}: parameters too close to zero.')

        hessian = numerical_hessian(objective, params, steps)
        _, se_original = covariance_and_se(hessian)

        # Repeat with smaller steps to check numerical stability
        hessian_small = numerical_hessian(
            objective, params, steps / 2
        )
        covariance, se = covariance_and_se(hessian_small)

        relative_change = np.max(np.abs(se - se_original) / se)

        if relative_change > 0.01:
            raise ValueError(
                f'{name}: SE changes by more than 1% when halving steps.'
            )

        row = {'index': name}

        for i, parameter in enumerate(PARAMETERS):
            row[parameter] = params[i]
            row[f'{parameter}_se'] = se[i]

        rows.append(row)

        pd.DataFrame(
            covariance,
            index=PARAMETERS,
            columns=PARAMETERS
        ).to_csv(
            RESULTS / f'{name}_covariance.csv',
            index_label='parameter'
        )

    summary = pd.DataFrame(rows).set_index('index')

    summary.to_csv(
        RESULTS / 'parameter_standard_errors.csv'
    )

    print(summary.to_string(float_format=lambda value: f'{value:.6f}'))


if __name__ == '__main__':
    main()