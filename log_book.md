## Implementation

I fitted a GARCH(1,1) model to daily returns for five indices:
the SSE Composite, EURO STOXX, FTSE, Nikkei and S&P 500.

The model is:

r_t = mu + epsilon_t
epsilon_t = sigma_t * z_t, where z_t are i.i.d. N(0, 1)
sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2

Returns are expressed in percentage units. 

For each index, I fixed mu at the sample mean and calculated the residuals epsilion_t as r_t - mu.
I initialized the first conditional variance using the sample variance.

For each candidate set of parameters (alpha, beta, omega), I recursively
calculated the daily conditional variances and evaluated the Gaussian
negative log-likelihood:

NLL = 0.5 * sum_t [
    log(2*pi) + log(sigma_t^2) + epsilon_t^2 / sigma_t^2
]

I minimized this function using scipy.optimize.minimize with L-BFGS-B.
The parameter bounds were alpha >= 0, beta >= 0 and omega >= 1e-8.

I chose initial values for alpha and beta manually, with their sum
below 1, and set the initial omega to:

omega_0 = (1 - alpha_0 - beta_0) * sample variance

This makes the unconditional variance implied by the starting
parameters equal to the sample variance, since:

unconditional variance = omega / (1 - alpha - beta)

For example, choosing alpha_0 = 0.08 and beta_0 = 0.90 gives
omega_0 = 0.02 * sample variance.

These are starting guesses, not fixed parameter values. Although
the starting values satisfy alpha_0 + beta_0 < 1, this restriction
is not imposed during optimization.

I did not impose alpha + beta < 1. Instead, I checked whether the
estimated parameters satisfied this condition after fitting.

To assess sensitivity to starting values, I will repeat the fit from
different initial parameter combinations and compare the estimates,
negative log-likelihood values and gradients.

result1 is obtained with initial a,b = 0.1, 0.8
result2 is obtained with initial a,b = 0.05, 0.90
result3 is obtained with initial a,b = 0.20, 0.50
result4 is obtained with initial a,b = 0.40, 0.20
result5 is obtained with initial a,b = 0.1, 0.89

to test the sensitivity of omega
result6 is obtained with a,b,w = [0.10, 0.80, 0.01 * v]
result7 is obtained with a,b,w = [0.10, 0.80, 1.00 * v]
