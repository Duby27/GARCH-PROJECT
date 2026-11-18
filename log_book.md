## Model

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

## Stage 0: Initial experiments

The results from the first seven runs are saved in `results/stage0/`.

Some starting points produced almost identical estimates, while others gave
different parameters and noticeably higher NLL values, even with
`success=True`. I therefore decided to check the numerical reliability of
the estimates before interpreting the market differences.

The rest of the project is organised into four stages.

I used multi-start optimization to check sensitivity to starting values.

The 164 starting points combine a coarse grid with denser sampling around
the alpha and beta ranges found in stage 0. For omega, I used multiples of
each index's sample return variance:

omega_0 = omega_scale * sample variance

The same starting-point list was used for all five indices.

The aim was to select the lowest-NLL estimates, inspect their gradients,
and check whether different starting points repeatedly reached similar
results.

### Implementation

The batch fitting code is in `fit_reliability_test.py`, results are saved in `results/stage1/reliability_results.csv`,
and the analysis code is in `reliability_analysis.py`.

During the initial attempt, I encountered numerical overflow in the
conditional variance recursion. I added an upper bound of 1 on beta to
prevent exponential amplification through the lagged variance term.

This changes the search region relative to stage 0, but does not impose
alpha + beta < 1.

For each index, I sorted the results by NLL and selected the lowest-NLL
candidate. I calculated its gradient euclidean norm.

I also counted how many runs had NLL values within 1e-4 of the lowest
value for that index. This is a numerical tolerance, not a statistical
significance threshold.

### Results and discussion

All 820 runs returned finite parameter estimates, NLL values and gradients,
with no missing values in the saved results.

The selected estimates were in `results/stage1/final_params.csv`
These estimates closely match the better results from stage 0.
All five selected estimates lie away from the parameter bounds and
satisfy alpha + beta < 1, even though this condition was not imposed.

The number of runs within 1e-4 of each index's minimum NLL was:

| Index | Runs near minimum | Percentage |
| --- | ---: | ---: |
| SSE | 134 / 164 | 81.7% |
| EURO STOXX | 150 / 164 | 91.5% |
| FTSE | 142 / 164 | 86.6% |
| Nikkei | 160 / 164 | 97.6% |
| S&P 500 | 139 / 164 | 84.8% |

The parameter estimates within these groups were also very similar.
This suggests that the selected solutions are reproducible across many
different starting points.

The selected candidates had gradient norms of approximately 0.0017–0.0062.
These were much smaller than those of some poorer runs. For example,
the worst SSE run returned an NLL of 6015.315 and a maximum absolute
gradient component of about 1954, despite reporting success.

All runs reported `success=True`, with the termination message:

`CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`


Overall, repeated agreement across starting points and the relatively
small gradients support using the selected parameters for the following
stages. This provides evidence of numerical stability, but does not prove
global optimality or establish that GARCH(1,1) adequately describes the
data.

## Stage 2: Volatility across markets

### Plan

Calculate summary statistics and plot daily returns, return distributions,
rolling volatility and fitted conditional volatility.

Compare volatility levels, responses to shocks and persistence across the
five indices, distinguishing volatility magnitude from persistence.

### Implementation

<!-- Add calculation methods and plotting choices here. -->

### Results and discussion

<!-- Add tables, figures and interpretations here. -->

## Stage 3: Residual diagnostics

### Plan

Calculate standardized residuals:

z_t = (r_t - mu) / sigma_t

Inspect the autocorrelation of both z_t and z_t^2 using ACF plots and
Ljung–Box tests.

Compare squared demeaned returns with squared standardized residuals to
assess how much volatility clustering the model explains.

### Implementation

<!-- Add diagnostic settings and tests here. -->

### Results and discussion

<!-- Add diagnostic results and remaining model limitations here. -->

## Stage 4: Normal vs Student-t innovations

### Plan

Fit Normal and Student-t GARCH(1,1) models using the same observations and
consistent mean specifications for each index.

Compare AIC and BIC to assess whether allowing heavier-tailed innovations
improves the fit enough to justify the additional parameter.

This comparison evaluates in-sample fit, not out-of-sample forecasting
performance.

### Implementation

<!-- Add model specifications and estimation details here. -->

### Results and discussion

<!-- Add model comparisons and conclusions here. -->