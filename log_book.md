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

Some starting points produced almost identical estimates, while others gave different parameters and noticeably higher negative log-likelihood (NLL) values, even with `success=True`. I therefore checked the numerical reliability of the estimates before interpreting differences across markets.

The rest of the project is organised into four stages.

## Stage 1: Numerical reliability and parameter uncertainty

### Plan

I investigated two questions. First, how often does the optimizer reach the same low-NLL solution when I change its starting point? Second, how precisely does each index's return sample identify the three GARCH parameters?

### Implementation

#### Multi-start optimization

I used 164 starting points for each index. These combined a coarse grid with denser sampling around the alpha and beta ranges found in Stage 0. For omega, I scaled each index's sample return variance by a set of starting multipliers:

$$
\omega_0 = \text{omega scale} \times \text{sample variance}.
$$

I used the same combinations of initial alpha, beta and omega scale for all five indices. The batch fitting code is in `fit_reliability_test.py`, the results are in `results/stage1/reliability_results.csv`, and the analysis code is in `reliability_analysis.py`.

During the initial attempt, I encountered numerical overflow in the conditional variance recursion. I added an upper bound of 1 on beta to prevent exponential amplification through the lagged variance term. This changed the search region relative to Stage 0 while still allowing $\alpha+\beta \geq 1$ during optimization.

For each index, I selected the run with the lowest NLL and calculated the Euclidean norm of its gradient. I also counted how many runs finished within $10^{-4}$ NLL units of that minimum and compared the parameter estimates within this group.

#### Parameter standard errors

I wrote `parameter_uncertainty.py` to estimate standard errors for $\alpha$, $\beta$ and $\omega$ at the selected solution for each index. The script keeps $\mu$ fixed at the sample mean, uses the same initial conditional variance as the fit, and checks that it reproduces the saved NLL before calculating derivatives.

Let $H$ be the $3 \times 3$ Hessian of the total Gaussian NLL with respect to $(\alpha,\beta,\omega)$, evaluated at the selected estimates. The script approximates $H$ using central finite differences, then calculates the model-based covariance matrix and standard errors as

$$
\widehat{\operatorname{Cov}}(\hat\theta)=H^{-1},
\qquad
\operatorname{SE}(\hat\theta_j)
=\sqrt{(H^{-1})_{jj}}.
$$

Because the objective is the **total** NLL, its curvature already accounts for sample size. The script checks that the Hessian is positive definite and repeats the calculation with half the step size. It uses the smaller-step result if every standard error changes by at most 1%.

The output is designed to include `results/stage1/parameter_standard_errors.csv` and one covariance matrix per index. The off-diagonal entries matter for Stage 2. For example, the standard error of estimated persistence is

$$
\operatorname{SE}(\hat\alpha+\hat\beta)
=
\sqrt{
\widehat{\operatorname{Var}}(\hat\alpha)
+\widehat{\operatorname{Var}}(\hat\beta)
+2\widehat{\operatorname{Cov}}(\hat\alpha,\hat\beta)
}.
$$

These standard errors use the fitted Gaussian model and treat the sample mean as fixed. They omit uncertainty from estimating $\mu$.

### Results and discussion

All 820 optimization runs returned finite parameter estimates, NLL values and gradients. The selected estimates, saved in `results/stage1/final_params.csv`, closely matched the better results from Stage 0. All five selected solutions were interior to the parameter bounds and satisfied $\hat\alpha+\hat\beta<1$.

| Index | Runs within 0.0001 of the minimum NLL | Percentage |
| --- | ---: | ---: |
| SSE Composite | 134 / 164 | 81.7% |
| EURO STOXX | 150 / 164 | 91.5% |
| FTSE | 142 / 164 | 86.6% |
| Nikkei | 160 / 164 | 97.6% |
| S&P 500 | 139 / 164 | 84.8% |

Within each group, the parameter estimates were also very similar. The selected candidates had gradient norms of approximately 0.0017–0.0062. By comparison, the worst SSE run finished at an NLL of 6015.315 with a maximum absolute gradient component of about 1954.

All runs reported `success=True` with the same termination message:

`CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`

The optimizer could therefore stop after a small change in the objective even when the gradient was large. Repeated agreement across starting points, together with the much smaller gradients at the selected solutions, gave me a basis for carrying those estimates into Stage 2. The run counts measure reproducibility on the same return data. Sampling uncertainty is a separate question addressed by the standard-error calculation above.

## Stage 2: Volatility across markets

### Plan

Compare fitted daily volatility with a 20-day rolling standard deviation, then compare how persistent volatility shocks are across the five indices.

### Implementation

For each index, I used the parameters from `results/stage1/final_params.csv` to reconstruct the daily conditional variance. I used the same sample mean and initial variance as in Stage 1, then took the square root to obtain daily conditional volatility.

I plotted this alongside the sample standard deviation of the most recent 20 daily returns. Both curves measure daily volatility in percentage units. At date $t$, the rolling window includes that day's return, while the GARCH recursion uses shocks only up to $t-1$. The GARCH parameters were estimated on the full sample, so the plots provide an in-sample comparison.

### Results and discussion

For the SSE Composite, both curves rose during the major high-volatility episodes. During quieter periods, the GARCH estimate was often above the rolling estimate. The two measures give past shocks different weights. A large return drops out of the rolling window after 20 trading days, while its influence on fitted GARCH variance decays gradually.

I compared the persistence of variance forecasts using $\hat\alpha+\hat\beta$.

| Index | Estimated persistence |
| --- | ---: |
| SSE Composite | 0.9870 |
| EURO STOXX | 0.9684 |
| S&P 500 | 0.9673 |
| FTSE | 0.9460 |
| Nikkei | 0.9438 |

The SSE had the highest estimated persistence. Its variance forecast therefore returns to its fitted long-run level most slowly after a shock. EURO STOXX and the S&P 500 had very similar point estimates, as did the FTSE and the Nikkei. I interpret these close rankings descriptively. Persistence measures the speed of mean reversion, while the level of the fitted volatility curve tells me how large daily fluctuations typically were.

## Stage 3: Residual diagnostics

### Plan

Check how much volatility clustering remains after scaling each day's shock by its fitted conditional volatility. Inspect the distribution of the standardized residuals and check for serial correlation in both the signed shocks and their squares.

### Implementation

In `residual_analysis.py`, I calculated the standardized residual for each trading day as

$$
\hat z_t = \frac{r_t-\hat\mu}{\hat\sigma_t}.
$$

For each index, I plotted the standardized residuals over time and as a histogram with bins symmetric around zero. I then compared the sample ACFs of three series at lags 1–20:

- Squared demeaned returns, $\hat\epsilon_t^2$, to measure volatility clustering before standardization.
- Standardized residuals, $\hat z_t$, to check for linear dependence between signed shocks.
- Squared standardized residuals, $\hat z_t^2$, to check whether large shocks still tended to occur close together after accounting for fitted volatility.

I used the same vertical scale for all three ACF panels within each index. The shaded bands show the approximate 95% pointwise iid reference range, $\pm 1.96/\sqrt{n}$. I also calculated Ljung–Box statistics for each series at 10 and 20 lags. Each test considers the autocorrelations from lag 1 through the stated lag jointly. The figures and `ljung_box_results.csv` are in `results/stage3/`.

### Results and discussion

All five indices showed positive autocorrelation in squared demeaned returns across multiple lags. This became much weaker after standardization. For the S&P 500, the lag-2 ACF fell from approximately 0.47 for squared demeaned returns to 0.02 for squared standardized residuals. Its 20-lag Ljung–Box statistic fell from 4749.24 to 13.27.

| Index | Nominal Ljung–Box p-value for squared standardized residuals, lags 1–10 | Lags 1–20 |
| --- | ---: | ---: |
| SSE Composite | 0.981 | 0.998 |
| EURO STOXX | 0.604 | 0.687 |
| FTSE | 0.757 | 0.926 |
| Nikkei | 0.113 | 0.146 |
| S&P 500 | 0.421 | 0.865 |

All ten nominal p-values exceeded 0.05. Together with the reduction in the ACFs, this suggests that GARCH(1,1) captured much of the observed volatility clustering in sample. I treat the p-values as approximate diagnostics because the ordinary chi-square reference distribution does not account for estimation of the GARCH parameters.

For the signed standardized residuals, the SSE showed some evidence of short-lag autocorrelation in the 10-lag test (nominal $p=0.033$). The 20-lag test gave weaker evidence ($p=0.223$). This finding concerns dependence in the signed shocks. The SSE's squared standardized residuals showed little remaining autocorrelation at either horizon.

Across the five indices, standardized residuals were concentrated near zero, but the most extreme negative observations extended further than the positive ones. For the SSE, the extremes were approximately −10 and +5.5 fitted conditional standard deviations. This motivates Stage 4. Once changes in daily volatility have been accounted for, does a heavier-tailed innovation distribution improve the fit? A symmetric Student-t distribution can accommodate more extreme shocks in both tails, although it still assigns equal probability to equally large positive and negative shocks.

## Stage 4: Normal vs Student-t 

### Plan

Stage 3 showed that standardizing returns removed much of the autocorrelation in squared shocks, but some large standardized residuals remained. I compared Normal and Student-t GARCH(1,1) models to see whether allowing heavier tails improved the in-sample fit enough to justify estimating one additional parameter.

### Implementation

In `student_t_comparison.py`, I used the selected Normal fits from `results/stage1/final_params.csv` as the baseline. For each index, both models used the same daily returns, the same sample mean for $\mu$, and the same sample variance for the initial conditional variance.

For the Student-t model, I kept the GARCH variance recursion and used Student-t innovations standardized to have variance one. For each candidate set of parameters $(\alpha,\beta,\omega,\nu)$, I calculated the daily conditional variances and evaluated the Student-t negative log-likelihood. With $\nu>2$, the scale passed to SciPy for day $t$ was

$$
\text{scale}_t = \sigma_t \sqrt{\frac{\nu-2}{\nu}},
$$

which gives the return shock a conditional variance of $\sigma_t^2$.

I initialized $(\alpha,\beta,\omega)$ at each index's selected Normal estimates and tried three starting values for $\nu$: 4, 8 and 20. From each starting point, I jointly optimized all four parameters using L-BFGS-B and retained the successful fit with the lowest NLL. This varied the starting value of $\nu$ while using a common starting point for the three volatility parameters.

I compared the models within each index using

$$
\mathrm{AIC}=2\,\mathrm{NLL}+2k,
\qquad
\mathrm{BIC}=2\,\mathrm{NLL}+k\log n.
$$

Here $n$ is the number of daily returns. I counted the estimated sample mean in both models, giving $k=4$ for Normal and $k=5$ for Student-t. The fitted parameters, NLL, AIC and BIC are saved in `results/stage4/model_comparison.csv`.

### Results and discussion

Both AIC and BIC favoured Student-t innovations for all five indices. The reductions below are calculated as the Normal value minus the Student-t value.

| Index | Fitted $\nu$ | AIC reduction | BIC reduction |
| --- | ---: | ---: | ---: |
| SSE Composite | 4.44 | 435.66 | 429.40 |
| EURO STOXX | 5.45 | 241.83 | 235.53 |
| FTSE | 5.83 | 205.31 | 199.01 |
| Nikkei | 6.40 | 166.90 | 160.63 |
| S&P 500 | 5.45 | 240.48 | 234.18 |

The SSE showed the largest improvement. Its NLL fell from 5880.58 to 5661.75, and its BIC fell by 429.40 after accounting for the additional parameter. It also had the lowest fitted $\nu$, corresponding to the heaviest fitted tails among the five indices. This is consistent with the extreme standardized shocks observed for the SSE in Stage 3. The symmetric Student-t model allows larger shocks in both directions, so this comparison leaves open whether the negative tail is heavier than the positive tail.

Estimated volatility persistence, $\hat\alpha+\hat\beta$, increased for all five indices under Student-t innovations. The largest change was for the S&P 500, from 0.9673 to 0.9897, bringing it close to the SSE estimate of 0.9898. The persistence differences reported in Stage 2 therefore depend in part on the assumed innovation distribution.

Overall, the five within-index comparisons support allowing heavier-tailed innovations in this sample. Assessing whether that improvement carries over to volatility forecasts would require an out-of-sample comparison.