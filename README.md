# Stochastic Data Analysis & Predictive Modeling Framework

A comprehensive, production-ready mathematical framework designed for exploring probability theory, evaluating statistical moments, and implementing predictive parametric models from scratch. The system focuses on absolute data verification against initial MS Excel prototypes and comparative algorithmic testing under stochastic (Monte-Carlo) simulation conditions.

## Key Modules & Features

### 1. Discrete Random Variables Analysis
Statistical profiling of discrete processes and validation against classical probability laws.
* **Supported Models:** Bernoulli, Binomial, Poisson, and Custom Discrete step-distributions.
* **Theoretical Probability Engine:** Custom combinatorial solvers (utilizing custom factorials and exponents) to evaluate exact theoretical probability mass functions (PMF).
* **Empirical Analysis:** Vectorized calculations of distinct values (`np.unique`), relative frequencies, and Empirical Cumulative Distribution Functions (ECDF).

### 2. Continuous Distributions & Descriptive Statistics
Advanced mathematical evaluation of continuous data arrays and testing of complex tail behaviors.
* **Advanced Distribution Fitting:** Integrates Uniform, Gaussian (Normal), Gamma, and asymmetric Beta (\(\alpha=\beta\), \(\alpha \neq \beta\)) distribution architectures.
* **Higher-Order Statistical Moments:** Custom computational logic for core descriptive indicators:
  * Expected Value (Mean) & Variance
  * Standard Deviation (\(\sigma\))
  * **Skewness:** Assessment of distribution asymmetry.
  * **Kurtosis:** Peakness evaluation with excess kurtosis adjustment (Leptokurtic/Platykurtic analysis).

### 3. Predictive Regression Modeling
Mathematical implementation of ordinary least squares (OLS) loss function optimization using the Nelder-Mead / BFGS algorithms (`scipy.optimize.minimize`) without relying on high-level ML libraries.
* **Multi-Model Regression:** Features standard Linear (\(y = a + bx\)), 2nd-degree Polynomial Non-Linear (\(y = a + bx + cx^2\)), and Multiple Polynomial feature interactions (\(z = a_0 + a_1x_1 + a_2x_2 + a_{11}x_1^2 + a_{22}x_2^2 + a_{12}x_1x_2\)).
* **Goodness-of-Fit Metrics:** Automated evaluation of Correlation Coefficients (\(R\)), Coefficients of Determination (\(R^2\)), and Residual Standard Error (RSE).

## Tech Stack
* **Language:** Python 3
* **Core Libraries:** NumPy (Vectorized array calculations), SciPy (Advanced mathematical optimization & statistical functions), Matplotlib (Data viz & frequency charts).
* **Prototyping:** MS Excel (Initial mathematical architecture validation).

## Operating Modes
1. **Test Mode (Deterministic):** Features a robust text-stream parser that extracts tabular variables from raw `input.txt` config files.
2. **Work Mode (Stochastic/Monte-Carlo):** Generates custom sample dimensions, applies controlled pseudo-random seeds (`np.random.seed`) for reproducibility, and injects controlled Gaussian/Uniform noise to benchmark parameter estimations against the true ground parameters.
