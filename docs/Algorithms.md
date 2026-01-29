# PyForecast Algorithms

Technical documentation of the algorithms used in PyForecast for decline curve analysis.

## Table of Contents

1. [Hyperbolic Decline Model](#hyperbolic-decline-model)
2. [Terminal Decline Switch](#terminal-decline-switch)
3. [Curve Fitting](#curve-fitting)
4. [Recency Weighting](#recency-weighting)
5. [Regime Change Detection](#regime-change-detection)
6. [Parameter Learning](#parameter-learning)

---

## Hyperbolic Decline Model

### The Arps Equation

PyForecast uses the Arps hyperbolic decline equation, the industry standard for production forecasting:

$$q(t) = \frac{q_i}{(1 + b \cdot D_i \cdot t)^{1/b}}$$

Where:
- $q(t)$ = Production rate at time $t$
- $q_i$ = Initial production rate at $t=0$
- $D_i$ = Initial nominal decline rate (fraction/time)
- $b$ = Hyperbolic exponent (dimensionless)
- $t$ = Time from start

### Special Cases

The hyperbolic equation includes two special cases:

#### Exponential Decline (b → 0)

When $b \rightarrow 0$, the equation simplifies to:

$$q(t) = q_i \cdot e^{-D_i \cdot t}$$

In practice, PyForecast uses exponential form when $b \leq 0.01$.

#### Harmonic Decline (b = 1)

When $b = 1$:

$$q(t) = \frac{q_i}{1 + D_i \cdot t}$$

### Cumulative Production

Total cumulative production from time 0 to $t$:

**General Hyperbolic ($b \neq 0, 1$):**

$$N_p(t) = \frac{q_i}{(1-b) \cdot D_i} \left[ 1 - (1 + b \cdot D_i \cdot t)^{\frac{b-1}{b}} \right]$$

**Exponential ($b \rightarrow 0$):**

$$N_p(t) = \frac{q_i}{D_i} \left( 1 - e^{-D_i \cdot t} \right)$$

**Harmonic ($b = 1$):**

$$N_p(t) = \frac{q_i}{D_i} \ln(1 + D_i \cdot t)$$

### Instantaneous Decline Rate

The instantaneous decline rate $D(t)$ at any time:

$$D(t) = \frac{D_i}{1 + b \cdot D_i \cdot t}$$

This decreases over time from $D_i$ toward zero (for hyperbolic) or remains constant (for exponential).

---

## Terminal Decline Switch

### Purpose

The hyperbolic equation with $b > 0$ implies that decline rate approaches zero as time increases, leading to unrealistically long well lives. The terminal decline switch addresses this by switching to exponential decline at a minimum rate $D_{min}$.

### Switch Time Calculation

The switch occurs when the instantaneous decline rate equals $D_{min}$:

$$D(t_{switch}) = D_{min}$$

Solving for $t_{switch}$:

$$t_{switch} = \frac{D_i / D_{min} - 1}{b \cdot D_i}$$

### Combined Model

For $t \leq t_{switch}$:
$$q(t) = \frac{q_i}{(1 + b \cdot D_i \cdot t)^{1/b}}$$

For $t > t_{switch}$:
$$q(t) = q_{switch} \cdot e^{-D_{min} \cdot (t - t_{switch})}$$

Where:
$$q_{switch} = \frac{q_i}{(1 + b \cdot D_i \cdot t_{switch})^{1/b}}$$

### Typical Values

| Parameter | Typical Value | Range |
|-----------|---------------|-------|
| $D_{min}$ | 6%/year | 3-10%/year |
| $t_{switch}$ | 3-10 years | Calculated |

---

## Curve Fitting

### Objective Function

PyForecast minimizes the weighted sum of squared residuals:

$$\text{minimize} \sum_{i=1}^{n} w_i \cdot (q_i - \hat{q}_i)^2$$

Where:
- $q_i$ = Observed production rate at month $i$
- $\hat{q}_i$ = Model predicted rate at month $i$
- $w_i$ = Weight for observation $i$ (see Recency Weighting)

### Optimization Method

The fitting uses `scipy.optimize.curve_fit` with:
- **Method**: Levenberg-Marquardt (trust region reflective)
- **Bounds**: Configurable bounds on $b$ ($b_{min}$ to $b_{max}$)
- **Initial guess**: Derived from log-linear regression

### Initial Guess Derivation

1. Apply log transform to positive rates: $\ln(q)$
2. Fit linear regression: $\ln(q) = a + m \cdot t$
3. Extract initial estimates:
   - $q_i \approx e^a$
   - $D_i \approx -m$ (for exponential approximation)
   - $b \approx 0.5$ (starting guess)

### Fit Quality Metrics

**R-squared (Coefficient of Determination):**

$$R^2 = 1 - \frac{\sum(q_i - \hat{q}_i)^2}{\sum(q_i - \bar{q})^2}$$

**RMSE (Root Mean Square Error):**

$$RMSE = \sqrt{\frac{1}{n}\sum(q_i - \hat{q}_i)^2}$$

**AIC (Akaike Information Criterion):**

$$AIC = n \cdot \ln\left(\frac{RSS}{n}\right) + 2k$$

Where $k = 3$ (number of parameters: $q_i$, $D_i$, $b$)

**BIC (Bayesian Information Criterion):**

$$BIC = n \cdot \ln\left(\frac{RSS}{n}\right) + k \cdot \ln(n)$$

---

## Recency Weighting

### Purpose

Recent production data often provides better indication of current decline behavior than older data. Recency weighting gives more influence to recent observations.

### Exponential Decay Weights

Weights follow an exponential decay from the most recent observation:

$$w_i = e^{-\lambda \cdot (t_{max} - t_i)}$$

Where:
- $t_i$ = Time of observation $i$
- $t_{max}$ = Time of most recent observation
- $\lambda$ = Decay constant

### Half-Life Parameterization

The decay is parameterized by half-life $\tau$ (months):

$$\lambda = \frac{\ln(2)}{\tau}$$

This means:
- Data $\tau$ months old has weight 0.5
- Data $2\tau$ months old has weight 0.25
- Data $3\tau$ months old has weight 0.125

### Example

With `recency_half_life = 12`:

| Months Ago | Weight |
|------------|--------|
| 0 (current) | 1.000 |
| 6 | 0.707 |
| 12 | 0.500 |
| 24 | 0.250 |
| 36 | 0.125 |

### Configuration

```yaml
fitting:
  recency_half_life: 12.0  # months
```

- **Lower values** (6-9): More reactive to recent changes
- **Higher values** (18-24): Smoother, more stable fits
- **Default**: 12 months

---

## Regime Change Detection

### Purpose

Production can change dramatically due to refracs, workovers, or other interventions. Regime detection identifies these changes so that fitting uses only the current decline regime.

### Algorithm Overview

1. Scan through production history chronologically
2. At each point, fit exponential trend to preceding window
3. Project trend forward and compare to actual production
4. Flag regime change if actual exceeds threshold

### Detailed Steps

#### Step 1: Window Fitting

For each point $i$ with sufficient history, fit exponential to window:

- Window: months $(i - w)$ to $(i - 1)$ where $w$ = window size
- Fit: $q(t) = q_0 \cdot e^{-D \cdot t}$
- Record residual standard deviation $\sigma$

#### Step 2: Projection

Project the fitted trend to point $i$:

$$\hat{q}_i = q_0 \cdot e^{-D \cdot w}$$

#### Step 3: Threshold Calculation

The threshold combines statistical and percentage criteria:

**Statistical threshold:**
$$T_{stat} = \hat{q}_i + n_\sigma \cdot \sigma$$

**Percentage threshold:**
$$T_{pct} = \hat{q}_i \cdot (1 + p_{min})$$

**Final threshold:**
$$T = \max(T_{stat}, T_{pct})$$

Where:
- $n_\sigma$ = Number of standard deviations (default: 2.5)
- $p_{min}$ = Minimum percentage increase (default: 1.0 = 100%)

#### Step 4: Sustained Detection

A regime change is confirmed when production exceeds threshold for $s$ consecutive months:

```
if q[i] > T for s consecutive months:
    regime_start = i
```

### Configuration

```yaml
regime:
  threshold: 1.0          # 100% increase required
  window: 6               # Months of trend data
  sustained_months: 2     # Consecutive months to confirm
```

### Algorithm Pseudocode

```python
def detect_regime_change(rates, config):
    n = len(rates)
    regime_start = 0

    i = config.window
    while i < n:
        # Fit window
        window = rates[i-config.window : i]
        qi, di, sigma = fit_exponential(window)

        # Project and threshold
        projected = qi * exp(-di * config.window)
        threshold = max(
            projected + config.n_sigma * sigma,
            projected * (1 + config.threshold)
        )

        if rates[i] > threshold:
            # Check sustained
            sustained = 1
            for j in range(i+1, i + config.sustained_months + 1):
                if j < n and rates[j] > threshold_at(j):
                    sustained += 1
                else:
                    break

            if sustained >= config.sustained_months:
                regime_start = i
                i += sustained
            else:
                i += 1
        else:
            i += 1

    return regime_start
```

### Calibration

Calibrate threshold using known events:

```bash
pyforecast calibrate-regime production.csv \
    --events known_events.csv \
    -o calibration.json
```

The calibration tests multiple thresholds and reports detection rates.

---

## Parameter Learning

### Purpose

After processing many wells, accumulated fit logs can be analyzed to suggest optimal fitting parameters for specific basins or formations.

### What Gets Learned

| Parameter | Description | Can Learn |
|-----------|-------------|-----------|
| `recency_half_life` | Data weighting | Yes |
| `regime_threshold` | Regime detection sensitivity | Yes |
| `regime_window` | Trend fitting window | Yes |
| `regime_sustained_months` | Confirmation period | Yes |
| `b_min`, `b_max` | B-factor bounds | No (physical) |
| `dmin` | Terminal decline | No (physical) |

Physical parameters are set based on reservoir physics, not learned.

### Learning Algorithm

#### Step 1: Accumulate Fit Logs

Each fit records:
- Fitting parameters used
- Fit quality ($R^2$, RMSE)
- Hindcast performance (MAPE, correlation)
- Context (basin, formation, product)

#### Step 2: Group Data

Group fits by:
- Basin + Formation + Product (if available)
- Basin + Product (fallback)
- Global (fallback)

#### Step 3: Performance-Weighted Average

For each group, compute weighted averages where weight = hindcast quality:

$$\bar{p} = \frac{\sum_i w_i \cdot p_i}{\sum_i w_i}$$

Where:
- $p_i$ = Parameter value for fit $i$
- $w_i$ = Weight based on hindcast MAPE

Weight function (lower MAPE = higher weight):

$$w_i = e^{-\alpha \cdot MAPE_i}$$

With $\alpha = 0.05$ typically.

#### Step 4: Confidence Assessment

| Sample Count | Confidence Level |
|--------------|------------------|
| < 20 | Low |
| 20 - 99 | Medium |
| ≥ 100 | High |

### Using Suggestions

```bash
# View suggestions
pyforecast suggest-params --basin "Permian" -p oil

# Output:
# Parameter Suggestion for Permian/oil
#   Based on 156 fits (confidence: high)
#
# Suggested values:
#   recency_half_life: 10.5
#   regime_threshold: 0.85
```

Apply to configuration:

```yaml
fitting:
  recency_half_life: 10.5

regime:
  threshold: 0.85
```

### Continuous Improvement Workflow

```bash
# 1. Process with logging and hindcast
pyforecast process batch.csv --log-fits --hindcast -o output/

# 2. After many batches, update suggestions
pyforecast suggest-params --update

# 3. View current suggestions
pyforecast suggest-params -p oil --basin "Permian"

# 4. Apply suggestions to future runs
pyforecast process new_batch.csv -c optimized_config.yaml -o output/
```

---

## Appendix: Mathematical Derivations

### Derivation of Switch Time

From the instantaneous decline rate:

$$D(t) = \frac{D_i}{1 + b \cdot D_i \cdot t}$$

Set $D(t_{switch}) = D_{min}$:

$$D_{min} = \frac{D_i}{1 + b \cdot D_i \cdot t_{switch}}$$

Solve for $t_{switch}$:

$$1 + b \cdot D_i \cdot t_{switch} = \frac{D_i}{D_{min}}$$

$$b \cdot D_i \cdot t_{switch} = \frac{D_i}{D_{min}} - 1$$

$$t_{switch} = \frac{D_i / D_{min} - 1}{b \cdot D_i}$$

### Derivation of Cumulative Production

Starting from:

$$N_p = \int_0^t q(\tau) d\tau = \int_0^t \frac{q_i}{(1 + b \cdot D_i \cdot \tau)^{1/b}} d\tau$$

Let $u = 1 + b \cdot D_i \cdot \tau$, then $du = b \cdot D_i \cdot d\tau$:

$$N_p = \frac{q_i}{b \cdot D_i} \int_1^{1+b \cdot D_i \cdot t} u^{-1/b} du$$

$$N_p = \frac{q_i}{b \cdot D_i} \cdot \frac{u^{1-1/b}}{1-1/b} \bigg|_1^{1+b \cdot D_i \cdot t}$$

$$N_p = \frac{q_i}{(1-b) \cdot D_i} \left[ (1 + b \cdot D_i \cdot t)^{\frac{b-1}{b}} - 1 \right]$$

Which is equivalent to:

$$N_p = \frac{q_i}{(1-b) \cdot D_i} \left[ 1 - (1 + b \cdot D_i \cdot t)^{\frac{b-1}{b}} \right]$$

### Exponential Limit

As $b \rightarrow 0$, using L'Hôpital's rule or Taylor expansion:

$$\lim_{b \to 0} (1 + b \cdot D_i \cdot t)^{1/b} = e^{D_i \cdot t}$$

Therefore:

$$\lim_{b \to 0} q(t) = q_i \cdot e^{-D_i \cdot t}$$
