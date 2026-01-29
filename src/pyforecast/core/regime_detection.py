"""Improved regime change detection for decline curve analysis.

Detects regime changes (RTP, refrac) by comparing production against
trend extrapolation, with adaptive thresholds based on data noise.
Requires sustained deviation to filter out single-month outliers.
"""

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass


@dataclass
class RegimeDetectionConfig:
    """Configuration for regime detection.

    Attributes:
        window_size: Number of months to use for trend fitting (default: 6)
        n_sigma: Number of standard deviations above projection to trigger (default: 2.5)
        min_pct_increase: Minimum percentage increase above projected to trigger (default: 0.5 = 50%)
        sustained_months: Consecutive months above threshold to confirm change (default: 2)
        min_data_points: Minimum points before attempting detection (default: 6)
    """
    window_size: int = 6
    n_sigma: float = 2.5
    min_pct_increase: float = 1.0  # 100% minimum increase required
    sustained_months: int = 2
    min_data_points: int = 6


def _fit_exponential_window(t: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """Fit exponential decline to a window of data.

    Args:
        t: Time array (relative)
        q: Production rate array

    Returns:
        Tuple of (qi, di, residual_std)
    """
    # Handle edge cases
    if len(t) < 3 or np.all(q <= 0):
        return q[0] if len(q) > 0 else 0, 0.01, np.std(q) if len(q) > 0 else 1.0

    # Filter positive values for log transform
    mask = q > 0
    if np.sum(mask) < 3:
        return q[0], 0.01, np.std(q)

    t_valid = t[mask]
    q_valid = q[mask]

    def exp_decline(t, qi, di):
        return qi * np.exp(-di * t)

    try:
        # Initial guess from log-linear regression
        log_q = np.log(q_valid)
        slope = (log_q[-1] - log_q[0]) / (t_valid[-1] - t_valid[0]) if t_valid[-1] != t_valid[0] else -0.05
        di_guess = max(0.001, -slope)
        qi_guess = q_valid[0]

        popt, _ = curve_fit(
            exp_decline,
            t_valid - t_valid[0],  # Normalize time to start at 0
            q_valid,
            p0=[qi_guess, di_guess],
            bounds=([0.1, 0.001], [qi_guess * 3, 2.0]),
            maxfev=1000
        )
        qi, di = popt

        # Calculate residuals
        predicted = exp_decline(t_valid - t_valid[0], qi, di)
        residuals = q_valid - predicted
        residual_std = np.std(residuals)

        return qi, di, max(residual_std, 1.0)  # Floor std at 1 to avoid division issues

    except (RuntimeError, ValueError):
        # Fallback to simple estimates
        return q_valid[0], 0.05, np.std(q_valid)


def _project_rate(qi: float, di: float, t_ahead: float) -> float:
    """Project rate forward from fitted parameters.

    Args:
        qi: Initial rate at t=0 of the fit window
        di: Decline rate
        t_ahead: Months ahead to project (from end of fit window)

    Returns:
        Projected rate
    """
    return qi * np.exp(-di * t_ahead)


def detect_regime_change_improved(
    rates: np.ndarray,
    config: RegimeDetectionConfig | None = None,
) -> int:
    """Detect the most recent regime change using trend extrapolation.

    A regime change is confirmed when:
    1. Production exceeds projected_rate + n_sigma * residual_std
    2. This elevated production is sustained for sustained_months

    Key: Once elevation is detected, we freeze the baseline and continue
    comparing against the pre-change trend (not a corrupted window).

    Args:
        rates: Production rates array (chronological order)
        config: Detection configuration

    Returns:
        Index of the start of the current regime (0 if no change detected)
    """
    if config is None:
        config = RegimeDetectionConfig()

    n = len(rates)

    if n < config.min_data_points:
        return 0

    # Track regime change candidates
    confirmed_regime_start = 0

    i = config.window_size
    while i < n:
        # Get window of data ending just before point i
        window_start = i - config.window_size
        window_end = i

        t_window = np.arange(config.window_size, dtype=float)
        q_window = rates[window_start:window_end]

        # Fit exponential to window (this is our baseline)
        qi, di, residual_std = _fit_exponential_window(t_window, q_window)

        # Project where rate should be at point i
        projected = _project_rate(qi, di, config.window_size)

        # Threshold is the HIGHER of:
        # 1. Statistical: projected + n_sigma * residual_std
        # 2. Percentage: projected * (1 + min_pct_increase)
        stat_threshold = projected + config.n_sigma * residual_std
        pct_threshold = projected * (1 + config.min_pct_increase)
        threshold = max(stat_threshold, pct_threshold)

        actual = rates[i]

        if actual > threshold:
            # Potential regime change detected at i
            # Now check if it's sustained by comparing subsequent points
            # against this SAME baseline (frozen)
            regime_start_candidate = i
            sustained_count = 1

            # Check subsequent points against the frozen baseline
            for j in range(i + 1, min(i + config.sustained_months + 2, n)):
                # Project further ahead using same baseline
                months_ahead = config.window_size + (j - i)
                projected_j = _project_rate(qi, di, months_ahead)

                # Same threshold logic: higher of statistical or percentage
                stat_threshold_j = projected_j + config.n_sigma * residual_std
                pct_threshold_j = projected_j * (1 + config.min_pct_increase)
                threshold_j = max(stat_threshold_j, pct_threshold_j)

                if rates[j] > threshold_j:
                    sustained_count += 1
                else:
                    # Dropped back below threshold
                    break

            if sustained_count >= config.sustained_months:
                # Confirmed regime change
                confirmed_regime_start = regime_start_candidate
                # Jump past this regime and continue scanning for more recent ones
                i = i + sustained_count
            else:
                # Not sustained, was just an outlier - skip past the spike
                i += 1
        else:
            i += 1

    return confirmed_regime_start


def _find_most_recent_regime(
    rates: np.ndarray,
    config: RegimeDetectionConfig,
) -> int:
    """Find the most recent sustained regime change.

    Scans backwards to find where the current production regime started.
    """
    n = len(rates)

    if n < config.min_data_points:
        return 0

    # Track state for each point: is it "elevated" relative to prior trend?
    elevated = np.zeros(n, dtype=bool)

    for i in range(config.window_size, n):
        window_start = i - config.window_size
        window_end = i

        t_window = np.arange(config.window_size, dtype=float)
        q_window = rates[window_start:window_end]

        qi, di, residual_std = _fit_exponential_window(t_window, q_window)
        projected = _project_rate(qi, di, config.window_size)
        threshold = projected + config.n_sigma * residual_std

        elevated[i] = rates[i] > threshold

    # Find runs of elevated points
    # A regime change is confirmed if we have sustained_months consecutive elevated points
    regime_start = 0
    i = n - 1

    while i >= config.window_size:
        if elevated[i]:
            # Found an elevated point, check if it's part of a sustained run
            run_end = i
            run_start = i

            while run_start > config.window_size and elevated[run_start - 1]:
                run_start -= 1

            run_length = run_end - run_start + 1

            if run_length >= config.sustained_months:
                # This is a confirmed regime change
                regime_start = run_start
                break

            i = run_start - 1
        else:
            i -= 1

    return regime_start


def analyze_regime_detection(
    rates: np.ndarray,
    config: RegimeDetectionConfig | None = None,
) -> dict:
    """Analyze regime detection with detailed diagnostics.

    Useful for understanding why a regime change was or wasn't detected.

    Args:
        rates: Production rates array
        config: Detection configuration

    Returns:
        Dictionary with detection details
    """
    if config is None:
        config = RegimeDetectionConfig()

    n = len(rates)
    results = {
        "regime_start_idx": 0,
        "points_analyzed": n,
        "config": {
            "window_size": config.window_size,
            "n_sigma": config.n_sigma,
            "sustained_months": config.sustained_months,
        },
        "point_analysis": []
    }

    if n < config.min_data_points:
        results["status"] = "insufficient_data"
        return results

    for i in range(config.window_size, n):
        window_start = i - config.window_size
        t_window = np.arange(config.window_size, dtype=float)
        q_window = rates[window_start:i]

        qi, di, residual_std = _fit_exponential_window(t_window, q_window)
        projected = _project_rate(qi, di, config.window_size)
        threshold = projected + config.n_sigma * residual_std
        actual = rates[i]

        results["point_analysis"].append({
            "index": i,
            "actual": float(actual),
            "projected": float(projected),
            "threshold": float(threshold),
            "residual_std": float(residual_std),
            "is_elevated": bool(actual > threshold),
            "deviation_sigma": float((actual - projected) / residual_std) if residual_std > 0 else 0,
        })

    results["regime_start_idx"] = detect_regime_change_improved(rates, config)
    results["status"] = "regime_detected" if results["regime_start_idx"] > 0 else "no_regime_change"

    return results
