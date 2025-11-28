"""
Mean Reversion & Volatility Dynamics of the VIX–VSTOXX Spread
Compressed Professional Version (Option 2)
Author: Andreas Lindgren

Pipeline:
- Download VIX (FRED / Yahoo) and VSTOXX (STOXX TXT)
- Construct spread = VIX - VSTOXX
- Stationarity tests (ADF, PP)
- AR(p) selection & estimation
- ARCH-LM test, GARCH(1,1) on residuals
- Z-score mean-reversion toy strategy
- Rolling AR forecast & evaluation
- Result tables (CSV + Excel workbook)
"""

# ============================================================
# 1. IMPORTS & BASIC SETTINGS
# ============================================================

import os
import warnings
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from arch import arch_model
from arch.unitroot import PhillipsPerron

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.display.float_format = "{:.4f}".format
plt.style.use("seaborn-v0_8-darkgrid")


# ============================================================
# 2. CONFIG
# ============================================================

VSTOXX_PATH = r"C:\Users\andre\OneDrive\Skrivbord\econometrics\vstoxx_1m.txt"
VIX_SOURCE = "fred"        # {"fred", "yfinance"}
SAMPLE_START = "2006-01-01"
SAMPLE_END = None          # latest available


# ============================================================
# 3. SMALL GENERIC HELPERS
# ============================================================

def clean_series(s: pd.Series) -> pd.Series:
    """
    Generic cleaning for time series:
    - ensure Series
    - drop +/- inf and NaN
    - cast to float
    """
    return (
        pd.Series(s)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )


def print_header(title: str) -> None:
    """Pretty section header for console output."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


# ============================================================
# 4. VIX DOWNLOAD FUNCTIONS
# ============================================================

def download_vix_fred(start: str = "1990-01-01",
                      end: Optional[str] = None) -> pd.DataFrame:
    """
    Download VIX from FRED (VIXCLS).
    Returns DataFrame with Date index and column 'VIX'.
    """
    print_header("Step 1: Downloading VIX from FRED (VIXCLS)")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    df = pd.read_csv(url)

    if df.empty:
        raise ValueError("FRED VIX CSV download failed or returned empty.")

    # Handle possible date column variants
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    if date_col not in df.columns or "VIXCLS" not in df.columns:
        raise ValueError(f"Unexpected columns in FRED CSV: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["VIX"] = pd.to_numeric(df["VIXCLS"].replace(".", np.nan), errors="coerce")
    df = df.dropna(subset=[date_col, "VIX"]).set_index(date_col).sort_index()

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]

    if df.empty:
        raise ValueError("No VIX data in selected sample from FRED.")

    print(
        f"FRED VIX: {len(df)} obs from {df.index.min().date()} "
        f"to {df.index.max().date()}."
    )
    return df[["VIX"]]


def download_vix_yfinance(start: str = "1990-01-01",
                          end: Optional[str] = None) -> pd.DataFrame:
    """
    Download VIX (^VIX) from Yahoo via yfinance.
    Returns DataFrame with Date index and column 'VIX'.
    """
    print_header("Step 1: Downloading VIX from Yahoo Finance (^VIX)")
    data = yf.download("^VIX", start=start, end=end, progress=False)

    if data.empty:
        raise ValueError("yfinance VIX download failed or returned empty.")

    # Prefer Close, fall back gently
    for col in ["Close", "Adj Close", "Price"]:
        if col in data.columns:
            vix = data[col]
            break
    else:
        raise ValueError(f"Could not find price column in VIX data: {data.columns}")

    vix = vix.to_frame("VIX").dropna()
    vix.index = pd.to_datetime(vix.index)
    vix = vix.sort_index()

    print(
        f"Yahoo VIX: {len(vix)} obs from {vix.index.min().date()} "
        f"to {vix.index.max().date()}."
    )
    return vix


def get_vix(start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Wrapper choosing VIX source based on global VIX_SOURCE."""
    if VIX_SOURCE.lower() == "fred":
        return download_vix_fred(start, end)
    elif VIX_SOURCE.lower() == "yfinance":
        return download_vix_yfinance(start, end)
    else:
        raise ValueError("VIX_SOURCE must be 'fred' or 'yfinance'.")


# ============================================================
# 5. VSTOXX LOADER
# ============================================================

def load_vstoxx(path: str) -> pd.DataFrame:
    """
    Load VSTOXX 1M from STOXX EUR Price TXT:
        DD.MM.YYYY;CODE;VALUE
    Return DataFrame with Date index and 'VSTOXX' column.
    """
    print_header("Step 2: Loading VSTOXX from file")

    if not os.path.exists(path):
        raise FileNotFoundError(f"VSTOXX file not found at: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["Date", "Code", "VSTOXX"],
        encoding="utf-8",
    )

    if df.empty:
        raise ValueError("VSTOXX file is empty.")

    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["VSTOXX"] = pd.to_numeric(df["VSTOXX"], errors="coerce")
    df = df.dropna(subset=["Date", "VSTOXX"]).set_index("Date").sort_index()

    if df.empty:
        raise ValueError("No valid VSTOXX observations after cleaning.")

    print(
        f"VSTOXX: {len(df)} obs from {df.index.min().date()} "
        f"to {df.index.max().date()}."
    )
    return df[["VSTOXX"]]


# ============================================================
# 6. SPREAD CONSTRUCTION & BASIC PLOT
# ============================================================

def construct_spread(vix: pd.DataFrame,
                     vstoxx: pd.DataFrame,
                     start: Optional[str] = None,
                     end: Optional[str] = None) -> pd.DataFrame:
    """
    Inner-join VIX & VSTOXX and construct:
        Spread = VIX - VSTOXX
    Return DataFrame with ['VIX', 'VSTOXX', 'Spread'].
    """
    print_header("Step 3: Aligning VIX and VSTOXX, constructing spread")
    df = vix.join(vstoxx, how="inner")

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise ValueError("No overlapping VIX-VSTOXX data in chosen sample window.")

    df["Spread"] = df["VIX"] - df["VSTOXX"]

    print(
        f"Aligned sample: {len(df)} days "
        f"from {df.index.min().date()} to {df.index.max().date()}."
    )
    print(
        f"Spread summary: "
        f"mean={df['Spread'].mean():.3f}, "
        f"std={df['Spread'].std():.3f}, "
        f"min={df['Spread'].min():.3f}, "
        f"max={df['Spread'].max():.3f}"
    )
    return df


def plot_indices_and_spread(df: pd.DataFrame) -> None:
    """Plot VIX, VSTOXX, and their spread (two subplots)."""
    print_header("Step 4: Plotting VIX, VSTOXX and spread")

    if df.empty:
        raise ValueError("No data to plot.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: levels
    axes[0].plot(df.index, df["VIX"], label="VIX (US, 30d)")
    axes[0].plot(df.index, df["VSTOXX"], label="VSTOXX (EU, 30d)")
    axes[0].set_ylabel("Implied Volatility (%)")
    axes[0].set_title("VIX vs VSTOXX (30d implied volatility)")
    axes[0].legend()

    # Bottom: spread
    axes[1].plot(df.index, df["Spread"], label="Spread = VIX - VSTOXX")
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Volatility Points")
    axes[1].set_title("VIX–VSTOXX Spread")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# =====================================================================
# Summary Statistics + Significance Tests for VIX, VSTOXX, Spread
# =====================================================================

from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron
import pandas as pd
import numpy as np

def compute_full_summary(vix, vstoxx):

    df = pd.DataFrame({
        "VIX": vix,
        "VSTOXX": vstoxx
    }).dropna()

    df["Spread"] = df["VIX"] - df["VSTOXX"]

    # 1. t-test for mean difference
    t_stat, t_p = stats.ttest_1samp(df["Spread"], popmean=0)

    # 2. F-test for variance difference
    f_stat = df["VIX"].var() / df["VSTOXX"].var()
    df1 = len(df["VIX"]) - 1
    df2 = len(df["VSTOXX"]) - 1
    f_p = stats.f.cdf(f_stat, df1, df2)
    f_p = 2 * min(f_p, 1 - f_p)

    # 3. Skewness & kurtosis tests
    skew_v, skew_p = stats.skewtest(df["Spread"])
    kurt_v, kurt_p = stats.kurtosistest(df["Spread"])

    # 4. ARCH LM Test
    arch_stat, arch_p, _, _ = het_arch(df["Spread"])

    # 5. ADF + PP Unit Root Tests
    adf_stat, adf_p, _, _, _, _ = adfuller(df["Spread"])

    pp = PhillipsPerron(df["Spread"])
    pp_stat = pp.stat
    pp_p = pp.pvalue

    # 6. Basic descriptive statistics
    desc = df["Spread"].describe()

    summary_table = pd.DataFrame({
        "Statistic": [
            "Mean (Spread)",
            "Std Dev (Spread)",
            "Min",
            "Max",
            "Skewness",
            "Skewness p-value",
            "Kurtosis",
            "Kurtosis p-value",
            "Mean Diff t-stat",
            "Mean Diff p-value",
            "Variance F-stat",
            "Variance F-test p-value",
            "ARCH LM stat",
            "ARCH LM p-value",
            "ADF stat",
            "ADF p-value",
            "PP stat",
            "PP p-value"
        ],
        "Value": [
            desc["mean"],
            desc["std"],
            desc["min"],
            desc["max"],
            skew_v,
            skew_p,
            kurt_v,
            kurt_p,
            t_stat,
            t_p,
            f_stat,
            f_p,
            arch_stat,
            arch_p,
            adf_stat,
            adf_p,
            pp_stat,
            pp_p
        ]
    })

    return df, summary_table
# =====================================================================
# ADDITIONAL CORRELATION ANALYSIS
# =====================================================================

def compute_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute:
    1) Pearson correlation (levels: VIX vs VSTOXX)
    2) Pearson correlation (first differences)
    3) Rolling correlations (60d, 120d, 252d)
    """
    # Ensure clean data
    df = df[["VIX", "VSTOXX", "Spread"]].dropna()

    # 1) Pearson correlation — LEVELS
    corr_levels = df["VIX"].corr(df["VSTOXX"])

    # 2) Pearson correlation — FIRST DIFFERENCES
    dvix = df["VIX"].diff().dropna()
    dvstoxx = df["VSTOXX"].diff().dropna()
    corr_diffs = dvix.corr(dvstoxx)

    # 3) Rolling correlations
    rolling_corr = pd.DataFrame(index=df.index)
    rolling_corr["Rolling_60d"] = df["VIX"].rolling(60).corr(df["VSTOXX"])
    rolling_corr["Rolling_120d"] = df["VIX"].rolling(120).corr(df["VSTOXX"])
    rolling_corr["Rolling_252d"] = df["VIX"].rolling(252).corr(df["VSTOXX"])

    # Build summary correlation table
    corr_table = pd.DataFrame({
        "Metric": [
            "Correlation (Levels: VIX vs VSTOXX)",
            "Correlation (1st Diff: ΔVIX vs ΔVSTOXX)"
        ],
        "Value": [
            corr_levels,
            corr_diffs
        ]
    })

    return corr_table, rolling_corr

# ============================================================
# 7. STATIONARITY TESTS: ADF & PP
# ============================================================

def adf_test_spread(spread: pd.Series) -> Dict[str, Any]:
    """
    Augmented Dickey–Fuller test on spread.
    H0: unit root (non-stationary)
    H1: stationary (mean-reverting)
    """
    print_header("Step 5: ADF test on spread (mean reversion check)")
    y = clean_series(spread)
    if y.empty:
        raise ValueError("Spread is empty or all NaN in ADF test.")

    result = adfuller(y, maxlag=20, regression="c", autolag="AIC")
    adf_stat, p_value, used_lag, nobs, crit_vals, icbest = result

    print("ADF Test Results (Spread):")
    print(f"  ADF statistic      : {adf_stat:.4f}")
    print(f"  p-value            : {p_value:.4f}")
    print(f"  # of lags used     : {used_lag}")
    print(f"  # of observations  : {nobs}")
    print("  Critical values    :")
    for level, cv in crit_vals.items():
        print(f"    {level}%: {cv:.4f}")
    print(f"  Best IC (AIC)      : {icbest:.4f}")

    if p_value < 0.05:
        print("\nConclusion: reject H0 → spread is STATIONARY (ADF).\n")
    else:
        print("\nConclusion: cannot reject H0 → no strong ADF evidence of stationarity.\n")

    return {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "nobs": nobs,
        "critical_values": crit_vals,
        "icbest": icbest,
    }


def pp_test_spread(spread: pd.Series) -> Dict[str, Any]:
    """
    Phillips–Perron test on spread.
    H0: unit root (non-stationary)
    H1: stationary.
    """
    print_header("Step 5b: Phillips–Perron (PP) test on spread")
    y = clean_series(spread)
    if y.empty:
        raise ValueError("Spread is empty or all NaN in PP test.")

    pp = PhillipsPerron(y)
    stat = float(pp.stat)
    pval = float(pp.pvalue)
    crit = pp.critical_values
    nobs = getattr(pp, "nobs", None)
    lags = getattr(pp, "lags", None)

    print("PP Test Results (Spread):")
    print(f"  PP statistic   : {stat:.4f}")
    print(f"  p-value        : {pval:.4f}")
    print("  Critical values:")
    for level, cv in crit.items():
        print(f"    {level}: {cv:.4f}")

    if pval < 0.05:
        print("\nConclusion: reject H0 → spread is STATIONARY (PP).\n")
    else:
        print("\nConclusion: cannot reject H0 → no strong PP evidence of stationarity.\n")

    return {
        "pp_statistic": stat,
        "p_value": pval,
        "critical_values": crit,
        "nobs": nobs,
        "lags": lags,
    }
# ============================================================
# 8. AR(p) MEAN MODEL
# ============================================================

def fit_ar_model(spread: pd.Series,
                 max_p: int = 6,
                 ic: str = "aic") -> Tuple[Any, int, pd.Series]:
    """
    Fit AR(p) with p in 1..max_p using ARIMA(p,0,0) and IC selection.
    Returns (fitted_result, best_p, residual_series).
    """
    print_header("Step 6: AR(p) model selection and estimation")

    ic = ic.lower()
    if ic not in {"aic", "bic"}:
        raise ValueError("ic must be 'aic' or 'bic'.")

    y = clean_series(spread)
    if y.empty:
        raise ValueError("Spread is empty in fit_ar_model.")

    best_p, best_ic, best_res = None, np.inf, None

    for p in range(1, max_p + 1):
        try:
            model = ARIMA(y, order=(p, 0, 0), trend="c")
            res = model.fit()
            value = res.aic if ic == "aic" else res.bic
            print(f"  AR({p}) {ic.upper()} = {value:.2f}")
            if value < best_ic:
                best_ic, best_p, best_res = value, p, res
        except Exception as e:
            print(f"  Warning: AR({p}) failed: {e}")

    if best_res is None:
        raise RuntimeError("AR(p) search failed for all p.")

    print(f"\nSelected AR({best_p}) with min {ic.upper()} = {best_ic:.2f}\n")

    # Print AR coefficients and roots
    ar_params = best_res.arparams
    print("Estimated AR coefficients:")
    for i, phi in enumerate(ar_params, start=1):
        print(f"  phi_{i} = {phi:.4f}")

    if hasattr(best_res, "arroots") and len(best_res.arroots) > 0:
        roots = best_res.arroots
        min_abs_root = np.min(np.abs(roots))
        print("\nAR polynomial roots (|z|):")
        for i, r in enumerate(roots, start=1):
            print(f"  root_{i}: |z|={np.abs(r):.4f}")
        if min_abs_root > 1:
            print("\nAll AR roots outside unit circle → STABLE, MEAN-REVERTING.\n")
        else:
            print("\nSome AR roots inside unit circle → possible NON-STATIONARITY.\n")
    else:
        print("\n(No AR roots reported; cannot verify stationarity explicitly.)\n")

    resid = pd.Series(best_res.resid, index=y.index, name="ar_resid")
    print("AR(p) estimation complete. Residuals ready for ARCH/GARCH.\n")
    return best_res, best_p, resid


# ============================================================
# 9. ARCH-LM TEST
# ============================================================

def arch_lm_test(residuals: pd.Series, lags: int = 12) -> Dict[str, Any]:
    """
    ARCH-LM test for conditional heteroskedasticity.
    H0: no ARCH effects.
    H1: ARCH effects (volatility clustering).
    """
    print_header("Step 7: ARCH-LM test on AR residuals")

    if residuals is None:
        raise ValueError("Residuals is None in arch_lm_test.")

    eps = clean_series(residuals)
    if eps.empty:
        raise ValueError("Residuals empty after cleaning in arch_lm_test.")

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(eps, nlags=lags)

    print("ARCH-LM Test Results:")
    print(f"  Lags tested         : {lags}")
    print(f"  Observations (clean): {len(eps)}")
    print(f"  LM statistic        : {lm_stat:.4f}")
    print(f"  LM p-value          : {lm_pvalue:.4f}")
    print(f"  F  statistic        : {f_stat:.4f}")
    print(f"  F  p-value          : {f_pvalue:.4f}")

    if lm_pvalue < 0.05:
        print("\nConclusion: reject H0 → ARCH effects present → volatility clustering.\n")
    else:
        print("\nConclusion: cannot reject H0 → weak evidence of volatility clustering.\n")

    return {
        "lm_stat": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
        "lags": int(lags),
        "nobs": int(len(eps)),
    }


# ============================================================
# 10. GARCH(1,1) ON RESIDUALS
# ============================================================

def fit_garch_on_resid(residuals: pd.Series,
                       p: int = 1,
                       q: int = 1,
                       dist: str = "t") -> Tuple[Any, pd.Series, pd.Series]:
    """
    Fit GARCH(p,q) with chosen innovation distribution to AR residuals.
    Returns (fitted_model, conditional_vol, standardized_resid).
    """
    print_header(f"Step 8: Fitting GARCH({p},{q}) on AR residuals (dist='{dist}')")

    if residuals is None:
        raise ValueError("Residuals is None in fit_garch_on_resid.")

    eps = clean_series(residuals)
    if eps.empty:
        raise ValueError("Residuals empty after cleaning in fit_garch_on_resid.")

    am = arch_model(eps, mean="zero", vol="GARCH", p=p, q=q, dist=dist)
    res = am.fit(disp="off", show_warning=False)

    print("GARCH parameter estimates:")
    print(res.params.to_string())

    params = res.params
    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha[1]", params.get("alpha1", np.nan)))
    beta = float(params.get("beta[1]", params.get("beta1", np.nan)))
    persistence = alpha + beta

    print("\nKey GARCH stats:")
    print(f"  omega (constant)     : {omega:.6f}")
    print(f"  alpha_1 (ARCH)       : {alpha:.4f}")
    print(f"  beta_1  (GARCH)      : {beta:.4f}")
    print(f"  persistence α+β      : {persistence:.4f}")

    if persistence < 1:
        half_life = np.log(0.5) / np.log(persistence)
        print("  variance is STATIONARY (α+β < 1).")
        print(f"  shock half-life      : {half_life:.1f} periods")
    else:
        print("  variance NON-STATIONARY or near unit (α+β ≥ 1). Very persistent.")

    cond_vol = pd.Series(res.conditional_volatility, index=eps.index, name="garch_sigma")
    std_resid = pd.Series(res.std_resid, index=eps.index, name="garch_std_resid")

    print("\nLjung-Box on standardized residuals (serial correlation):")
    for lag in (5, 10, 20):
        lb = acorr_ljungbox(std_resid.dropna(), lags=[lag], return_df=True)
        print(f"  lag {lag}: Q={lb['lb_stat'].iloc[0]:.2f}, p={lb['lb_pvalue'].iloc[0]:.4f}")

    print("Ljung-Box on squared standardized residuals (remaining ARCH):")
    for lag in (5, 10, 20):
        lb2 = acorr_ljungbox((std_resid.dropna() ** 2), lags=[lag], return_df=True)
        print(f"  lag {lag}: Q={lb2['lb_stat'].iloc[0]:.2f}, p={lb2['lb_pvalue'].iloc[0]:.4f}")

    print("\nGARCH(1,1) estimation complete.\n")
    return res, cond_vol, std_resid


def plot_garch_volatility(cond_vol: pd.Series,
                          title: str = "Conditional Volatility (GARCH)") -> None:
    """Plot GARCH conditional volatility over time."""
    if cond_vol is None or cond_vol.empty:
        print("Warning: No conditional volatility to plot.")
        return
    plt.figure(figsize=(12, 4))
    plt.plot(cond_vol.index, cond_vol.values, label="σ_t (GARCH)")
    plt.title(title)
    plt.ylabel("Volatility (σ)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 11. Z-SCORE STRATEGY & BACKTEST
# ============================================================

def compute_zscore(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Rolling z-score of the spread:
        z_t = (Spread_t - rolling_mean_t) / rolling_std_t
    """
    x = clean_series(spread)
    mu = x.rolling(lookback, min_periods=lookback // 2).mean()
    sd = x.rolling(lookback, min_periods=lookback // 2).std(ddof=0)
    z = (x - mu) / sd
    return z.rename(f"z_{lookback}")


def backtest_zscore_strategy(spread: pd.Series,
                             lookback: int = 60,
                             z_entry: float = 1.5,
                             z_exit: float = 0.5,
                             cost_bp: float = 5.0) -> pd.DataFrame:
    """
    Toy mean-reversion strategy on the spread with positions in {-1,0,+1}.
    - Go long when z < -z_entry
    - Go short when z >  z_entry
    - Exit when |z| < z_exit
    PnL is based on change in spread. Transaction costs in bp per position change.
    """
    z = compute_zscore(spread, lookback=lookback).dropna()
    if z.empty:
        raise ValueError("No z-score values available for backtest.")

    pos = pd.Series(0.0, index=z.index)

    long_sig = z < -z_entry
    short_sig = z > z_entry
    exit_sig = z.abs() < z_exit

    state = 0.0
    for t in z.index:
        if state == 0.0:
            if long_sig.loc[t]:
                state = +1.0
            elif short_sig.loc[t]:
                state = -1.0
        else:
            if exit_sig.loc[t]:
                state = 0.0
            elif state == +1.0 and short_sig.loc[t]:
                state = -1.0
            elif state == -1.0 and long_sig.loc[t]:
                state = +1.0
        pos.loc[t] = state

    pos = pos.reindex(spread.index).ffill().fillna(0.0)

    dS = spread.diff()
    ret_raw = pos.shift(1) * dS

    dpos = pos.diff().abs().fillna(0.0)
    cost = (cost_bp / 1e4) * dpos
    ret_net = ret_raw - cost

    equity = ret_net.cumsum().rename("equity")

    out = pd.DataFrame({
        "spread": spread,
        "z": z.reindex(spread.index),
        "pos": pos,
        "dS": dS,
        "ret_raw": ret_raw,
        "cost": cost,
        "ret": ret_net,
        "equity": equity,
    })

    valid = out["ret"].dropna()
    if valid.std(ddof=0) > 0:
        sharpe = valid.mean() / valid.std(ddof=0) * np.sqrt(252)
    else:
        sharpe = np.nan

    trades = int((dpos > 0).sum())
    print_header("Z-Score Backtest (Toy Mean-Reversion on Spread)")
    print(f"  lookback      : {lookback}")
    print(f"  entry z-score : {z_entry}")
    print(f"  exit  z-score : {z_exit}")
    print(f"  cost          : {cost_bp} bp per trade")
    print(f"  trades        : {trades}")
    print(f"  mean daily ret: {valid.mean():.4f}")
    print(f"  stdev daily   : {valid.std(ddof=0):.4f}")
    print(f"  Sharpe (ann.) : {sharpe:.2f}")
    print(f"  final equity  : {equity.iloc[-1]:.2f}\n")

    return out


def plot_strategy_equity(bt_df: pd.DataFrame,
                         title: str = "Z-Score Strategy Equity") -> None:
    """Plot cumulative PnL of the z-score strategy."""
    if bt_df is None or bt_df.empty:
        print("Warning: No backtest results to plot.")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(bt_df.index, bt_df["equity"], label="Equity (cum PnL)")
    plt.title(title)
    plt.ylabel("Cumulative PnL (spread points)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_backtest_stats(bt_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute simple performance statistics from backtest output.
    """
    if bt_df is None or bt_df.empty:
        return {}

    valid = bt_df["ret"].dropna()
    dpos = bt_df["pos"].diff().abs().fillna(0.0)
    trades = int((dpos > 0).sum())

    if valid.std(ddof=0) > 0:
        sharpe = valid.mean() / valid.std(ddof=0) * np.sqrt(252)
    else:
        sharpe = np.nan

    return {
        "n_obs": len(valid),
        "mean_daily_ret": valid.mean(),
        "std_daily_ret": valid.std(ddof=0),
        "sharpe_annual": sharpe,
        "trades": trades,
        "final_equity": bt_df["equity"].iloc[-1],
    }
# ============================================================
# 12. ROLLING AR FORECAST
# ============================================================

def rolling_ar_forecast(spread: pd.Series,
                        p: int,
                        window: int = 90,
                        max_obs: Optional[int] = 1500) -> pd.DataFrame:
    """
    Rolling 1-step-ahead AR(p) forecast:
    - For each t >= window, fit AR(p) on last 'window' observations.
    - Forecast Spread_{t+1}.
    Return DataFrame with columns: ['Spread', 'AR_forecast_1d'].
    """
    print_header(f"Step 9: Rolling AR({p}) 1-step forecast (window={window})")

    y = clean_series(spread)
    if max_obs is not None and len(y) > max_obs:
        y = y.iloc[-max_obs:]

    if len(y) <= window + p:
        raise ValueError(
            f"Not enough data for rolling AR forecast "
            f"(need > {window + p}, have {len(y)})."
        )

    dates = y.index
    forecasts, fc_dates = [], []

    for i in range(window, len(y) - 1):
        y_win = y.iloc[i - window:i]

        try:
            model = AutoReg(y_win, lags=p, trend="c", old_names=False)
            res = model.fit()
            params = res.params

            if "const" in params.index:
                const = params["const"]
                coefs = params.drop("const").values
            else:
                const = params.iloc[0]
                coefs = params.iloc[1:].values

            # Last p values up to time i (exclusive), reversed for dot product
            y_lags = y.iloc[i - p:i].iloc[::-1].values
            fc_val = const + float(np.dot(coefs, y_lags))

            forecasts.append(fc_val)
            fc_dates.append(dates[i + 1])

        except Exception as e:
            print(f"  Warning: Rolling AR window ending {dates[i].date()} failed: {e}")

    if not forecasts:
        print("Warning: No successful windows in rolling_ar_forecast.")
        return pd.DataFrame()

    fc_series = pd.Series(forecasts, index=pd.DatetimeIndex(fc_dates), name="AR_forecast_1d")
    realized = y.reindex(fc_series.index)

    result = pd.DataFrame({"Spread": realized, "AR_forecast_1d": fc_series}).dropna()
    print(f"Rolling AR forecast generated for {len(result)} days.")
    return result


def plot_forecast(fc_df: pd.DataFrame,
                  title: str = "Rolling AR Forecast (1-step)") -> None:
    """Plot actual spread vs rolling AR 1-step ahead forecast."""
    print_header("Step 10: Plotting rolling AR forecast vs actual spread")

    if fc_df is None or fc_df.empty:
        print("Warning: Forecast DataFrame empty – nothing to plot.")
        return

    v = fc_df.dropna()
    if v.empty:
        print("Warning: All forecast rows are NaN – nothing to plot.")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(v.index, v["Spread"], label="Spread (actual)")
    plt.plot(v.index, v["AR_forecast_1d"], label="AR Forecast (1-step)", alpha=0.8)
    plt.title(title)
    plt.ylabel("Spread")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 13. FORECAST EVALUATION
# ============================================================

def evaluate_forecast(df_fc: pd.DataFrame,
                      actual_col: str = "Spread",
                      fc_col: str = "AR_forecast_1d") -> Dict[str, Any]:
    """
    Compute MSE, MAE and directional hit-rate from rolling AR forecasts.
    """
    if df_fc is None:
        print("Warning: df_fc is None in evaluate_forecast.")
        return {}
    if not isinstance(df_fc, pd.DataFrame):
        print(f"Warning: df_fc is not DataFrame (type={type(df_fc)}).")
        return {}
    if df_fc.empty:
        print("Warning: df_fc empty in evaluate_forecast.")
        return {}
    if actual_col not in df_fc.columns or fc_col not in df_fc.columns:
        print(
            "Warning: expected columns "
            f"'{actual_col}' and '{fc_col}' not found. "
            f"Available: {list(df_fc.columns)}"
        )
        return {}

    valid = df_fc.dropna(subset=[actual_col, fc_col]).copy()
    if valid.empty:
        print("Warning: no non-NaN observations in evaluate_forecast after dropna.")
        return {}

    err = valid[actual_col] - valid[fc_col]
    mse = (err ** 2).mean()
    mae = err.abs().mean()

    # Directional accuracy: compare sign of actual vs predicted change
    sign_actual = np.sign(valid[actual_col].diff())
    sign_fc = np.sign(valid[fc_col] - valid[actual_col].shift(1))

    da = sign_actual.dropna()
    df_ = sign_fc.dropna()
    common = da.index.intersection(df_.index)

    if common.empty:
        hit_rate = np.nan
        directional_hits = 0
    else:
        da = da.loc[common]
        df_ = df_.loc[common]
        hits = (da == df_).sum()
        hit_rate = (da == df_).mean()
        directional_hits = int(hits)

    stats = {
        "n_obs": len(valid),
        "mse": mse,
        "mae": mae,
        "directional_hits": directional_hits,
        "directional_hit_rate": hit_rate,
    }

    print_header("Rolling AR Forecast Evaluation")
    print(f"  n_obs               : {stats['n_obs']}")
    print(f"  MSE                 : {stats['mse']:.4f}")
    print(f"  MAE                 : {stats['mae']:.4f}")
    print(f"  Directional hits    : {stats['directional_hits']}")
    print(f"  Directional hit-rate: {stats['directional_hit_rate']:.2%}\n")

    return stats


# ============================================================
# 14. SUMMARY TABLES
# ============================================================

def summarize_results(
    adf_results: Dict[str, Any],
    pp_results: Dict[str, Any],
    ar_results: Any,
    best_p: int,
    arch_results: Dict[str, Any],
    garch_res: Any,
    backtest_stats: Dict[str, Any],
    forecast_stats: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Build pandas DataFrame tables summarizing the main empirical results.
    """
    # 1) Stationarity: ADF & PP
    station_rows = []
    if adf_results:
        station_rows.append({
            "Test": "ADF",
            "Statistic": adf_results.get("adf_statistic"),
            "p_value": adf_results.get("p_value"),
            "lags_used": adf_results.get("used_lag"),
            "n_obs": adf_results.get("nobs"),
            "IC_best": adf_results.get("icbest"),
            "Conclusion": "Reject unit root (stationary)"
            if adf_results.get("p_value", 1.0) < 0.05 else
               "Cannot reject unit root",
        })
    if pp_results:
        station_rows.append({
            "Test": "PP",
            "Statistic": pp_results.get("pp_statistic"),
            "p_value": pp_results.get("p_value"),
            "lags_used": pp_results.get("lags"),
            "n_obs": pp_results.get("nobs"),
            "IC_best": None,
            "Conclusion": "Reject unit root (stationary)"
            if pp_results.get("p_value", 1.0) < 0.05 else
               "Cannot reject unit root",
        })
    stationarity_results = pd.DataFrame(station_rows)

    # 2) AR(p) model
    ar_rows = []
    if ar_results is not None:
        phi = getattr(ar_results, "arparams", np.array([]))
        roots = getattr(ar_results, "arroots", np.array([]))
        min_abs_root = np.min(np.abs(roots)) if roots.size > 0 else np.nan
        stable = bool(min_abs_root > 1) if np.isfinite(min_abs_root) else None

        row = {
            "Selected_p": best_p,
            "AIC": ar_results.aic,
            "BIC": ar_results.bic,
            "min_|root|": min_abs_root,
            "Stable_AR": stable,
        }
        for i, coeff in enumerate(phi, start=1):
            row[f"phi_{i}"] = coeff
        ar_rows.append(row)
    ar_model_results = pd.DataFrame(ar_rows)

    # 3) ARCH-LM test
    arch_rows = []
    if arch_results:
        arch_rows.append({
            "lags_tested": arch_results.get("lags"),
            "LM_stat": arch_results.get("lm_stat"),
            "LM_p_value": arch_results.get("lm_pvalue"),
            "F_stat": arch_results.get("f_stat"),
            "F_p_value": arch_results.get("f_pvalue"),
            "n_obs": arch_results.get("nobs"),
            "Conclusion": "Reject no-ARCH (vol clustering)"
            if arch_results.get("lm_pvalue", 1.0) < 0.05 else
               "Cannot reject no-ARCH",
        })
    arch_lm_results = pd.DataFrame(arch_rows)

    # 4) GARCH(1,1) summary
    garch_rows = []
    if garch_res is not None:
        params = garch_res.params
        alpha1 = params.get("alpha[1]", params.get("alpha1", np.nan))
        beta1 = params.get("beta[1]", params.get("beta1", np.nan))
        persistence = alpha1 + beta1 if np.isfinite(alpha1) and np.isfinite(beta1) else np.nan
        stationary_var = bool(persistence < 1) if np.isfinite(persistence) else None

        if np.isfinite(persistence) and 0 < persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.nan

        garch_rows.append({
            "omega": params.get("omega", np.nan),
            "alpha_1": alpha1,
            "beta_1": beta1,
            "alpha+beta": persistence,
            "Stationary_variance": stationary_var,
            "Half_life_days": half_life,
            "nu_df": params.get("nu", np.nan),
            "distribution": getattr(getattr(garch_res.model, "distribution", None), "name", None),

        })
    garch_results = pd.DataFrame(garch_rows)

    # 5) Backtest results
    backtest_results = pd.DataFrame([backtest_stats]) if backtest_stats else pd.DataFrame()

    # 6) Forecast results
    forecast_results = pd.DataFrame([forecast_stats]) if forecast_stats else pd.DataFrame()

    return {
        "stationarity_results": stationarity_results,
        "ar_model_results": ar_model_results,
        "arch_lm_results": arch_lm_results,
        "garch_results": garch_results,
        "backtest_results": backtest_results,
        "forecast_results": forecast_results,
    }


# ============================================================
# 15. MAIN PIPELINE
# ============================================================

def main() -> None:
    print_header("VIX–VSTOXX PROJECT: FULL PIPELINE")
    print(f"Using VIX source: {VIX_SOURCE.upper()}")
    print(f"Sample window   : {SAMPLE_START} to {SAMPLE_END or 'latest available'}")

    # 1) VIX
    vix = get_vix(start="1990-01-01", end=SAMPLE_END)
    print("VIX sample (head):")
    print(vix.head(), "\n")

    # 2) VSTOXX
    if not os.path.exists(VSTOXX_PATH):
        print(f"WARNING: VSTOXX file not found at:\n  {VSTOXX_PATH}")
        print(
            "Download 'EURO STOXX 50 Volatility (VSTOXX 1 month)' EUR Price TXT "
            "from STOXX and save it at this path (or update VSTOXX_PATH), then re-run."
        )
        return

    vstoxx = load_vstoxx(VSTOXX_PATH)
    print("VSTOXX sample (head):")
    print(vstoxx.head(), "\n")

    # 3) Spread
    df = construct_spread(vix, vstoxx, start=SAMPLE_START, end=SAMPLE_END)

    # >>> ADD THIS BLOCK (you were missing it)
    summary_df, summary_stats_table = compute_full_summary(
        df["VIX"], df["VSTOXX"]
    )


    # 3b) Correlation analysis
    corr_table, rolling_corr = compute_correlations(df)

    print("\n" + "="*70)
    print("Correlation Analysis: VIX vs VSTOXX")
    print("="*70)
    print(corr_table.to_string(index=False))
    print("\nLast rows of rolling correlations:")
    print(rolling_corr.tail(), "\n")


    # 5) Plots
    plot_indices_and_spread(df)


    # 5) Stationarity tests
    adf_results = adf_test_spread(df["Spread"])
    pp_results = pp_test_spread(df["Spread"])

    # 6) AR(p) model
    ar_results, best_p, ar_resid = fit_ar_model(df["Spread"], max_p=6, ic="aic")

    # 7) ARCH-LM
    arch_results = arch_lm_test(ar_resid, lags=12)

    # 8) GARCH(1,1)
    garch_res, garch_sigma, garch_std_resid = fit_garch_on_resid(ar_resid, p=1, q=1, dist="t")
    plot_garch_volatility(garch_sigma, title="VIX–VSTOXX Spread: GARCH(1,1) σ_t")

    # 9) Z-score strategy
    bt = backtest_zscore_strategy(
        df["Spread"],
        lookback=60,
        z_entry=1.5,
        z_exit=0.5,
        cost_bp=5.0,
    )
    plot_strategy_equity(bt, title="VIX–VSTOXX Spread: Z-Score Strategy Equity")

    backtest_stats = compute_backtest_stats(bt)
    backtest_stats.update({
        "lookback": 60,
        "z_entry": 1.5,
        "z_exit": 0.5,
        "cost_bp": 5.0,
    })

    # 10) Rolling AR forecast
    df_fc = rolling_ar_forecast(df["Spread"], window=90, p=best_p)
    plot_forecast(df_fc, title="VIX–VSTOXX Spread: Rolling AR Forecast (1-step)")

    forecast_stats = evaluate_forecast(df_fc)
    forecast_stats.update({
        "window": 90,
        "ar_order_p": best_p,
    })

    # 11) Summary tables
    tables = summarize_results(
        adf_results=adf_results,
        pp_results=pp_results,
        ar_results=ar_results,
        best_p=best_p,
        arch_results=arch_results,
        garch_res=garch_res,
        backtest_stats=backtest_stats,
        forecast_stats=forecast_stats,
    )

    for name, tbl in tables.items():
        print_header(name)
        print(tbl)
        tbl.to_csv(f"{name}.csv", index=True)

    # 12) Excel workbook
    excel_path = "vix_vstoxx_results.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        for name, tbl in tables.items():
            tbl.to_excel(writer, sheet_name=name, index=True)

    print_header("SUCCESS")
    print(
        "Data prepared; ADF & PP run; AR(p) fitted; ARCH-LM & GARCH(1,1) estimated;\n"
        "z-score strategy backtested; rolling AR forecast evaluated.\n"
        "Summary tables saved as individual CSVs and combined Excel workbook."
    )


if __name__ == "__main__":
    main()
