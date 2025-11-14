print("DEBUG: script file is executing")  # proves the script actually runs

"""
Project: Mean Reversion and Volatility Dynamics of the VIX–VSTOXX Spread
Author: Andreas Lindgren
Environment: Python 3.13, VS Code (Windows)

Current pipeline:
- Download VIX (30d implied vol on S&P 500).
    - Default: FRED VIXCLS via direct CSV (1990+).
    - Fallback: Yahoo Finance (^VIX) via yfinance.
- Load EURO STOXX 50 Volatility (VSTOXX 1 month, 30d) from STOXX TXT.
- Align series on overlapping dates (default from 2006+ to include 2008–2009).
- Construct Spread_t = VIX_t - VSTOXX_t.
- Plot VIX, VSTOXX, and the spread.

Next steps (to be added on this base):
- ADF test on spread.
- AR(p) model.
- ARCH test + GARCH(1,1).
- Simple z-score trading rule.
"""

# ==== 1. IMPORT LIBRARIES ====
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.unitroot import PhillipsPerron

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)







# ==== 2. BASIC SETTINGS ====
plt.style.use("seaborn-v0_8-darkgrid")
pd.options.display.float_format = "{:.4f}".format

# ==== 3. CONFIG ====

# STOXX "EUR Price" TXT file path for:
# EURO STOXX 50 Volatility (VSTOXX 1 month)
VSTOXX_PATH = r"C:\Users\andre\OneDrive\Skrivbord\econometrics\vstoxx_1m.txt"

# Choose VIX source:
# "fred"     -> FRED VIXCLS via CSV (preferred, long history)
# "yfinance" -> Yahoo Finance ^VIX (backup)
VIX_SOURCE = "fred"

# Analysis sample window (we want 2008–2009 included)
SAMPLE_START = "2006-01-01"
SAMPLE_END = None  # use latest available if None


# ==== 4. VIX DOWNLOAD FUNCTIONS ====

def download_vix_fred(start: str = "1990-01-01",
                      end: str | None = None) -> pd.DataFrame:
    """
    Download daily VIX data from FRED (series: VIXCLS) via CSV.

    Returns DataFrame with:
        index: Date
        column: 'VIX'
    """
    print("Step 1: Downloading VIX data from FRED (VIXCLS) via CSV...")

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    df = pd.read_csv(url)

    if df.empty:
        raise ValueError("ERROR: FRED VIX CSV download failed or returned empty.")

    # Handle possible column name variants
    if "observation_date" in df.columns:
        date_col = "observation_date"
    elif "DATE" in df.columns:
        date_col = "DATE"
    else:
        raise ValueError(f"ERROR: No date column in FRED VIX CSV. Columns: {df.columns.tolist()}")

    if "VIXCLS" not in df.columns:
        raise ValueError(f"ERROR: No VIXCLS column in FRED VIX CSV. Columns: {df.columns.tolist()}")

    # Parse dates and values
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["VIX"] = pd.to_numeric(df["VIXCLS"].replace(".", np.nan), errors="coerce")
    df = df.dropna(subset=[date_col, "VIX"])

    vix = df[[date_col, "VIX"]].set_index(date_col).sort_index()

    # Apply start/end filters
    if start is not None:
        vix = vix[vix.index >= pd.to_datetime(start)]
    if end is not None:
        vix = vix[vix.index <= pd.to_datetime(end)]

    if vix.empty:
        raise ValueError("ERROR: No VIX data in selected sample from FRED.")

    print(
        f"FRED VIX: downloaded {len(vix)} observations "
        f"from {vix.index.min().date()} to {vix.index.max().date()}."
    )
    return vix



def download_vix_yfinance(start: str = "1990-01-01",
                          end: str | None = None) -> pd.DataFrame:
    """
    Download daily VIX (^VIX) from Yahoo Finance via yfinance.

    Returns DataFrame:
        index: Date
        column: 'VIX'
    """
    print("Step 1: Downloading VIX data from Yahoo Finance (^VIX)...")

    data = yf.download(
        "^VIX",
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("ERROR: yfinance VIX download failed or returned empty.")

    # Handle MultiIndex or single-level columns
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].iloc[:, 0]
        else:
            # fallback: first block
            close = data.xs(data.columns.levels[0][0], axis=1).iloc[:, 0]
    else:
        for candidate in ["Close", "Adj Close", "Price"]:
            if candidate in data.columns:
                close = data[candidate]
                break
        else:
            raise ValueError(
                f"ERROR: Could not find a price column in VIX data. "
                f"Columns: {list(data.columns)}"
            )

    vix = close.to_frame(name="VIX")
    vix.index = pd.to_datetime(vix.index)
    vix = vix.sort_index().dropna()

    print(
        f"Yahoo VIX: downloaded {len(vix)} observations "
        f"from {vix.index.min().date()} to {vix.index.max().date()}."
    )
    return vix


def get_vix(start: str, end: str | None = None) -> pd.DataFrame:
    """
    Wrapper to select VIX source based on VIX_SOURCE.
    """
    src = VIX_SOURCE.lower()
    if src == "fred":
        return download_vix_fred(start=start, end=end)
    elif src == "yfinance":
        return download_vix_yfinance(start=start, end=end)
    else:
        raise ValueError(f"Unknown VIX_SOURCE: {VIX_SOURCE}. Use 'fred' or 'yfinance'.")


# ==== 5. VSTOXX LOADER ====

def load_vstoxx(file_path: str) -> pd.DataFrame:
    """
    Load VSTOXX 1M index data from STOXX .txt (or .csv) file.

    Expected STOXX "EUR Price" format:
        DD.MM.YYYY;CODE;VALUE

    Returns DataFrame with:
        index: Date
        column: 'VSTOXX'
    """
    print("Step 2: Loading VSTOXX data from file...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"VSTOXX file not found at: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=";",                 # semicolon-separated
        header=None,             # no header row
        names=["Date", "Code", "VSTOXX"],
        encoding="utf-8",
    )

    if df.empty:
        raise ValueError("ERROR: VSTOXX file is empty.")

    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["VSTOXX"] = pd.to_numeric(df["VSTOXX"], errors="coerce")
    df = df.dropna(subset=["Date", "VSTOXX"])

    vstoxx = df[["Date", "VSTOXX"]].set_index("Date").sort_index()

    if vstoxx.empty:
        raise ValueError("ERROR: No valid VSTOXX observations after cleaning.")

    print(
        f"VSTOXX: loaded {len(vstoxx)} observations "
        f"from {vstoxx.index.min().date()} to {vstoxx.index.max().date()}."
    )
    return vstoxx


# ==== 6. SPREAD CONSTRUCTION & PLOT ====

def construct_spread(vix: pd.DataFrame,
                     vstoxx: pd.DataFrame,
                     start: str | None = None,
                     end: str | None = None) -> pd.DataFrame:
    """
    Align VIX and VSTOXX and construct:
        Spread_t = VIX_t - VSTOXX_t

    Returns DataFrame with ['VIX', 'VSTOXX', 'Spread'].
    """
    print("Step 3: Aligning VIX and VSTOXX and constructing spread...")

    df = vix.join(vstoxx, how="inner")

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]

    df = df[["VIX", "VSTOXX"]].replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        raise ValueError("ERROR: No overlapping data in the chosen sample window.")

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
    """
    Plot VIX, VSTOXX, and their spread.
    """
    print("Step 4: Plotting VIX, VSTOXX, and spread...")

    if df.empty:
        raise ValueError("ERROR: No data to plot.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: levels
    axes[0].plot(df.index, df["VIX"], label="VIX (US, 30d)")
    axes[0].plot(df.index, df["VSTOXX"], label="VSTOXX (EU, 30d)")
    axes[0].set_ylabel("Implied Volatility (%)")
    axes[0].set_title("VIX vs VSTOXX (30-day implied volatility)")
    axes[0].legend()

    # Bottom: spread
    axes[1].plot(df.index, df["Spread"], label="Spread = VIX - VSTOXX")
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Volatility Points")
    axes[1].set_title("VIX - VSTOXX Spread")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def adf_test_spread(spread: pd.Series) -> dict:
    """
    Perform Augmented Dickey-Fuller test on the spread series.

    H0 (null): spread has a unit root (non-stationary, no mean reversion)
    H1 (alt): spread is stationary (mean-reverting)

    We use:
    - regression="c": constant only (spread may have non-zero mean)
    - autolag="AIC": lag length chosen by AIC
    - maxlag=20: reasonable upper bound for daily data

    Returns
    -------
    results : dict with ADF statistic, p-value, lags, nobs, critical values.
    """
    print("Step 5: ADF test on spread (mean reversion check)...")

    # Basic sanity check
    spread_clean = spread.replace([np.inf, -np.inf], np.nan).dropna()
    if spread_clean.empty:
        raise ValueError("ERROR: Spread series is empty or all NaN in adf_test_spread().")

    result = adfuller(
        spread_clean,
        maxlag=20,
        regression="c",
        autolag="AIC"
    )

    adf_stat, p_value, used_lag, nobs, crit_vals, icbest = result

    print("\nADF Test Results for Spread:")
    print(f"  ADF statistic      : {adf_stat:.4f}")
    print(f"  p-value            : {p_value:.4f}")
    print(f"  # of lags used     : {used_lag}")
    print(f"  # of observations  : {nobs}")
    print("  Critical values    :")
    for level, cv in crit_vals.items():
        print(f"    {level}%: {cv:.4f}")
    print(f"  Best information criterion (AIC-based): {icbest:.4f}")

    # Simple interpretation for the console
    if p_value < 0.05:
        print("\nConclusion: p < 0.05 → reject H0 of unit root.")
        print("The spread appears STATIONARY → evidence of mean reversion.\n")
    else:
        print("\nConclusion: p >= 0.05 → cannot reject H0 of unit root.")
        print("No strong evidence of stationarity → spread may not be mean-reverting.\n")

    return {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "nobs": nobs,
        "critical_values": crit_vals,
        "icbest": icbest,
    }
def pp_test_spread(spread: pd.Series) -> dict:
    """
    Phillips–Perron (PP) unit root test on the spread.

    H0 (null): spread has a unit root (non-stationary).
    H1 (alt): spread is stationary.

    PP is robust to autocorrelation and heteroskedasticity in the error term.
    """
    print("Step 5b: Phillips–Perron (PP) test on spread...")

    spread_clean = spread.replace([np.inf, -np.inf], np.nan).dropna()
    if spread_clean.empty:
        raise ValueError("ERROR: Spread series is empty or all NaN in pp_test_spread().")

    pp = PhillipsPerron(spread_clean)

    stat = float(pp.stat)
    pval = float(pp.pvalue)
    crit = pp.critical_values  # dict: {"1%": ..., "5%": ..., "10%": ...}

    print("\nPP Test Results for Spread:")
    print(f"  PP statistic   : {stat:.4f}")
    print(f"  p-value        : {pval:.4f}")
    print("  Critical values:")
    for level, cv in crit.items():
        print(f"    {level}: {cv:.4f}")

    if pval < 0.05:
        print("\nConclusion: p < 0.05 → reject H0 of unit root.")
        print("The spread appears STATIONARY (PP) → further evidence of mean reversion.\n")
    else:
        print("\nConclusion: p >= 0.05 → cannot reject H0 of unit root.")
        print("PP finds no strong evidence of stationarity.\n")

    return {
        "pp_statistic": stat,
        "p_value": pval,
        "critical_values": crit,
    }


def fit_ar_model(spread: pd.Series,
                 max_p: int = 6,
                 ic: str = "aic"):
    """
    Fit an AR(p) model to the spread using information-criterion selection.
    """
    print("Step 6: Estimating AR(p) model for the spread...")

    ic = ic.lower()
    if ic not in {"aic", "bic"}:
        raise ValueError("ic must be 'aic' or 'bic'.")

    # Clean spread
    y = spread.replace([np.inf, -np.inf], np.nan).dropna()
    if y.empty:
        raise ValueError("ERROR: Spread series is empty in fit_ar_model().")

    best_p = None
    best_ic = np.inf
    best_res = None

    # Grid search over p = 1..max_p
    for p in range(1, max_p + 1):
        try:
            # AR(p) with constant term, no differencing, no MA
            model = ARIMA(y, order=(p, 0, 0), trend="c")
            res = model.fit()  # default MLE
        except Exception as e:
            print(f"  Warning: AR({p}) estimation failed: {e}")
            continue

        value = res.aic if ic == "aic" else res.bic
        print(f"  AR({p}) {ic.upper()} = {value:.2f}")

        if value < best_ic:
            best_ic = value
            best_p = p
            best_res = res

    if best_res is None or best_p is None:
        raise RuntimeError("AR(p) search failed for all p in 1..max_p.")

    print(f"\nSelected AR({best_p}) model based on minimum {ic.upper()} = {best_ic:.2f}.\n")

    # AR coefficients
    ar_params = best_res.arparams
    print("Estimated AR coefficients (phi):")
    for i, phi in enumerate(ar_params, start=1):
        print(f"  phi_{i} = {phi:.4f}")

    # Check stationarity via roots
    if hasattr(best_res, "arroots") and len(best_res.arroots) > 0:
        roots = best_res.arroots
        min_abs_root = np.min(np.abs(roots))
        print("\nAR polynomial roots (modulus):")
        for i, r in enumerate(roots, start=1):
            print(f"  root_{i} |z| = {np.abs(r):.4f}")
        if min_abs_root > 1:
            print(
                "\nAll AR roots lie outside the unit circle → "
                "the estimated AR process is STABLE / MEAN-REVERTING."
            )
        else:
            print(
                "\nAt least one AR root lies inside the unit circle → "
                "model may be NON-STATIONARY (re-specification needed)."
            )
    else:
        print("\n(No AR roots reported; cannot explicitly verify stationarity.)")

    # Residuals
    resid = pd.Series(best_res.resid, index=y.index, name="ar_resid")

    print("\nAR(p) estimation complete. Residuals ready for ARCH/GARCH diagnostics.\n")

    return best_res, best_p, resid
def arch_lm_test(residuals: pd.Series, lags: int = 12) -> dict:
    """
    ARCH-LM test for conditional heteroskedasticity in AR residuals.

    H0: No ARCH effects (homoskedastic errors)
    H1: ARCH effects present (volatility clustering)

    Parameters
    ----------
    residuals : pd.Series
        Residuals from your mean model (e.g., AR(p)); index should be dates.
    lags : int, default 12
        Number of ARCH lags to test. For daily data, 12 ≈ ~2 trading weeks.

    Returns
    -------
    results : dict with LM/F statistics and p-values.
    """
    print("Step 7: ARCH-LM test on AR residuals (volatility clustering)...")

    if residuals is None:
        raise ValueError("ERROR: residuals is None in arch_lm_test().")

    eps = pd.Series(residuals, copy=True).astype(float)
    eps = eps.replace([np.inf, -np.inf], np.nan).dropna()
    if eps.empty:
        raise ValueError("ERROR: residuals are empty after cleaning in arch_lm_test().")

    # statsmodels returns: (lm_stat, lm_pvalue, f_stat, f_pvalue)
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(eps, nlags=lags)

    print("\nARCH-LM Test Results (AR residuals):")
    print(f"  Lags tested         : {lags}")
    print(f"  Observations (clean): {len(eps)}")
    print(f"  LM statistic        : {lm_stat:.4f}")
    print(f"  LM p-value          : {lm_pvalue:.4f}")
    print(f"  F  statistic        : {f_stat:.4f}")
    print(f"  F  p-value          : {f_pvalue:.4f}")

    if lm_pvalue < 0.05:
        print("\nConclusion: p < 0.05 → reject H0 of no ARCH effects.")
        print("Residual variance appears time-varying → proceed to GARCH modeling.\n")
    else:
        print("\nConclusion: p ≥ 0.05 → cannot reject H0 of no ARCH effects.")
        print("Little evidence of volatility clustering in residuals.\n")

    return {
        "lm_stat": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
        "lags": int(lags),
        "nobs": int(len(eps)),
    }
def fit_garch_on_resid(residuals: pd.Series,
                       p: int = 1,
                       q: int = 1,
                       dist: str = "t"):
    """
    Fit a GARCH(p,q) to mean-model residuals (here: AR residuals).

    Parameters
    ----------
    residuals : pd.Series
        Zero-mean residuals from the AR(p) mean model. Index should be dates.
    p, q : int
        ARCH and GARCH orders (default GARCH(1,1)).
    dist : {"normal", "t", "skewt"}
        Innovation distribution. 't' is robust for fat tails in vol data.

    Returns
    -------
    res : arch.univariate.base.ARCHModelResult
        Fitted model result.
    cond_vol : pd.Series
        Conditional volatility (sigma_t), aligned to residuals index.
    std_resid : pd.Series
        Standardized residuals (eps_t / sigma_t), aligned to index.
    """
    print(f"Step 8: Fitting GARCH({p},{q}) on AR residuals (dist='{dist}')...")

    if residuals is None:
        raise ValueError("ERROR: residuals is None in fit_garch_on_resid().")

    eps = pd.Series(residuals, copy=True).astype(float)
    eps = eps.replace([np.inf, -np.inf], np.nan).dropna()
    if eps.empty:
        raise ValueError("ERROR: residuals are empty after cleaning in fit_garch_on_resid().")

    # Mean is zero because we've already modeled the conditional mean (AR)
    am = arch_model(
        eps,
        mean="zero",
        vol="GARCH",
        p=p,
        q=q,
        dist=dist
    )

    # Quiet, robust fit
    res = am.fit(disp="off", show_warning=False)

    print("\nGARCH parameter estimates:")
    print(res.params.to_string())

    # Core diagnostics
    params = res.params
    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha[1]", params.get("alpha1", np.nan)))
    beta  = float(params.get("beta[1]",  params.get("beta1",  np.nan)))
    persistence = alpha + beta

    print(f"\nKey stats:")
    print(f"  omega (constant)     : {omega:.6f}")
    print(f"  alpha_1 (ARCH)       : {alpha:.4f}")
    print(f"  beta_1  (GARCH)      : {beta:.4f}")
    print(f"  persistence α+β      : {persistence:.4f}")

    # Stationarity (variance exists) requires persistence < 1
    if persistence < 1:
        # half-life (in periods) ≈ ln(0.5) / ln(persistence)
        hl = np.log(0.5) / np.log(persistence)
        hl_txt = f"{hl:.1f} periods"
        print("  variance is STATIONARY (α+β < 1).")
        print(f"  shock half-life      : {hl_txt}")
    else:
        print("  variance is NON-STATIONARY or near-unit (α+β ≥ 1) — shocks highly persistent.")

    # Series outputs
    cond_vol = pd.Series(res.conditional_volatility, index=eps.index, name="garch_sigma")
    std_resid = pd.Series(res.std_resid, index=eps.index, name="garch_std_resid")

    # Quick residual diagnostics
    print("\nLjung-Box test on standardized residuals (serial correlation):")
    for lag in (5, 10, 20):
        lb = acorr_ljungbox(std_resid.dropna(), lags=[lag], return_df=True)
        print(f"  lag {lag}: Q={lb['lb_stat'].iloc[0]:.2f}, p={lb['lb_pvalue'].iloc[0]:.4f}")

    print("Ljung-Box test on squared standardized residuals (remaining ARCH effects):")
    for lag in (5, 10, 20):
        lb2 = acorr_ljungbox((std_resid.dropna() ** 2), lags=[lag], return_df=True)
        print(f"  lag {lag}: Q={lb2['lb_stat'].iloc[0]:.2f}, p={lb2['lb_pvalue'].iloc[0]:.4f}")

    print("\nGARCH(1,1) estimation complete. Conditional volatility series ready.\n")
    return res, cond_vol, std_resid

def plot_garch_volatility(index: pd.DatetimeIndex,
                          cond_vol: pd.Series,
                          title: str = "Conditional Volatility (GARCH)") -> None:
    """Simple plot of GARCH conditional volatility."""
    if cond_vol is None or cond_vol.empty:
        print("Warning: No conditional volatility to plot.")
        return
    plt.figure(figsize=(12, 4))
    plt.plot(index, cond_vol, label="σ_t (GARCH)")
    plt.title(title)
    plt.ylabel("Volatility (σ)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_zscore(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Rolling z-score of the spread using rolling mean and std over 'lookback' days.

    z_t = (Spread_t - rolling_mean_t) / rolling_std_t

    If the spread is stationary and mean-reverting, large |z_t| should be rare
    and followed by moves back toward zero.
    """
    x = spread.astype(float)
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
    Toy mean-reversion rule on the spread.

    Position_t ∈ {-1, 0, +1}:
      - Go SHORT spread if z_t >  z_entry  (expect spread to FALL)
      - Go LONG  spread if z_t < -z_entry  (expect spread to RISE)
      - EXIT/FLAT when |z_t| < z_exit      (mean reversion achieved)

    PnL is from change in spread:
        ret_{t+1} = position_t * (Spread_{t+1} - Spread_t)

    cost_bp: transaction cost in basis points per change in position (entry/flip/exit).
    """
    z = compute_zscore(spread, lookback=lookback)
    z = z.dropna().copy()

    pos = pd.Series(0.0, index=z.index)

    long_sig  = z < -z_entry
    short_sig = z >  z_entry
    exit_sig  = z.abs() < z_exit

    state = 0.0
    for idx in z.index:
        if state == 0.0:
            if long_sig.loc[idx]:
                state = +1.0
            elif short_sig.loc[idx]:
                state = -1.0
        else:
            if exit_sig.loc[idx]:
                state = 0.0
            elif state == +1.0 and short_sig.loc[idx]:
                state = -1.0
            elif state == -1.0 and long_sig.loc[idx]:
                state = +1.0

        pos.loc[idx] = state

    # Align position to full spread index
    pos = pos.reindex(spread.index).fillna(method="ffill").fillna(0.0)

    # Spread changes (in points)
    dS = spread.diff()

    # Strategy PnL: yesterday's position * today's change in spread
    strat_ret_raw = pos.shift(1) * dS

    # Transaction costs (per change in position), approximated in bp of notional 1
    dpos = pos.diff().abs().fillna(0.0)
    cost = (cost_bp / 1e4) * dpos
    strat_ret = strat_ret_raw - cost

    equity = strat_ret.cumsum().rename("equity")

    out = pd.DataFrame({
        "spread": spread,
        "z": z.reindex(spread.index),
        "pos": pos,
        "dS": dS,
        "ret_raw": strat_ret_raw,
        "cost": cost,
        "ret": strat_ret,
        "equity": equity,
    })

    # Simple performance stats
    valid = out["ret"].dropna()
    if valid.std(ddof=0) > 0:
        sharpe = valid.mean() / valid.std(ddof=0) * np.sqrt(252)
    else:
        sharpe = np.nan

    trades = int((dpos > 0).sum())
    print("\nZ-Score Backtest (toy mean-reversion on spread):")
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

def rolling_ar_forecast(spread: pd.Series,
                        p: int,
                        window: int = 90,
                        max_obs: int = 1500) -> pd.DataFrame:
    """
    Rolling one-step-ahead AR(p) forecast of the spread.

    For each day t >= window, we estimate an AR(p) on the previous `window`
    observations and forecast Spread_{t+1}. This mimics a real-time setting.

    Parameters
    ----------
    spread : pd.Series
        The spread series.
    p : int
        AR order (use best_p from fit_ar_model).
    window : int
        Rolling estimation window length in days (e.g. 90).
    max_obs : int
        Limit evaluation to last `max_obs` observations for speed.
        If None, use the full history.

    Returns
    -------
    pd.DataFrame
        index: forecast date (t+1)
        columns:
            'Spread'         – realized spread at t+1
            'AR_forecast_1d' – forecast of spread at t+1 based on data up to t
    """
    print(f"Step 9: Rolling AR({p}) 1-step forecast with window={window}...")

    # Clean series
    y = spread.replace([np.inf, -np.inf], np.nan).dropna()

    # Optionally restrict to last max_obs points
    if max_obs is not None and len(y) > max_obs:
        y = y.iloc[-max_obs:]

    if len(y) <= window + p:
        raise ValueError(
            f"Not enough data for rolling AR forecast: "
            f"need > {window + p} obs, have {len(y)}."
        )

    dates = y.index
    forecasts = []
    forecast_dates = []

    # For i = window .. len(y)-2:
    #   use y[i-window : i] to forecast y[i+1]
    for i in range(window, len(y) - 1):
        y_window = y.iloc[i - window : i]

        try:
            # Fit AR(p) with constant via OLS
            model = AutoReg(y_window, lags=p, trend="c", old_names=False)
            res = model.fit()

            # Manually compute 1-step-ahead forecast:
            #  ŷ_{t+1} = const + Σ_{j=1}^p φ_j * y_{t+1-j}
            params = res.params

            if "const" in params.index:
                const = params["const"]
                coefs = params.drop("const").values
            else:
                # Fallback: first element as constant
                const = params.iloc[0]
                coefs = params.iloc[1:].values

            # Take last p observations up to time i (y[i-1], y[i-2], ..., y[i-p])
            y_lags = y.iloc[i - p : i].iloc[::-1].values  # reverse to match φ_1*y_t + φ_2*y_{t-1}+...

            fc_val = const + float(np.dot(coefs, y_lags))
            forecasts.append(fc_val)
            forecast_dates.append(dates[i + 1])

        except Exception as e:
            print(f"  Warning (rolling AR): window ending {dates[i].date()} failed: {e}")
            continue

    if not forecasts:
        print("Warning: No successful forecast windows in rolling_ar_forecast().")
        return pd.DataFrame()

    fc_series = pd.Series(
        forecasts,
        index=pd.DatetimeIndex(forecast_dates),
        name="AR_forecast_1d",
    )

    realized = y.reindex(fc_series.index)

    result = pd.DataFrame({
        "Spread": realized,
        "AR_forecast_1d": fc_series
    }).dropna()

    print(f"Rolling AR forecast generated for {len(result)} days.")
    return result



def plot_forecast(fc_df: pd.DataFrame,
                  title: str = "Rolling AR Forecast (1-step)") -> None:
    """
    Plot actual spread vs rolling AR 1-step-ahead forecast.
    """
    print("Step 10: Plotting rolling AR forecast vs actual spread...")

    if fc_df is None or fc_df.empty:
        print("Warning: rolling forecast DataFrame is empty – nothing to plot.")
        return

    v = fc_df.dropna()
    if v.empty:
        print("Warning: all forecast rows are NaN – nothing to plot.")
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

def compute_backtest_stats(bt_df: pd.DataFrame) -> dict:
    """
    Compute simple performance statistics from the z-score backtest output.
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

    stats = {
        "n_obs": len(valid),
        "mean_daily_ret": valid.mean(),
        "std_daily_ret": valid.std(ddof=0),
        "sharpe_annual": sharpe,
        "trades": trades,
        "final_equity": bt_df["equity"].iloc[-1],
    }
    return stats

def evaluate_forecast(df_fc: pd.DataFrame) -> dict:
    """
    Compute MSE, MAE and directional hit-rate from rolling AR forecast results.

    Expects a DataFrame with at least two columns:
      - 'y'       : actual spread
      - 'fc_mean' : one-step-ahead forecast of the spread

    If those columns are missing or the DataFrame is empty, returns {} and
    prints a warning instead of raising.
    """
    # Basic validity checks
    if df_fc is None:
        print("Warning in evaluate_forecast(): df_fc is None.")
        return {}

    if not isinstance(df_fc, pd.DataFrame):
        print(f"Warning in evaluate_forecast(): df_fc is not a DataFrame (type={type(df_fc)}).")
        return {}

    if df_fc.empty:
        print("Warning in evaluate_forecast(): df_fc is empty.")
        return {}

    # Check expected columns
    cols = list(df_fc.columns)
    if ("y" not in cols) or ("fc_mean" not in cols):
        print(
            "Warning in evaluate_forecast(): expected columns 'y' and 'fc_mean' "
            f"not found. Available columns: {cols}"
        )
        return {}

    # Drop rows where either actual or forecast is missing
    valid = df_fc.dropna(subset=["y", "fc_mean"]).copy()
    if valid.empty:
        print("Warning in evaluate_forecast(): no non-NaN observations after dropna.")
        return {}

    # Forecast error
    err = valid["y"] - valid["fc_mean"]
    mse = (err ** 2).mean()
    mae = err.abs().mean()

    # Directional accuracy: compare the sign of changes
    # Actual change: y_t - y_{t-1}
    sign_actual = np.sign(valid["y"].diff())
    # Forecasted change: fc_mean_t - y_{t-1} (forecast vs last actual)
    sign_fc = np.sign(valid["fc_mean"] - valid["y"].shift(1))

    # Align and drop NaNs from sign series
    da = sign_actual.dropna()
    df_ = sign_fc.dropna()
    common_index = da.index.intersection(df_.index)

    if common_index.empty:
        hit_rate = np.nan
        directional_hits = 0
    else:
        da = da.loc[common_index]
        df_ = df_.loc[common_index]
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

    print("Rolling AR forecast evaluation:")
    print(f"  n_obs               : {stats['n_obs']}")
    print(f"  MSE                 : {stats['mse']:.4f}")
    print(f"  MAE                 : {stats['mae']:.4f}")
    print(f"  Directional hits    : {stats['directional_hits']}")
    print(f"  Directional hit-rate: {stats['directional_hit_rate']:.2%}\n")

    return stats


def summarize_results(
    adf_results: dict,
    pp_results: dict,
    ar_results,
    best_p: int,
    arch_results: dict,
    garch_res,
    backtest_stats: dict,
    forecast_stats: dict,
) -> dict:
    """
    Build pandas DataFrame tables summarizing the main empirical results.

    Returns
    -------
    tables : dict[str, pd.DataFrame]
        Keys:
          - 'stationarity_results'
          - 'ar_model_results'
          - 'arch_lm_results'
          - 'garch_results'
          - 'backtest_results'
          - 'forecast_results'
    """

    # ---- 1) Stationarity: ADF & PP ----
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
            "lags_used": pp_results.get("lags", None),
            "n_obs": pp_results.get("nobs"),
            "IC_best": None,
            "Conclusion": "Reject unit root (stationary)"
            if pp_results.get("p_value", 1.0) < 0.05 else
               "Cannot reject unit root",
        })

    stationarity_results = pd.DataFrame(station_rows)

    # ---- 2) AR(p) model ----
    ar_rows = []
    if ar_results is not None:
        phi = getattr(ar_results, "arparams", np.array([]))
        roots = getattr(ar_results, "arroots", np.array([]))
        min_abs_root = np.min(np.abs(roots)) if roots.size > 0 else np.nan
        stable = bool(min_abs_root > 1) if not np.isnan(min_abs_root) else None

        row = {
            "Selected_p": best_p,
            "AIC": ar_results.aic,
            "BIC": ar_results.bic,
            "min_|root|": min_abs_root,
            "Stable_AR": stable,
        }
        # add phi_1 ... phi_p
        for i, coeff in enumerate(phi, start=1):
            row[f"phi_{i}"] = coeff

        ar_rows.append(row)

    ar_model_results = pd.DataFrame(ar_rows)

    # ---- 3) ARCH-LM test ----
    arch_rows = []
    if arch_results:
        arch_rows.append({
            "lags_tested": arch_results.get("lags"),
            "LM_stat": arch_results.get("LM_stat"),
            "LM_p_value": arch_results.get("LM_p_value"),
            "F_stat": arch_results.get("F_stat"),
            "F_p_value": arch_results.get("F_p_value"),
            "n_obs": arch_results.get("n_obs"),
            "Conclusion": "Reject no-ARCH (vol clustering)"
            if arch_results.get("LM_p_value", 1.0) < 0.05 else
               "Cannot reject no-ARCH",
        })

    arch_lm_results = pd.DataFrame(arch_rows)

    # ---- 4) GARCH(1,1) ----
    garch_rows = []
    if garch_res is not None:
        params = garch_res.params
        omega = params.get("omega", np.nan)
        alpha1 = params.get("alpha[1]", np.nan)
        beta1 = params.get("beta[1]", np.nan)
        nu = params.get("nu", np.nan)  # df for t-distribution, if present
        persistence = alpha1 + beta1 if np.isfinite(alpha1) and np.isfinite(beta1) else np.nan
        stationary_var = bool(persistence < 1) if np.isfinite(persistence) else None

        # half-life in days: log(0.5) / log(alpha+beta)
        if np.isfinite(persistence) and persistence > 0 and persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.nan

        garch_rows.append({
            "omega": omega,
            "alpha_1": alpha1,
            "beta_1": beta1,
            "alpha+beta": persistence,
            "Stationary_variance": stationary_var,
            "Half_life_days": half_life,
            "nu_df": nu,
            "distribution": garch_res.distribution.name if hasattr(garch_res, "distribution") else None,
        })

    garch_results = pd.DataFrame(garch_rows)

    # ---- 5) Backtest (z-score strategy) ----
    bt_rows = []
    if backtest_stats:
        bt_rows.append(backtest_stats)
    backtest_results = pd.DataFrame(bt_rows)

    # ---- 6) Forecast results ----
    fc_rows = []
    if forecast_stats:
        fc_rows.append(forecast_stats)
    forecast_results = pd.DataFrame(fc_rows)

    # ---- Collect all tables ----
    tables = {
        "stationarity_results": stationarity_results,
        "ar_model_results": ar_model_results,
        "arch_lm_results": arch_lm_results,
        "garch_results": garch_results,
        "backtest_results": backtest_results,
        "forecast_results": forecast_results,
    }

    return tables


# ==== 7. MAIN EXECUTION ====

def main():
    print("==== VIX–VSTOXX PROJECT: DATA PIPELINE ====")
    print(f"Using VIX source: {VIX_SOURCE.upper()}")
    print(f"Sample window: {SAMPLE_START} to {SAMPLE_END or 'latest available'}")

    # 1) VIX from chosen source (FRED default, long history)
    vix = get_vix(start="1990-01-01", end=SAMPLE_END)

    print("\nVIX sample (head):")
    print(vix.head(), "\n")

    # 2) VSTOXX from STOXX file
    if not os.path.exists(VSTOXX_PATH):
        print(f"WARNING: VSTOXX file not found at:\n  {VSTOXX_PATH}")
        print(
            "Download 'EURO STOXX 50 Volatility (VSTOXX 1 month)' EUR Price TXT from STOXX\n"
            "and save it at this path (or update VSTOXX_PATH), then re-run."
        )
        return

    vstoxx = load_vstoxx(VSTOXX_PATH)

    print("\nVSTOXX sample (head):")
    print(vstoxx.head(), "\n")

    # 3) Construct spread using explicit sample window (incl. 2008–09)
    df = construct_spread(vix, vstoxx, start=SAMPLE_START, end=SAMPLE_END)

    # 4) Visualize
    plot_indices_and_spread(df)
    # 5) ADF unit root test on the spread
    adf_results = adf_test_spread(df["Spread"])
    # 5b) Phillips–Perron unit root test on the spread
    pp_results = pp_test_spread(df["Spread"])


    # 6) AR(p) model for the stationary spread
    ar_results, best_p, ar_resid = fit_ar_model(df["Spread"], max_p=6, ic="aic")
    # 7) ARCH-LM test on AR residuals
    arch_results = arch_lm_test(ar_resid, lags=12)

    # 8) GARCH(1,1) on AR residuals (Student-t innovations for robustness)
    garch_res, garch_sigma, garch_std_resid = fit_garch_on_resid(ar_resid, p=1, q=1, dist="t")

    # (Optional) Visualize conditional volatility
    plot_garch_volatility(garch_sigma.index, garch_sigma, title="VIX–VSTOXX Spread: GARCH(1,1) σ_t")

      # 9) Simple z-score mean-reversion strategy on the spread (toy example)
    bt = backtest_zscore_strategy(
        df["Spread"],
        lookback=60,      # 60 trading days (~3 months)
        z_entry=1.5,      # enter when deviation is large
        z_exit=0.5,       # exit when it mean-reverts
        cost_bp=5.0       # 5 basis points per trade
    )
    plot_strategy_equity(bt, title="VIX–VSTOXX Spread: Z-Score Strategy Equity")

    # Compute backtest stats for summary table
    backtest_stats = compute_backtest_stats(bt)
    # add meta-parameters
    backtest_stats["lookback"] = 60
    backtest_stats["z_entry"] = 1.5
    backtest_stats["z_exit"] = 0.5
    backtest_stats["cost_bp"] = 5.0

    # 10) Rolling AR(p) 1-step-ahead forecast with 90-day rolling window
    df_fc = rolling_ar_forecast(
        df["Spread"],
        window=90,        # 90-day rolling estimation window
        p=best_p          # use the AR order we selected earlier
    )
    plot_forecast(df_fc, title="VIX–VSTOXX Spread: Rolling AR Forecast (1-step)")

    forecast_stats = evaluate_forecast(df_fc)
    forecast_stats["window"] = 90
    forecast_stats["ar_order_p"] = best_p

    # === Build and print/save summary tables ===
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
        print(f"\n==== {name} ====")
        print(tbl)
        # also save as CSV for use in the report (Excel / LaTeX / etc.)
        tbl.to_csv(f"{name}.csv", index=True)

 # === NEW: Save all tables into a single Excel file ===
    excel_path = "vix_vstoxx_results.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        for name, tbl in tables.items():
            tbl.to_excel(writer, sheet_name=name, index=True)

    print(
        "SUCCESS: Data prepared, ADF & PP run, AR(p) fitted, ARCH-LM and GARCH(1,1) estimated,\n"
        "z-score strategy backtested, and rolling AR forecast evaluated. Summary tables saved as CSV."
    )


if __name__ == "__main__":
    main()
