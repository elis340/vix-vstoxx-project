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

    print(
        "SUCCESS: Data prepared and spread constructed.\n"
        "You now have a clean dataset incl. 2008–2009 crisis for further modeling."
    )


if __name__ == "__main__":
    main()
