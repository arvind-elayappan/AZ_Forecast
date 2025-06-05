from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go
import plotly as px


# ------------------------------------------------------------
# decomposition chart
# ------------------------------------------------------------
# 12) SEASONAL DECOMPOSITION PLOT
# ------------------------------------------------------------


def plot_seasonal_decompose(
    df: pd.DataFrame,
    column: str,
    model: str = "additive",
    period: int = None,
    figsize: tuple = (12, 8),
    title: str = None,
) -> None:
    """
    Perform and plot seasonal decomposition of a single time‐series column.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame whose index is (or can be converted to) a DatetimeIndex and
        which contains the specified `column`.
    column : str
        Name of the numeric column to decompose.
    model : str, default "additive"
        Either "additive" or "multiplicative". Use "additive" if seasonal variations
        are roughly constant over time, or "multiplicative" if they grow/shrink.
    period : int, optional
        The periodicity of the seasonality. For example:
          - 7  for weekly seasonality on daily data,
          - 144 for a 24-hour cycle on 10-minute data,
          - 365 for yearly seasonality on daily data, etc.
        If None, the function will attempt to infer a period of 1 (not recommended).
    figsize : tuple, default (12, 8)
        Figure size for the 4-panel decomposition plot.
    title : str, optional
        Main title for the decomposition figure. If None, defaults to
        "Seasonal Decomposition of <column>".

    Usage Example:
    --------------
    # Suppose `df_daily` is a daily‐aggregated DataFrame with a DatetimeIndex:
    plot_seasonal_decompose(
        df_daily,
        column="Zone 1 Power Consumption",
        model="additive",
        period=7,
        figsize=(14, 10),
        title="Weekly Decomposition of Zone 1"
    )
    """
    # 1) Basic sanity checks
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' is not numeric and cannot be decomposed.")
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            # Attempt to convert the index to datetime if it isn’t already
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise TypeError(
                "Index must be a DatetimeIndex or convertible to datetime."
            ) from e

    # 2) Ensure the series is sorted by date
    series = df[column].sort_index()

    # 3) If no period is provided, warn the user
    if period is None:
        raise ValueError(
            "You must specify a `period` (e.g. 7 for weekly on daily data, "
            "144 for daily seasonality on 10-minute data, etc.)."
        )

    # 4) Perform the decomposition
    decomposition = seasonal_decompose(series, model=model, period=period)

    # 5) Plot the result
    plt.rcParams.update({"figure.figsize": figsize})
    fig = decomposition.plot()
    fig.suptitle(title or f"Seasonal Decomposition of '{column}'", fontsize=16)
    plt.tight_layout()
    plt.show()


# stationarity_check.py

# from statsmodels.tsa.stattools import adfuller, kpss


def test_stationarity(
    df: pd.DataFrame,
    column: str,
    window: int = None,
    freq: str = None,
    adf_regression: str = "c",
    kpss_regression: str = "c",
) -> None:
    """
    Perform a combined stationarity check on a single numeric column:
      1) Plot rolling mean & rolling std
      2) Run Augmented Dickey-Fuller test (null: non-stationary)
      3) Run KPSS test (null: stationary)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex or convertible index.
    column : str
        Name of the numeric column to test.
    window : int, optional
        Window size for rolling mean/std. If None, defaults to 144 if freq="10min" or
        defaults to 7 if freq="D". You can override as desired.
    freq : str, optional
        Resampling frequency string. E.g., "D" for daily. If provided, this function
        will first resample `df[column]` at that freq by taking the mean.
    adf_regression : str, default "c"
        Regression parameter for ADF ("c" for constant, "ct" for constant+trend, "nc" for none).
    kpss_regression : str, default "c"
        Regression parameter for KPSS ("c" for constant, "ct" for constant+trend).

    Prints
    ------
    - Rolling‐mean & rolling‐std plot (matplotlib)
    - ADF test statistic, p-value, used regression, and a quick stationarity interpretation
    - KPSS test statistic, p-value, used regression, and a quick stationarity interpretation

    Usage Example
    -------------
    # For daily‐aggregated data (7-day window), test stationarity:
    test_stationarity(df_daily, column="Zone 2 Power Consumption", freq=None, window=7)

    # For raw 10-minute data, first resample daily then test:
    test_stationarity(df, column="Zone 2 Power Consumption", freq="D", window=7)
    """
    # 1) Validate column
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in the DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(
            f"Column '{column}' is not numeric and cannot be tested for stationarity."
        )

    # 2) Build the series to test: optionally resample first
    series = df[column]
    if freq is not None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DatetimeIndex to resample.")
        try:
            series = series.resample(freq).mean()
        except Exception as e:
            raise ValueError(f"Could not resample series at freq='{freq}': {e}")

    series = series.dropna().sort_index()

    # 3) Determine rolling window if none provided
    if window is None:
        # If freq="D", default to a 7-day window; else if measurement freq looks like 10-min, window=144
        if freq == "D":
            window = 7
        else:
            # Try to infer: if the index frequency is 10 minutes, then 24h window = 144 points
            inferred = pd.infer_freq(series.index)
            if inferred and ("T" in inferred or "min" in inferred):
                window = 144
            else:
                # fallback
                window = max(3, min(len(series) // 10, 30))

    # 4) Plot rolling mean & rolling std
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    plt.figure(figsize=(10, 4))
    plt.plot(series, color="blue", label="Original")
    plt.plot(rolling_mean, color="red", label=f"Rolling Mean (window={window})")
    plt.plot(rolling_std, color="green", label=f"Rolling Std  (window={window})")
    plt.title(f"Stationarity Check (Rolling Mean & Std) for '{column}'")
    plt.legend(loc="best")
    plt.show()

    # 5) Augmented Dickey-Fuller Test
    print("----- Augmented Dickey–Fuller Test (Null: unit root, non-stationary) -----")
    adf_result = adfuller(series, regression=adf_regression, autolag="AIC")
    adf_stat, adf_pvalue, usedlag, nobs, crit_vals, icbest = adf_result
    print(f"ADF Statistic   : {adf_stat:.6f}")
    print(f"p-value         : {adf_pvalue:.6f}")
    print(f"Used lag        : {usedlag}")
    print(f"Number of obs   : {nobs}")
    print("Critical Values :")
    for key, val in crit_vals.items():
        print(f"    {key}: {val:.3f}")
    if adf_pvalue < 0.05:
        print(
            f"=> ADF p-value < 0.05 → reject null. '{column}' is stationary by ADF test."
        )
    else:
        print(
            f"=> ADF p-value >= 0.05 → fail to reject null. '{column}' may be non-stationary by ADF."
        )

    print()

    # 6) KPSS Test
    print("----- KPSS Test (Null: stationary) -----")
    # “regression='c'” tests level stationarity. To test for trend stationarity, use regression='ct'.
    try:
        kpss_stat, kpss_pvalue, kpss_lags, kpss_crit_vals = kpss(
            series, regression=kpss_regression, nlags="auto"
        )
    except ValueError as e:
        print(f"KPSS test error: {e}")
        return

    print(f"KPSS Statistic : {kpss_stat:.6f}")
    print(f"p-value         : {kpss_pvalue:.6f}")
    print(f"Used lags       : {kpss_lags}")
    print("Critical Values :")
    for key, val in kpss_crit_vals.items():
        print(f"    {key}: {val:.3f}")
    if kpss_pvalue < 0.05:
        print(
            f"=> KPSS p-value < 0.05 → reject null. '{column}' is non-stationary by KPSS test."
        )
    else:
        print(
            f"=> KPSS p-value >= 0.05 → fail to reject null. '{column}' is stationary by KPSS test."
        )

    print()
    print("----- Interpretation Summary -----")
    if adf_pvalue < 0.05 and kpss_pvalue >= 0.05:
        print(f"Both ADF and KPSS agree: '{column}' is stationary.")
    elif adf_pvalue < 0.05 and kpss_pvalue < 0.05:
        print(
            f"ADF says stationary but KPSS says non-stationary: borderline. You may want to difference or investigate further."
        )
    elif adf_pvalue >= 0.05 and kpss_pvalue < 0.05:
        print(
            f"Both ADF and KPSS agree: '{column}' is non-stationary. Consider differencing or detrending."
        )
    else:  # adf_pvalue >= 0.05 and kpss_pvalue >= 0.05
        print(
            f"ADF is inconclusive / fails to reject non-stationary, but KPSS fails to reject stationarity. Mixed signals—visual inspection is recommended."
        )

    print("----------------------------------------------------------------------")


# --------------------------------------------------------------------------------------
# ACF and PACF plots


def plot_acf_pacf(
    df: pd.DataFrame,
    column: str,
    lags: int = 28,
    pacf_method: str = "ywm",
    figsize: tuple = (14, 4),
    title_suffix: str = "",
) -> None:
    """
    Plot the ACF and PACF for a single numeric time‐series column.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame whose index should be a DatetimeIndex (or convertible to datetime)
        and which contains the specified `column`.
    column : str
        Name of the numeric column to analyze. This column will be plotted.
    lags : int, default 28
        Number of lags (in periods of the index) to show in both the ACF and PACF plots.
    pacf_method : str, default "ywm"
        Method for computing the PACF: commonly "ywm" or "ols". Use whichever works best
        for your data.
    figsize : tuple, default (14, 4)
        Figure size (width, height) for the combined ACF/PACF plot.
    title_suffix : str, default ""
        Optional text to append to each subplot’s title (e.g. "(weekly)" or "(daily)").

    Usage Example
    -------------
    # Suppose you have a daily‐aggregated DataFrame `df_daily` with a DatetimeIndex:
    plot_acf_pacf(df_daily, column="Zone 2 Power Consumption", lags=28, title_suffix="(daily agg)")

    What It Does
    ------------
    1) Verifies that `column` exists and is numeric.
    2) Attempts to cast the DataFrame’s index to a DatetimeIndex if it isn’t already.
    3) Sorts the series by date.
    4) Creates two subplots side by side:
       - Left:  ACF (autocorrelation) up to `lags`.
       - Right: PACF (partial autocorrelation) up to `lags`, using the chosen `pacf_method`.
    5) Labels each subplot clearly, showing which column and any `title_suffix`.
    6) Calls plt.tight_layout() and plt.show().

    Notes
    -----
    - The ACF shows how yesterday’s value, two days ago, … up to `lags`‐days ago, correlate
      with today’s value (including indirect pathways through intervening days).
    - The PACF shows the “direct” correlation at each lag after accounting for all shorter lags.
      In other words, PACF at lag k is the correlation between x_t and x_{t-k} once you remove
      the influence of x_{t-1}, x_{t-2}, … x_{t-(k-1)}.
    - For a pure AR(p) process, the PACF will “cut off” after lag p. For a pure MA(q),
      the ACF will cut off after lag q.

    """
    # 1) Column existence & numeric check
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' is not numeric and cannot be plotted.")

    # 2) Ensure the index is a DatetimeIndex
    series = df[column]
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            raise TypeError(
                "Index must be a DatetimeIndex or convertible to datetime."
            ) from e

    # 3) Sort by date
    series = series.sort_index()

    # 4) Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 5) Plot ACF
    plot_acf(
        series,
        lags=lags,
        ax=axes[0],
        alpha=0.05,  # 95% confidence interval
    )
    axes[0].set_title(f"ACF ({column}) {title_suffix}".strip())

    # 6) Plot PACF
    plot_pacf(series, lags=lags, method=pacf_method, ax=axes[1], alpha=0.05)
    axes[1].set_title(f"PACF ({column}) {title_suffix}".strip())

    # 7) Final layout adjustments
    plt.tight_layout()
    plt.show()


# Below is a Plotly‐based function you can add to your eda_tools.py (or any notebook) that will overlay each block of a specified length (e.g. 28 days) as a separate, interactive line.


def plotly_overlay_seasonality(
    series: pd.Series,
    block_length: int,
    title: str = None,
    line_opacity: float = 0.6,
    width: int = 900,
    height: int = 500,
) -> None:
    """
    Create an interactive Plotly figure that overlays each 'block_length'-day slice
    of 'series' on the same set of axes. Hovering shows (day_index, value).

    Parameters
    ----------
    series : pd.Series
        A daily‐indexed Pandas Series (or convertible to daily frequency). Each value
        should be numeric. The index will be cast to a DatetimeIndex if needed, then
        resampled to 'D' frequency and interpolated to fill gaps.
    block_length : int
        How many days each block should contain. For example, if block_length=28, then
        the function will split the first (n_blocks * 28) days into rows of length 28.
    title : str, optional
        Figure title. If None, defaults to
        "{block_length}-Day Overlay Seasonality (n_blocks blocks)".
    line_opacity : float, default 0.6
        Opacity for each overlaid line (0.0 to 1.0). Lower values make it easier to see
        where many lines overlap.
    width : int, default 900
        Width of the figure in pixels.
    height : int, default 500
        Height of the figure in pixels.
    """
    # 1) Prepare a clean, daily‐frequency series
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            raise TypeError(
                "Index must be a DatetimeIndex or convertible to datetime."
            ) from e

    # Sort by date, reindex to fill any missing calendar days, and then interpolate
    s = series.sort_index().asfreq("D")
    s = s.interpolate(method="linear").dropna()

    # 2) Determine how many full blocks of length 'block_length' we can extract
    N = len(s)
    n_blocks = N // block_length
    if n_blocks < 1:
        raise ValueError(
            f"Series is too short for even one block of length {block_length} days."
        )

    # 3) Trim to an exact multiple of block_length
    trimmed = s.iloc[: n_blocks * block_length]
    arr = trimmed.values.reshape((n_blocks, block_length))

    # 4) Build an interactive Plotly figure
    fig = go.Figure()

    # Add one trace per block
    for i in range(n_blocks):
        y = arr[i]  # values for block i
        x = np.arange(block_length)  # day‐within‐block: 0,1,...,block_length-1
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Block {i + 1}",
                opacity=line_opacity,
                hovertemplate="Day Index: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )
        )

    # 5) Final layout touches
    fig.update_layout(
        title=title or f"{block_length}-Day Overlay Seasonality ({n_blocks} blocks)",
        xaxis_title=f"Day Index within {block_length}-Day Block",
        yaxis_title=series.name or "Value",
        width=width,
        height=height,
        legend=dict(title="Block Number", orientation="v", x=1.02, y=1),
        margin=dict(l=60, r=200, t=60, b=60),
    )

    fig.show()


# plotly_average_seasonality.py
# This function computes the average value for each day‐index across all blocks
# of a specified length in a Pandas Series, then displays an interactive Plotly chart
# showing that mean curve and a ±1 standard‐deviation band.
def plotly_average_seasonality(
    series: pd.Series,
    block_length: int = 91,
    title: str = None,
    width: int = 900,
    height: int = 500,
) -> None:
    """
    Compute the average value for each day‐index (0 through block_length-1)
    across all consecutive blocks in `series`, then display an interactive
    Plotly chart showing that mean curve and a ±1 standard‐deviation band.

    Parameters
    ----------
    series : pd.Series
        A Pandas Series indexed by datetime (or convertible to daily). The function
        will reindex to one row per calendar day (filling any gaps by linear interpolation).
    block_length : int, default 91
        Number of days in each “block.” The series will be split into floor(N/91) blocks.
    title : str, optional
        Title for the figure. Defaults to
        "Average {block_length}-Day Seasonality (n={n_blocks} blocks)".
    width : int, default 900
        Width of the Plotly figure in pixels.
    height : int, default 500
        Height of the Plotly figure in pixels.
    """
    # 1) Ensure the index is daily‐frequency and sorted
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            raise TypeError(
                "Index must be a DatetimeIndex or convertible to datetime."
            ) from e

    s = series.sort_index().asfreq("D")  # reindex daily
    s = s.interpolate(method="linear").dropna()

    # 2) Determine how many full blocks we can extract
    N = len(s)
    n_blocks = N // block_length
    if n_blocks < 1:
        raise ValueError(
            f"Series is too short for even one block of length {block_length} days."
        )

    # 3) Trim off the extra days so that we have an exact multiple of block_length
    trimmed = s.iloc[: n_blocks * block_length]
    arr = trimmed.values.reshape((n_blocks, block_length))

    # 4) Compute per-day-of-block mean and standard deviation
    day_means = arr.mean(axis=0)  # shape = (block_length,)
    day_stds = arr.std(axis=0)  # shape = (block_length,)

    x = np.arange(block_length)

    # 5) Build Plotly figure
    fig = go.Figure()

    # (a) Plot the mean line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=day_means,
            mode="lines+markers",
            line=dict(color="blue"),
            name="Mean",
            hovertemplate="Day Index: %{x}<br>Mean: %{y:.2f}<extra></extra>",
        )
    )

    # (b) Plot the ±1 std-dev shaded area
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([day_means + day_stds, (day_means - day_stds)[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="±1 Std. Dev.",
        )
    )

    # 6) Final layout settings
    fig.update_layout(
        title=title or f"Average {block_length}-Day Seasonality (n={n_blocks} blocks)",
        xaxis_title=f"Day Index within {block_length}-Day Block",
        yaxis_title=series.name or "Value",
        width=width,
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(x=0.01, y=0.99),
    )

    fig.show()


# detrended_seasonality.py
# This function splits a Pandas Series into consecutive blocks of a specified length,
# subtracts each block’s own mean to “detrend” it, and then overlays the results


def plotly_detrended_seasonality(
    series: pd.Series,
    block_length: int = 7,
    title: str = None,
    line_opacity: float = 0.6,
    width: int = 900,
    height: int = 500,
) -> None:
    """
    Split the input series into consecutive blocks of `block_length` days,
    subtract each block’s own mean, and overlay the results as interactive Plotly lines
    so the repeated shape is easy to inspect.

    Parameters
    ----------
    series : pd.Series
        A Pandas Series indexed by datetime (or convertible to daily). This function
        will reindex it to daily frequency (filling any gaps via linear interpolation).
    block_length : int, default 7
        Number of days per block. The series is split into floor(N / block_length) blocks.
    title : str, optional
        Title for the plot. If None, defaults to
        "Detrended {block_length}-Day Seasonality (n={n_blocks} blocks overlaid)".
    line_opacity : float, default 0.6
        Opacity of each overlaid line (0.0 to 1.0). Lower values help visualize overlap.
    width : int, default 900
        Width of the figure in pixels.
    height : int, default 500
        Height of the figure in pixels.
    """
    # 1) Ensure the index is a daily‐frequency DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            raise TypeError(
                "Index must be a DatetimeIndex or convertible to datetime."
            ) from e

    # Sort by date, reindex to one row per calendar day, then interpolate gaps
    s = series.sort_index().asfreq("D")
    s = s.interpolate(method="linear").dropna()

    # 2) Determine how many full blocks we can extract
    N = len(s)
    n_blocks = N // block_length
    if n_blocks < 1:
        raise ValueError(
            f"Series is too short for even one block of length {block_length} days."
        )

    # 3) Trim off any extra days so the length is exactly n_blocks * block_length
    trimmed = s.iloc[: n_blocks * block_length]
    arr = trimmed.values.reshape((n_blocks, block_length))

    # 4) Subtract each block’s own mean to “detrend” by block
    block_means = arr.mean(axis=1).reshape(n_blocks, 1)  # shape = (n_blocks, 1)
    detrended = arr - block_means  # shape = (n_blocks, block_length)

    x = np.arange(block_length)  # day indices: 0, 1, ..., block_length-1

    # 5) Build the Plotly figure
    fig = go.Figure()

    for i in range(n_blocks):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=detrended[i],
                mode="lines",
                name=f"Block {i + 1}",
                opacity=line_opacity,
                hovertemplate="Day Index: %{x}<br>Value (detrended): %{y:.2f}<extra></extra>",
            )
        )

    # 6) Add a horizontal zero‐line for reference
    fig.add_trace(
        go.Scatter(
            x=[0, block_length - 1],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 7) Final layout adjustments
    fig.update_layout(
        title=title
        or f"Detrended {block_length}-Day Seasonality ({n_blocks} blocks overlaid)",
        xaxis_title=f"Day Index within {block_length}-Day Block",
        yaxis_title=f"{series.name} (demeaned by block)",
        width=width,
        height=height,
        margin=dict(l=60, r=200, t=60, b=60),
        legend=dict(title="Block Number", orientation="v", x=1.02, y=1),
    )

    fig.show()
