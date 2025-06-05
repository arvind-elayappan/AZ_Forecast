# auto_ts_analysis.py

"""
This module automates running a suite of EDA and time‐series tests (including Plotly‐based
seasonality visualizations) on multiple target columns. It imports helper functions from
eda_tools.py and tests.py (your time‐series test module).

Usage Example:
--------------
from auto_ts_analysis import run_full_analysis
import pandas as pd

df = pd.read_csv("tetuan_city.csv")
results = run_full_analysis(
    df=df,
    datetime_col="date_time",
    target_cols=["zone1_consumption", "zone2_consumption", "zone3_consumption"],
    other_numeric_cols=["temperature", "humidity"],
    decomposition_period=7,   # for classical seasonal_decompose and for plotly block_length
    adf_regression="c",
    kpss_regression="c",
    output_dir="eda_outputs"
)

After running, you will see:
  - Inline matplotlib plots: raw time series, decomposition, ACF/PACF
  - Inline Plotly figures: overlay seasonality, average seasonality, detrended seasonality
  - A folder "eda_outputs" with saved PNGs of the matplotlib‐based plots
  - A dictionary `results` containing summary statistics and decomposition objects
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Import EDA helper functions
from eda_tools import (
    set_datetime_index,
    plot_time_series_for_column,
    get_correlation_matrix,
    plot_correlation_heatmap,
)

# Import classical decomposition and stationarity tests
from tests import (
    plot_seasonal_decompose,
    test_stationarity,
    plot_acf_pacf,
    plotly_overlay_seasonality,
    plotly_average_seasonality,
    plotly_detrended_seasonality,
)


def run_full_analysis(
    df: pd.DataFrame,
    datetime_col: str,
    target_cols: list,
    other_numeric_cols: list = None,
    decomposition_period: int = None,
    adf_regression: str = "c",
    kpss_regression: str = "c",
    output_dir: str = None,
):
    """
    Run a full suite of time‐series EDA, stationarity tests, ACF/PACF, and Plotly‐based seasonality
    visualizations on multiple target columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least one datetime column and numeric columns.
    datetime_col : str
        Column name in `df` that holds datetime values. Will be converted to index.
    target_cols : list of str
        List of column names to treat as separate time series (e.g., ["zone1", "zone2", "zone3"]).
    other_numeric_cols : list of str, optional
        Additional numeric columns to summarize (e.g., ["temperature", "humidity"]).
    decomposition_period : int, optional
        Periodicity parameter for classical seasonal decomposition (e.g., 7 for weekly). Also used
        as `block_length` for Plotly‐based seasonality functions. If None, decomposition and Plotly
        seasonality steps are skipped.
    adf_regression : str, default "c"
        Regression parameter for the ADF test ("c", "ct", "nc").
    kpss_regression : str, default "c"
        Regression parameter for the KPSS test ("c", "ct").
    output_dir : str, optional
        If provided, matplotlib‐based plots will also be saved under this directory. The directory
        is created if it does not exist.

    Returns
    -------
    results_dict : dict
        A dictionary where each key is a column name from `target_cols` or `other_numeric_cols`.
        For each target column, the sub‐dictionary contains:
          - "decomposition": statsmodels SeasonalDecomposeResult (or None if skipped/failed)
          - "adf": None (stationarity test prints to console)
          - "kpss": None (stationarity test prints to console)
          - "summary": pd.Series.describe() for that column
        For each other_numeric_col, the sub‐dictionary contains:
          - "summary": pd.Series.describe()
    """
    # 1) Create output directory if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 2) Copy DataFrame and convert `datetime_col` to DatetimeIndex
    df_copy = df.copy()
    if datetime_col not in df_copy.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not found in DataFrame.")
    df_copy = set_datetime_index(df_copy, datetime_col=datetime_col)

    # 3) Validate target columns
    for col in target_cols:
        if col not in df_copy.columns:
            raise KeyError(f"Target column '{col}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            raise TypeError(f"Target column '{col}' is not numeric.")

    # 4) Validate other numeric columns (if any)
    if other_numeric_cols:
        for col in other_numeric_cols:
            if col not in df_copy.columns:
                raise KeyError(f"Other numeric column '{col}' not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                raise TypeError(f"Column '{col}' is not numeric.")

    results_dict = {}

    # 5) Loop over each target time‐series column
    for col in target_cols:
        col_dict = {}

        # 5.a) Plot raw time series (matplotlib)
        print(f"\n========= Plotting Raw Time Series for '{col}' =========")
        plot_time_series_for_column(
            df_copy, column=col, title=f"Raw Time Series: {col}"
        )
        # Save raw time series plot
        if output_dir:
            fig = df_copy[col].plot(title=f"Raw Time Series: {col}", figsize=(12, 4))
            fig.figure.savefig(os.path.join(output_dir, f"{col}_raw_timeseries.png"))
            plt.close(fig.figure)

        # 5.b) Classical seasonal decomposition (statsmodels) if period provided
        decom_result = None
        if decomposition_period:
            print(
                f"\n--- Classical Seasonal Decomposition for '{col}' (period={decomposition_period}) ---"
            )
            # Use the existing plot_seasonal_decompose to show the plot inline
            try:
                plot_seasonal_decompose(
                    df=df_copy,
                    column=col,
                    model="additive",
                    period=decomposition_period,
                    figsize=(12, 8),
                    title=f"Seasonal Decomposition: {col}",
                )
            except Exception as e:
                print(f"Could not plot decomposition for '{col}': {e}")

            # Capture the decomposition object
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose

                series = df_copy[col].dropna().sort_index()
                decom_result = seasonal_decompose(
                    series, model="additive", period=decomposition_period
                )
                # Save the decomposition plot
                fig = decom_result.plot()
                fig.suptitle(f"Seasonal Decomposition: {col}", fontsize=16)
                fig.tight_layout()
                if output_dir:
                    fig.savefig(os.path.join(output_dir, f"{col}_decomposition.png"))
                plt.close(fig)
            except Exception as e:
                print(f"Could not capture decomposition object for '{col}': {e}")
                decom_result = None

        col_dict["decomposition"] = decom_result

        # 5.c) Stationarity tests (ADF & KPSS)
        print(f"\n--- Stationarity Tests for '{col}' ---")
        test_stationarity(
            df=df_copy,
            column=col,
            window=None,
            freq=None,
            adf_regression=adf_regression,
            kpss_regression=kpss_regression,
        )
        # Note: test_stationarity() prints results; it does not return values
        col_dict["adf"] = None
        col_dict["kpss"] = None

        # 5.d) ACF & PACF plots (matplotlib)
        print(f"\n--- ACF & PACF for '{col}' ---")
        plot_acf_pacf(
            df=df_copy,
            column=col,
            lags=28,
            pacf_method="ywm",
            figsize=(14, 4),
            title_suffix="",
        )
        # Save ACF & PACF plots
        if output_dir:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

            series = df_copy[col].dropna().sort_index()

            # ACF
            fig_acf = plot_acf(series, lags=28, alpha=0.05)
            fig_acf.figure.savefig(os.path.join(output_dir, f"{col}_acf.png"))
            plt.close(fig_acf.figure)

            # PACF
            fig_pacf = plot_pacf(series, lags=28, method="ywm", alpha=0.05)
            fig_pacf.figure.savefig(os.path.join(output_dir, f"{col}_pacf.png"))
            plt.close(fig_pacf.figure)

        # 5.e) Summary statistics
        summary_series = df_copy[col].dropna().describe()
        print(f"\n--- Summary Statistics for '{col}' ---\n{summary_series}\n")
        col_dict["summary"] = summary_series

        # 5.f) Plotly‐based seasonality visualizations (if period provided)
        if decomposition_period:
            series = df_copy[col].dropna().sort_index()

            # Overlay Seasonality
            print(
                f"\n--- Plotly Overlay Seasonality for '{col}' "
                f"(block_length={decomposition_period}) ---"
            )
            try:
                plotly_overlay_seasonality(
                    series,
                    block_length=decomposition_period,
                    title=f"{col}: {decomposition_period}-Day Overlay Seasonality",
                )
            except Exception as e:
                print(f"Could not plot overlay seasonality for '{col}': {e}")

            # Average Seasonality
            print(
                f"\n--- Plotly Average Seasonality for '{col}' "
                f"(block_length={decomposition_period}) ---"
            )
            try:
                plotly_average_seasonality(
                    series,
                    block_length=decomposition_period,
                    title=f"{col}: {decomposition_period}-Day Average Seasonality",
                )
            except Exception as e:
                print(f"Could not plot average seasonality for '{col}': {e}")

            # Detrended Seasonality
            print(
                f"\n--- Plotly Detrended Seasonality for '{col}' "
                f"(block_length={decomposition_period}) ---"
            )
            try:
                plotly_detrended_seasonality(
                    series,
                    block_length=decomposition_period,
                    title=f"{col}: {decomposition_period}-Day Detrended Seasonality",
                )
            except Exception as e:
                print(f"Could not plot detrended seasonality for '{col}': {e}")

        # Add the results for this column
        results_dict[col] = col_dict

    # 6) Summaries for other numeric columns (if any)
    if other_numeric_cols:
        for col in other_numeric_cols:
            print(f"\n========= Summary for Other Numeric Column '{col}' =========")
            summary_series = df_copy[col].dropna().describe()
            print(summary_series)
            results_dict[col] = {"summary": summary_series}

    # 7) Correlation matrix & heatmap (targets + other numeric)
    if other_numeric_cols:
        print("\n========= Correlation Matrix (Targets + Other Numerics) =========")
        combined_cols = target_cols + other_numeric_cols
        corr_mat = get_correlation_matrix(df_copy, cols=combined_cols)
        print(corr_mat)
        plot_correlation_heatmap(
            df_copy,
            cols=combined_cols,
            figsize=(8, 6),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            title="Correlation Matrix (Targets + Other Numerics)",
        )
        if output_dir:
            plt.figure(figsize=(8, 6))
            import seaborn as sns

            sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Correlation Matrix (Targets + Other Numerics)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
            plt.close()

    return results_dict
