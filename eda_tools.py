# eda_tools.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns


# removes extra spaces and '#' characters from column names
import re


def clean_column_names(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Normalize column names by:
      - Collapsing any sequence of whitespace into a single space
      - Stripping leading/trailing whitespace
    If inplace=True (default), modifies df.columns directly and returns None.
    If inplace=False, returns a new DataFrame with cleaned column names.

    Usage:
        # In‐place rename:
        clean_column_names(df)

        # Or get a new renamed DataFrame:
        df2 = clean_column_names(df, inplace=False)
    """
    # Build cleaned names
    cleaned = [re.sub(r"\s+", " ", col).strip() for col in df.columns]

    if inplace:
        df.columns = cleaned
        return None
    else:
        return df.rename(columns={old: new for old, new in zip(df.columns, cleaned)})


# ------------------------------------------------------------
# 1) DATETIME CONVERSION & INDEXING
# ------------------------------------------------------------
def set_datetime_index(
    df: pd.DataFrame,
    datetime_col: str,
    infer_format: bool = False,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Convert a column to datetime (if not already), then set it as the DataFrame index.
    - datetime_col: name of the column to convert & set.
    - infer_format: if True, uses pd.to_datetime(..., infer_datetime_format=True).
    - drop_original: included for API compatibility—but set_index already “moves”
                     the column into the index, so no explicit drop is needed.
    Returns a new DataFrame with a sorted DatetimeIndex.

    Usage example:
        df = set_datetime_index(df, datetime_col="DateTime", infer_format=True)
    """
    df = df.copy()

    # 1) Convert to datetime dtype if needed
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(
            df[datetime_col], infer_datetime_format=infer_format, errors="coerce"
        )

    # 2) Set as index
    df = df.set_index(datetime_col)

    # 3) Sort the index
    df = df.sort_index()

    return df


# 4) MISSING / NULL VALUE CHECKS & HANDLING
def drop_missing_rows(
    df: pd.DataFrame, subset: list = None, how: str = "any"
) -> pd.DataFrame:
    """
    Return a new DataFrame with rows dropped according to missing‐value rules.
    - subset: list of columns to consider (default: all columns).
    - how: 'any' or 'all' (passed to df.dropna).
    """
    return df.dropna(subset=subset, how=how)


def fill_missing(
    df: pd.DataFrame, method: str = "ffill", limit: int = None
) -> pd.DataFrame:
    """
    Return a new DataFrame with missing values filled.
    - method: 'ffill' (forward fill), 'bfill' (backward fill), or 'interpolate'
    - limit: maximum number of consecutive NaNs to fill (None = no limit)
    """
    df = df.copy()
    if method in ["ffill", "pad", "bfill", "backfill"]:
        df = df.fillna(method=method, limit=limit)
    elif method == "interpolate":
        df = df.interpolate(limit=limit)
    else:
        raise ValueError(
            "Unsupported fill method: choose 'ffill', 'bfill', or 'interpolate'."
        )
    return df


# ------------------------------------------------------------
# 5) TIME‐SERIES PLOTTING (ALL OR INDIVIDUAL COLUMNS)
# ------------------------------------------------------------
def plot_time_series_for_column(
    df: pd.DataFrame,
    column: str,
    figsize: tuple = (12, 4),
    title: str = None,
    xlabel: str = "Index",
    ylabel: str = None,
) -> None:
    """
    Plot a single numeric column versus its index. If the index is a DateTimeIndex,
    this yields a proper time‐series. Otherwise, it plots vs. the row index.

    Usage:
       plot_time_series_for_column(df, column="Temperature")
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column '{column}' is not numeric; skipping plot.")
        return

    plt.figure(figsize=figsize)
    df[column].plot(
        title=title or f"{column} (Time‐Series)",
        xlabel=xlabel,
        ylabel=ylabel or column,
    )
    plt.tight_layout()
    plt.show()


def plot_each_time_series(
    df: pd.DataFrame, figsize: tuple = (10, 4), xlabel: str = "DateTime"
) -> None:
    """
    For each numeric column in df, create a separate time‐series plot (one figure per column).
    - df: DataFrame whose index is either a DatetimeIndex or something plottable.
    - figsize: tuple controlling the size of each individual figure.
    - xlabel: label to put on the x-axis (defaults to "DateTime").

    Example usage:
        plot_each_time_series(df)
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        print("No numeric columns to plot.")
        return

    for col in numeric_df.columns:
        plt.figure(figsize=figsize)
        numeric_df[col].plot(title=f"Time Series: {col}", xlabel=xlabel, ylabel=col)
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# single column time‐series plotting with resampling/aggregation
# ------------------------------------------------------------


def plot_time_series_for_column(
    df: pd.DataFrame,
    column: str,
    figsize: tuple = (12, 4),
    title: str = None,
    xlabel: str = "Index",
    ylabel: str = None,
    freq: str = None,
    how: str = "mean",
) -> None:
    """
    Plot a single numeric column versus its index. If `freq` is provided, resample
    the series at that frequency using the specified aggregation `how` before plotting.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    column : str
        Name of the numeric column to plot.
    figsize : tuple, default=(12, 4)
        Size of the figure.
    title : str, optional
        Title of the plot. If None, defaults to "{column} (Time‐Series)".
    xlabel : str, default="Index"
        Label for the x‐axis.
    ylabel : str, optional
        Label for the y‐axis. If None, uses the column name.
    freq : str, optional
        Resampling frequency string (e.g., 'D', 'H', 'M'). If provided, the series
        is resampled before plotting. Requires a DatetimeIndex.
    how : str, default="mean"
        Aggregation method to use when resampling (e.g., 'mean', 'sum', 'max', 'min').

    Usage Examples:
    ---------------
    # Plot raw time‐series:
    plot_time_series_for_column(df, column="Temperature")

    # Plot daily mean:
    plot_time_series_for_column(df, column="Temperature", freq="D", how="mean")

    # Plot monthly sum:
    plot_time_series_for_column(df, column="Zone 1 Power Consumption", freq="M", how="sum")
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column '{column}' is not numeric; skipping plot.")
        return

    series = df[column]
    if freq is not None:
        # Ensure index is a DatetimeIndex before resampling
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex to resample.")
        # Resample using the specified aggregation
        try:
            series = getattr(series.resample(freq), how)()
        except AttributeError:
            raise ValueError(f"Aggregation method '{how}' is not valid for resampling.")

    plt.figure(figsize=figsize)
    series.plot(
        title=title
        or f"{column} (Time‐Series){' [' + freq + ' ' + how + ']' if freq else ''}",
        xlabel=xlabel,
        ylabel=ylabel or column,
    )
    plt.tight_layout()
    plt.show()


# 6) RESAMPLING / AGGREGATION PLOTTING (ALL COLUMNS)


def plot_resampled_split(
    df: pd.DataFrame,
    freq: str,
    how: str = "mean",
    figsize: tuple = (14, 6),
    title_high: str = None,
    title_low: str = None,
    xlabel: str = "Date",
    ylabel_high: str = "Value (High-Range Variables)",
    ylabel_low: str = "Value (Low-Range Variables)",
) -> None:
    """
    Resample `df` at the given frequency (e.g. "D", "H", "M") using the specified
    aggregation method (e.g. "mean", "sum", "max", etc.). Then automatically split
    the resampled numeric columns into two groups—those with a “large” range vs. those
    with a “small” range—using the MEDIAN of all column‐ranges as the threshold.

    Finally, plot them side-by-side:
      - Left subplot: high-range columns (range ≥ median_range)
      - Right subplot: low-range columns (range <  median_range)

    This ensures that columns whose values differ by orders of magnitude do not
    overwhelm smaller-magnitude columns in the same plot.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame (index can be anything—if you want a true time-series,
        set df.index to a DatetimeIndex beforehand).
    freq : str
        Resampling frequency string (e.g. "D" for daily, "H" for hourly, "M" for monthly).
    how : str, default "mean"
        Aggregation method to apply during resample (any valid DataFrame.resample(...).how()).
    figsize : tuple, default (14, 6)
        Overall figure size (width, height). Each subplot will get half the width.
    title_high : str, optional
        Title for the high-range subplot. If None, defaults to "High-Range Variables (Resampled)".
    title_low : str, optional
        Title for the low-range subplot. If None, defaults to "Low-Range Variables (Resampled)".
    xlabel : str, default "Date"
        Label for the shared x-axis (typically a date).
    ylabel_high : str, default "Value (High-Range Variables)"
        Y-axis label for the high-range subplot.
    ylabel_low : str, default "Value (Low-Range Variables)"
        Y-axis label for the low-range subplot.

    Usage Example
    -------------
    # 1) If needed, set your DataFrame’s index to be a DatetimeIndex:
    #    df = set_datetime_index(df, datetime_col="DateTime", infer_format=True)

    # 2) Call plot_resampled_split:
    plot_resampled_split(
        df,
        freq="D",
        how="sum",
        figsize=(16, 6),
        title_high="Daily Sum: High-Range Variables",
        title_low="Daily Sum: Low-Range Variables",
        xlabel="Date"
    )
    """
    # 1) Select numeric columns and resample
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        print("No numeric columns to resample/plot.")
        return

    # Perform resample + aggregation
    resampled = getattr(numeric_df.resample(freq), how)()

    # 2) Compute range (max – min) for each column in the resampled data
    col_ranges = resampled.max() - resampled.min()

    # 3) Find the median of those ranges
    median_range = col_ranges.median()

    # 4) Split columns into high vs low range
    high_cols = col_ranges[col_ranges >= median_range].index.tolist()
    low_cols = col_ranges[col_ranges < median_range].index.tolist()

    # 5) Prepare subplot grid: 1 row, 2 columns
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharex=True,  # share the same x‐axis scale
    )

    # Left subplot: HIGH-RANGE
    ax_high = axes[0]
    if len(high_cols) > 0:
        resampled[high_cols].plot(
            ax=ax_high,
            legend=True,
            title=title_high or f"High-Range Columns (Range ≥ {median_range:.2f})",
            xlabel=xlabel,
            ylabel=ylabel_high,
        )
    else:
        ax_high.text(
            0.5,
            0.5,
            "No high-range columns found",
            ha="center",
            va="center",
            transform=ax_high.transAxes,
        )
        ax_high.set_title("High-Range Columns (None)")
        ax_high.set_xlabel("")
        ax_high.set_ylabel(ylabel_high)

    # Right subplot: LOW-RANGE
    ax_low = axes[1]
    if len(low_cols) > 0:
        resampled[low_cols].plot(
            ax=ax_low,
            legend=True,
            title=title_low or f"Low-Range Columns (Range < {median_range:.2f})",
            xlabel=xlabel,
            ylabel=ylabel_low,
        )
    else:
        ax_low.text(
            0.5,
            0.5,
            "No low-range columns found",
            ha="center",
            va="center",
            transform=ax_low.transAxes,
        )
        ax_low.set_title("Low-Range Columns (None)")
        ax_low.set_xlabel("")
        ax_low.set_ylabel(ylabel_low)

    plt.tight_layout()
    plt.show()


def plot_each_hourly_aggregation(
    df: pd.DataFrame, how: str = "mean", figsize: tuple = (10, 4), xlabel: str = "Hour"
) -> None:
    """
    For each numeric column in df, resample to hourly frequency and plot the aggregated series.
    - how: aggregation method, e.g. 'mean', 'sum', 'max', 'min', etc.
    - figsize: size of each individual plot.
    - xlabel: label for the x-axis (defaults to "Hour").

    Requirements:
    - df.index must be a DatetimeIndex.

    Example:
        plot_each_hourly_aggregation(df, how="mean")
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex to use hourly aggregation.")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        print("No numeric columns found to aggregate.")
        return

    for col in numeric_df.columns:
        hourly_series = getattr(numeric_df[col].resample("H"), how)()

        plt.figure(figsize=figsize)
        hourly_series.plot(
            title=f"Hourly {how.capitalize()} of '{col}'", xlabel=xlabel, ylabel=col
        )
        plt.tight_layout()
        plt.show()


def plot_each_daily_aggregation(
    df: pd.DataFrame, how: str = "mean", figsize: tuple = (10, 4), xlabel: str = "Date"
) -> None:
    """
    For each numeric column in df, resample to daily frequency and plot the aggregated series.
    - how: aggregation method, e.g. 'mean', 'sum', 'max', 'min', etc.
    - figsize: size of each individual plot.
    - xlabel: label for the x-axis (defaults to "Date").

    Requirements:
    - df.index must be a DatetimeIndex.

    Example:
        plot_each_daily_aggregation(df, how="mean")
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex to use daily aggregation.")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        print("No numeric columns found to aggregate.")
        return

    for col in numeric_df.columns:
        daily_series = getattr(numeric_df[col].resample("D"), how)()

        plt.figure(figsize=figsize)
        daily_series.plot(
            title=f"Daily {how.capitalize()} of '{col}'", xlabel=xlabel, ylabel=col
        )
        plt.tight_layout()
        plt.show()


def plot_each_monthly_aggregation(
    df: pd.DataFrame, how: str = "mean", figsize: tuple = (10, 4), xlabel: str = "Month"
) -> None:
    """
    For each numeric column in df, resample to monthly frequency and plot the aggregated series.
    - how: aggregation method, e.g. 'mean', 'sum', 'max', 'min', etc.
    - figsize: size of each individual plot.
    - xlabel: label for the x-axis (defaults to "Month").

    Requirements:
    - df.index must be a DatetimeIndex.

    Example:
        plot_each_monthly_aggregation(df, how="mean")
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex to use monthly aggregation.")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        print("No numeric columns found to aggregate.")
        return

    for col in numeric_df.columns:
        monthly_series = getattr(numeric_df[col].resample("M"), how)()

        plt.figure(figsize=figsize)
        monthly_series.plot(
            title=f"Monthly {how.capitalize()} of '{col}'", xlabel=xlabel, ylabel=col
        )
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------------
# ------------------------------------------------------------
# 9) DATAFRAME AGGREGATION BY FREQUENCY
# ------------------------------------------------------------
def aggregate_time_series(
    df: pd.DataFrame, freq: str, how: str = "mean"
) -> pd.DataFrame:
    """
    Resample and aggregate the DataFrame at a given frequency, returning a new DataFrame.
    Only numeric columns are aggregated; non-numeric columns are dropped.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame. Its index must be a DatetimeIndex if you want true time-based
        aggregation. If not, this function will attempt to convert the index to datetime first.
    freq : str
        A pandas offset alias string for resampling. Examples:
          - "H"  → hourly
          - "D"  → daily
          - "M"  → month end
          - "MS" → month start
          - "Y"  → year end
          - "YS" → year start
          - "Q"  → quarterly
        (See pandas documentation for all valid offset aliases.)
    how : str, default "mean"
        Aggregation method to apply. Must correspond to a valid pandas Series method under
        `.resample(freq)`, e.g., "mean", "sum", "max", "min", "median", "first", "last".

    Returns:
    --------
    pd.DataFrame
        A new DataFrame whose index is the resampled frequency and whose columns are
        the numeric columns from the original `df`, aggregated by `how`.

    Raises:
    -------
    TypeError:
        If the index is not a DatetimeIndex and cannot be converted to one.
    ValueError:
        If `how` is not a valid aggregation method.

    Usage Examples:
    ---------------
    # Suppose df has a DatetimeIndex and numeric columns like "Temperature", "Zone 1 Power".
    # 1) Hourly mean:
    hourly_df = aggregate_time_series(df, freq="H", how="mean")

    # 2) Daily sum:
    daily_df = aggregate_time_series(df, freq="D", how="sum")

    # 3) Monthly max:
    monthly_df = aggregate_time_series(df, freq="M", how="max")

    # 4) Yearly median:
    yearly_df = aggregate_time_series(df, freq="Y", how="median")
    """
    df_copy = df.copy()

    # Ensure the index is datetime
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        try:
            df_copy.index = pd.to_datetime(df_copy.index)
        except Exception as e:
            raise TypeError(
                "The DataFrame index must be a DatetimeIndex or convertible to datetime."
            ) from e

    # Select only numeric columns for aggregation
    numeric_df = df_copy.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns available to aggregate.")

    # Attempt to resample + aggregate
    try:
        aggregated = getattr(numeric_df.resample(freq), how)()
    except AttributeError as e:
        raise ValueError(
            f"Aggregation method '{how}' is not valid for resampling."
        ) from e

    return aggregated


# def get_hourly_aggregation(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
#     """
#     Convenience wrapper to get an hourly‐aggregated DataFrame.

#     Equivalent to:
#         aggregate_time_series(df, freq="H", how=how)

#     Example:
#         hourly_mean_df = get_hourly_aggregation(df, how="mean")
#         hourly_sum_df  = get_hourly_aggregation(df, how="sum")
#     """
#     return aggregate_time_series(df, freq="H", how=how)


# def get_daily_aggregation(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
#     """
#     Convenience wrapper to get a daily‐aggregated DataFrame.

#     Equivalent to:
#         aggregate_time_series(df, freq="D", how=how)

#     Example:
#         daily_mean_df = get_daily_aggregation(df, how="mean")
#         daily_sum_df  = get_daily_aggregation(df, how="sum")
#     """
#     return aggregate_time_series(df, freq="D", how=how)


# def get_monthly_aggregation(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
#     """
#     Convenience wrapper to get a monthly‐aggregated DataFrame (calendar‐month end).

#     Equivalent to:
#         aggregate_time_series(df, freq="M", how=how)

#     Example:
#         monthly_max_df   = get_monthly_aggregation(df, how="max")
#         monthly_median_df = get_monthly_aggregation(df, how="median")
#     """
#     return aggregate_time_series(df, freq="M", how=how)


# def get_yearly_aggregation(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
#     """
#     Convenience wrapper to get a yearly‐aggregated DataFrame (calendar‐year end).

#     Equivalent to:
#         aggregate_time_series(df, freq="Y", how=how)

#     Example:
#         yearly_sum_df = get_yearly_aggregation(df, how="sum")
#         yearly_min_df = get_yearly_aggregation(df, how="min")
#     """
#     return aggregate_time_series(df, freq="Y", how=how)


# ------------------------------------------------------------
# 7) CORRELATION MATRIX & HEATMAP
# ------------------------------------------------------------
def get_correlation_matrix(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Return the correlation matrix (Pearson) for the specified columns.
    If cols is None, uses all numeric columns.
    """
    if cols is None:
        numeric_df = df.select_dtypes(include="number")
    else:
        missing_cols = set(cols) - set(df.columns)
        if missing_cols:
            raise KeyError(f"Columns not found: {missing_cols}")
        numeric_df = df[cols].select_dtypes(include="number")
    return numeric_df.corr()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list = None,
    figsize: tuple = (8, 6),
    annot: bool = True,
    cmap: str = "coolwarm",
    fmt: str = ".2f",
    title: str = "Correlation Matrix",
) -> None:
    """
    Compute the correlation matrix among numeric columns (or a subset) and display
    it as a heatmap. By default, annotates each cell with the correlation coefficient.

    Example:
        plot_correlation_heatmap(df, cols=['Temperature','Humidity','Zone 1 Power Consumption'])
    """
    corr_mat = get_correlation_matrix(df, cols=cols)
    plt.figure(figsize=figsize)
    sns.heatmap(corr_mat, annot=annot, fmt=fmt, cmap=cmap, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Create a seaborn pairplot (scatterplot matrix) of the specified columns
def plot_pairplot(
    df: pd.DataFrame,
    columns: list = None,
    max_vars: int = 5,
    dropna: bool = True,
    **pairplot_kwargs,
) -> None:
    """
    Create a seaborn pairplot (scatterplot matrix) of the specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing your data.
    columns : list, optional (default=None)
        A list of column names to include in the pairplot. If None, the function will:
            1) Automatically select numeric columns
            2) If the number of numeric columns <= max_vars, use all of them
            3) Otherwise, skip plotting and print a message.
    max_vars : int, default=5
        The maximum number of numeric columns to plot if `columns` is None.
        Only used in the “auto‐select” case.
    dropna : bool, default=True
        Whether to drop rows with any NaN in the chosen columns before plotting.
        If False, NaNs remain (may cause seaborn to error or plot oddly).
    **pairplot_kwargs :
        Any additional keyword arguments you wish to forward to `sns.pairplot()`.
        For example, `hue='Category'`, `diag_kind='hist'`, etc.

    Usage Examples:
    ---------------
    # 1) Auto‐select up to max_vars numeric columns:
    plot_pairplot(df, max_vars=4)

    # 2) Explicitly specify which columns to plot (bypasses max_vars):
    plot_pairplot(df, columns=['Temperature','Humidity','Wind Speed'])

    # 3) Add a hue and specify diag_kind:
    plot_pairplot(
        df,
        columns=['Temperature','Humidity','Zone 1 Power Consumption'],
        hue='CategoryColumn',
        diag_kind='kde'
    )
    """
    # If the user explicitly provided columns, validate them:
    if columns is not None:
        # 1) Check that all columns exist in df
        missing = set(columns) - set(df.columns)
        if missing:
            raise KeyError(
                f"The following columns were not found in the DataFrame: {missing}"
            )
        # 2) Optionally—or obligatorily—ensure they are numeric (unless the user wants a hue)
        #    We'll allow non‐numeric if user also passes a valid hue in pairplot_kwargs.
        numeric_cols = [
            col for col in columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        non_numeric_cols = [
            col for col in columns if not pd.api.types.is_numeric_dtype(df[col])
        ]

        if non_numeric_cols and "hue" not in pairplot_kwargs:
            raise TypeError(
                f"Columns {non_numeric_cols} are not numeric. "
                "If you want to include a non‐numeric column, pass it as the `hue` argument instead."
            )

        plot_cols = columns[:]  # Use the user‐provided list verbatim

    else:
        # No columns explicitly provided → auto‐select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            print("No numeric columns available to plot.")
            return

        if len(numeric_cols) <= max_vars:
            plot_cols = numeric_cols
        else:
            print(
                f"Number of numeric columns ({len(numeric_cols)}) exceeds max_vars ({max_vars}).\n"
                "To override, explicitly pass a smaller `columns=[...]` list to this function."
            )
            return

    # At this point, plot_cols is the final list of columns to include in the pairplot.
    if dropna:
        data_to_plot = df[plot_cols].dropna()
    else:
        data_to_plot = df[plot_cols]

    if data_to_plot.shape[0] == 0:
        print("After dropping NaNs, no rows remain. Pairplot cannot be created.")
        return

    # Finally, create the pairplot
    sns.pairplot(data_to_plot, **pairplot_kwargs)
    plt.tight_layout()
    plt.show()


# 8) BOX PLOTS FOR OUTLIER DETECTION (NUMERIC)
def plot_boxplots(df: pd.DataFrame, figsize: tuple = (8, 5)) -> None:
    """
    For each numeric column:
      - Plot a vertical boxplot.

    Usage:
        plot_boxplots(df)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns to plot boxplots.")
        return

    for col in numeric_cols:
        plt.figure(figsize=figsize)
        sns.boxplot(y=df[col].dropna())
        plt.title(f"Box Plot of '{col}'")
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# 8) “ALL‐IN‐ONE” OVERVIEW: SHAPE, DTYPES, INFO, HEAD, DESCRIBE, MISSING SUMMARY
# ------------------------------------------------------------


def summarize_dataframe(df: pd.DataFrame, head_n: int = 5) -> None:
    """
    Print a complete overview of the DataFrame, including:
      1) Shape
      2) Dtypes
      3) df.info()
      4) First `head_n` rows
      5) Descriptive statistics (df.describe().T)
      6) Missing‐value summary

    All logic is contained within this single function, without calls to other helpers.
    """
    # 1) Shape
    print("----- 1) SHAPE -----")
    print(df.shape)
    print()

    # 2) Dtypes
    print("----- 2) DTYPES -----")
    print(df.dtypes)
    print()

    # 3) Info
    print("----- 3) INFO -----")
    df.info()
    print()

    # 4) Head
    print(f"----- 4) HEAD (first {head_n} rows) -----")
    print(df.head(head_n))
    print()

    # 5) Descriptive Statistics
    print("----- 5) DESCRIPTIVE STATISTICS -----")
    print(df.describe().T)
    print()

    # 6) Missing‐Value Summary
    print("----- 6) MISSING‐VALUE SUMMARY -----")
    total_rows = len(df)
    missing_count = df.isna().sum()
    missing_percent = (missing_count / total_rows) * 100
    missing_summary = pd.DataFrame(
        {"missing_count": missing_count, "missing_percent": missing_percent}
    )
    print(missing_summary)
    print()
