# eda_tools.py and 

import pandas as pd
import numpy as np
import re

# For interactive plotting:
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------------------------------------
# 1) CLEANING COLUMN NAMES (UNCHANGED)
# ------------------------------------------------------------
def clean_column_names(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Normalize column names by:
      - Collapsing any sequence of whitespace into a single space
      - Stripping leading/trailing whitespace
    If inplace=True (default), modifies df.columns directly and returns None.
    If inplace=False, returns a new DataFrame with cleaned column names.
    """
    cleaned = [re.sub(r"\s+", " ", col).strip() for col in df.columns]
    if inplace:
        df.columns = cleaned
        return None
    else:
        return df.rename(columns={old: new for old, new in zip(df.columns, cleaned)})


# ------------------------------------------------------------
# 2) DATETIME CONVERSION & INDEXING (UNCHANGED)
# ------------------------------------------------------------
def set_datetime_index(
    df: pd.DataFrame,
    datetime_col: str,
    infer_format: bool = False,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Convert a column to datetime (if not already), then set it as the DataFrame index.
    Returns a new DataFrame with a sorted DatetimeIndex.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(
            df[datetime_col], infer_datetime_format=infer_format, errors="coerce"
        )
    df = df.set_index(datetime_col).sort_index()
    return df


# ------------------------------------------------------------
# 3) MISSING / NULL VALUE CHECKS & HANDLING (UNCHANGED)
# ------------------------------------------------------------
def drop_missing_rows(
    df: pd.DataFrame, subset: list = None, how: str = "any"
) -> pd.DataFrame:
    return df.dropna(subset=subset, how=how)


def fill_missing(
    df: pd.DataFrame, method: str = "ffill", limit: int = None
) -> pd.DataFrame:
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
# 4) SUMMARY STATISTICS & OVERVIEW (UNCHANGED)
# ------------------------------------------------------------
def summarize_dataframe(df: pd.DataFrame, head_n: int = 5) -> None:
    print("----- 1) SHAPE -----")
    print(df.shape, "\n")

    print("----- 2) DTYPES -----")
    print(df.dtypes, "\n")

    print("----- 3) INFO -----")
    df.info()
    print()

    print(f"----- 4) HEAD (first {head_n} rows) -----")
    print(df.head(head_n), "\n")

    print("----- 5) DESCRIPTIVE STATISTICS -----")
    print(df.describe().T, "\n")

    print("----- 6) MISSING‐VALUE SUMMARY -----")
    total_rows = len(df)
    missing_count = df.isna().sum()
    missing_percent = (missing_count / total_rows) * 100
    missing_summary = pd.DataFrame(
        {"missing_count": missing_count, "missing_percent": missing_percent}
    )
    print(missing_summary, "\n")


# ------------------------------------------------------------
# 5) INTERACTIVE TIME‐SERIES PLOTTING (PLOTLY)
# ------------------------------------------------------------
def plot_time_series_for_column(
    df: pd.DataFrame,
    column: str,
    title: str = None,
    xlabel: str = "Index",
    ylabel: str = None,
    freq: str = None,
    how: str = "mean",
) -> go.Figure:
    """
    Plot a single numeric column versus its index using Plotly. If `freq` is provided,
    resample the series at that frequency using the specified aggregation `how` before plotting.

    Returns:
        A Plotly Figure object (interactive).
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column '{column}' is not numeric; skipping plot.")
        return None

    series = df[column]

    if freq is not None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex to resample.")
        try:
            series = getattr(series.resample(freq), how)()
        except AttributeError:
            raise ValueError(f"Aggregation method '{how}' is not valid for resampling.")

    fig = px.line(
        x=series.index,
        y=series.values,
        labels={"x": xlabel, "y": ylabel or column},
        title=title
        or f"{column} (Time‐Series){' [' + freq + ' ' + how + ']' if freq else ''}",
    )
    fig.update_layout(hovermode="x unified")
    fig.show()
    return fig


def plot_each_time_series(
    df: pd.DataFrame,
    xlabel: str = "DateTime",
) -> list:
    """
    For each numeric column in df, create a separate interactive time‐series plot (Plotly).
    Returns a list of Plotly Figure objects (one per numeric column).
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    figs = []
    if not numeric_cols:
        print("No numeric columns to plot.")
        return figs

    for col in numeric_cols:
        fig = px.line(
            x=df.index,
            y=df[col],
            labels={"x": xlabel, "y": col},
            title=f"Time Series: {col}",
        )
        fig.update_layout(hovermode="x unified")
        fig.show()
        figs.append(fig)

    return figs


# ------------------------------------------------------------
# 6) INTERACTIVE RESAMPLED SPLIT PLOTS (PLOTLY)
# ------------------------------------------------------------
def plot_resampled_split(
    df: pd.DataFrame,
    freq: str,
    how: str = "mean",
    title_high: str = None,
    title_low: str = None,
    xlabel: str = "Date",
    ylabel_high: str = "Value (High-Range Variables)",
    ylabel_low: str = "Value (Low-Range Variables)",
) -> go.Figure:
    """
    Resample df at `freq` using aggregation `how`. Split numeric columns into high-range
    vs low-range groups by median(range). Plot side-by-side interactive subplots in Plotly.

    Returns:
        A Plotly Figure object containing two subplots (high-range on left, low-range on right).
    """
    # 1) Select numeric columns & resample
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        print("No numeric columns to resample/plot.")
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex to resample.")

    resampled = getattr(numeric_df.resample(freq), how)()

    # 2) Compute each column’s range & median threshold
    col_ranges = resampled.max() - resampled.min()
    median_range = col_ranges.median()
    high_cols = col_ranges[col_ranges >= median_range].index.tolist()
    low_cols = col_ranges[col_ranges < median_range].index.tolist()

    # 3) Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        subplot_titles=[
            title_high or f"High-Range (≥ {median_range:.2f})",
            title_low or f"Low-Range (< {median_range:.2f})",
        ],
    )

    # 4) Add traces for high-range group
    if high_cols:
        for col in high_cols:
            fig.add_trace(
                go.Scatter(x=resampled.index, y=resampled[col], mode="lines", name=col),
                row=1,
                col=1,
            )
        fig.update_yaxes(title_text=ylabel_high, row=1, col=1)
    else:
        fig.add_annotation(
            text="No high-range columns",
            xref="paper",
            yref="paper",
            x=0.25,
            y=0.5,
            showarrow=False,
        )

    # 5) Add traces for low-range group
    if low_cols:
        for col in low_cols:
            fig.add_trace(
                go.Scatter(
                    x=resampled.index,
                    y=resampled[col],
                    mode="lines",
                    name=col,
                    showlegend=False,  # legend on left only (to avoid duplication)
                ),
                row=1,
                col=2,
            )
        fig.update_yaxes(title_text=ylabel_low, row=1, col=2)
    else:
        fig.add_annotation(
            text="No low-range columns",
            xref="paper",
            yref="paper",
            x=0.75,
            y=0.5,
            showarrow=False,
        )

    fig.update_layout(
        height=500,
        width=1100,
        title_text=f"Resampled ({freq}) → {how}",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text=xlabel)
    fig.show()


# ------------------------------------------------------------
# 7) INTERACTIVE AGGREGATION PLOTS (HOURLY, DAILY, MONTHLY)
# ------------------------------------------------------------
def plot_each_hourly_aggregation(
    df: pd.DataFrame, how: str = "mean", xlabel: str = "Hour"
) -> list:
    """
    For each numeric column in df, resample hourly (freq='H') and plot interactive line charts.
    Returns a list of Plotly Figures.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex to use hourly aggregation.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    figs = []
    for col in numeric_cols:
        series = getattr(df[col].resample("H"), how)()
        fig = px.line(
            x=series.index,
            y=series.values,
            labels={"x": xlabel, "y": col},
            title=f"Hourly {how.capitalize()} of '{col}'",
        )
        fig.update_layout(hovermode="x unified")
        fig.show()
        figs.append(fig)
    return figs


def plot_each_daily_aggregation(
    df: pd.DataFrame, how: str = "mean", xlabel: str = "Date"
) -> list:
    """
    For each numeric column in df, resample daily (freq='D') and plot interactive line charts.
    Returns a list of Plotly Figures.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex to use daily aggregation.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    figs = []
    for col in numeric_cols:
        series = getattr(df[col].resample("D"), how)()
        fig = px.line(
            x=series.index,
            y=series.values,
            labels={"x": xlabel, "y": col},
            title=f"Daily {how.capitalize()} of '{col}'",
        )
        fig.update_layout(hovermode="x unified")
        fig.show()
        figs.append(fig)
    return figs


def plot_each_monthly_aggregation(
    df: pd.DataFrame, how: str = "mean", xlabel: str = "Month"
) -> list:
    """
    For each numeric column in df, resample monthly (freq='M') and plot interactive line charts.
    Returns a list of Plotly Figures.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex to use monthly aggregation.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    figs = []
    for col in numeric_cols:
        series = getattr(df[col].resample("M"), how)()
        fig = px.line(
            x=series.index,
            y=series.values,
            labels={"x": xlabel, "y": col},
            title=f"Monthly {how.capitalize()} of '{col}'",
        )
        fig.update_layout(hovermode="x unified")
        fig.show()
        figs.append(fig)
    return figs


# ------------------------------------------------------------
# 8) INTERACTIVE CORRELATION HEATMAP (PLOTLY)
# ------------------------------------------------------------
def plot_correlation_heatmap(
    df: pd.DataFrame, cols: list = None, title: str = "Correlation Matrix"
) -> go.Figure:
    """
    Compute the correlation matrix among numeric columns (or a subset) and display
    it as an interactive Plotly heatmap.
    Returns a Plotly Figure.
    """
    if cols is None:
        numeric_df = df.select_dtypes(include="number")
    else:
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        numeric_df = df[cols].select_dtypes(include="number")

    corr_mat = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_mat.values,
            x=corr_mat.columns,
            y=corr_mat.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title=title, xaxis=dict(tickangle=45), yaxis=dict(autorange="reversed")
    )
    fig.show()
    return fig


# ------------------------------------------------------------
# 9) INTERACTIVE PAIRPLOT / SCATTER MATRIX (PLOTLY)
# ------------------------------------------------------------
def plot_pairplot(
    df: pd.DataFrame,
    columns: list = None,
    max_vars: int = 5,
    dropna: bool = True,
    **px_kwargs,
) -> go.Figure:
    """
    Create an interactive scatter‐matrix (pairplot) using Plotly.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing your data.
    columns : list, optional
        A list of column names to include in the pairplot. If None:
          1) Auto‐select numeric columns
          2) If len(numeric_cols) <= max_vars, use them
          3) Else, abort with a warning
    max_vars : int, default=5
        Maximum number of numeric columns to auto‐select if `columns` is None.
    dropna : bool, default=True
        Whether to drop rows with NaNs in the chosen columns before plotting.
    **px_kwargs :
        Additional keyword args forwarded to px.scatter_matrix
        (e.g. `dimensions_color_continuous_scale='Viridis'`, `symbol='CategoryColumn'`, etc.)

    Returns:
    --------
    A Plotly Figure object containing the scatter‐matrix. If aborted, returns None.
    """
    if columns is not None:
        missing = set(columns) - set(df.columns)
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")

        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]

        if non_numeric and "color" not in px_kwargs and "symbol" not in px_kwargs:
            raise TypeError(
                f"Columns {non_numeric} are not numeric. "
                "If you want to include them, pass them via `color=` or `symbol=`."
            )

        plot_cols = columns[:]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            print("No numeric columns available to plot.")
            return None
        if len(numeric_cols) <= max_vars:
            plot_cols = numeric_cols
        else:
            print(
                f"Found {len(numeric_cols)} numeric columns, which is > max_vars ({max_vars}).\n"
                "Either reduce the number of columns or explicitly pass a `columns=[...]` list."
            )
            return None

    data_to_plot = df[plot_cols].dropna() if dropna else df[plot_cols]
    if data_to_plot.empty:
        print("No data after dropping NaNs; pairplot cannot be created.")
        return None

    fig = px.scatter_matrix(data_to_plot, dimensions=plot_cols, **px_kwargs)
    fig.update_traces(diagonal_visible=True)  # show histograms on diagonal
    fig.update_layout(height=600, width=600, title="Scatter Matrix")
    fig.show()
    return fig


# ------------------------------------------------------------
# 10) INTERACTIVE BOXPLOTS (PLOTLY)
# ------------------------------------------------------------
def plot_boxplots(df: pd.DataFrame) -> list:
    """
    For each numeric column in df, create an interactive boxplot using Plotly.
    Returns a list of Plotly Figure objects (one per numeric column).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    figs = []
    if not numeric_cols:
        print("No numeric columns to plot boxplots.")
        return figs

    for col in numeric_cols:
        fig = px.box(
            y=df[col].dropna(),
            points="outliers",  # show outliers
            labels={"y": col},
            title=f"Boxplot of {col}",
        )
        fig.show()
        figs.append(fig)

    return figs


# ------------------------------------------------------------
# 11) AGGREGATION BY FREQUENCY (UNCHANGED)
# ------------------------------------------------------------
def aggregate_time_series(
    df: pd.DataFrame, freq: str, how: str = "mean"
) -> pd.DataFrame:
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        try:
            df_copy.index = pd.to_datetime(df_copy.index)
        except Exception as e:
            raise TypeError(
                "Index must be a DatetimeIndex or convertible to datetime."
            ) from e

    numeric_df = df_copy.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns available to aggregate.")

    try:
        aggregated = getattr(numeric_df.resample(freq), how)()
    except AttributeError as e:
        raise ValueError(
            f"Aggregation method '{how}' is not valid for resampling."
        ) from e

    return aggregated


def get_hourly_aggregation(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    return aggregate_time_series(df, freq="H", how=how)


def get_daily_aggregation(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    return aggregate_time_series(df, freq="D", how=how)


def get_monthly_aggregation(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    return aggregate_time_series(df, freq="M", how=how)


def get_yearly_aggregation(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    return aggregate_time_series(df, freq="Y", how=how)
