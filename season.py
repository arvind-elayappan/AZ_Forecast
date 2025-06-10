import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _get_freq(series: pd.Series) -> str:
    """
    Infer a uniform frequency string (e.g. 'H','D','M','Q','A') from series.index.
    Raises if it cannot.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    freq = pd.infer_freq(series.index)
    if freq is None:
        raise ValueError(f"Cannot infer a uniform frequency for '{series.name}'.")
    return freq


def plotly_overlay_seasonality(
    series: pd.Series,
    block_length: int,
    title: str = None,
    line_opacity: float = 0.6,
    width: int = 900,
    height: int = 500,
) -> None:
    """
    Overlay each consecutive block of length `block_length` (in the series' own freq units)
    without detrending.
    """
    freq = _get_freq(series)
    s = series.sort_index().asfreq(freq).interpolate()
    N = len(s)
    n_blocks = N // block_length
    if n_blocks < 1:
        raise ValueError(f"Need at least {block_length}×{freq} points; only have {N}.")
    trimmed = s.iloc[: n_blocks * block_length]
    arr = trimmed.values.reshape((n_blocks, block_length))
    x = np.arange(block_length)

    fig = go.Figure()
    for i in range(n_blocks):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=arr[i],
                mode="lines",
                name=f"Block {i + 1}",
                opacity=line_opacity,
                hovertemplate=f"{series.name}<br>Idx: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title or f"{series.name}: Raw Overlay ({block_length}×{freq} blocks)",
        xaxis_title=f"Index within {block_length}×{freq} block",
        yaxis_title=series.name,
        width=width,
        height=height,
    )
    fig.show()


def plotly_average_seasonality(
    series: pd.Series,
    block_length: int,
    title: str = None,
    width: int = 900,
    height: int = 500,
) -> None:
    """
    Compute & plot the mean ±1 std-dev across each position in blocks of length `block_length`
    in the series' own frequency.
    """
    freq = _get_freq(series)
    s = series.sort_index().asfreq(freq).interpolate()
    N = len(s)
    n_blocks = N // block_length
    if n_blocks < 1:
        raise ValueError(f"Need at least {block_length}×{freq} points; only have {N}.")
    trimmed = s.iloc[: n_blocks * block_length]
    arr = trimmed.values.reshape((n_blocks, block_length))

    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    x = np.arange(block_length)

    fig = go.Figure()
    # mean line
    fig.add_trace(go.Scatter(x=x, y=means, mode="lines+markers", name="Mean"))
    # ±1 std dev band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([means + stds, (means - stds)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,0,255,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1 Std Dev",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title or f"{series.name}: Average Seasonality ({block_length}×{freq})",
        xaxis_title=f"Index within {block_length}×{freq} block",
        yaxis_title=series.name,
        width=width,
        height=height,
    )
    fig.show()


def plotly_detrended_seasonality(
    series: pd.Series,
    block_length: int,
    title: str = None,
    line_opacity: float = 0.6,
    width: int = 900,
    height: int = 500,
) -> None:
    """
    Subtract each block’s own mean in blocks of length `block_length` (in series’ freq units),
    then overlay the demeaned lines.
    """
    freq = _get_freq(series)
    s = series.sort_index().asfreq(freq).interpolate()
    N = len(s)
    n_blocks = N // block_length
    if n_blocks < 1:
        raise ValueError(f"Need at least {block_length}×{freq} points; only have {N}.")
    trimmed = s.iloc[: n_blocks * block_length]
    arr = trimmed.values.reshape((n_blocks, block_length))

    block_means = arr.mean(axis=1).reshape(n_blocks, 1)
    detrended = arr - block_means
    x = np.arange(block_length)

    fig = go.Figure()
    for i in range(n_blocks):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=detrended[i],
                mode="lines",
                name=f"Block {i + 1}",
                opacity=line_opacity,
                hovertemplate=f"{series.name}<br>Idx: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
            )
        )
    # zero line
    fig.add_trace(
        go.Scatter(
            x=[0, block_length - 1],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title or f"{series.name}: Detrended Seasonality ({block_length}×{freq})",
        xaxis_title=f"Index within {block_length}×{freq} block",
        yaxis_title=series.name,
        width=width,
        height=height,
    )
    fig.show()
