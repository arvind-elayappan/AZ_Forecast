# ```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_seasonality(series, period):
    """
    Plots overlaid lines for each 'period'-day block in a daily time series.

    Parameters:
    - series: Pandas Series with a DatetimeIndex at daily frequency.
    - period: integer number of days for seasonality (e.g., 7, 30, 90).
    """
    # Ensure the series is sorted by date
    series = series.sort_index()

    # Reindex to a complete daily range, filling any missing dates by interpolation
    full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
    series = series.reindex(full_idx).interpolate()

    N = len(series)
    n_periods = N // period
    if n_periods < 1:
        print(f"Not enough data for a single period of length {period} days.")
        return

    # Trim to a multiple of 'period' days
    trimmed = series.iloc[: n_periods * period]

    # Convert to a NumPy array and reshape into (n_periods, period)
    arr = trimmed.to_numpy()
    shaped = arr.reshape((n_periods, period))

    # Plot each period on the same axes
    plt.figure(figsize=(10, 6))
    x = range(period)
    for i in range(n_periods):
        plt.plot(x, shaped[i], alpha=0.7)

    # Label axes and title
    plt.xlabel(f"Day Index within {period}-Day Cycle")
    ylabel = series.name if series.name else "Value"
    plt.ylabel(ylabel)
    plt.title(f"{period}-Day Seasonality Plot ({n_periods} blocks)")
    plt.tight_layout()
    plt.show()


# ─── USAGE EXAMPLE ───────────────────────────────────────────────────────────────
# Assume you have already loaded your dataset into a DataFrame called `df`,
# and that it has a DateTime index. For example:
#
#     df = pd.read_csv(
#         r'C:\mlaz.explore\datasets\tetuan\Tetuan_City_power_consumption.csv',
#         parse_dates=['Date'],
#         index_col='Date',
#         dayfirst=True  # or False depending on your CSV’s date format
#     )
#
# And suppose your column of interest is named 'Global_active_power'.
#
# Then you would call:
#
#     series = df['Global_active_power']
#     plot_seasonality(series, 7)    # 7-day seasonality
#     plot_seasonality(series, 30)   # 30-day seasonality
#     plot_seasonality(series, 90)   # 90-day seasonality
