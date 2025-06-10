import auto_ts_analysis as ata

df = (df_daily,)
datetime_col = ("Date",)
target_cols = (
    [
        "Zone 1 Power Consumption",
        "Zone 2 Power Consumption",
        "Zone 3 Power Consumption",
    ],
)
decomposition_period = (7,)
adf_regression = ("c",)
kpss_regression = "c"

ata.run_full_analysis(
    df, datetime_col, target_cols, decomposition_period, adf_regression, kpss_regression
)
