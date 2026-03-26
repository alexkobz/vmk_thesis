import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import date


def exponential_smoothing(
    s: pd.Series,
    trend: str = None,           # None, 'add', 'mul'
    seasonal: str = None,        # None, 'add', 'mul'
    seasonal_periods: int = None,
    alpha: float = None,         # smoothing_level
    beta: float = None,          # smoothing_slope
    gamma: float = None          # smoothing_seasonal
) -> pd.Series:
    """
    Exponential smoothing (Single, Double, Triple) for a pandas Series.
    Returns Series with the same index.
    """

    model = ExponentialSmoothing(
        s,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )

    fit = model.fit(
        smoothing_level=alpha,
        smoothing_slope=beta,
        smoothing_seasonal=gamma,
        optimized=(alpha is None and beta is None and gamma is None)
    )

    return pd.Series(fit.fittedvalues, index=s.index, name=s.name)


TARGET_DATES = pd.to_datetime(
    [date(y, 3, 15) for y in range(2011, 2025)]
)

mults = [
    'capex_revenue_rsbu', 'capex_revenue_msfo', 'capital_rsbu', 'capital_msfo', 'current_ratio_rsbu', 'current_ratio_msfo', 'debt_equity_rsbu', 'debt_equity_msfo', 'debt_ratio_rsbu', 'debt_ratio_msfo', 'debtebitda_rsbu', 'debtebitda_msfo', 'dpr_rsbu', 'dpr_msfo', 'ebitda_margin_rsbu', 'ebitda_margin_msfo', 'ev_ebit_rsbu', 'ev_ebit_msfo', 'evebitda_rsbu', 'evebitda_msfo', 'evs_rsbu', 'evs_msfo', 'gross_margin_rsbu', 'gross_margin_msfo', 'interest_coverage_rsbu', 'interest_coverage_msfo', 'net_margin_rsbu', 'net_margin_msfo', 'net_working_capital_rsbu', 'net_working_capital_msfo', 'netdebt_ebitda_rsbu', 'netdebt_ebitda_msfo', 'operation_margin_rsbu', 'operation_margin_msfo', 'pbv_rsbu', 'pbv_msfo', 'pcf_rsbu', 'pcf_msfo', 'pe_rsbu', 'pe_msfo', 'pfcf_rsbu', 'pfcf_msfo', 'pffo_rsbu', 'pffo_msfo', 'ps_rsbu', 'ps_msfo', 'roa_rsbu', 'roa_msfo', 'roce_rsbu', 'roce_msfo', 'roe_rsbu', 'roe_msfo', 'roic_rsbu', 'roic_msfo', 'ros_rsbu', 'ros_msfo',
]
funds = [
    'line_1100', 'line_1110', 'line_1120', 'line_1130', 'line_1140', 'line_1150', 'line_1160', 'line_1170', 'line_1180', 'line_1190', 'line_1200', 'line_1210', 'line_1220', 'line_1230', 'line_1240', 'line_1250', 'line_1260', 'line_1300', 'line_1310', 'line_1320', 'line_1340', 'line_1350', 'line_1360', 'line_1370', 'line_1400', 'line_1410', 'line_1420', 'line_1430', 'line_1450', 'line_1500', 'line_1510', 'line_1520', 'line_1530', 'line_1540', 'line_1550', 'line_1600', 'line_1700', 'line_2100', 'line_2110', 'line_2120', 'line_2200', 'line_2210', 'line_2220', 'line_2300', 'line_2310', 'line_2320', 'line_2330', 'line_2340', 'line_2350', 'line_2400', 'line_2410', 'line_2411', 'line_2412', 'line_2421', 'line_2430', 'line_2450', 'line_2460', 'line_2500', 'line_2510', 'line_2520', 'line_2530', 'line_2900', 'line_2910', 'line_3100', 'line_3200', 'line_3210', 'line_3211', 'line_3212', 'line_3213', 'line_3214', 'line_3215', 'line_3216', 'line_321x', 'line_3220', 'line_3221', 'line_3222', 'line_3223', 'line_3224', 'line_3225', 'line_3226', 'line_3227', 'line_322x', 'line_3230', 'line_3240', 'line_3300', 'line_3310', 'line_3311', 'line_3312', 'line_3313', 'line_3314', 'line_3315', 'line_3316', 'line_331x', 'line_3320', 'line_3321', 'line_3322', 'line_3323', 'line_3324', 'line_3325', 'line_3326', 'line_3327', 'line_332x', 'line_3330', 'line_3340', 'line_3400', 'line_3401', 'line_3402', 'line_3410', 'line_3411', 'line_3412', 'line_3420', 'line_3421', 'line_3422', 'line_3500', 'line_3501', 'line_3502', 'line_3600', 'line_4100', 'line_4110', 'line_4111', 'line_4112', 'line_4113', 'line_4119', 'line_411x', 'line_4120', 'line_4121', 'line_4122', 'line_4123', 'line_4124', 'line_4129', 'line_412x', 'line_4200', 'line_4210', 'line_4211', 'line_4212', 'line_4213', 'line_4214', 'line_4219', 'line_421x', 'line_4220', 'line_4221', 'line_4222', 'line_4223', 'line_4224', 'line_4229', 'line_422x', 'line_4300', 'line_4310', 'line_4311', 'line_4312', 'line_4313', 'line_4314', 'line_4319', 'line_431x', 'line_4320', 'line_4321', 'line_4322', 'line_4323', 'line_4329', 'line_432x', 'line_4400', 'line_4450', 'line_4490', 'line_4500', 'line_6100', 'line_6200', 'line_6210', 'line_6215', 'line_6220', 'line_6230', 'line_6240', 'line_6250', 'line_6300', 'line_6310', 'line_6311', 'line_6312', 'line_6313', 'line_6320', 'line_6321', 'line_6322', 'line_6323', 'line_6324', 'line_6325', 'line_6326', 'line_6330', 'line_6350', 'line_6400',
]

def smooth_report_data(
    s: pd.Series,
    report_dates: pd.DatetimeIndex = TARGET_DATES
) -> pd.Series:
    """"""

    if s.empty:
        return s

    idx = pd.DatetimeIndex(s.index)
    s = s.astype(float).fillna(0.0)

    anchors = {}

    for d in report_dates:
        mask = idx.year == d.year
        if not mask.any():
            continue

        year_idx = idx[mask]
        nearest = year_idx[np.abs(year_idx - d).argmin()]

        value = s.loc[nearest]

        anchors[nearest] = float(value)

    anchors = pd.Series(anchors).sort_index()

    if len(anchors) < 2:
        return s

    t_anchor = anchors.index.view("int64")
    t_all = idx.view("int64")

    if np.any(np.diff(t_anchor) <= 0):
        return s

    y_smooth = CubicSpline(
        t_anchor,
        anchors.values,
        bc_type="natural"
    )(t_all)

    return pd.Series(y_smooth, index=s.index, name=s.name)

def smooth_report(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = {}
    for col in mults + funds:
        print(col)
        smoothed = (
            df
            .groupby(level=0, group_keys=False)
            .apply(lambda x: smooth_report_data(x.droplevel('secid')[col]))
            .values
        )
        smoothed = pd.Series(smoothed, index=df.index, name=f'smoothed_{col}')
        new_cols[smoothed.name] = smoothed
        new_cols[f'pct_{col}'] = smoothed.pct_change() * 100
        new_cols[f'log_{col}'] = np.log(smoothed).diff(1)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
