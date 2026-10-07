"""Arithmetic checks behind docs/audit.md.

The PEBSA figures below are transcribed from Chart 1 of legacy/pebsa_ijecs_2024.pdf
(page 26671). The S&P 500 price returns come from the Welch-Goyal monthly file and the
total returns from the ^SP500TR candidate sample, both saved by sentcast.datacheck.
Run: python -m sentcast.audit
"""
from __future__ import annotations

import pandas as pd

from .config import SAMPLES, write_result

# month: (S&P 500 % change as printed, PEBSA % change as printed, % error as printed)
PEBSA_CHART1 = {
    "2024-01": (1.7, 1.76, 3.53),
    "2024-02": (5.3, 5.11, 3.59),
    "2024-03": (3.2, 3.19, 0.31),
    "2024-04": (-4.1, -4.20, 2.44),
}
PEBSA_PRINTED_AVERAGE_ERROR = 2.4675
PEBSA_PRINTED_ACCURACY = 97.5325


def _goyal() -> pd.DataFrame:
    g = pd.read_excel(SAMPLES / "goyal_predictor_data.xlsx", sheet_name="Monthly")
    g.index = pd.PeriodIndex(g["yyyymm"].astype(int).astype(str), freq="M")
    return g


def main() -> None:
    rows = []
    for month, (actual, pred, err) in PEBSA_CHART1.items():
        recomputed = abs(pred - actual) / abs(actual) * 100
        rows.append({"month": month, "sp500_printed_pct": actual, "pebsa_printed_pct": pred,
                     "error_printed_pct": err, "error_recomputed_pct": round(recomputed, 4),
                     "abs_gap_pct_points": round(abs(pred - actual), 4)})
    mean_recomputed = sum(r["error_recomputed_pct"] for r in rows) / len(rows)
    mean_printed = sum(r["error_printed_pct"] for r in rows) / len(rows)
    g = _goyal()
    price = (g["Index"] / g["Index"].shift(1) - 1) * 100   # price index, month-end to month-end
    tr = pd.read_csv(SAMPLES / "yahoo_SP500TR.csv", index_col=0, parse_dates=True)["Close"].dropna()
    tr = tr.groupby(tr.index.to_period("M")).last()
    total = (tr / tr.shift(1) - 1) * 100                   # S&P 500 total return index
    for r in rows:
        p = pd.Period(r["month"], "M")
        r["sp500_price_return_pct"] = round(float(price[p]), 4)
        r["sp500_total_return_pct"] = round(float(total[p]), 4)
    hist = price[(price.index >= "1950-01") & (price.index <= "2023-12")].dropna()
    payload = {
        "source": "legacy/pebsa_ijecs_2024.pdf, Chart 1 (page 26671)",
        "n_observations": len(rows),
        "rows": rows,
        "mean_error_printed_rows_pct": round(mean_printed, 4),
        "mean_error_recomputed_pct": round(mean_recomputed, 4),
        "printed_average_error_pct": PEBSA_PRINTED_AVERAGE_ERROR,
        "printed_accuracy_pct": PEBSA_PRINTED_ACCURACY,
        "accuracy_as_100_minus_mean_printed_error_pct": round(100 - mean_printed, 4),
        "sp500_reference": {
            "source": "price: Welch-Goyal monthly file, column Index; total return: ^SP500TR sample",
            "months_1950_2023": int(len(hist)),
            "share_up_months_1950_2023": round(float((hist > 0).mean()), 4),
            "mean_abs_monthly_return_pct_1950_2023": round(float(hist.abs().mean()), 4),
        },
    }
    path = write_result("audit_checks", payload)
    print("wrote", path)
    for r in rows:
        print(r)
    print("mean printed", mean_printed, "mean recomputed", mean_recomputed)


if __name__ == "__main__":
    main()
