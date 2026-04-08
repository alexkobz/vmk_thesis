from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_ticker(
    cap_y_test: pd.Series,
    cap_y_pred: pd.Series,
    ticker: str,
    mape: float,
    f1: float,
    save_path: str | Path | None = None,
) -> None:
    cap_y_test_ticker = cap_y_test.xs(ticker, level="secid")
    cap_y_pred_ticker = cap_y_pred.xs(ticker, level="secid")
    title: str = f"{ticker} cap forecast\n"
    if mape > 0:
        title += f"MAPE: {mape * 100:.2f}% "
    if f1 > 0:
        title += f"F1: {f1 * 100:.2f}% "
    fig = plt.figure(figsize=(8, 3))
    plt.title(title)
    plt.plot(cap_y_test_ticker, "r--", alpha=0.7)
    plt.plot(cap_y_pred_ticker, "g--", alpha=0.7)
    plt.legend(["actual", "forecast"])

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname=save_path / f'{ticker}.jpeg', bbox_inches="tight", dpi=150)

    plt.show()

    plt.close(fig)
