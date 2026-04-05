from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_ticker(
    cap_y_test: pd.Series,
    cap_y_pred: pd.Series,
    mape_secid: pd.Series,
    score: float,
    ticker: str,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 3))
    plt.title(
        f"{ticker} cap forecast\n"
        f"MAPE: {mape_secid[ticker] * 100:.2f}% "
        f"WMAPE: {score * 100:.2f}%"
    )
    plt.plot(cap_y_test.xs(ticker, level="secid"), "r--", alpha=0.7)
    plt.plot(cap_y_pred.xs(ticker, level="secid"), "g--", alpha=0.7)
    plt.legend(["actual", "forecast"])

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    plt.close(fig)
