# backtest.py
import numpy as np
import pandas as pd
import config as cfg

PERIODS_PER_YEAR = 252 / cfg.LABEL_HORIZON_DAYS  # ~monthly periods

def inverse_vol_weights(vol, n, cap_mult=3.0):
    """1/vol weights, capped at cap_mult x the equal-weight share and renormalized,
    so one unusually low-vol name can't dominate the basket."""
    w = 1.0 / vol.clip(lower=vol.quantile(0.05))
    w = w / w.sum()
    cap = cap_mult / n
    w = w.clip(upper=cap)
    return w / w.sum()

def run_backtest(preds, n_long=None):
    """preds: columns [date, symbol, pred, fwd_ret, vol_3m_raw] from a walk-forward run.
    Each period, go long the top n_long names by predicted score, inverse-vol weighted
    (capped), and realize `fwd_ret` (the actual forward return) as that period's
    portfolio return. Also reports an equal-weight-universe benchmark for comparison."""
    n_long = n_long or cfg.N_LONG

    rows = []
    for dt, grp in preds.groupby("date"):
        top = grp.nlargest(n_long, "pred")
        w = inverse_vol_weights(top["vol_3m_raw"], len(top))
        rows.append({
            "date": dt,
            "strategy_ret": (w * top["fwd_ret"]).sum(),
            "benchmark_ret": grp["fwd_ret"].mean(),
            "n_names": len(top),
        })

    port = pd.DataFrame(rows).set_index("date").sort_index()
    port["strategy_cum"] = (1 + port["strategy_ret"]).cumprod()
    port["benchmark_cum"] = (1 + port["benchmark_ret"]).cumprod()
    return port

def _annualized_stats(ret_series):
    mean = ret_series.mean() * PERIODS_PER_YEAR
    vol = ret_series.std() * np.sqrt(PERIODS_PER_YEAR)
    sharpe = mean / vol if vol > 0 else float("nan")
    cum = (1 + ret_series).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()
    return mean, vol, sharpe, max_dd

def print_report(port):
    s_mean, s_vol, s_sharpe, s_dd = _annualized_stats(port["strategy_ret"])
    b_mean, b_vol, b_sharpe, b_dd = _annualized_stats(port["benchmark_ret"])

    print("\n=== Walk-forward backtest report ===")
    print(f"Periods: {len(port)}  ({port.index.min().date()} -> {port.index.max().date()})")
    print(f"{'':12}{'Strategy':>12}{'Benchmark':>12}")
    print(f"{'Ann. return':12}{s_mean:>12.2%}{b_mean:>12.2%}")
    print(f"{'Ann. vol':12}{s_vol:>12.2%}{b_vol:>12.2%}")
    print(f"{'Sharpe':12}{s_sharpe:>12.2f}{b_sharpe:>12.2f}")
    print(f"{'Max drawdown':12}{s_dd:>12.2%}{b_dd:>12.2%}")
    print(f"{'Total return':12}{port['strategy_cum'].iloc[-1]-1:>12.2%}"
          f"{port['benchmark_cum'].iloc[-1]-1:>12.2%}")
