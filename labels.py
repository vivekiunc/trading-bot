# labels.py
import pandas as pd
import config as cfg
from features import returns_matrix

def build_labels(bars, universe, horizon=None):
    """Forward return over the next `horizon` trading days, for each (date, symbol)
    in the as-of universe. Dates too close to the end of the data (not enough
    forward days available yet) are dropped."""
    horizon = horizon or cfg.LABEL_HORIZON_DAYS
    rets = returns_matrix(bars)
    idx = rets.index

    out = []
    for dt, grp in universe.groupby("date"):
        future = idx[idx > dt]
        if len(future) < horizon:
            continue
        window = future[:horizon]
        syms = [s for s in grp.symbol if s in rets.columns]
        fwd = (1 + rets.loc[window, syms]).prod() - 1
        lbl = fwd.rename("fwd_ret").reset_index()
        lbl.columns = ["symbol", "fwd_ret"]
        lbl["date"] = dt
        out.append(lbl)

    labels = pd.concat(out, ignore_index=True)
    # cross-sectional percentile rank of forward return: the training target.
    # Bounded in [0, 1] so a handful of extreme-return outliers each month can't
    # dominate the loss the way raw returns would -- we only need relative ranking
    # for stock selection anyway. `fwd_ret` is kept raw for backtest P&L.
    labels["label"] = labels.groupby("date")["fwd_ret"].rank(pct=True)
    return labels
