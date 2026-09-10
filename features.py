# features.py
import pandas as pd
import numpy as np
import config as cfg

def returns_matrix(bars):
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_localize(None)
    px = bars.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    return px.pct_change()

def build_features(bars, universe):
    rets = returns_matrix(bars)
    mkt = rets.mean(axis=1)

    out = []
    for dt, grp in universe.groupby("date"):
        syms = [s for s in grp.symbol if s in rets.columns]
        r = rets.loc[rets.index <= dt, syms]
        if len(r) < 252:
            continue
        r = r.tail(252)

        f = pd.DataFrame(index=syms)
        f["mom_12_1"]    = (1 + r.iloc[:-21]).prod() - 1
        f["reversal_1m"] = (1 + r.tail(21)).prod() - 1
        f["vol_3m"]      = r.tail(63).std()
        m = mkt.loc[r.index]
        f["beta"]        = r.apply(lambda c: c.cov(m) / m.var())
        f["illiq"]       = r.tail(63).abs().mean()

        f["date"], f["symbol"] = dt, f.index
        out.append(f.reset_index(drop=True))

    return pd.concat(out, ignore_index=True)

def cross_sectional_rank(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df.groupby("date")[c].rank(pct=True)
    return df