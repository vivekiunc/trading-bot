# model.py
import json
import pandas as pd
import xgboost as xgb
import config as cfg

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
)

def walk_forward_backtest(merged, feature_cols, train_min_months=None):
    """Expanding-window walk-forward: for each rebalance date, train on all strictly
    earlier dates and predict on that date. No future data ever touches training."""
    train_min_months = train_min_months or cfg.TRAIN_MIN_MONTHS
    dates = sorted(merged["date"].unique())

    out = []
    for i, dt in enumerate(dates):
        if i < train_min_months:
            continue
        train = merged[merged["date"] < dt]
        test = merged[merged["date"] == dt]
        if train.empty or test.empty:
            continue

        reg = xgb.XGBRegressor(**XGB_PARAMS)
        reg.fit(train[feature_cols], train["label"])
        preds = reg.predict(test[feature_cols])

        res = test[["date", "symbol", "fwd_ret", "vol_3m_raw"]].copy()
        res["pred"] = preds
        out.append(res)
        print(f"[{i+1}/{len(dates)}] trained through {dt.date()} on {len(train)} rows, "
              f"scored {len(test)} names")

    return pd.concat(out, ignore_index=True)

def train_live_model(merged, feature_cols):
    """Train on ALL available labeled history for use in live scoring."""
    reg = xgb.XGBRegressor(**XGB_PARAMS)
    reg.fit(merged[feature_cols], merged["label"])
    return reg

def score_latest(reg, feats_latest, feature_cols):
    """Score the most recent (unlabeled) feature snapshot."""
    out = feats_latest[["date", "symbol", "vol_3m_raw"]].copy()
    out["pred"] = reg.predict(feats_latest[feature_cols])
    return out.sort_values("pred", ascending=False)

def save_model(reg, feature_cols, path):
    reg.save_model(path)
    with open(path + ".cols.json", "w") as f:
        json.dump(feature_cols, f)

def load_model(path):
    reg = xgb.XGBRegressor()
    reg.load_model(path)
    with open(path + ".cols.json") as f:
        feature_cols = json.load(f)
    return reg, feature_cols
