import os
import pandas as pd
import config as cfg
import universe as u
import features as ft
import labels as lb
import model as md
import backtest as bt

FEATURE_COLS = ["mom_12_1", "reversal_1m", "vol_3m", "beta", "illiq"]

def stage1_data():
    """Download asset universe + historical bars, build monthly as-of universe. Run once."""
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    assets = u.get_asset_universe()
    assets.to_parquet(f"{cfg.DATA_DIR}/assets.parquet")
    bars = u.download_bars(assets.symbol.tolist(), out_path=f"{cfg.DATA_DIR}/bars.parquet")
    univ = u.build_asof_universe(bars, assets, rebalance=cfg.REBALANCE,
                                  out_path=f"{cfg.DATA_DIR}/universe.parquet")
    return bars, univ, assets

def stage2_features(bars, univ):
    feats = ft.build_features(bars, univ)
    feats = ft.cross_sectional_rank(feats, FEATURE_COLS)
    feats.to_parquet(f"{cfg.DATA_DIR}/features.parquet")
    print(feats.groupby("date").size().describe())
    return feats

def stage3_labels(bars, univ):
    lbls = lb.build_labels(bars, univ)
    lbls.to_parquet(f"{cfg.DATA_DIR}/labels.parquet")
    return lbls

def stage4_backtest(feats, lbls):
    merged = feats.merge(lbls, on=["date", "symbol"])
    preds = md.walk_forward_backtest(merged, FEATURE_COLS)
    preds.to_parquet(f"{cfg.DATA_DIR}/predictions.parquet")
    results = bt.run_backtest(preds)
    bt.print_report(results)
    return preds, results

if __name__ == "__main__":
    bars_path = f"{cfg.DATA_DIR}/bars.parquet"
    univ_path = f"{cfg.DATA_DIR}/universe.parquet"
    assets_path = f"{cfg.DATA_DIR}/assets.parquet"

    if os.path.exists(bars_path) and os.path.exists(univ_path):
        print("Found existing data, skipping download. Delete data/ to force a refresh.")
        bars = pd.read_parquet(bars_path)
        univ = pd.read_parquet(univ_path)
    else:
        bars, univ, assets = stage1_data()

    feats = stage2_features(bars, univ)
    lbls = stage3_labels(bars, univ)
    preds, results = stage4_backtest(feats, lbls)

    # train the final live model on all available labeled history
    merged = feats.merge(lbls, on=["date", "symbol"])
    live_model = md.train_live_model(merged, FEATURE_COLS)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    md.save_model(live_model, FEATURE_COLS, f"{cfg.MODEL_DIR}/xgb_live.json")
    print(f"Saved live model to {cfg.MODEL_DIR}/xgb_live.json")