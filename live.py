# live.py — score the current universe with the trained model and rebalance
# the Alpaca PAPER account to the resulting top-N long portfolio.
import argparse
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import config as cfg
import universe as u
import features as ft
import model as md
import backtest as bt

FEATURE_COLS = ["mom_12_1", "reversal_1m", "vol_3m", "beta", "illiq"]
LIVE_LOOKBACK_DAYS = 500  # calendar days; comfortably covers the 252 trading-day feature window

def get_live_snapshot():
    """Pull a fresh recent window of bars and build today's as-of universe + features."""
    print("Fetching tradable asset list...")
    assets = u.get_asset_universe()

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=LIVE_LOOKBACK_DAYS)
    print(f"Downloading bars {start.date()} -> {end.date()} for {len(assets)} symbols "
          f"(this takes a few minutes)...")
    bars = u.download_bars(
        assets.symbol.tolist(),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        out_path=f"{cfg.DATA_DIR}/live_bars.parquet",
    )
    bars_ts = pd.to_datetime(bars["timestamp"]).dt.tz_localize(None)
    as_of = bars_ts.max()

    univ = u.build_asof_universe(
        bars, assets, dates=[as_of], out_path=f"{cfg.DATA_DIR}/live_universe.parquet"
    )
    feats = ft.build_features(bars, univ)
    feats = ft.cross_sectional_rank(feats, FEATURE_COLS)
    if feats.empty:
        raise RuntimeError(
            "No features could be built for the latest date — not enough history in the "
            "lookback window. Try increasing LIVE_LOOKBACK_DAYS."
        )
    return feats, as_of

def pick_targets(feats, model_path):
    reg, feature_cols = md.load_model(model_path)
    scored = md.score_latest(reg, feats, feature_cols)
    return scored.head(cfg.N_LONG)

def rebalance(targets, dry_run=False):
    client = TradingClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET, paper=True)
    account = client.get_account()
    equity = float(account.equity)
    gross = equity * cfg.MAX_GROSS_EXPOSURE
    weights = bt.inverse_vol_weights(targets["vol_3m_raw"], len(targets))

    target_dollars = {
        row.symbol: gross * w for row, w in zip(targets.itertuples(), weights)
    }

    positions = {p.symbol: float(p.market_value) for p in client.get_all_positions()}

    print(f"\nEquity: ${equity:,.2f}  Target gross: ${gross:,.2f}  Names: {len(targets)}")

    # close positions no longer in the target set
    for sym, mv in positions.items():
        if sym not in target_dollars:
            print(f"CLOSE {sym} (current ${mv:,.2f})")
            if not dry_run:
                client.close_position(sym)

    # bring target names to their target dollar amount
    for sym, target in target_dollars.items():
        current = positions.get(sym, 0.0)
        diff = target - current
        if abs(diff) < 1.0:
            continue
        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        print(f"{side.value.upper():5} {sym:6} ${abs(diff):,.2f}  (current ${current:,.2f} -> target ${target:,.2f})")
        if not dry_run:
            order = MarketOrderRequest(
                symbol=sym,
                notional=round(abs(diff), 2),
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            try:
                client.submit_order(order)
            except Exception as e:
                print(f"  !! order failed for {sym}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="compute targets but don't submit orders")
    parser.add_argument("--model", default=f"{cfg.MODEL_DIR}/xgb_live.json")
    args = parser.parse_args()

    feats, as_of = get_live_snapshot()
    latest = feats[feats["date"] == feats["date"].max()]
    print(f"\nScoring {len(latest)} eligible names as of {as_of.date()}...")

    targets = pick_targets(latest, args.model)
    print("\nTarget portfolio:")
    print(targets.to_string(index=False))

    rebalance(targets, dry_run=args.dry_run)
