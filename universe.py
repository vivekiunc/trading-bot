from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
import pandas as pd
import config as cfg
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import os

def get_asset_universe():
    client = TradingClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET, paper=True)
    assets = client.get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    )
    rows = [{
        "symbol": a.symbol,
        "exchange": a.exchange.value,
        "tradable": a.tradable,
        "shortable": a.shortable,
        "easy_to_borrow": a.easy_to_borrow,
        "fractionable": a.fractionable,
    } for a in assets]
    df = pd.DataFrame(rows)
    # keep common exchanges, tradable names; exclude the obvious non-common-stock junk later
    df = df[df.tradable & df.exchange.isin(["NYSE", "NASDAQ", "ARCA", "AMEX"])]
    return df.reset_index(drop=True)

def download_bars(symbols, start=None, end=None, out_path=None):
    start = start or cfg.START
    end = end or cfg.END
    client = StockHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)
    frames = []
    for i in range(0, len(symbols), 200):          # chunk to stay under limits
        chunk = symbols[i:i+200]
        req = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="all",                        # split + dividend adjusted
        )
        bars = client.get_stock_bars(req).df         # MultiIndex (symbol, timestamp)
        if len(bars):
            frames.append(bars)
        print(f"{i+len(chunk)}/{len(symbols)} symbols")
    df = pd.concat(frames).reset_index()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_parquet(out_path)
    return df

def build_asof_universe(bars, assets, rebalance="ME", dates=None, out_path=None):
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_localize(None)
    bars["dollar_vol"] = bars["close"] * bars["volume"]

    if dates is None:
        dates = pd.date_range(bars.timestamp.min(), bars.timestamp.max(), freq=rebalance)
    shortable = set(assets.loc[assets.shortable, "symbol"])

    out = []
    for dt in dates:
        # STRICT: only bars on or before the rebalance date
        hist = bars[bars.timestamp <= dt]
        recent = hist[hist.timestamp > dt - pd.Timedelta(days=cfg.ADV_WINDOW * 2)]

        adv = recent.groupby("symbol")["dollar_vol"].median()
        last_px = hist.sort_values("timestamp").groupby("symbol")["close"].last()

        elig = pd.DataFrame({"adv": adv, "price": last_px}).dropna()
        elig = elig[elig.price >= cfg.MIN_PRICE]
        elig = elig.nlargest(cfg.UNIVERSE_SIZE, "adv")

        elig = elig.reset_index()
        elig["date"] = dt
        elig["shortable"] = elig.symbol.isin(shortable)
        out.append(elig[["date", "symbol", "price", "adv", "shortable"]])

    universe = pd.concat(out, ignore_index=True)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        universe.to_parquet(out_path)
    return universe