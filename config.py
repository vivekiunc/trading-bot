import os

ALPACA_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]

START = "2016-01-01"       # backtest start
END   = "2024-12-31"
MIN_PRICE = 5.0            # drop sub-$5 names
UNIVERSE_SIZE = 1000       # top-N by dollar volume each rebalance
ADV_WINDOW = 63            # ~3 months of trading days for liquidity
REBALANCE = "ME"           # month-end (matches FF monthly cadence)
DATA_DIR = "data"