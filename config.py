import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]

START = "2016-01-01"       # backtest start
END   = "2024-12-31"
MIN_PRICE = 5.0            # drop sub-$5 names
UNIVERSE_SIZE = 1000       # top-N by dollar volume each rebalance
ADV_WINDOW = 63            # ~3 months of trading days for liquidity
REBALANCE = "ME"           # month-end (matches FF monthly cadence)
DATA_DIR = "data"
MODEL_DIR = "models"

# --- modeling ---
LABEL_HORIZON_DAYS = 21    # ~1 month forward return, matches monthly rebalance
TRAIN_MIN_MONTHS = 36      # need at least this many monthly snapshots before training
N_LONG = 30                # number of names to hold long each rebalance

# --- live trading ---
MAX_GROSS_EXPOSURE = 0.95  # fraction of equity to deploy (leave cash buffer)
