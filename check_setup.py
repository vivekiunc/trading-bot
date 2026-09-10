import os
from alpaca.trading.client import TradingClient

client = TradingClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    paper = True,
)

acct = client.get_account()
print("Status:" + acct.status)
print("Buying Power: " + acct.buying_power)
print("Cash: " + acct.cash)