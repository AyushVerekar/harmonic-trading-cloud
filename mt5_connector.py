# mt5_connector.py
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd

# 🔹 PUT YOUR DEMO ACCOUNT DETAILS HERE
LOGIN = 103185444
PASSWORD = "_qBu5fXg"
SERVER = "MetaQuotes-Demo"


def initialize():
    if not mt5.initialize():
        print("Initialize failed:", mt5.last_error())
        return False

    if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
        print("Login failed:", mt5.last_error())
        return False

    print("Connected to MT5")
    return True


# 🔹 Historical from date
def get_rates_from_date(symbol, timeframe, year=2010):
    from_date = datetime(year, 1, 1)

    rates = mt5.copy_rates_from(symbol, timeframe, from_date, 100000)

    if rates is None:
        print("Failed to get rates:", mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df


# 🔹 Live recent candles
def get_recent_rates(symbol, timeframe, n=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)

    if rates is None:
        print("Failed to get rates:", mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df


def shutdown():
    mt5.shutdown()
