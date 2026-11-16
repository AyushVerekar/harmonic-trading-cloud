import MetaTrader5 as mt5
mt5.initialize()
print([s.name for s in mt5.symbols_get() if "XAU" in s.name])
