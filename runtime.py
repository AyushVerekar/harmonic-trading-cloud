# runtime.py

import pandas as pd
from harmonic.pattern_engine import HarmonicEngine

SYMBOL = "XAUUSD"
PATTERNS = ["gartley", "bat", "butterfly", "crab", "shark"]

engine = HarmonicEngine(PATTERNS)


def load_and_clean_csv(path):
    # 🔥 IMPORTANT: MT5 exports TAB separated files
    df = pd.read_csv(path, sep="\t")

    # Normalize column names
    df.columns = [c.strip().lower().replace("<", "").replace(">", "") for c in df.columns]

    # Combine date + time
    if "date" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["date"] + " " + df["time"])
        df.drop(columns=["date"], inplace=True)

    # Required OHLC columns
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing column: {col}. Found columns: {df.columns.tolist()}"
            )

    # Convert to numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols)

    return df



def scan_and_save_from_csv():
    print("\nScanning CSV historical data...\n")

    file_map = {
        "5m": "data/xauusd_m5.csv",
        "15m": "data/xauusd_m15.csv",
        "1h": "data/xauusd_h1.csv"
    }

    all_patterns = []

    for tf_name, path in file_map.items():
        print(f"Scanning {tf_name} from CSV")

        df = load_and_clean_csv(path)

        print("Candles loaded:", len(df))

        patterns = engine.detect(df)

        print(f"Found {len(patterns)} patterns on {tf_name}")

        for p in patterns:
            row = {
                "symbol": SYMBOL,
                "timeframe": tf_name,
                "pattern": p["pattern"],
                "direction": p["direction"],
                "confidence": p["confidence"],
                "D_price": p["D_price"],
                "prz_low": p["PRZ"]["prz_low"],
                "prz_high": p["PRZ"]["prz_high"],
                "tp1": p["risk"]["tp1"],
                "tp2": p["risk"]["tp2"],
                "stop_loss": p["risk"]["stop_loss"],
                "index": p["index"]
            }
            all_patterns.append(row)

    if all_patterns:
        df_out = pd.DataFrame(all_patterns)
        df_out.to_csv("harmonic_lifetime_data.csv", index=False)
        print("\nSaved harmonic_lifetime_data.csv\n")
    else:
        print("No patterns found.")


if __name__ == "__main__":
    scan_and_save_from_csv()
