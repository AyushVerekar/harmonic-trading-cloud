# harmonic/swing_detector.py

import numpy as np

def detect_swings(df, window=3):
    highs = df["high"].values
    lows = df["low"].values

    swing_points = []

    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i-window:i+window]):
            swing_points.append((i, highs[i], "H"))

        if lows[i] == min(lows[i-window:i+window]):
            swing_points.append((i, lows[i], "L"))

    return swing_points
