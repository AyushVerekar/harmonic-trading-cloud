# harmonic/pattern_engine.py

from .swing_detector import detect_swings
from .pattern_rules import validate_pattern
from .prz import calculate_prz
from .risk_manager import compute_risk
from .deduplicator import remove_duplicates

class HarmonicEngine:

    def __init__(self, patterns):
        self.patterns = patterns

    def detect(self, df):
        swings = detect_swings(df)
        found = []

        if len(swings) < 5:
            return []

        for i in range(len(swings) - 4):
            idxs = swings[i:i+5]
            prices = [p[1] for p in idxs]

            for pattern in self.patterns:
                result = validate_pattern(pattern, prices)

                if result:
                    direction = "bullish" if prices[-1] < prices[-2] else "bearish"
                    prz = calculate_prz(pattern, prices)
                    risk = compute_risk(prices, direction)

                    found.append({
                        "pattern": pattern,
                        "direction": direction,
                        "confidence": result["confidence"],
                        "ratios": result["ratios"],
                        "D_price": prices[-1],
                        "PRZ": prz,
                        "risk": risk,
                        "index": idxs[-1][0]
                    })

        return remove_duplicates(found)
