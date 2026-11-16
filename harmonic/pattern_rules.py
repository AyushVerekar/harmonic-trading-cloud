# harmonic/pattern_rules.py

from .fib_utils import fib_ratio, in_range, ratio_error
from .config import PATTERN_RULES

def validate_pattern(name, prices):
    X, A, B, C, D = prices
    rules = PATTERN_RULES[name]

    ab = fib_ratio(X, A, B)
    bc = fib_ratio(A, B, C)
    cd = fib_ratio(B, C, D)
    ad = abs(D - X) / (abs(A - X) + 1e-9)

    checks = {
        "AB": (ab, rules["AB"]),
        "BC": (bc, rules["BC"]),
        "CD": (cd, rules["CD"]),
        "AD": (ad, rules["AD"]),
    }

    total_error = 0

    for key, (val, rule) in checks.items():
        if not in_range(val, *rule):
            return None
        total_error += ratio_error(val, *rule)

    confidence = max(0, 1 - total_error)

    return {
        "confidence": round(confidence, 3),
        "ratios": {"AB": ab, "BC": bc, "CD": cd, "AD": ad}
    }
