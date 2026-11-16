# harmonic/prz.py

def calculate_prz(pattern_name, prices):
    X, A, B, C, D = prices

    prz_low = min(D, C)
    prz_high = max(D, C)

    return {
        "prz_low": prz_low,
        "prz_high": prz_high,
        "mid": (prz_low + prz_high) / 2
    }
