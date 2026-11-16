# harmonic/risk_manager.py

def compute_risk(prices, direction):
    X, A, B, C, D = prices

    if direction == "bullish":
        stop_loss = D * 0.995
        tp1 = C
        tp2 = A
    else:
        stop_loss = D * 1.005
        tp1 = C
        tp2 = A

    return {
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2
    }
