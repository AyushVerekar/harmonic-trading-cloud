# harmonic/signal_generator.py

def generate_signal(pattern_data):
    if pattern_data["direction"] == "bullish":
        return "BUY"
    else:
        return "SELL"

