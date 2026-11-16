# harmonic/deduplicator.py

def remove_duplicates(patterns, price_tol=0.001):
    unique = []

    for p in patterns:
        exists = False
        for u in unique:
            if (
                p["pattern"] == u["pattern"]
                and abs(p["D_price"] - u["D_price"]) < price_tol
            ):
                exists = True
                break
        if not exists:
            unique.append(p)

    return unique
