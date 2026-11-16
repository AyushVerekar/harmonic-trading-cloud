# harmonic/fib_utils.py

import numpy as np

def fib_ratio(p1, p2, p3):
    return abs(p3 - p2) / (abs(p2 - p1) + 1e-9)

def in_range(val, min_val, max_val, tol=0.03):
    return (min_val - tol) <= val <= (max_val + tol)

def ratio_error(val, min_val, max_val):
    mid = (min_val + max_val) / 2
    return abs(val - mid)
