import numpy as np
from scipy.optimize import linprog


def center(A: np.array, b: np.array) -> tuple:
    A_ = np.concatenate([A, np.sqrt(np.sum(A**2, 1, keepdims=True))], 1)
    c = np.array([0, 0, -1])

    res = linprog(c, A_ub=A_, b_ub=b)
    x = res.x[:2]
    r = res.x[2]

    return x, r
