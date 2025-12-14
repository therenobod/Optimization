# 最速下降法(steepest desent, SD)

"""
迭代步骤：
x_k+1 = x_k + alpha_k * d_k
"""
import numpy as np

def SD_solve(funcation, x_0, linesearch, epsilon = 1e-6, k_max = 1000,):
    x_history = []
    g_history = []
    k = 0
    # 确保是 numpy 向量，避免 list 和 ndarray 混用导致运算异常
    x_k = np.array(x_0, dtype=float)
    g_k = funcation.grad(x_k)
    x_history.append(x_k.copy())
    g_history.append(g_k.copy())

    while np.linalg.norm(g_k) > epsilon and k < k_max:
        d_k = -g_k
        alpha_k = linesearch(funcation.f, x_k, d_k)
        x_k = x_k + alpha_k * d_k
        g_k = funcation.grad(x_k)

        x_history.append(x_k.copy())
        g_history.append(g_k.copy())

        k += 1

    return x_history, g_history, k