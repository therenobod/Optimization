# 线搜索
import numpy as np

# 黄金分割法
def _golden_section(phi, a, b, tol=1e-6, max_iter=100):
    """在给定区间 [a, b] 上对 phi(alpha) 做黄金分割搜索。"""
    gr = (np.sqrt(5) - 1) / 2  # ≈ 0.618

    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = phi(c)
    fd = phi(d)

    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = phi(d)

    return 0.5 * (a + b)


def exact_line_search(f, x, d,
                      alpha_init=1.0,
                      expansion_factor=2.0,
                      max_expand=10,
                      tol=1e-6,
                      max_iter=100):
    """带自适应区间扩展的精确线搜索。

    先从 alpha_init 开始沿方向 d 扩展步长，直到不再下降（或达到最大扩展次数），
    在得到的 [alpha_left, alpha_right] 区间上用黄金分割搜索近似最优步长。

    :param f: 目标函数 f(x)
    :param x: 当前点，numpy 向量
    :param d: 搜索方向，numpy 向量
    :param alpha_init: 初始步长
    :param expansion_factor: 区间扩展倍率 (>1)
    :param max_expand: 最多扩展次数
    :param tol: 黄金分割法区间长度容忍度
    :param max_iter: 黄金分割法最大迭代次数
    :return: 近似最优步长 alpha_star
    """

    phi = lambda a: f(x + a * d)

    # 自适应扩展区间，寻找包含最小值的 [alpha_left, alpha_right]
    alpha_left = 0.0
    alpha_right = alpha_init

    f0 = phi(0.0)
    f_right = phi(alpha_right)

    # 如果一开始就没下降，就缩小初始步长，直到区间右端点小于左端点
    if f_right >= f0:
        for _ in range(max_expand):
            alpha_right /= expansion_factor
            if alpha_right <= 1e-12:
                # 退化为几乎不动
                return 0.0
            f_right = phi(alpha_right)
            if f_right < f0:
                break

    # 如果下降，就尝试向更大的步长扩展，直到右端点超过左端点
    # 记录原本区间
    alpha_prev = alpha_left
    alpha_curr = alpha_right
    f_curr = f_right

    for _ in range(max_expand):
        alpha_next = alpha_curr * expansion_factor
        f_next = phi(alpha_next)

        # 只要还在继续下降，就继续扩展
        if f_next < f_curr:
            alpha_prev = alpha_curr
            alpha_curr, f_curr = alpha_next, f_next
        else:
            # 找到一个区间 [alpha_prev, alpha_next] 包含极小值
            alpha_left = alpha_prev
            alpha_right = alpha_next
            break
    else:
        # 扩展到上限也一直在下降，就用 [alpha_prev, alpha_curr] 做搜索
        alpha_left = alpha_prev
        alpha_right = alpha_curr

    # 在 [alpha_left, alpha_right] 上做黄金分割搜索
    alpha_star = _golden_section(phi, alpha_left, alpha_right,
                                 tol=tol, max_iter=max_iter)
    return alpha_star

# 精确线搜索
# def exact_line_search(f, x, d):
#     """
#     exact_line_search 的 Docstring
    
#     :param f: 精确线搜索函数
#     :param x: 初始点
#     :param d: 线搜索方向
#     """
#     dir = {}
#     alpha = 0



#     return alpha

# Armijo+Wolfe 非精确线搜索
