# 中心差分法数值求解函数梯度
import numpy as np

# 中心差分法求解梯度
def numerical_gradient(f, x, h=1e-5):
    """
    numerical_gradient 的 Docstring
    
    :param f: 求解函数
    :param x: 计算梯度的点
    :param h: 中心差分的偏移
    """
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


if __name__ == "__main__":
    from funcation import f_modified_rosenbrock, f_rosenbrock

    # 测试 1：简单二次函数 f(x, y) = x^2 + 3y^2，解析梯度为 [2x, 6y]
    def quad_func(x):
        return x[0] ** 2 + 3 * x[1] ** 2

    x_test = np.array([1.0, -2.0])
    grad_num = numerical_gradient(quad_func, x_test)
    grad_true = np.array([2 * x_test[0], 6 * x_test[1]])
    print("=== Test 1: simple quadratic ===")
    print("x          =", x_test)
    print("grad_num   =", grad_num)
    print("grad_true  =", grad_true)
    print("error_norm =", np.linalg.norm(grad_num - grad_true))

    # 测试 2：在若干点上Modified Rosenbrock 
    print("\n=== Test 2: modified Rosenbrock (shape check) ===")
    test_points_MR = [
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([-1.0, 1.0]),
    ]

    for p in test_points_MR:
        g = numerical_gradient(f_modified_rosenbrock, p)
        print(f"x = {p}, grad = {g}")

    # 测试 3：4 维 Rosenbrock 函数
    print("\n=== Test 3: 4D Rosenbrock (shape check) ===")
    test_points_R = [
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-1.0, 1.0, -1.0, 1.0]),
        np.array([0.5, 0.25, 0.75, 0.1]),
    ]
    for p in test_points_R:
        g = numerical_gradient(f_rosenbrock, p)
        print(f"x = {p}, grad = {g}")

    