# 二维Modified-Rosenbrock函数和低维 Rosenbrock 函数扩展版

import numpy as np

class f_modified_rosenbrock():
    # Modified_Rosenbrock
    def f(self, x):
        """
        2维 Modified Rosenbrock 函数
        
        参数:
        x: 长度为2的数组或列表 [x1, x2]
        
        返回:
        f: 函数值
        """
        x1, x2 = x[0], x[1]
        term1 = (1 - x1) ** 2
        term2 = 100 * (x2 - x1 ** 2) ** 2
        term3 = 5 * np.sin(2 * np.pi * x1) * np.sin(2 * np.pi * x2)
        
        return term1 + term2 + term3

    # MR函数梯度
    def grad(self, x):
        x1, x2 = x[0], x[1]
        grad = np.zeros_like(x)
        
        grad[0] = 2*(x1-1) + 400*(x1 ** 2 - x2)*x1 + 10 * np.pi * np.cos(2*np.pi*x1)*np.sin(2*np.pi*x2)
        grad[1] = 200 * (x2 - x1**2) + 10*np.pi*np.sin(2*np.pi*x1)*np.cos(2*np.pi*x2)

        return grad
    
class f_rosenbrock():
    # R函数
    def f(self, x):
        """
        4维标准Rosenbrock函数（相邻维度耦合）
        
        参数:
        x: 长度为4的数组 [x1, x2, x3, x4]
        
        返回:
        f: 函数值
        """
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        
        term1 = (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2
        term2 = (1 - x2) ** 2 + 100 * (x3 - x2 ** 2) ** 2
        term3 = (1 - x3) ** 2 + 100 * (x4 - x3 ** 2) ** 2
        
        return term1 + term2 + term3

    def grad(self, x):
        """
        f_rosenbrock_grad 的 Docstring
        
        :param x: 求解梯度对应点
        """
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        grad = np.zeros_like(x)

        def model1(a,b):
            return 2*(a-1) + 400*(a**2-b)*a
        
        def model2(a,c):
            return 200*(a-c**2)
        
        grad[0] = model1(x1,x2)
        grad[1] = model1(x2,x3) + model2(x2,x1)
        grad[2] = model1(x3,x4) + model2(x3,x2)
        grad[3] = model2(x4,x3)
        
        return grad

if __name__ == "__main__":
    # MR测试
    test_points_MR = [
        [1.0, 1.0],      # 全局极小值点附近
        [0.0, 0.0],      # 原点
        [-1.0, 1.0],     # 其他点
        [-1.1, 1.2],
        [0.65, 0.3]
    ]

    f_MR = f_modified_rosenbrock()

    print("MR函数测试")
    for point in test_points_MR:
        value = f_MR.f(point)
        print(f"f({point[0]:.2f}, {point[1]:.2f}) = {value:.4f}")

    """
    f(1.00, 1.00) = 0.0000
    f(0.00, 0.00) = 1.0000
    f(-1.00, 1.00) = 4.0000
    f(-1.10, 1.20) = 1.6249
    f(0.65, 0.30) = -2.2240
    """

    print("MR函数梯度测试")
    for point in test_points_MR:
        grad = f_MR.grad(point)
        print(f"f({point[0]:.2f}, {point[1]:.2f}) = {grad}")

    """
    f(1.00, 1.00) = [-7.69468277e-15 -7.69468277e-15]
    f(0.00, 0.00) = [-2.  0.]
    f(-1.00, 1.00) = [-4.00000000e+00  7.69468277e-15]
    """


    f_R = f_rosenbrock()
    # R测试
    test_points_R = np.array([
        [1.0, 1.0, 1.0, 1.0],      # 全局极小值
        [0.0, 0.0, 0.0, 0.0],      # 原点
        [-1.0, 1.0, -1.0, 1.0],    # 其他点
        [0.5, 0.25, 0.75, 0.1]     
    ])

    print("R函数测试")
    for point in test_points_R:
        value = f_R.f(point)
        print(f"f({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}, {point[3]:.2f}) = {value:.4f}")
    """
    f(1.00, 1.00, 1.00, 1.00) = 0.0000
    f(0.00, 0.00, 0.00, 0.00) = 3.0000
    f(-1.00, 1.00, -1.00, 1.00) = 408.0000
    f(0.50, 0.25, 0.75, 0.10) = 69.5312
    """

    print("R函数梯度测试")
    for point in test_points_R:
        grad = f_R.grad(point)
        print(f"f({point[0]:.2f}, {point[1]:.2f}) = {grad}")
    
    """
    f(1.00, 1.00) = [0. 0. 0. 0.]
    f(0.00, 0.00) = [-2. -2. -2.  0.]
    f(-1.00, 1.00) = [  -4.  800. -404.    0.]
    f(0.50, 0.25) = [ -1.   -70.25 275.75 -92.5 ]
    """

