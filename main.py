import numpy as np
from SD import SD_solve
from funcation import f_modified_rosenbrock,f_rosenbrock
from line_search import exact_line_search
from plot_utils import plot_xy

class optimization():
    """
    optimization 的 Docstring

    传入
    function:目标函数
    x0:初始点
    solve:求解器/求解方法
    linesearch:求解器使用的线搜索
    plot:绘图方法
    """
    def __init__(self, 
                 function,
                 x0,
                 solve,
                 linesearch,
                 plot,
                 ):
        self.function = function
        self.x0 = x0
        self.solve = solve
        self.linesearch = linesearch
        self.plot = plot
        self.x_list = []
        self.g_list = []
        self.k = None
    
    # 使用优化器优化函数
    def optimize(self):
        self.x_list, self.g_list, self.k = self.solve(self.function, self.x0,self.linesearch)

    # 绘制梯度模的图像
    def draw_norm(self):
        g_norms = self.g_to_norm()
        iters = np.arange(len(g_norms))
        self.plot(iters, g_norms)

    # 绘制梯度模长的对数的图像
    def draw_log_norm(self):
        g_log_norms = self.g_to_log_norm()
        iters = np.arange(len(g_log_norms))
        self.plot(iters, g_log_norms)

    def g_to_norm(self):
        g_norms = [np.linalg.norm(g) for g in self.g_list]
        return g_norms
    
    def g_to_log_norm(self):
        g_norms = self.g_to_norm()
        g_log_norms = [np.log(g) for g in g_norms]
        return g_log_norms
    
    # 打印优化的最新信息
    def print_last(self):
        print(f"k = {self.k}")
        print("last x =", self.x_list[-1])
        print("last grad =", self.g_list[-1])


f_MR = f_rosenbrock()
x_MR = np.array([-1.2,1,-0.8, 0.6])
optimization_ = optimization(function=f_MR, x0=x_MR,solve=SD_solve, linesearch=exact_line_search,plot=plot_xy)

optimization_.optimize()
optimization_.draw_log_norm()




# # MR函数优化，会陷入局部最小值点
# f_MR = f_modified_rosenbrock()
# x_MR = np.array([-1.2, 1.0])

# x_list, g_list, k = SD_solve(funcation=f_MR, x_0=x_MR,linesearch=exact_line_search,k_max=1000)
# print(f"k = {k}")
# print("last x =", x_list[-1])
# print("last grad =", g_list[-1])

# grad_norms = [np.linalg.norm(g) for g in g_list]
# grad_norms_log = [np.log(g) for g in grad_norms]
# iters = np.arange(len(grad_norms))

# plot_xy(x=iters, y= grad_norms_log)

# # R函数优化，收纳速度慢，在第15000次迭代附近收敛
# f_R = f_rosenbrock()
# x_R = np.array([-1.2,1,-0.8, 0.6])

# x_list, g_list, k = SD_solve(funcation=f_R, x_0=x_R, linesearch=exact_line_search,k_max=100000)
# print(f"k = {k}")
# print("last x =", x_list[-1])
# print("last grad =", g_list[-1])
# g_norm =np.linalg.norm(g_list[-1])
# print(f"last grad norm = {g_norm:.6e}")