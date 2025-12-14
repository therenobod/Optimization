import numpy as np
from SD import SD_solve
from funcation import f_modified_rosenbrock,f_rosenbrock
from line_search import exact_line_search


# MR函数优化，会陷入局部最小值点
f_MR = f_modified_rosenbrock()
x_MR = np.array([-1.2, 1.0])

x_list, g_list, k = SD_solve(funcation=f_MR, x_0=x_MR,linesearch=exact_line_search,k_max=1000)
print(f"k = {k}")
print("last x =", x_list[-1])
print("last grad =", g_list[-1])


# R函数优化，收纳速度慢，在第15000次迭代附近收敛
f_R = f_rosenbrock()
x_R = np.array([-1.2,1,-0.8, 0.6])

x_list, g_list, k = SD_solve(funcation=f_R, x_0=x_R, linesearch=exact_line_search,k_max=100000)
print(f"k = {k}")
print("last x =", x_list[-1])
print("last grad =", g_list[-1])
g_norm =np.linalg.norm(g_list[-1])
print(f"last grad norm = {g_norm:.6e}")