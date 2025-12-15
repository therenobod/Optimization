import numpy as np
import matplotlib.pyplot as plt


def plot_xy(x=None, y=None,
            xlabel="x",
            ylabel="y",
            title="", 
            use_marker=True):
    """通用二维折线/散点图绘制函数。

    参数
    ------
    x : 1D array-like 或 None
        横坐标数据；若为 None，则使用 0..len(y)-1。
    y : 1D array-like
        纵坐标数据（必填）。
    xlabel : str
        x 轴标签。
    ylabel : str
        y 轴标签。
    title : str
        图像标题。
    use_marker : bool
        是否在折线上加点标记。
    """
    if y is None:
        raise ValueError("y 不能为空")

    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)

    if x.shape != y.shape:
        raise ValueError("x 和 y 的形状不一致，无法绘图")

    plt.figure()
    marker = "o" if use_marker else None
    plt.plot(x, y, marker=marker)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 简单示例：y = x^2
    xs = np.linspace(-2, 2, 50)
    ys = xs ** 2
    plot_xy(xs, ys, xlabel="x", ylabel="x^2", title="Demo Plot")
