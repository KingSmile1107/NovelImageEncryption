import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def chaotic_map(r, x):
    if x < 0.5:
        return (r * x + np.cos(r * np.arccos(x))) % 1
    else:
        return (r * (1 - x) + np.cos(r * np.arccos(x))) % 1


def derivative_chaotic_map(r, x):
    if x < 0.5:
        inner = np.arccos(x)
        term1 = r
        term2 = -r * np.sin(r * inner) / np.sqrt(1 - x ** 2)
    else:
        inner = np.arccos(x)
        term1 = -r
        term2 = -r * np.sin(r * inner) / np.sqrt(1 - x ** 2)
    return np.abs(term1 + term2)


def Lyapunov(x, r):
    epsilon = 1e-10  # 添加一个小的epsilon以避免除零错误
    return np.log(derivative_chaotic_map(r, x) + epsilon)


if __name__ == '__main__':
    iteration = 1000  # 增加迭代次数
    n_points = 1000
    listx = []
    listy = []

    for r in tqdm(np.linspace(0, 8, n_points)):
        x_i = 0.1  # 改变初始条件
        ly = 0
        for i in range(iteration):
            x_i = chaotic_map(r, x_i)
            ly += Lyapunov(x_i, r)
        listx.append(r)
        listy.append(ly / iteration)  # 不添加常数，Lyapunov指数是多少就是多少

    plt.plot(listx, listy)
    plt.axhline(0, color='red', lw=0.7, alpha=0.5)
    plt.xlabel(r'$r$')
    plt.ylabel('Lyapunov Exponents')
    plt.title('Lyapunov exponent curves of the chaotic map (5 < r < 6)')
    plt.show()
