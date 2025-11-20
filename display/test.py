import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as mpl

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


def sine_tent_cosine_map(r, x, beta):
    if x < 0.5:
        return np.cos(np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * x + beta))
    else:
        return np.cos(np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * (1 - x) + beta))


def derivative_sine_tent_cosine_map(r, x, beta):
    if x < 0.5:
        inner = np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * x + beta)
        return -np.pi * np.sin(inner) * (np.pi * r * np.cos(np.pi * x) + 2 * (1 - r))
    else:
        inner = np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * (1 - x) + beta)
        return -np.pi * np.sin(inner) * (np.pi * r * np.cos(np.pi * x) - 2 * (1 - r))


def Lyapunov(x, r, beta):
    return np.log(abs(derivative_sine_tent_cosine_map(r, x, beta)))


if __name__ == '__main__':
    iteration = 1000
    n = 1000
    beta_values = [0.9, 0.5, 0.3, 0.1]
    plt.figure(figsize=(12, 8))

    for beta in beta_values:
        listx = []
        listy = []
        for r in tqdm(np.linspace(0, 1, n)):
            x = 10e-5
            ly = 0
            for i in range(iteration):
                x1 = sine_tent_cosine_map(r, x, beta)
                ly += Lyapunov(x, r, beta)
                x = x1
            listx.append(r)
            listy.append(ly / iteration)
        plt.plot(listx, listy, label=f'β = {beta}')

    plt.axhline(0, color='red', lw=0.7, alpha=0.5)
    plt.xlabel("α", fontsize=16)
    plt.ylabel("LE", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=16, framealpha=1, borderpad=1.5)
    # plt.title('Lyapunov exponent curves of the Sine-Tent-Cosine map for different β values', fontsize=16)
    plt.show()