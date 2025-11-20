import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as mpl

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

listx = []
listy = []

def sine_tent_cosine_map(r, x, b):
    if x < 0.5:
        return np.cos(np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * x + b))
    else:
        return np.cos(np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * (1 - x) + b))

def derivative_sine_tent_cosine_map(r, x, b):
    if x < 0.5:
        inner = np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * x + b)
        return -np.pi * np.sin(inner) * (np.pi * r * np.cos(np.pi * x) + 2 * (1 - r))
    else:
        inner = np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * (1 - x) + b)
        return -np.pi * np.sin(inner) * (np.pi * r * np.cos(np.pi * x) - 2 * (1 - r))

def Lyapunov(x, r, b):
    return np.log(abs(derivative_sine_tent_cosine_map(r, x, b)))

if __name__ == '__main__':
    iteration = 1000
    n = 1000
    for r in tqdm(np.linspace(0, 1, n)):
        x = 10e-5
        ly = 0
        for i in range(iteration):
            x1 = sine_tent_cosine_map(r, x, b)
            ly += Lyapunov(x, r, b)
            x = x1
        listx.append(r)
        listy.append(ly / iteration)
    plt.plot(listx, listy)
    plt.axhline(0, color='red', lw=0.7, alpha=0.5)
    plt.xlabel("α")
    plt.ylabel("LE")
    # plt.title('Lyapunov exponent curves of the Sine-Tent-Cosine map')
    plt.show()
