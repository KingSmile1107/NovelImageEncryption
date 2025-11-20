import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as mpl

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

listx = []
listy = []
listy_sine = []
listy_tent = []

def sine_tent_cosine_map(r, x):
    if x < 0.5:
        return np.cos(np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * x - 0.5))
    else:
        return np.cos(np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * (1 - x) - 0.5))

def derivative_sine_tent_cosine_map(r, x):
    if x < 0.5:
        inner = np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * x - 0.5)
        return -np.pi * np.sin(inner) * (np.pi * r * np.cos(np.pi * x) + 2 * (1 - r))
    else:
        inner = np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * (1 - x) - 0.5)
        return -np.pi * np.sin(inner) * (np.pi * r * np.cos(np.pi * x) - 2 * (1 - r))

def Lyapunov(x, r, map_func, derivative_func):
    return np.log(abs(derivative_func(r, x)))

def sine_map(r, x):
    return r * np.sin(np.pi * x)

def derivative_sine_map(r, x):
    return np.pi * r * np.cos(np.pi * x)

def tent_map(r, x):
    if x < 0.5:
        return 2 * r * x
    else:
        return 2 * r * (1 - x)

def derivative_tent_map(r, x):
    if x < 0.5:
        return 2 * r
    else:
        return -2 * r

if __name__ == '__main__':
    iteration = 1000
    n = 1000
    for r in tqdm(np.linspace(0, 1, n)):
        x = 10e-5
        ly_cosine = 0
        ly_sine = 0
        ly_tent = 0
        for i in range(iteration):
            x1 = sine_tent_cosine_map(r, x)
            ly_cosine += Lyapunov(x, r, sine_tent_cosine_map, derivative_sine_tent_cosine_map)
            x = x1
        listx.append(r)
        listy.append(ly_cosine / iteration)

        x = 10e-5
        for i in range(iteration):
            x1 = sine_map(r, x)
            ly_sine += Lyapunov(x, r, sine_map, derivative_sine_map)
            x = x1
        listy_sine.append(ly_sine / iteration)

        x = 10e-5
        for i in range(iteration):
            x1 = tent_map(r, x)
            ly_tent += Lyapunov(x, r, tent_map, derivative_tent_map)
            x = x1
        listy_tent.append(ly_tent / iteration)

    plt.plot(listx, listy, label='STC map', linestyle='-', color='red')
    plt.plot(listx, listy_sine, label='Sine map', linestyle='--', color='royalblue')
    plt.plot(listx, listy_tent, label='Tent map', linestyle='-.', color='forestgreen')
    plt.axhline(0, color='gray', lw=0.7, alpha=0.5)
    plt.xlabel("α")
    plt.ylabel("LE")
    plt.ylim(-2, 2)
    plt.legend()
    plt.show()
