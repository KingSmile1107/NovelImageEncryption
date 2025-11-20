import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as mpl
mpl.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
listx=[]
listy=[]
def logistics(r,x):
    x1 = r * x * (1 - x)
    return x1
def Lyapunov(x, r):
    return np.log(abs(r - 2 * r * x))
if __name__ == '__main__':
    iteration=1000
    n=1000
    for r in tqdm(np.linspace(2.50, 4.00, n)):
        x = 10e-5
        ly = 0
        for i in range(iteration):
             x1 = logistics(r, x)
             x = x1
             ly += Lyapunov(x, r)
        listx.append(r)
        listy.append(ly / iteration)
    plt.plot(listx, listy)
    plt.axhline(0, color='red', lw=0.7, alpha=0.5)
    plt.xlabel("μ")
    plt.ylabel("Liapunov Exponents")
    plt.show()