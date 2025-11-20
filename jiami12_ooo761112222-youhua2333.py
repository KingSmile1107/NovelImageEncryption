import time
from math import floor

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import img_bit_decomposition, sine_tent_cosine_map, chebyshev_tent_map, logistic_map, \
    test_img_bit_decomposition, seeded_shuffle, seeded_unshuffle

# from analysis import *

img_path_base = "./img/"
img_name = "Lena.png"  # 自行修改图片名，默认为 lena 图
img_name1 = "img_1.png"
img_name2 = "test1.png"
img_name3 = "test1_differencial1.png"
img_path = img_path_base + img_name

filename, ext = img_path.rsplit('.', 1)
encrypt_img_path = f"{filename}_233345678.png"
encrypt_img_path1 = f"{filename}_encrypt1.png"

def cubic_map(x):
    return x ** 3 - x  # Cubic 映射公式


# def sine_cubic_map(x, omega):
#     # 防止 f(x) = 0 时分母为零
#     fx = cubic_map(x)
#     return (np.sin(omega * (10 - x) / cubic_map(x))) % 1


def sine_cubic_map(x0, omega, sequence_length):
    sequence = np.zeros(sequence_length)  # 初始化序列数组
    x = x0  # 设置初始值x

    for i in range(sequence_length):  # 进行迭代
        if x < 0.5:
            # 对于x < 0.5的情况，按照STC映射公式计算
            x = (np.sin(omega * (10 - x) / cubic_map(x))) % 1
        else:
            # 对于x >= 0.5的情况，按照STC映射公式计算
            x = (np.sin(omega * (10 - x) / cubic_map(x))) % 1
        sequence[i] = x  # 将计算结果存入序列数组

    return sequence


x01 = 0.2 #0.20000000000001
r01 = 0.3

x11 = 0.4
r11 = 0.5

x21 = (x01 + x11) % 1
r21 = (r01 + r11) % 1

x0 = 0.1 #0.10000000000001
u0 = 2.2  # 2.20000000000001
beta0 = 0.5 # 0.50000000000001

x1 = 0.7
u1 = 3.3
beta1 = 0.4

x2 = (x0 + x1) % 1
u2 = (u0 + u1) % 4
beta2 = (beta0 + beta1) % 1

def reconstruct_image(bit_planes):
    """Reconstruct a single-channel image from bit planes"""
    return np.sum([(bit_planes[i] * (2 ** i)).astype(np.uint8) for i in range(8)], axis=0)

def img_bit_decomposition_color(img):
    """Decompose a color image into bit planes"""
    return [img_bit_decomposition(img[:, :, channel]) for channel in range(3)]

def reconstruct_image_color(bit_planes_color):
    """Reconstruct a color image from bit planes"""
    channels = [reconstruct_image(bit_planes) for bit_planes in bit_planes_color]
    channels = [channel.astype(np.uint8) for channel in channels]
    return cv2.merge(channels)


def jiami(img):
    # 读取彩色图像
    example_img_color = cv2.imread(img, cv2.IMREAD_COLOR)

    m, n, _ = example_img_color.shape

    # 记录开始时间
    start_time = time.time()
    start_time1 = time.time()

    N0 = 1000
    sine_tent_cosine_map_x0 = sine_cubic_map(x01, r01, N0 + 3 * m * n)[N0:]
    sine_tent_cosine_map_x2 = sine_cubic_map(x21, r21, N0 + 3 * m * n)[N0:]

    X0 = [floor(j * 1e14) % 256 for j in sine_tent_cosine_map_x0]
    X2 = [floor(j * 3 * m * n) for j in sine_tent_cosine_map_x2]

    N01 = 1000
    chebyshev_tent_map_x0 = sine_cubic_map(x0, beta0, N01 + m * 8 * 3)[N01:]
    chebyshev_tent_map_x1 = sine_cubic_map(x1, beta1, N01 + n * 8 * 3)[N01:]
    chebyshev_tent_map_x2 = sine_cubic_map(x2, beta2, N01 + m * n)[N01:]

    hang_random = [floor(j * 1e14) % 256 for j in chebyshev_tent_map_x0]
    hang_random_split = np.array_split(hang_random, 24)
    lie_random = [floor(j * 1e14) % 256 for j in chebyshev_tent_map_x1]
    lie_random_split = np.array_split(lie_random, 24)

    X01 = [floor(j * 1e14) % 256 for j in chebyshev_tent_map_x2]

    X01_reshape1 = np.mat(X01, dtype=np.uint8).reshape(m, n)

    X01_bitplanes1 = img_bit_decomposition(X01_reshape1)

    # 记录结束时间
    end_time1 = time.time()
    print("产生混沌序列运行时间:", end_time1 - start_time1, "秒")

    # -----------------------------------时间一样

    start_time2 = time.time()

    # 拆分图像并转换为一维数组
    B, G, R = cv2.split(example_img_color)
    all1 = np.concatenate((R.ravel(), G.ravel(), B.ravel())).astype(np.int64)
    # all1 = np.concatenate((R.ravel(), G.ravel(), B.ravel()))

    # print(type(all1[0]))

    for j in range(3):
        for i in range(3 * m * n):
            if i == 0:
                all1[i] = ((all1[i] + all1[-1]) % 256) ^ X0[i]
            else:
                all1[i] = ((all1[i] + all1[i - 1])% 256) ^ X0[i]

    all1 = all1.astype(np.uint8)  # 转换回 uint8，确保适用于 OpenCV

    end_time2 = time.time()
    print("扩散运行时间:", end_time2 - start_time2, "秒")

    start_time3 = time.time()
    seeded_shuffle(all1, X2)
    end_time3 = time.time()
    print("fy置乱运行时间:", end_time3 - start_time3, "秒")

    all_split = np.array_split(all1, 3)

    B11 = all_split[2]
    G11 = all_split[1]
    R11 = all_split[0]

    B2 = B11.reshape((m, n))
    G2 = G11.reshape((m, n))
    R2 = R11.reshape((m, n))

    E = cv2.merge([B2, G2, R2])
    end_time4 = time.time()

    start_time5 = time.time()
    # 应用位平面分解于彩色图像
    decomposed_bit_planes_color = img_bit_decomposition_color(E)

    for i in range(3):
        for j in range(4, 8):
            decomposed_bit_planes_color[i][j] = np.bitwise_xor(decomposed_bit_planes_color[i][j],
                                                               X01_bitplanes1[j, :, :])

    # 记录结束时间
    end_time5 = time.time()
    print("比特级1111运行时间:", end_time5 - start_time5, "秒")

    # 记录开始时间
    start_time6 = time.time()

    for i in range(2):
        for j in range(3):
            for k in range(m):
                seeded_shuffle(decomposed_bit_planes_color[i][j][k], hang_random_split[(i+1)*j])
            for z in range(n):
                seeded_shuffle(decomposed_bit_planes_color[i][j][:][z], lie_random_split[(i+1)*j])

    # 记录结束时间
    end_time6 = time.time()
    print("比特级2222运行时间:", end_time6 - start_time6, "秒")

    # 从位平面重构彩色图像
    reconstructed_img_color = reconstruct_image_color(decomposed_bit_planes_color)
    # # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    print("总运行时间:", end_time - start_time, "秒")
    cv2.imwrite(encrypt_img_path, reconstructed_img_color)  # 保存图像
    return reconstructed_img_color


def main():
    # 执行函数
    jiami(img_path)

if __name__ == '__main__':
    main()