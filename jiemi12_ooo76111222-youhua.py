import time
from math import floor

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import img_bit_decomposition, sine_tent_cosine_map, logistic_map, \
    chebyshev_tent_map, seeded_unshuffle, seeded_shuffle

# from analysis import *

img_path_base = "./img/"
# img_name = "Lossy_Lena_encrypt111111111111233333.png"
img_name = "Lena_233345678.png"
img_name1 = "Female_encrypt.png"
img_name2 = "House_encrypt.png"
img_name3 = "Lena_encrypt.png"
img_name4 = "Peppers_encrypt.png"
img_path = img_path_base + img_name

filename, ext = img_path.rsplit('.', 1)
encrypt_img_path = f"{filename}_save.png"

x01 = 0.2 #0.20000000000001
r01 = 0.3

x11 = 0.4
r11 = 0.5

x21 = (x01 + x11) % 1
r21 = (r01 + r11) % 1

x0 = 0.1
u0 = 2.2
beta0 = 0.5

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

def jiemi(img):
    # 读取彩色图像
    example_img_color = cv2.imread(img, cv2.IMREAD_COLOR)

    m, n, _ = example_img_color.shape

    # 记录开始时间
    start_time = time.time()
    start_time1 = time.time()

    # 应用位平面分解于彩色图像
    decomposed_bit_planes_color = img_bit_decomposition_color(example_img_color)

    N0 = 1000
    sine_tent_cosine_map_x0 = sine_tent_cosine_map(x01, r01, N0 + 3 * m * n)[N0:]
    # sine_tent_cosine_map_x1 = sine_tent_cosine_map(x11, r11, N0 + 3 * m * n)[N0:]
    sine_tent_cosine_map_x2 = sine_tent_cosine_map(x21, r21, N0 + 3 * m * n)[N0:]

    X0 = [floor(j * 1e14) % 256 for j in sine_tent_cosine_map_x0]
    # X1 = [floor(j * 1e14) % 256 for j in sine_tent_cosine_map_x1]
    X2 = [floor(j * 3 * m * n) for j in sine_tent_cosine_map_x2]

    N01 = 1000
    chebyshev_tent_map_x0 = chebyshev_tent_map(u0, x0, beta0, N01 + m * 8 * 3)[N01:]
    chebyshev_tent_map_x1 = chebyshev_tent_map(u1, x1, beta1, N01 + n * 8 * 3)[N01:]
    chebyshev_tent_map_x2 = chebyshev_tent_map(u2, x2, beta2, N01 + m * n)[N01:]

    hang_random = [floor(j * 1e14) % 256 for j in chebyshev_tent_map_x0]
    hang_random_split = np.array_split(hang_random, 24)
    lie_random = [floor(j * 1e14) % 256 for j in chebyshev_tent_map_x1]
    lie_random_split = np.array_split(lie_random, 24)

    X01 = [floor(j * 1e14) % 256 for j in chebyshev_tent_map_x2]

    X01_reshape1 = np.mat(X01, dtype=np.uint8).reshape(m, n)

    X01_bitplanes1 = img_bit_decomposition(X01_reshape1)

    end_time1 = time.time()
    print("产生混沌序列运行时间:", end_time1 - start_time1, "秒")

    # -----------------------------------时间一样

    # 记录开始时间
    start_time2 = time.time()

    for i in range(3):
        for j in range(8):
            for z in range(n):
                seeded_unshuffle(decomposed_bit_planes_color[i][j][:][z], lie_random_split[(i+1)*j])
            for k in range(m):
                seeded_unshuffle(decomposed_bit_planes_color[i][j][k], hang_random_split[(i+1)*j])

    # 记录结束时间
    end_time2 = time.time()
    print("逆FY置乱22222运行时间:", end_time2 - start_time2, "秒")

    # 记录开始时间
    start_time3 = time.time()
    for i in range(3):
        for j in range(4, 8):
            # 解密步骤
            decomposed_bit_planes_color[i][j] = np.bitwise_xor(decomposed_bit_planes_color[i][j],
                                                               X01_bitplanes1[j, :, :])

    end_time3 = time.time()
    print("逆比特级xor11111运行时间:", end_time3 - start_time3, "秒")

    # 从位平面重构彩色图像
    reconstructed_img_color = reconstruct_image_color(decomposed_bit_planes_color)
    # cv2.imwrite(encrypt_img_path, reconstructed_img_color)  # 保存图像

    # 记录开始时间
    start_time4 = time.time()

    B, G, R = cv2.split(reconstructed_img_color)

    B1 = B.ravel()
    G1 = G.ravel()
    R1 = R.ravel()

    all1 = np.concatenate((R1, G1, B1))

    seeded_unshuffle(all1, X2)

    end_time4 = time.time()
    print("逆像素级FY置乱运行时间:", end_time4 - start_time4, "秒")

    # 记录开始时间
    start_time5 = time.time()

    all1 = all1.astype(np.int64)  # 转换为 int32，以避免溢出问题

    for j in reversed(range(3)):
        # 逆向遍历进行异或和减法操作
        for i in range(3 * m * n - 1, 0, -1):
            all1[i] = all1[i] ^ X0[i]
            all1[i] = (all1[i] - all1[i - 1]) % 256

        # 处理 i == 0 的情况
        all1[0] = all1[0] ^ X0[0]
        all1[0] = (all1[0] - all1[-1]) % 256

    all1 = all1.astype(np.uint8)  # 转换回 uint8，确保适用于 OpenCV

    # # 转换为 int16，以避免溢出问题
    # all1 = all1.astype(np.int16)
    #
    # # 使用 NumPy 的向量化操作来代替循环
    # for j in range(3):
    #     # 处理所有元素（从索引 1 到末尾）
    #     all1[1:] = (all1[1:] ^ X0[1:]) - all1[:-1]
    #     all1[1:] %= 256
    #
    #     # 单独处理第一个元素
    #     all1[0] = (all1[0] ^ X0[0]) - all1[-1]
    #     all1[0] %= 256
    #
    # # 转换回 uint8，确保适用于 OpenCV
    # all1 = all1.astype(np.uint8)

    end_time5 = time.time()
    print("逆像素级扩散运行时间:", end_time5 - start_time5, "秒")

    all_decrypted_split = np.array_split(all1, 3)

    B2 = all_decrypted_split[2].reshape((m, n))
    G2 = all_decrypted_split[1].reshape((m, n))
    R2 = all_decrypted_split[0].reshape((m, n))

    E = cv2.merge([B2, G2, R2])
    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    print("运行时间:", end_time - start_time, "秒")
    cv2.imwrite(encrypt_img_path, E)  # 保存图像

    cv2.imshow("E", E)
    cv2.waitKey(0)


def main():

    # 执行函数
    jiemi(img_path)
if __name__ == '__main__':
    main()