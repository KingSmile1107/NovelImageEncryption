import time
from math import floor

import cv2
import numpy as np

from utils import img_bit_decomposition, sine_tent_cosine_map, logistic_map, chebyshev_tent_map, seeded_unshuffle, seeded_shuffle

img_path_base = "./img/"
img_name = "Lena_23334567.png"
img_path = img_path_base + img_name

filename, ext = img_path.rsplit('.', 1)
encrypt_img_path = f"{filename}_save.png"

x01, r01 = 0.2, 0.3
x11, r11 = 0.4, 0.5
x0, u0, beta0 = 0.1, 2.2, 0.5
x1, u1, beta1 = 0.7, 3.3, 0.4

x21 = (x01 + x11) % 1
r21 = (r01 + r11) % 1
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
    return cv2.merge([channel.astype(np.uint8) for channel in channels])

def jiemi(img):
    example_img_color = cv2.imread(img, cv2.IMREAD_COLOR)
    m, n, _ = example_img_color.shape

    start_time = time.time()
    start_time1 = time.time()

    # 位平面分解
    decomposed_bit_planes_color = img_bit_decomposition_color(example_img_color)

    # 生成混沌序列
    sine_tent_cosine_map_x0 = sine_tent_cosine_map(x01, r01, 1000 + 3 * m * n)[1000:]
    sine_tent_cosine_map_x2 = sine_tent_cosine_map(x21, r21, 1000 + 3 * m * n)[1000:]
    X0 = [floor(j * 1e14) % 256 for j in sine_tent_cosine_map_x0]
    X2 = [floor(j * 3 * m * n) for j in sine_tent_cosine_map_x2]

    chebyshev_tent_map_x0 = chebyshev_tent_map(u0, x0, beta0, 1000 + m * 8 * 3)[1000:]
    chebyshev_tent_map_x1 = chebyshev_tent_map(u1, x1, beta1, 1000 + n * 8 * 3)[1000:]
    chebyshev_tent_map_x2 = chebyshev_tent_map(u2, x2, beta2, 1000 + m * n)[1000:]

    hang_random_split = np.array_split([floor(j * 1e14) % 256 for j in chebyshev_tent_map_x0], 24)
    lie_random_split = np.array_split([floor(j * 1e14) % 256 for j in chebyshev_tent_map_x1], 24)

    X01 = [floor(j * 1e14) % 256 for j in chebyshev_tent_map_x2]
    X01_reshape1 = np.mat(X01, dtype=np.uint8).reshape(m, n)
    X01_bitplanes1 = img_bit_decomposition(X01_reshape1)

    end_time1 = time.time()
    print("混沌序列生成时间:", end_time1 - start_time1, "秒")

    # 逆位平面置乱
    start_time2 = time.time()
    for i in range(3):
        for j in range(8):
            for z in range(n):
                seeded_unshuffle(decomposed_bit_planes_color[i][j][:, z], lie_random_split[(i + 1) * j])
            for k in range(m):
                seeded_unshuffle(decomposed_bit_planes_color[i][j][k], hang_random_split[(i + 1) * j])
    end_time2 = time.time()
    print("逆位平面置乱运行时间:", end_time2 - start_time2, "秒")

    # 逆比特级xor
    start_time3 = time.time()
    for i in range(3):
        for j in range(4, 8):
            decomposed_bit_planes_color[i][j] = np.bitwise_xor(decomposed_bit_planes_color[i][j], X01_bitplanes1[j, :, :])
    end_time3 = time.time()
    print("逆比特级xor运行时间:", end_time3 - start_time3, "秒")

    # 重构图像
    reconstructed_img_color = reconstruct_image_color(decomposed_bit_planes_color)

    # 逆像素级置乱
    start_time4 = time.time()
    B, G, R = cv2.split(reconstructed_img_color)
    all1 = np.concatenate((R.ravel(), G.ravel(), B.ravel()))
    seeded_unshuffle(all1, X2)
    end_time4 = time.time()
    print("逆像素级FY置乱运行时间:", end_time4 - start_time4, "秒")

    # 逆像素级扩散
    start_time5 = time.time()
    all1 = all1.astype(np.int32)
    # end_index = 3 * m * n - 1
    # for i in reversed(range(1, end_index + 1)):
    #     all1[i] = (all1[i] ^ X0[i]) - all1[i - 1]
    #     all1[i] = all1[i] % 256
    # all1[0] = (all1[0] ^ X0[0]) - all1[-1]
    # all1[0] = all1[0] % 256
    for j in reversed(range(3)):
        # all1 = np.roll(all1, -m * n)  # 注意滚动的方向和步数
        for i in reversed(range(1, 3 * m * n)):
            # 逆向异或操作
            all1[i] = all1[i] ^ X0[(i) % len(X0)]
            # 逆向加法操作，恢复原始像素值
            if i > 0:  # 避免索引越界
                all1[i] = (((all1[i]) - (all1[i - 1])) % 256)
    all1 = all1.astype(np.uint8)
    end_time5 = time.time()
    print("逆像素级扩散运行时间:", end_time5 - start_time5, "秒")

    # 重构解密图像
    all_decrypted_split = np.array_split(all1, 3)
    B2 = all_decrypted_split[2].reshape((m, n))
    G2 = all_decrypted_split[1].reshape((m, n))
    R2 = all_decrypted_split[0].reshape((m, n))
    E = cv2.merge([B2, G2, R2])

    # 保存并显示解密图像
    end_time = time.time()
    print("运行时间:", end_time - start_time, "秒")
    cv2.imwrite(encrypt_img_path, E)
    cv2.imshow("E", E)
    cv2.waitKey(0)

def main():
    jiemi(img_path)

if __name__ == '__main__':
    main()