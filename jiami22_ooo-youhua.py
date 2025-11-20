import time
from math import floor

import cv2
import numpy as np

from utils import img_bit_decomposition, sine_tent_cosine_map, chebyshev_tent_map, seeded_shuffle

img_path_base = "./img/"
img_name = "Lena.png"  # 自行修改图片名，默认为 lena 图
img_path = img_path_base + img_name

filename, ext = img_path.rsplit('.', 1)
encrypt_img_path = f"{filename}_23334567.png"

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
    return np.sum([(bit_planes[i] * (2 ** i)).astype(np.uint8) for i in range(8)], axis=0)


def img_bit_decomposition_color(img):
    return [img_bit_decomposition(img[:, :, channel]) for channel in range(3)]


def reconstruct_image_color(bit_planes_color):
    channels = [reconstruct_image(bit_planes) for bit_planes in bit_planes_color]
    return cv2.merge([channel.astype(np.uint8) for channel in channels])


def jiami(img):
    example_img_color = cv2.imread(img, cv2.IMREAD_COLOR)
    m, n, _ = example_img_color.shape

    start_time = time.time()
    start_time1 = time.time()

    # Generate chaotic sequences
    sine_tent_cosine_map_x0 = sine_tent_cosine_map(x01, r01, 1000 + 3 * m * n)[1000:]
    sine_tent_cosine_map_x2 = sine_tent_cosine_map(x21, r21, 1000 + 3 * m * n)[1000:]
    X0 = [floor(j * 1e14) % 256 for j in sine_tent_cosine_map_x0]
    X2 = [floor(j * 3 * m * n) for j in sine_tent_cosine_map_x2]

    # print(X2)

    chebyshev_tent_map_x0 = chebyshev_tent_map(u0, x0, beta0, 1000 + m * 8 * 3)[1000:]
    chebyshev_tent_map_x1 = chebyshev_tent_map(u1, x1, beta1, 1000 + n * 8 * 3)[1000:]
    chebyshev_tent_map_x2 = chebyshev_tent_map(u2, x2, beta2, 1000 + m * n)[1000:]

    hang_random_split = np.array_split([floor(j * 1e14) % 256 for j in chebyshev_tent_map_x0], 24)
    lie_random_split = np.array_split([floor(j * 1e14) % 256 for j in chebyshev_tent_map_x1], 24)

    X01 = [floor(j * 1e14) % 256 for j in chebyshev_tent_map_x2]
    X01_reshape1 = np.mat(X01, dtype=np.uint8).reshape(m, n)
    X01_bitplanes1 = img_bit_decomposition(X01_reshape1)

    end_time1 = time.time()
    print("Chaotic sequence generation time:", end_time1 - start_time1, "seconds")

    # Diffusion
    start_time2 = time.time()
    B, G, R = cv2.split(example_img_color)
    all1 = np.concatenate((R.ravel(), G.ravel(), B.ravel())).astype(np.int32)

    # for i in range(3 * m * n):
    #     all1[i] = ((all1[i] + (all1[i - 1] if i > 0 else all1[-1])) % 256) ^ X0[i]

    for j in range(3):
        for i in range(3 * m * n):
            if i == 0:
                all1[i] = ((all1[i] + all1[-1]) % 256) ^ X0[i]
            else:
                all1[i] = ((all1[i] + all1[i - 1])% 256) ^ X0[i]

    all1 = all1.astype(np.uint8)
    end_time2 = time.time()
    print("Diffusion time:", end_time2 - start_time2, "seconds")

    # Shuffling
    start_time3 = time.time()
    seeded_shuffle(all1, X2)
    end_time3 = time.time()
    print("Shuffle time:", end_time3 - start_time3, "seconds")

    # Reconstruct shuffled image
    start_time4 = time.time()
    B2, G2, R2 = np.array_split(all1, 3)
    E = cv2.merge([B2.reshape((m, n)), G2.reshape((m, n)), R2.reshape((m, n))])
    end_time4 = time.time()
    print("Reconstruction time (shuffled image):", end_time4 - start_time4, "seconds")

    # Bit-plane decomposition and modification
    start_time5 = time.time()
    decomposed_bit_planes_color = img_bit_decomposition_color(E)
    for i in range(3):
        for j in range(4, 8):
            decomposed_bit_planes_color[i][j] = np.bitwise_xor(decomposed_bit_planes_color[i][j],
                                                               X01_bitplanes1[j, :, :])
    end_time5 = time.time()
    print("Bit-plane modification time:", end_time5 - start_time5, "seconds")

    # Shuffle bit-planes
    start_time6 = time.time()
    for i in range(3):
        for j in range(8):
            for k in range(m):
                seeded_shuffle(decomposed_bit_planes_color[i][j][k], hang_random_split[(i + 1) * j])
            for z in range(n):
                seeded_shuffle(decomposed_bit_planes_color[i][j][:, z], lie_random_split[(i + 1) * j])
    end_time6 = time.time()
    print("Bit-plane shuffle time:", end_time6 - start_time6, "seconds")

    # Reconstruct encrypted image
    start_time7 = time.time()
    reconstructed_img_color = reconstruct_image_color(decomposed_bit_planes_color)
    end_time7 = time.time()
    print("Reconstruction time (encrypted image):", end_time7 - start_time7, "seconds")

    # Save encrypted image
    cv2.imwrite(encrypt_img_path, reconstructed_img_color)
    print("Total runtime:", time.time() - start_time, "seconds")

    return reconstructed_img_color


def main():
    jiami(img_path)


if __name__ == '__main__':
    main()