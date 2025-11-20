import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

def F(x, L):
    """定义混沌映射函数。假设为逻辑映射函数。"""
    return L * x * (1 - x)


img_path_base = "./images/"
img_name = "lena.png"  # 自行修改图片名，默认为 lena 图[
img_name1 = "lena_encrypt_diff.png"
img_path = img_path_base + img_name

# 读取图像并转换为浮点数
X = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(15, 10))
plt.subplot(221)
plt.imshow(X, cmap='gray')
plt.title("原图")
S = X.shape[1]

# 密钥生成
x2 = 0.4
L2 = 0.3
x2 = (x2 + np.sum(X) / 256) % 1

for i in range(300):
    x2 = F(x2, L2)

y3 = np.zeros(S * S)
y4 = np.zeros(S * S)
y3[0] = x2
for i in range(1, S * S):
    y3[i] = F(y3[i - 1], L2)
y4[0] = y3[-1]
for i in range(1, S * S):
    y4[i] = F(y4[i - 1], L2)

# 混沌序列标准化
z3 = np.floor((y3 * 10**14) % 256).astype(int)
z4 = np.floor((y4 * 10**14) % 256).astype(int)

# 图像加密
H = np.reshape(np.flipud(np.rot90(X)), (1, S**2))

H1 = np.zeros(S * S, dtype=int)
H1[0] = H[0, 0] ^ np.sum(H) ^ z3[0]
for i in range(1, S * S):
    H1[i] = H[0, i] ^ H1[i - 1] ^ z3[i]

H1 = np.reshape(H1, (S, S))
H1 = np.flipud(np.fliplr(H1))
V = np.reshape(H1, (1, S**2))
E = np.zeros(S * S, dtype=int)
E[0] = V[0] ^ np.sum(V) ^ z4[0]
for i in range(1, S * S):
    E[i] = E[i - 1] ^ V[i] ^ z4[i]

e = np.reshape(E, (S, S))
plt.subplot(222)
plt.imshow(e, cmap='gray')
plt.title("像素级环形扩散后")

# # 图像解密
# V1 = np.zeros(S * S, dtype=int)
# V1[0] = E[0] ^ z4[0] ^ np.sum(V)
# for i in range(1, S * S):
#     V1[i] = E[i] ^ z4[i] ^ E[i - 1]
#
# h22 = np.reshape(V1, (S, S))
# h11 = np.reshape(np.fliplr(np.flipud(h22)), (1, S**2))
#
# H2 = np.zeros(S * S, dtype=int)
# H2[0] = z3[0] ^ h11[0] ^ np.sum(H)
# for i in range(1, S * S):
#     H2[i] = z3[i] ^ h11[i] ^ h11[i - 1]
#
# H2 = np.reshape(H2, (S, S))
# X1 = np.rot90(np.flipud(H2), 3)
#
# plt.subplot(223)
# plt.imshow(X1, cmap='gray')
# plt.title("还原图")
# plt.show()
