import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('./images/img.png')

# 转换为RGB格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 构建高斯金字塔
def gaussian_pyramid(image, levels=6):
    pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)  # 高斯降采样
        pyramid.append(image)
    return pyramid

# 显示金字塔
def show_pyramid(pyramid):
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(pyramid):
        plt.subplot(1, len(pyramid), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Level {i}')
    plt.show()

# 创建金字塔
pyramid = gaussian_pyramid(image)

# 显示金字塔
show_pyramid(pyramid)
