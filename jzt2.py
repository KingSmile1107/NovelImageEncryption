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

# 高斯金字塔合并（恢复图像）
def reconstruct_gaussian_pyramid(pyramid):
    # 从最后一层开始，逐步上采样并合并
    reconstructed_image = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        reconstructed_image = cv2.pyrUp(reconstructed_image)  # 上采样
        reconstructed_image = cv2.add(reconstructed_image, pyramid[i])  # 合并
    return reconstructed_image

# 构建拉普拉斯金字塔
def laplacian_pyramid(image, levels=6):
    gaussian_pyr = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)  # 高斯降采样
        gaussian_pyr.append(image)

    laplacian_pyr = []
    for i in range(levels, 0, -1):
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i])  # 上采样
        laplacian = cv2.subtract(gaussian_pyr[i-1], gaussian_expanded)  # 拉普拉斯层
        laplacian_pyr.append(laplacian)

    # 最后一层直接使用高斯金字塔中的最小层（没有细节）
    laplacian_pyr.append(gaussian_pyr[0])
    return laplacian_pyr

# 拉普拉斯金字塔合并（恢复图像）
def reconstruct_laplacian_pyramid(laplacian_pyr):
    reconstructed_image = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        reconstructed_image = cv2.pyrUp(reconstructed_image)  # 上采样
        reconstructed_image = cv2.add(reconstructed_image, laplacian_pyr[i])  # 合并
    return reconstructed_image

# 显示金字塔
def show_pyramid(pyramid, title='Pyramid'):
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(pyramid):
        plt.subplot(1, len(pyramid), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Level {i}')
    plt.suptitle(title)
    plt.show()

# 生成高斯金字塔
gaussian_pyr = gaussian_pyramid(image)

# 显示高斯金字塔
show_pyramid(gaussian_pyr, title='Gaussian Pyramid')

# 从高斯金字塔恢复原图
reconstructed_image_gaussian = reconstruct_gaussian_pyramid(gaussian_pyr)

# 显示恢复的图像
plt.figure(figsize=(8, 8))
plt.imshow(reconstructed_image_gaussian)
plt.title("Reconstructed Image from Gaussian Pyramid")
plt.axis('off')
plt.show()

# 生成拉普拉斯金字塔
laplacian_pyr = laplacian_pyramid(image)

# 显示拉普拉斯金字塔
show_pyramid(laplacian_pyr, title='Laplacian Pyramid')

# 从拉普拉斯金字塔恢复原图
reconstructed_image_laplacian = reconstruct_laplacian_pyramid(laplacian_pyr)

# 显示恢复的图像
plt.figure(figsize=(8, 8))
plt.imshow(reconstructed_image_laplacian)
plt.title("Reconstructed Image from Laplacian Pyramid")
plt.axis('off')
plt.show()
