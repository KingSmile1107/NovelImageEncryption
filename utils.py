import cv2
from numpy import uint8, zeros, sin, pi, cos
import numpy as np


def test_img_bit_decomposition(img):
    '''图像的位平面分解'''
    m, n = img.shape  # 获取图像的尺寸
    r = np.zeros((8, m, n), dtype=np.uint8)  # 初始化结果矩阵，8个位平面
    for i in range(8):
        # 对于每一位，使用给定公式计算对应的位平面
        r[i, :, :] = (np.floor_divide(img, 2**i)) % 2
    return r


def seeded_shuffle(items, chaosList):
    """
    使用指定的种子来洗牌。
    """
    n = 0
    for i in range(len(items) - 1, -1, -1):
        j = chaosList[n]
        items[i], items[j] = items[j], items[i]
        n = n + 1

# def seeded_shuffle(items, chaosList):
#     """
#     使用指定的种子进行高效洗牌，使用 NumPy 进行批量处理。
#     """
#     items_np = np.array(items)
#     chaosList_np = np.array(chaosList)
#
#     # 初始化一个等长的索引数组
#     indices = np.arange(len(items))
#
#     # 按照 chaosList 中指定的顺序交换位置
#     indices = indices[chaosList_np]
#
#     # 根据重新排列的索引数组对items进行重新排序
#     shuffled_items = items_np[indices]
#     return shuffled_items.tolist(), indices
#
#
# def seeded_unshuffle(shuffled_items, shuffle_indices):
#     """
#     使用已知的洗牌索引来逆向恢复原始顺序，使用 NumPy 进行批量处理。
#     """
#     shuffled_items_np = np.array(shuffled_items)
#
#     # 使用 shuffle_indices 生成逆向索引
#     unshuffle_indices = np.argsort(shuffle_indices)
#
#     # 通过逆向索引直接对元素进行排序，恢复原始顺序
#     unshuffled_items = shuffled_items_np[unshuffle_indices]
#     return unshuffled_items.tolist()




def seeded_unshuffle(items, chaosList):
    """
    使用指定的种子来逆向洗牌，恢复原始顺序（优化版）。
    """
    # 直接按 chaosList 值的逆序来执行交换
    n = len(items) - 1
    for i in range(len(items)):
        j = chaosList[n - i]
        items[i], items[j] = items[j], items[i]



def img_bit_decomposition(img):
    '''图像的位平面分解'''
    m, n = img.shape
    r = zeros((8, m, n), dtype=uint8)
    for i in range(8):
        r[i, :, :] = cv2.bitwise_and(img, 2**i)
        mask = r[i, :, :] > 0
        r[i, mask] = 1
    return r


def logistic_map(x, r=3.9999):
    '''Logistic映射函数
    x: 当前值
    r: 控制参数
    '''
    return r * x * (1 - x)

def logistic_map(x, r=3.9999, sequence_length=10):
    '''Logistic映射函数
    x: 当前值
    r: 控制参数
    sequence_length: 生成的序列长度
    '''
    sequence = [x]
    for _ in range(sequence_length - 1):
        x = r * x * (1 - x)
        sequence.append(x)
    return sequence


def sine_tent_cosine_map(r, x0, sequence_length):
    """
    实现Sine-Tent-Cosine (STC)混沌映射。

    :param r: 参数r，取值范围为[0, 1]
    :param x0: 初始值x，取值范围为[0, 1]
    :param sequence_length: 迭代次数
    :return: 包含混沌序列的numpy数组
    """
    sequence = np.zeros(sequence_length)  # 初始化序列数组
    x = x0  # 设置初始值x

    for i in range(sequence_length):  # 进行迭代
        if x < 0.5:
            # 对于x < 0.5的情况，按照STC映射公式计算
            x = np.cos(np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * x - 0.5))
        else:
            # 对于x >= 0.5的情况，按照STC映射公式计算
            x = np.cos(np.pi * (r * np.sin(np.pi * x) + 2 * (1 - r) * (1 - x) - 0.5))
        sequence[i] = x  # 将计算结果存入序列数组

    return sequence

# def chebyshev_tent_map(mu, x0, sequence_length):
#     """
#     实现一个自定义的混沌映射。
#
#     :param mu: 控制参数mu，取值范围为[0, 4]
#     :param x0: 初始值x，取值范围为[0, 1]
#     :param sequence_length: 迭代次数
#     :return: 包含混沌序列的numpy数组
#     """
#     sequence = np.zeros(sequence_length)  # 初始化序列数组
#     x_i = x0  # 设置初始值x
#
#     for n in range(sequence_length):  # 进行迭代
#         if x_i < 0.5:
#             # 对于x_i < 0.5的情况，按照给定的混沌映射公式计算
#             x_i = (np.cos(n * np.arccos(x_i)) + (4 - mu) * (x_i / 2)) % 1
#         else:
#             # 对于x_i >= 0.5的情况，按照给定的混沌映射公式计算
#             x_i = (np.cos(n * np.arccos(x_i)) + (4 - mu) * (1 - x_i) * 2) % 1
#         sequence[n] = x_i  # 将计算结果存入序列数组
#
#     return sequence

def chebyshev_tent_map(mu, x0, beta, sequence_length):
    """
    实现一个自定义的混沌映射。

    :param mu: 控制参数mu，取值范围为[0, 4]
    :param x0: 初始值x，取值范围为[0, 1]
    :param sequence_length: 迭代次数
    :return: 包含混沌序列的numpy数组
    """
    sequence = np.zeros(sequence_length)  # 初始化序列数组
    x = x0  # 设置初始值x

    for n in range(sequence_length):  # 进行迭代
        if x < 0.5:
            # 对于x_i < 0.5的情况，按照给定的混沌映射公式计算
            x = (mu * np.cos(np.pi*beta) * np.sin(x) + np.cos(n * np.arccos(x)) + ((4 - mu) * x)) % 1
        else:
            # 对于x_i >= 0.5的情况，按照给定的混沌映射公式计算
            x = (mu * np.cos(np.pi*beta) * np.sin(x) + np.cos(n * np.arccos(x)) + ((4 - mu) * (1 - x))) % 1
        sequence[n] = x  # 将计算结果存入序列数组

    return sequence
