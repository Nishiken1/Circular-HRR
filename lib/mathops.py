"""
Library functions to perform circular convolution operations.
"""

__author__ = "Ashwinkumar Ganesan, Sunil Gandhi, Hang Gao"
__email__ = "gashwin1@umbc.edu,sunilga1@umbc.edu,hanggao@umbc.edu"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Pytorch functions.
"""
def complex_multiplication(left, right):
    """
    Multiply two vectors in complex domain.
    """
    #left_real, left_complex = left[..., 0], left[..., 1]
    #right_real, right_complex = right[..., 0], right[..., 1]
    left_real, left_complex = left.real, left.imag
    right_real, right_complex = right.real, right.imag

    output_real = left_real * right_real - left_complex * right_complex
    output_complex = left_real * right_complex + left_complex * right_real
    return torch.stack([output_real, output_complex], dim=-1)

def complex_division(left, right):
    """
    Divide two vectors in complex domain.
    """
    left_real, left_complex = left[..., 0], left[..., 1]
    right_real, right_complex = right[..., 0], right[..., 1]

    output_real = torch.div((left_real * right_real + left_complex * right_complex),(right_real**2 + right_complex**2))
    output_complex = torch.div((left_complex * right_real - left_real * right_complex ),(right_real**2 + right_complex**2))
    return torch.stack([output_real, output_complex], dim=-1)

def circular_conv(a, b):
    """ Defines the circular convolution operation
    a: tensor of shape (batch, D)
    b: tensor of shape (batch, D)
    """
    # left = torch.fft.rfft(a, 1, onesided=False)
    #left = torch.view_as_real(torch.fft.fft2(a))
    left = torch.fft.rfft(a)
    
    # right = torch.fft.rfft(b, 1, onesided=False)
    #right = torch.view_as_real(torch.fft.fft2(b))
    right = torch.fft.rfft(b)
    # print(left)
    # print(right)

    output = complex_multiplication(left, right)
    
    # print(output)
    output = torch.view_as_complex(output)
    #output = torch.view_as_complex(output)
    #print(output)
    output = torch.fft.irfft(output)
    # print(output)

    # output = torch.fft.irfft(torch.view_as_complex(output), n=a.shape[1], dim=1)
    
    # output = complex_multiplication(left, right)
    # output = torch.irfft(output, 1, signal_sizes=a.shape[-1:], onesided=False)
    return output

def modulo_pi_to_negpi(theta):
    result = torch.fmod(theta + np.pi, 2.0 * np.pi) - np.pi
    return result

def circular_binding(theta,phi):
  return modulo_pi_to_negpi(theta+phi)

# def circular_circular_conv(a, b):
#     """ Defines the circular convolution operation
#     a: tensor of shape (batch, D)
#     b: tensor of shape (batch, D)
#     """
#     output = circular_binding(a, b)
#     return output


def circular_circular_conv(a, b):
    result = a + b
    result = (result + np.pi) % (2 * np.pi) - np.pi
    return result

def get_appx_inv(a):
    """
    Compute approximate inverse of vector a.
    """
    return torch.roll(torch.flip(a, dims=[-1]), 1,-1)

def get_inv(a, typ=torch.DoubleTensor):
    """
    Compute exact inverse of vector a.
    """
    left = torch.fft.rfft(a, 1, onesided=False)
    complex_1 = np.zeros(left.shape)
    complex_1[...,0] = 1
    op = complex_division(typ(complex_1),left)
    return torch.fft.irfft(op,1,onesided=False)

def complexMagProj(x):
    """
    Normalize a vector x in complex domain.
    """
    c = torch.view_as_real(torch.fft.fft2(x))
    c_ish=c/torch.norm(c, dim=-1,keepdim=True)
    output = torch.fft.irfft(torch.view_as_complex(c_ish), n=x.shape[1], dim=1)

    # c = torch.rfft(x, 1, onesided=False)
    # c_ish=c/torch.norm(c, dim=-1,keepdim=True)
    # output = torch.irfft(c_ish, 1, signal_sizes=x.shape[1:], onesided=False)
    return output

def normalize(x):
    return x/torch.norm(x)

"""
Numpy Functions.
"""
# Make them work with batch dimensions
def cc(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b))

def np_inv(a):
    return np.fft.irfft((1.0/np.fft.rfft(a)),n=a.shape[-1])

def np_appx_inv(a):
    #Faster implementation
    return np.roll(np.flip(a, axis=-1), 1,-1)

def npcomplexMagProj(x):
    """
    Normalize a vector x in complex domain.
    """
    c = np.fft.rfft(x)

    # Look at real and image as if they were real
    c_ish = np.vstack([c.real, c.imag])

    # Normalize magnitude of each complex/real pair
    c_ish=c_ish/np.linalg.norm(c_ish, axis=0)
    c_proj = c_ish[0,:] + 1j * c_ish[1,:]
    return np.fft.irfft(c_proj,n=x.shape[-1])

def nrm(a):
    return a / np.linalg.norm(a)

# def dot(theta_array_1, theta_array_2):
#     d = theta_array_1.size(2) # 400の部分
#     cos_values = torch.cos(theta_array_1 - theta_array_2.unsqueeze(1))
#     return cos_values.sum(dim=2) / d

# def dot(theta_array_1, theta_array_2):
#     # print(theta_array_1.shape)
#     # print(theta_array_2.unsqueeze(1).shape)
#     non_zero_counts = []
#     for batch in theta_array_1:
#         count = 0
#         for vector in batch:
#             if not torch.all(vector == 0):
#                 count += 1
#         non_zero_counts.append(count)

#     d = theta_array_1.size(2)  # 400の部分
#     cos_values = torch.cos(theta_array_1 - theta_array_2.unsqueeze(1))

#     result = cos_values.sum(dim=2) / d
#     # print(cos_values.shape)
#     # print(cos_values.sum(dim=2).shape)
#     # exit()
#     # print("Number of non-zero vectors in each tensor:", non_zero_counts)
#     for i, count in enumerate(non_zero_counts):
#         if count < result.size(1):
#             result[i, count:] = 0

#     return result

def dot(theta_array_1, theta_array_2):
    d = theta_array_1.size(2)

    # ゼロでないベクトルの数をカウント
    non_zero_counts = (theta_array_1 != 0).any(dim=2).sum(dim=1)

    cos_values = torch.cos(theta_array_1 - theta_array_2.unsqueeze(1))
    result = cos_values.sum(dim=2) / d

    # ゼロでないベクトルの数を元に、結果を修正
    mask = torch.arange(result.size(1)).expand(result.size(0), -1).to(theta_array_1.device)
    mask = mask < non_zero_counts.unsqueeze(1)
    result *= mask.float()

    return result


def custom_dot(tensor_1, tensor_2):
    # tensor_1のサイズ: (64, 400)
    # tensor_2のサイズ: (400, 160)

    # 式がコサイン類似度に似ているので、ここではコサイン類似度の計算を使用します。
    # ただし、厳密にコサイン類似度と一致するわけではないため注意が必要です。

    # tensor_1とtensor_2の差分を計算
    #tensor_diff = tensor_1.unsqueeze(1) - tensor_2.unsqueeze(0)
    tensor_diff = tensor_2.unsqueeze(0) - tensor_1.unsqueeze(1)
    # cos計算
    dot_similarity = torch.sum(torch.cos(tensor_diff), dim=2) / tensor_1.size()[1]
    return dot_similarity


def sim(pos_classes,s_r):
    device = pos_classes.device 
    d = (pos_classes.size()[2])
    non_zero_counts = (pos_classes != 0).any(dim=2).sum(dim=1)
    pos = torch.stack((torch.cos(pos_classes), torch.sin(pos_classes)), dim=3)

    similarity = torch.sum(pos * s_r.unsqueeze(1), dim=-1)
    similarity = torch.sum(similarity, dim=-1) / d
    
    mask = torch.arange(similarity.size(1), device=device).unsqueeze(0) < non_zero_counts.unsqueeze(1).to(device)
    
    # マスクを使用して、特定の位置から0に設定
    similarity = similarity * mask.float()

    return similarity

def custom_sim(y,class_vec):
    batch_size = y.size()[0]
    d = y.size()[1]
    label_size = class_vec.size()[0]
    class_vec = torch.stack((torch.cos(class_vec), torch.sin(class_vec)), dim=2)
    y = torch.stack((torch.cos(y), torch.sin(y)), dim=2)
    y = y[:,:,[1,0]]

    # print(y[0].shape)
    # print(y[0][0])
    # print(class_vec[0][0])
    
    # inner_product = torch.zeros((64, 160, 400))

    # for batch in range(64):
    #     for label in range(160):
    #         for dim in range(400):
    #             inner_product[batch][label][dim] = torch.dot(y[batch][dim], class_vec[label][dim])
    

    # #inner_product = torch.sum(y * class_vec[:, None, :, :], dim=2)
    # print(inner_product[0][0])

    # # 各400次元ベクトルの内積の合計を計算し、400で割る
    # result = inner_product.sum(dim=2) / d

    # inner_product = torch.einsum('bik,ljk->bil', y, class_vec)

    # Summing along the last dimension and dividing by 400
    # result = inner_product.sum(dim=2) / y.size(1)
    # print(inner_product[0][0][0])
    result = torch.zeros((batch_size, label_size))
    
    for batch in range(batch_size):
        for label in range(label_size):
            result[batch, label] = torch.sum(y[batch] * class_vec[label]) / d
    # inner_product = torch.matmul(y, class_vec.permute(0, 2, 1))

    # 最後の次元での合計と平均を計算

    return result