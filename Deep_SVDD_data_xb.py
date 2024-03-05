import pandas as pd
import librosa
import numpy as np
import os
import tensorflow as tf
import pywt
from sklearn.preprocessing import MinMaxScaler
"""
音频数据的预处理
小波变换
Qingxin_Lu Jiangnan University
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# sess1 = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess1 = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))  # 启用设备的GPU进行训练

def reshape_wave(waveform, target_samples):
    if waveform.ndim == 1:
        waveform = np.reshape(waveform, (1, len(waveform)))
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, 0:target_samples]
    elif waveform.shape[1] < target_samples:
        num_pad = target_samples - waveform.shape[1]  # 补0
        waveform = np.pad(waveform, ((0, 0), (0, num_pad)), mode='constant')
    return waveform


def resample_wave(waveform, sr, target_sr):
    if sr != target_sr:
        waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return waveform


def rechannel_wave(waveform):
    if waveform.shape[0] > 1:
        # waveform = tf.reduce_mean(waveform, axis=0, keepdims=True)
        waveform = np.mean(waveform, axis=0, dtype=float, keepdims=True)
    return waveform


def audio_data_to_net_xb(excel_dir, target_samples, target_sr, root_dir, hop_size):
    excel = pd.read_csv(excel_dir)  # 读入csv文件
    N = len(excel)  # data number
    X = []
    label = []
    for i in range(N):
        # audio_name = excel.iloc[i, 5]  # 获取音频名称
        # audio_path = os.path.join(root_dir, audio_name)  # 获取音频路径
        # waveform, sr = librosa.load(audio_path, mono=False)  # 读入该音频文件
        # fold = f"fold{excel.iloc[i, 5]}"  # index行，5列，获取fold文件夹序号 urbansound8k用
        audio_name = f"{excel.iloc[i, 0]}"  # 获取文件该音频文件名
        target = excel.iloc[i, 1]  # 获取该音频文件标签 urbansound8k为[i, 6]
        # target = excel.iloc[i, 6]
        # audio_path = os.path.join(root_dir, fold)  # 获取该文件所在的fold文件夹路径 urbansound8k用
        audio_path = root_dir
        # audio_file = os.listdir(audio_path)  # 获取fold文件夹内所有文件
        wave_path = os.path.join(audio_path, audio_name)  # 获取该文件绝对路径
        waveform, sr = librosa.load(wave_path, mono=False)  # 读入该音频文件
        # 归一化
        # waveform = waveform / np.max(np.abs(waveform))

        # 数据增强
        length = len(waveform)
        n = (length - 30000) // hop_size + 1
        for j in range(n):
            waveform_i = waveform[hop_size*j:hop_size*j+30000]
            # waveform_i = waveform

            waveform_i = reshape_wave(waveform_i, target_samples)  # 长度统一
            waveform_i = resample_wave(waveform_i, sr, target_sr)  # 采样率统一
            waveform_i = rechannel_wave(waveform_i)  # 通道数统一
            pre_emphasis = 0.97
            waveform_i = np.append(waveform_i, waveform_i[1:] - pre_emphasis * waveform_i[:-1], axis=0)  # 预加重

            """
            信号变换
            """
            wavelet_name = 'db4'  # 小波基函数
            level = 4  # 变换级数
            # coeffs = pywt.wavedec(waveform, wavelet=wavelet_name, level=level)  # 小波系数
            # 获取每个尺度的小波系数
            """
            approximation = coeffs[0]  # 近似系数
            details = coeffs[1:]  # 尺度从最低到最高的细节系数
            """
            wp = pywt.WaveletPacket(data=waveform_i, wavelet=wavelet_name, mode='symmetric', maxlevel=level)  # 小波包分解
            coefficients = [node.data for node in wp.get_level(level, 'natural')]
            coefficients_matrix = np.vstack(coefficients)  # 构建小波包系数矩阵
            # coefficients_matrix = np.expand_dims(coefficients_matrix, axis=-1)  # 一维使用

            delta = librosa.feature.delta(coefficients_matrix, order=1)  #delta
            delta_deltas = librosa.feature.delta(coefficients_matrix, order=2)  #delta-deltas
            coeffs_3 = np.concatenate((coefficients_matrix[:, :, np.newaxis], delta[:, :, np.newaxis],
                                       delta_deltas[:, :, np.newaxis]), axis=-1) #
            label.append(target)
            X.append(coeffs_3)
            print(i*n + j)
            # print(i)
    X = np.array(X)
    label = np.array(label)
    n_y = len(label)
    label = np.reshape(label, (n_y, 1))
    return X, label