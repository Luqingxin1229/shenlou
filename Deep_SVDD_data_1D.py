import pandas as pd
import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import librosa.display
import cv2
from tensorflow.keras import layers, models

"""
音频数据的预处理
STFT
Qingxin_Lu Jiangnan University
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


# sess1 = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
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


def audio_data_to_net_1D(excel_dir, target_samples, target_sr, root_dir, hop_size):
    excel = pd.read_csv(excel_dir)  # 读入csv文件
    N = len(excel)  # data number
    X = []
    label = []
    for i in range(N):
        # fold = f"fold{excel.iloc[i, 5]}"  # index行，5列，获取fold文件夹序号 urbansound8k用
        audio_name = f"{excel.iloc[i, 0]}"  # 获取文件该音频文件名
        target = excel.iloc[i, 1]  # 获取该音频文件标签 urbansound8k为[i, 6]
        # target = excel.iloc[i, 6]
        # audio_path = os.path.join(root_dir, fold)  # 获取该文件所在的fold文件夹路径 urbansound8k用
        # audio_path = root_dir
        # audio_file = os.listdir(audio_path)  # 获取fold文件夹内所有文件
        wave_path = os.path.join(root_dir, audio_name)  # 获取该文件绝对路径
        waveform, sr = librosa.load(wave_path, mono=False, dtype=np.float32)  # 读入该音频文件

        # 数据增强
        length = len(waveform)
        n = (length - 10000) // hop_size + 1
        for j in range(n):
            waveform_i = waveform[hop_size * j:hop_size * j + 10000 + 1]
            waveform_i = waveform
            waveform_i = reshape_wave(waveform_i, target_samples)  # 长度统一
            waveform_i = resample_wave(waveform_i, sr, target_sr)  # 采样率统一
            waveform_i = rechannel_wave(waveform_i)  # 通道数统一
            pre_emphasis = 0.97
            waveform_i = np.append(waveform_i, waveform_i[1:] - pre_emphasis * waveform_i[:-1], axis=0)  # 预加重

            waveform_i = np.expand_dims(waveform_i, axis=-1)
            label.append(target)
            X.append(waveform_i)
            print(i * n + j)
            # print(i)
    X = np.array(X)
    label = np.array(label)
    n_y = len(label)
    label = np.reshape(label, (n_y, 1))
    return X, label

def model_1D(inputshape, H):
    inputs = layers.Input(shape=inputshape)

    # 3*3

    branch1 = layers.Conv2D(32, (1, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    branch1 = layers.MaxPooling2D((1, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)

    branch1 = layers.Conv2D(64, (1, 3), strides=(1, 1), activation='relu', padding='same')(branch1)
    branch1 = layers.MaxPooling2D((1, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)

    # 5*5
    branch2 = layers.Conv2D(32, (1, 5), strides=(1, 1), activation='relu', padding='same')(inputs)
    branch2 = layers.MaxPooling2D((1, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)

    branch2 = layers.Conv2D(64, (1, 5), strides=(1, 1), activation='relu', padding='same')(branch2)
    branch2 = layers.MaxPooling2D((1, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)
    # 7*7
    branch3 = layers.Conv2D(32, (1, 7), strides=(1, 1), activation='relu', padding='same')(inputs)
    branch3 = layers.MaxPooling2D((1, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)

    branch3 = layers.Conv2D(64, (1, 7), strides=(1, 1), activation='relu', padding='same')(branch3)
    branch3 = layers.MaxPooling2D((1, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)

    """
    merged = layers.Concatenate()([branch1, branch2, branch3])
    x1 = layers.Flatten()(merged)
    x2 = layers.Dense(4096, activation='relu')(x1)
    x3 = layers.Dropout(0.5)(x2)
    x3 = layers.Dense(2048, activation='relu')(x3)
    x = layers.Dropout(0.5)(x3)
    """

    x1 = layers.GlobalAveragePooling2D()(branch1)
    x2 = layers.GlobalAveragePooling2D()(branch2)
    x3 = layers.GlobalAveragePooling2D()(branch3)
    merged = layers.Concatenate(axis=-1)([x1, x2, x3])  # [:,np.newaxis]

    # x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merged)
    # x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.BatchNormalization(epsilon=1e-4)(x)
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.5)(x)
    # x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(H)(x)  # kernel_initializer='random_uniform'
    # outputs = layers.Dense(32)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model
# model = model_1D((1, 30000, 1), H=64)