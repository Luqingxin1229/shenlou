import pandas as pd
import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import librosa.display
import cv2
"""
音频数据的预处理
STFT
Qingxin_Lu Jiangnan University
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sess1 = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess1 = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))  # 启用设备的GPU进行训练

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


def audio_data_to_net_pro(excel_dir, target_samples, target_sr, root_dir, hop_size, n_mfcc):
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

        """        length = len(waveform)
                n = (length - 30000) // hop_size + 1
                for j in range(n):
                    waveform_i = waveform[hop_size*j:hop_size*j+30000+1]"""

        # 数据增强
        length = len(waveform)
        n = (length - target_samples) // hop_size + 1
        for j in range(n):

            waveform_i = waveform[hop_size*j:hop_size*j+target_samples]
            # waveform_i = waveform
            waveform_i = reshape_wave(waveform_i, target_samples)  # 长度统一
            waveform_i = resample_wave(waveform_i, sr, target_sr)  # 采样率统一
            waveform_i = rechannel_wave(waveform_i)  # 通道数统一
            pre_emphasis = 0.97
            waveform_i = np.append(waveform_i, waveform_i[1:] - pre_emphasis * waveform_i[:-1], axis=0)  # 预加重


            # waveform_i = scaler.fit_transform(waveform_i.reshape(-1, 1)).flatten()
            # waveform_i = np.array(waveform_i)
            # waveform_i = np.reshape(waveform_i, (1, len(waveform_i)))

            # mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=target_sr, n_fft=1024, hop_length=512)  # 获取梅尔谱图
            # mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # 转dB图
            # Reshape the MFCC matrix to (n_samples, n_features) format

            """mel_spec = librosa.feature.melspectrogram(y=waveform_i, sr=target_sr)
            # mel_spec = np.transpose(mel_spec, (1, 2, 0))
            delta = librosa.feature.delta(mel_spec, order=1)  # delta
            delta_deltas = librosa.feature.delta(mel_spec, order=2)  # delta-deltas
            mel_spectrogram_3 = np.concatenate((mel_spec, delta, delta_deltas))
            mel_spectrogram_3 = np.transpose(mel_spectrogram_3, (1, 2, 0))"""

            """
            # 归一化
            scaler = MinMaxScaler()
            MFCCs = np.reshape(MFCCs, (n_mfcc, 98))
            n_samples, n_features = MFCCs.shape[0], MFCCs.shape[1]
            mfcc_matrix_reshaped = MFCCs.reshape(n_samples, -1)  # Flattening MFCC coefficients for each frame
    
            # Fit the scaler on the MFCC data to compute the minimum and maximum values for each feature
            scaler.fit(mfcc_matrix_reshaped)
    
            # Transform the MFCC data to the [0, 1] range
            mfcc_normalized = scaler.transform(mfcc_matrix_reshaped)
    
            # Reshape the normalized data back to the original format
            mfcc_normalized = mfcc_normalized.reshape(1, n_samples, n_features)
            """
            """
            MFCCs = np.reshape(MFCCs,(n_mfcc, 98))
            # 将MFCC矩阵显示为图像
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(MFCCs, x_axis='time', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title('MFCC')
            plt.tight_layout()
    
            # 显示图像
            plt.show()
            """

            MFCCs = librosa.feature.mfcc(y=waveform_i, sr=target_sr, n_mfcc=n_mfcc)
            delta = librosa.feature.delta(MFCCs, order=1)  #delta
            delta_deltas = librosa.feature.delta(MFCCs, order=2)  #delta-deltas
            mel_spectrogram_3 = np.concatenate((MFCCs, delta, delta_deltas))
            mel_spectrogram_3 = np.transpose(mel_spectrogram_3, (1, 2, 0))

            label.append(target)
            X.append(mel_spectrogram_3)
            print(i * n + j)
    X = np.array(X)
    label = np.array(label)
    n_y = len(label)
    label = np.reshape(label, (n_y, 1))
    return X, label