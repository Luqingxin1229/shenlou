import pandas as pd
import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
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


def audio_data_to_net(excel_dir, target_samples, target_sr, root_dir, n_mfcc):
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
        # audio_path = os.path.join(root_dir, fold)  # 获取该文件所在的fold文件夹路径 urbansound8k用
        audio_path = root_dir
        # audio_file = os.listdir(audio_path)  # 获取fold文件夹内所有文件
        wave_path = os.path.join(audio_path, audio_name)  # 获取该文件绝对路径
        waveform, sr = librosa.load(wave_path, mono=False)  # 读入该音频文件

        waveform = reshape_wave(waveform, target_samples)  # 长度统一
        waveform = resample_wave(waveform, sr, target_sr)  # 采样率统一
        waveform = rechannel_wave(waveform)  # 通道数统一
        pre_emphasis = 0.97
        waveform = np.append(waveform, waveform[1:] - pre_emphasis * waveform[:-1], axis=0)  # 预加重

        # mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=target_sr, n_fft=1024, hop_length=512)  # 获取梅尔谱图
        # mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # 转dB图
        mel_spectrogram = librosa.feature.mfcc(y=waveform, sr=target_sr, n_mfcc=n_mfcc)
        # X.append(mel_spectrogram)
        delta = librosa.feature.delta(mel_spectrogram, order=1)  #delta
        delta_deltas = librosa.feature.delta(mel_spectrogram, order=2)  #delta-deltas
        mel_spectrogram_3 = np.concatenate((mel_spectrogram, delta, delta_deltas))
        mel_spectrogram_3 = np.transpose(mel_spectrogram_3, (1, 2, 0))
        label.append(target)
        X.append(mel_spectrogram_3)
        print(i)
    X = np.array(X)
    label = np.array(label)
    n_y = len(label)
    label = np.reshape(label, (n_y, 1))
    return X, label