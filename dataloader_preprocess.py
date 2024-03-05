import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
import librosa
import pandas as pd
# from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')



"""
音频预处理和训练批次生成程序
注释：
audio_shape = (channel, samples)
data_shape(batch_size, 3:channel, 128, 45)
"""


class data_generator(Sequence):
    def __init__(self, root_dir, excel_dir, target_samples, target_sr, batch_size, shuffle, spectrogram_dim, n_class):
        super().__init__()
        self.root_dir = root_dir  # 音频文件上级目录
        self.excel = pd.read_csv(excel_dir)  # 读入csv文件
        self.target_samples = target_samples  # 希望的采样点数
        self.target_sr = target_sr  # 希望的采样率
        self.batch_size = batch_size  # 每个batch的音频数量
        self.shuffle = shuffle  # 每个batch是否打乱顺序
        self.dim = spectrogram_dim  # 语谱图的大小
        self.n_class = n_class  # 类别数
        self.on_epoch_end()
    def reshape_wave(self, waveform):
        if waveform.ndim == 1:
            waveform = np.reshape(waveform, (1, len(waveform)))
        if waveform.shape[1] > self.target_samples:
            waveform = waveform[:, 0:self.target_samples]
        elif waveform.shape[1] < self.target_samples:
            num_pad = self.target_samples - waveform.shape[1]  # 补0
            waveform = np.pad(waveform, ((0, 0), (0, num_pad)), mode='constant')
        return waveform

    def resample_wave(self, waveform, sr):
        if sr != self.target_sr:
            waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=self.target_sr)
        return waveform

    def rechannel_wave(self, waveform):
        if waveform.shape[0] > 1:
            # waveform = tf.reduce_mean(waveform, axis=0, keepdims=True)
            waveform = np.mean(waveform, axis=0, dtype=float, keepdims=True)
        return waveform

    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size] # 生成批次索引
        # list_IDs_batch = [k for k in indexes]
        list_IDs_batch = indexes

        X = self._generate_x(list_IDs_batch)
        Y = self._generate_y(list_IDs_batch)
        Y = self.label_to_onehot(Y)
        return X, Y

    def _generate_x(self, list_IDs):
        X = np.empty((self.batch_size, 3, *self.dim))  # Initialize
        # data_generation
        for i, k in enumerate(list_IDs):
            # one_batch
            X[i, ] = self.pre_processing(k)

        return X
    def _generate_y(self, list_IDs):
        Y = np.empty((self.batch_size, 1), dtype=int)
        for i, k in enumerate(list_IDs):
            # one_batch
            Y[i, ] = self.excel.iloc[k, 6]  # 获取音频文件标签
        return Y

    def on_epoch_end(self):
        # 每个epoch之后更新索引, 打乱顺序训练
        self.indexes = np.arange(len(self.excel))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def label_to_onehot(self, Y_0):
        """
        将标签转换为one-hot-code
        :param Y: 原始标签
        :return: one-hot编码后的标签
        """
        Y = np.zeros((self.batch_size, self.n_class), dtype=int)#TODO:格式是否为int呢？
        for i in range(self.batch_size):
            Y[i, Y_0[i]] = 1
        return Y
    def pre_processing(self, index):
        fold = f"fold{self.excel.iloc[index, 5]}"  # index行，5列，获取fold文件夹序号
        audio_name = f"{self.excel.iloc[index, 0]}"  # 获取文件该音频文件名
        # target = self.excel.iloc[index, 6]  # 获取该音频文件标签
        audio_path = os.path.join(self.root_dir, fold)  # 获取该文件所在的fold文件夹路径
        self.audio_file = os.listdir(audio_path)  # 获取fold文件夹内所有文件
        wave_path = os.path.join(audio_path, audio_name)  # 获取该文件绝对路径
        waveform, sr = librosa.load(wave_path, mono=False)  # 读入该音频文件
        """
        预处理部分
        """
        waveform = self.reshape_wave(waveform)  # 长度统一
        waveform = self.resample_wave(waveform, sr)  # 采样率统一
        waveform = self.rechannel_wave(waveform)  # 通道数统一
        pre_emphasis = 0.97
        waveform = np.append(waveform, waveform[1:] - pre_emphasis * waveform[:-1], axis=0)  # 预加重
        """
        特征提取部分
        log-mel-spectrogram
        deltas log-mel一阶差分（微分系数）
        deltas-deltas log-mel二阶差分（加速度系数）
        """
        mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=self.target_sr, n_fft=1024, hop_length=512)  # 获取梅尔谱图
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # 转dB图
        delta = librosa.feature.delta(mel_spectrogram, order=1)  #delta
        delta_deltas = librosa.feature.delta(mel_spectrogram, order=2)  #delta-deltas
        mel_spectrogram_3 = np.concatenate((mel_spectrogram, delta, delta_deltas))


        return mel_spectrogram_3
        # return waveform, mel_spectrogram, sr, target, audio_name
    def __len__(self):
        return int(len(self.excel)/self.batch_size)
