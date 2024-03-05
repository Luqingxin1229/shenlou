from tensorflow import keras
import os
import numpy as np
import librosa
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

"""
加载训练好的模型用于预测_single
"""
### 模型加载 ###
model_file = r'D:\python_documents\shenlou\models'  # 模型所在文件夹
model_name = 'VGGish_us_328_1.h5'  # model_name
model_path = os.path.join(model_file, model_name)  # model_path
VGGish_model = keras.models.load_model(model_path)  # load model

### 文件及标签加载 ###
audio_path = r'D:\python_documents\shenlou\UrbanSound8k\UrbanSound8K\audio'
excel_dir = r'D:\python_documents\shenlou\UrbanSound8k\UrbanSound8K\metadata\UrbanSound8K.csv'
class Get_audio():
    def __init__(self, audio_path, excel_dir, target_samples, target_sr):
        self.audio_path = audio_path
        self.excel = pd.read_csv(excel_dir)
        self.target_samples = target_samples
        self.target_sr = target_sr

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
        fold = f"fold{self.excel.iloc[index, 5]}"  # index行，5列，获取fold文件夹序号
        audio_name = f"{self.excel.iloc[index, 0]}"  # 获取文件该音频文件名
        print(audio_name)
        target = self.excel.iloc[index, 6]  # 获取该音频文件标签
        audio_path = os.path.join(self.audio_path, fold)  # 获取该文件所在的fold文件夹路径
        self.audio_file = os.listdir(audio_path)  # 获取fold文件夹内所有文件
        wave_path = os.path.join(audio_path, audio_name)  # 获取该文件绝对路径
        waveform, sr = librosa.load(wave_path, mono=False)  # 读入该音频文件
        # 读取慢可以使用cache缓存进行加速，见‘前端-基于梅尔频谱的音频信号分类识别’
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
        mel_spectrogram = librosa.feature.melspectrogram(waveform, self.target_sr, n_fft=1024, hop_length=512)  # 获取梅尔谱图
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # 转dB图
        delta = librosa.feature.delta(mel_spectrogram, order=1)  #delta
        delta_deltas = librosa.feature.delta(mel_spectrogram, order=2)  #delta-deltas
        mel_spectrogram_3 = np.concatenate((mel_spectrogram, delta, delta_deltas))
        return mel_spectrogram_3, target
    def __len__(self):
        return len(self.excel)

if __name__ == '__main__':
    index = 22
    get_audio = Get_audio(audio_path, excel_dir, target_samples=50_000, target_sr=10_000)
    mel_spectrogram, label = get_audio[index]
    mel_spectrogram = np.reshape(mel_spectrogram, (1, 3, 128, 45))
    print('The True label of this audio is:', label)
    label_pred = VGGish_model.predict(mel_spectrogram)
    print('network output:', label_pred)
    label_pred = np.argmax(label_pred, axis=1)
    print('The predicted label of this audio is:', label_pred)