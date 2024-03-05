import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import librosa
from scipy.fft import fft

plt.figure(dpi=600) # 将显示的所有图分辨率调高
matplotlib.rc("font",family='SimHei') # 显示中文
matplotlib.rcParams['axes.unicode_minus']=False # 显示符号

def displaySpectrum(filepath): # 显示语音频域谱线
    x, sr = librosa.load(filepath, sr=16000) # sr 采样率
    print(len(x))
    # ft = librosa.stft(x)
    # magnitude = np.abs(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    # frequency = np.angle(ft)  # (0, 16000, 121632)

    ft = fft(x)
    print(len(ft), type(ft), np.max(ft), np.min(ft))
    magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)

    print(len(magnitude), type(magnitude), np.max(magnitude), np.min(magnitude))
    print(len(frequency), type(frequency), np.max(frequency), np.min(frequency))

    # plot spectrum，限定[:40000]
    # plt.figure(figsize=(18, 8))
    plt.plot(frequency[:10000], magnitude[:10000])  # magnitude spectrum
    plt.title("spectrum")
    plt.xlabel("Hz")
    plt.ylabel("amplitude")
    # plt.savefig("your dir\语音信号频谱图", dpi=600)
    plt.show()

def displayWaveform(filepath): # 显示语音时域波形
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    samples, sr = librosa.load(filepath, sr=16000)
    # samples = samples[6000:16000]

    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)

    plt.plot(time, samples)
    plt.title("语音信号时域波形")
    plt.xlabel("时长（秒）")
    plt.ylabel("振幅")
    # plt.savefig("your dir\语音信号时域波形图", dpi=600)
    plt.show()

if __name__ == '__main__':
    displaySpectrum('1910080346_2023-02-01_正常.wav')
    displayWaveform('1910080346_2023-02-01_正常.wav')