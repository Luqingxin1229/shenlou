import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    path = "1910080626_2023-02-01_渗漏.wav"

    # sr=None声音保持原采样频率， mono=False声音保持原通道数
    data, fs = librosa.load(path, sr=None, mono=False)  # fs为输入信号的采样频率，data为音频信号值
    print(fs)
    pre_emphasis = 0.97
    data = np.append(data[0], data[1:] - pre_emphasis * data[:-1])  # 预加重
    L = len(data)
    print('Time:', L / fs)  # 音频时长

    print(len(data), fs)
    # time = np.arange(0, len(data)) * (1.0 / fs)

    # plt.plot(time, data)
    # plt.title("语音信号时域波形")
    # plt.xlabel("时长（秒）")
    # plt.ylabel("振幅")
    # plt.show()
    # 0.03s
    framesize = 0.03  # 桢长度
    # NFFT点数=0.025*fs
    framelength = int(framesize * fs)  # 每帧包含多少采样点
    print("framelength:", framelength)

    #提取mel特征
    # mel_spect = librosa.feature.melspectrogram(data, sr=fs, n_fft=framelength, win_length=framelength, hop_length=128, window='hamming')
    mel_spect = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=1024)
    print(mel_spect)
    #转化为log形式
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)  # 将幅度频谱转换为dB标度频谱。也就是对mel_spect取对数
    delta = librosa.feature.delta(mel_spect, order=1)
    delta_deltas = librosa.feature.delta(mel_spect, order=2)
    print(mel_spect.shape)
    #画mel谱图

    plt.imshow(mel_spect)
    # plt.imshow(delta)
    # plt.imshow(delta_deltas)
    """
    librosa.display.specshow(mel_spect, sr=fs, x_axis='time', y_axis='mel')
    librosa.display.specshow(delta, sr=fs)
    librosa.display.specshow(delta_deltas, sr=fs)
    # plt.savefig('1.png', bbox_inches='tight', pad_inches=0)
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    """
    plt.show()
def get_mel(file_path, save_path):
    # sr=None声音保持原采样频率， mono=False声音保持原通道数
    data, fs = librosa.load(file_path, sr=None, mono=False)  # fs为输入信号的采样频率，data为音频信号值
    # print(fs)

    L = len(data)
    # print('Time:', L / fs)  # 音频时长

    # 0.025s
    framesize = 0.03  # 桢长度
    # NFFT点数=0.025*fs
    framelength = int(framesize * fs)  # 每帧包含多少采样点
    # print("NFFT:", framelength)

    # 提取mel特征
    mel_spect = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=512, win_length=framelength, hop_length=128, window='hamming')
    # 转化为log形式
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)  # 将幅度频谱转换为dB标度频谱。也就是对mel_spect取对数
    # 画mel谱图
    plt.imshow(mel_spect)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, transparent=True, dpi=300, pad_inches=0)
    plt.clf()