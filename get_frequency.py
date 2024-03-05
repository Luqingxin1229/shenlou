import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 读取音频文件
sample_rate, audio_data = wavfile.read('aug_abnormal927/1910080353_20230310_空.wav')

# 计算音频信号的傅里叶变换
freq_domain = np.fft.fft(audio_data)
freq_domain = np.abs(freq_domain)  # 获取幅度谱

# 获取频率轴
freqs = np.fft.fftfreq(len(freq_domain), 1 / sample_rate)

# 仅保留正频率部分
positive_freq_domain = freq_domain[freqs >= 0]
positive_freqs = freqs[freqs >= 0]

# 创建频率图，只显示正频率部分
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, positive_freq_domain)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Positive Frequency Spectrum of Audio')
plt.grid()
plt.show()


# 获取时间轴
time_axis = np.arange(0, len(audio_data)) / sample_rate

# 创建音频时域波形图
plt.figure(figsize=(10, 6))
plt.plot(time_axis, audio_data, lw=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.grid()
plt.show()

