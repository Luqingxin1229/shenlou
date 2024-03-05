import numpy as np
from scipy.io.wavfile import write

"""# 设置音频参数
sample_rate = 44100  # 采样率（每秒样本数）
duration = 5  # 音频时长（秒）
frequency = 440  # 生成的音频频率（Hz）

# 生成时间轴
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# 生成音频数据（示例中生成了一个简单的440Hz正弦波）
audio_data = np.sin(2 * np.pi * frequency * t)

# 将音频数据扩展到16位整数范围（可根据需要进行调整）
audio_data = np.int16(audio_data * 32767)

# 保存音频文件
write('whiteaudio/whiteaudio_1.wav', sample_rate, audio_data)"""

import numpy as np
from scipy.io.wavfile import write

# 设置音频参数
sample_rate = 44100  # 采样率（每秒样本数）
duration = 5  # 音频时长（秒）
n = 13
for i in range(n):
        # 生成随机白噪声
        audio_data = np.random.normal(0, 1, int(sample_rate * duration))

        # 将音频数据扩展到16位整数范围（可根据需要进行调整）
        audio_data = np.int16(audio_data * 32767)

        # 保存音频文件
        write('whiteaudio/whiteaudio_%d.wav' % i, sample_rate, audio_data)
