import muda
import soundfile as sf
import os
import numpy as np
import librosa

# 定义数据集的输入与输出路径
input_dir = 'DeepSAD_tf/a'
output_dir = 'DeepSAD_tf/b'
noise_files_1 = 'aug_abnormal927/noise/2005082447_230908030000.wav'
noise_files_2 = 'aug_abnormal927/noise/2108306676_230909030000.wav'
noise_files_3 = 'aug_abnormal927/noise/whiteaudio_2.wav'

"""# 定义变形器列表
deformers = [
    muda.deformers.PitchShift(n_semitones=-0.3),  # 设置PitchShift参数
    muda.deformers.BackgroundNoise(n_samples=1, files=[noise_files_1], weight_min=0.05, weight_max=0.3),
    muda.deformers.BackgroundNoise(n_samples=1, files=[noise_files_2], weight_min=0.05, weight_max=0.3),
    muda.deformers.BackgroundNoise(n_samples=1, files=[noise_files_3], weight_min=0.05, weight_max=0.3)
]"""


# 定义音高变换的半音数目（正数表示升高音调，负数表示降低音调）
semitones = -4  # 2个半音的升高音调，可以根据需求调整
count = 0
# 遍历音频文件并进行音高变换增强
audio_files = os.listdir(input_dir)

for audio_file in audio_files:
    input_path = os.path.join(input_dir, audio_file)
    output_path = os.path.join(output_dir, audio_file)

    # 读取音频文件
    y, sr = librosa.load(input_path, sr=None)

    """    # 进行音高变换增强
    y_pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

    # 保存增强后的音频
    sf.write(output_path, y_pitch_shifted, sr)"""
    count = count+1
    print(count)
    # 随机噪音叠加增强
    # 随机选择一个噪音音频文件并读取
    # noise_files = os.listdir(noise_files_1)
    # selected_noise_file = np.random.choice(noise_files)
    # noise_path = os.path.join(noise_dir, selected_noise_file)
    noise_path = noise_files_1
    y_noise, _ = librosa.load(noise_path, sr=sr)

    # 随机生成叠加权重在（0.1, 0.3）之间
    weight = np.random.uniform(0.1, 0.3)

    # 进行音频叠加
    y_augmented = (y*(1-weight)) + (weight * y_noise)

    # 保存叠加后的音频
    sf.write(output_path, y_augmented, sr)


