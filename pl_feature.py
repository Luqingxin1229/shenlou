import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def get_spectrogram(audio_path, save_path):
    # audio_path = 'path/to/your/audio/file.wav'
    audio, sr = librosa.load(audio_path)

    D = librosa.stft(audio)
    spectrogram = librosa.amplitude_to_db(abs(D), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(save_path)

if __name__ == '__main__':
    path = r'feature/0'
    file = os.listdir(path)
    for i in range(len(file)):
        audio_path = os.path.join(path, file[i])
        save_path = r'feature/pinpu/0'
        get_spectrogram(audio_path=audio_path, save_path=os.path.join(path, file[i][:-4] + '.png'))
        print('%d completed' % (i+1))