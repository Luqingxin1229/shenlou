from get_Mel import get_mel
import os


if __name__ == '__main__':
    file_list = os.listdir('D:/python_documents/shenlou/audio_file')
    print(file_list)
    n = len(file_list)
    print('音频文件个数为：', n)
    for i in range(0, n):
        print("正在转换第%d段音频" % i)
        get_mel('D:/python_documents/shenlou/audio_file/' + file_list[i], 'D:/python_documents/shenlou/audio_mel/' + file_list[i] + '.png')

