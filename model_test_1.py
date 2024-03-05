import numpy as np
import tensorflow as tf
from math import ceil
from Deep_SVDD_data import audio_data_to_net
from Deep_SVDD_data_xb import audio_data_to_net_xb
from Deep_SVDD_data_pro import audio_data_to_net_pro
from extract_feature import get_conv_output, create_conv_model

if __name__ == '__main__':
    x_test, y_test = audio_data_to_net_pro(excel_dir='LD_18_test/LD18_test.csv',target_samples=50000,
                                             target_sr=22050, root_dir='LD_18_test', hop_size=20000, n_mfcc=None)
    # x_test, y_test= audio_data_to_net_xb(excel_dir='LD_18_test/LD18_test.csv', target_samples=30000, target_sr=22_050,
    #                            root_dir='LD_18_test', hop_size=20000)
    # print(y_test)
    # jiance(x_test, 64)
    deep_model = tf.keras.models.load_model('VGGish_SVDD_885.h5')
    x_test_conv = get_conv_output(x_test, model_path='VGGish_SVDD_885.h5')
    svdd = tf.saved_model.load('deep_svdd_models/VGGish+deep_svdd_8_8_5')
    print(svdd.c)
    # x_test = np.array(x_test)
    x_test_conv = tf.cast(x_test_conv, dtype=tf.float64)
    k = 0
    q = 0
    n = 512
    l = np.zeros_like(y_test)
    y_pred = np.zeros_like(y_test)
    for i in range(512):
        a = x_test_conv[i]
        b = x_test[i]
        a = np.expand_dims(a, axis=0)
        b = np.expand_dims(b, axis=0)
        score = svdd.predict_fcn(a)
        lei = deep_model.predict(b)
        l[i] = np.argmax(lei)
        print(l[i], score, y_test[i])
        # 精度计算
        if score >= 0.003 :
            y_pred[i] = 1
        else:y_pred[i] = 0

    for i in range(512):
        if int(y_pred[i]) == int(y_test[i]):
            k = k + 1
        if int(l[i]) == int (y_test[i]):
            q = q + 1
    acc = k / n * 100
    acc_1 = q / n * 100
    print('该模型测试精度为：', acc, '%')
    print('该深度模型测试精度为：', acc_1, '%')

