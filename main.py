# from VGGish import VGGish_train

# VGGish_train()

"""
测试dataloader代码
config = tf.compat.v1.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage 限制CPU使用的核的个数
                inter_op_parallelism_threads = 1,
                intra_op_parallelism_threads = 1,
                log_device_placement=True)
with tf.compat.v1.Session(config = config) as sess:

    root_dir = r'D:/python_documents/shenlou/UrbanSound8k/UrbanSound8K/audio'  # 音频文件根目录
    excel_dir = r'D:/python_documents/shenlou/UrbanSound8k/UrbanSound8K/metadata/UrbanSound8K.csv'  # csv文件目录
    index = 5
    print(len(pd.read_csv(excel_dir)))
    mydata = data_generator(root_dir=root_dir, excel_dir=excel_dir, target_sr=10_000, target_samples=50_000, batch_size=5,
                       shuffle=False, spectrogram_dim=(128, 45), n_class=10)
    for data in mydata:
        x, y = data
        print(x.shape)
        print(y)
"""

from Deep_SVDD_data import audio_data_to_net
from SVDD_models import VGGish, VGGish_1, VGGish_2, VGGish_multi, model_test, model_test_m, model_test_m1, model_test_m2
from sklearn.metrics import roc_auc_score
from Deep_SVDD_2 import DeepSVDD
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Deep_SVDD_data_pro import audio_data_to_net_pro
from Deep_SVDD_data_xb import audio_data_to_net_xb
from Deep_SVDD_data_1D import audio_data_to_net_1D, model_1D
import os



def cifar_lenet(H=128):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (5, 5), strides=(3, 3), padding='same', use_bias=False, input_shape=(20, 98, 3)))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Conv2D(64, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(H, use_bias=False))

    return model
if __name__ == '__main__':
    # air_c_debug.csv   air_c_debug_t.csv
    # air_c.csv   air_test.csv
    #  MFCC & Mel
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    tf.keras.backend.clear_session()
    """
    X, t = audio_data_to_net(excel_dir='normal_LD18/normal_LD18_1.csv', target_samples=70_250, target_sr=22_050,
                          root_dir='normal_LD18')  # 50_000, 10_000 ub
    x_test, y_test= audio_data_to_net(excel_dir='LD_18_test/LD18_test.csv', target_samples=70_250, target_sr=22_050,
                               root_dir='LD_18_test')
    """
    nmfcc = 20
    samples_sr = 22050
    #  正常样本数据增强
    '''
    X, t = audio_data_to_net_pro(excel_dir='normal_LD18/normal_LD18_2.csv', target_samples=70125, target_sr=samples_sr,
                          root_dir='normal_LD18', hop_size=30000, n_mfcc=nmfcc)  # 50_000, 10_000 ub
    x_test, y_test= audio_data_to_net_pro(excel_dir='LD_18_test/LD18_test.csv', target_samples=70125, target_sr=samples_sr,
                               root_dir='LD_18_test', hop_size=30000, n_mfcc=nmfcc)
    '''
    #  数据增强MFCC
    X, t = audio_data_to_net_pro(excel_dir='DeepSAD_tf/aug_abnormal_total/aug_abnormal_DeepSVDD.csv', target_samples=51000, target_sr=samples_sr,
                          root_dir='DeepSAD_tf/aug_abnormal_total', hop_size=20000, n_mfcc=nmfcc)  # 50_000, 10_000 ub
    x_test, y_test= audio_data_to_net_pro(excel_dir='DeepSAD_tf/testnormalpro/testnormalpro.csv', target_samples=51000, target_sr=samples_sr,
                               root_dir='DeepSAD_tf/testnormalpro', hop_size=20000, n_mfcc=nmfcc)
    #  异常样本数据增强xb
    # X, t = audio_data_to_net_xb(excel_dir='aug_abnormal_total/aug_abnormal.csv', target_samples=30000, target_sr=samples_sr,
    #                       root_dir='aug_abnormal_total', hop_size=20000)  # 50_000, 10_000 ub
    # x_test, y_test= audio_data_to_net_xb(excel_dir='testnormalpro/testnormalpro.csv', target_samples=30000, target_sr=samples_sr,
    #                            root_dir='testnormalpro', hop_size=20000)
    #  Urbansound xb
    # X, t = audio_data_to_net_xb(excel_dir='normal_LD18/normal_LD18_2.csv', target_samples=30000, target_sr=samples_sr,
    #                       root_dir='normal_LD18', hop_size=20000)  # 50_000, 10_000 ub
    # x_test, y_test= audio_data_to_net_xb(excel_dir='LD_18_test/LD18_test.csv', target_samples=30000, target_sr=samples_sr,
    #                            root_dir='LD_18_test', hop_size=20000)
    #  小波变换
    # X, t = audio_data_to_net_xb(excel_dir='normal_LD18/normal_LD18_2.csv', target_samples=30000, target_sr=22_050,
    #                       root_dir='normal_LD18', hop_size=20000)  # 50_000, 10_000 ub
    # x_test, y_test= audio_data_to_net_xb(excel_dir='LD_18_test/LD18_test.csv', target_samples=30000, target_sr=22_050,
    #                            root_dir='LD_18_test', hop_size=20000)
    # 1D
    # X, t = audio_data_to_net_pro(excel_dir='abnormal_LD18_1/abnormal_train.csv', target_samples=10000, target_sr=samples_sr,
    #                              root_dir='abnormal_LD18_1', hop_size=5000)  # 50_000, 10_000 ub
    # x_test, y_test = audio_data_to_net_1D(excel_dir='LD_18_test1/test1.csv', target_samples=10000, target_sr=samples_sr,
    #                                        root_dir='LD_18_test1', hop_size=5000)
    model = model_test_m1(inputshape=(20, 100, 3), H=32)  # ub 3,20,45
    # model_Mel = VGGish_1(inputshape=(128, 216, 3))
    # model_xb = model_test_m1(inputshape=(16, 4389, 3))
    svdd = DeepSVDD(deep_model=model, objective='one-class', batch_size=64, nu=0.1,
                    representation_dim=32)  # mfcc(3, 20 , 45)  mel(3, 128, 45)
    # train Deep SVDD
    auc = 0
#    while auc < 0.93:
    svdd.fit(X, x_test, y_test, epochs=4, verbose=True)
    print(svdd.c)
    # test Deep SVDD
    score = svdd.predict(x_test)
    auc = roc_auc_score(y_test, score)
    _ = svdd.predict_fcn(np.zeros((1, 20, 100, 3)))  # the same as model above
    tf.saved_model.save(svdd, 'deep_svdd_models/deep_svdd_saved_model_10_28_a')
    # deep_svdd = tf.saved_model.load('')
    print(score)
    print('AUROC: %.3f' % auc)

# 10/28
# 0.878
# 0.892
# 0.890
# 0.911
# 0.898
# 0.918
# 0.899
# 0.896
# 0.895
# 0.915
#
#
