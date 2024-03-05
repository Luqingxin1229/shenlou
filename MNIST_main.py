from SVDD_models import VGGish, VGGish_1, VGGish_2, VGGish_multi, model_test, model_test_m, model_test_m1
from sklearn.metrics import roc_auc_score
from Deep_SVDD_2 import DeepSVDD
import numpy as np
import tensorflow as tf
import cv2
import os
import pandas as pd
from tensorflow import keras

def mnist_lenet(H=32):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(8, (5, 5), padding='same', use_bias=False, input_shape=(28, 28, 1)))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Conv2D(4, (5, 5), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(H, use_bias=False))

    return model

def data_read(root_dir, excel_dir):
    excel = pd.read_csv(excel_dir)  # 读入csv文件
    N = len(excel)
    X = []
    label = []
    for i in range(N):
        pic_filename = f"{excel.iloc[i, 0]}"  # 获取文件名
        label_i = excel.iloc[i, 1]
        pic_path = os.path.join(root_dir, pic_filename)
        img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        print(img.shape)
        X.append(img)
        label.append(label_i)
        print(i)
    X = np.array(X)
    label = np.array(label)
    n_label = len(label)
    label = np.reshape(label, (n_label, 1))

    return X, label


if __name__ == '__main__':

    x_test, y_test = data_read(root_dir='MNIST_data/mnist_train/ceshi',
                     excel_dir='MNIST_data/mnist_train/ceshi/ceshi.csv')
    X, Y = data_read(root_dir='MNIST_data/mnist_train/0_train',
                     excel_dir='MNIST_data/mnist_train/0_train/train_MT.csv')
    model_MFCC = mnist_lenet(32) # ub 3,20,45
    # model_Mel = VGGish_1(inputshape=(128, 216, 3))
    # model_xb = model_test_m1(inputshape=(16, 4389, 3))
    svdd = DeepSVDD(deep_model=model_MFCC, objective='one-class', batch_size=64, nu=0.1,
                    representation_dim=32)  # mfcc(3, 20 , 45)  mel(3, 128, 45)
    # train Deep SVDD
    auc = 0
#    while auc < 0.93:
    svdd.fit(X, x_test, y_test, epochs=200, verbose=True)
    print(svdd.c)
    # test Deep SVDD
    score = svdd.predict(x_test)
    auc = roc_auc_score(y_test, score)
    _ = svdd.predict_fcn(np.zeros((1, 28, 28, 3)))  # the same as model above
    tf.saved_model.save(svdd, 'deep_svdd_models/deep_svdd_saved_model_8_2_1_MT')
    # deep_svdd = tf.saved_model.load('')
    print(score)
    print('AUROC: %.3f' % auc)