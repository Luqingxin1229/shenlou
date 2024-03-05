import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from .preprocessing import create_semisupervised_setting, load_tfdataset


def load_mnist(cfg):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    # y_train = np.expand_dims(y_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # y_test = np.expand_dims(y_test, -1)
     
    normal_classes = tuple(cfg['normal_class'])
    known_outlier_classes = tuple(cfg['known_outlier_class'])
    outlier_classes = list(range(0, 10))
    for i in normal_classes: outlier_classes.remove(i)  # 剔除正常类
    outlier_classes = tuple(outlier_classes)
    """
    上边代码功能为创建正常类别与异常类别
    """

    ratio_known_normal = cfg['ratio_known_normal']
    ratio_known_outlier = cfg['ratio_known_outlier']
    ratio_pollution = cfg['ratio_pollution']

    # 创建半监督标签列表
    idx, _, semi_targets = create_semisupervised_setting(y_train, normal_classes,
                                                             outlier_classes, known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)


    y_train[idx] = np.array([int(x in outlier_classes) for x in  y_train[idx]])  # 标签变为异常1, 正常0
    y_test = np.array([int(x in outlier_classes) for x in  y_test])
    train_data = load_tfdataset(cfg, x_train[idx], y_train[idx], semi_targets)
    test_data = load_tfdataset(cfg, x_test, y_test, np.zeros_like(y_test))
    
    return train_data, test_data