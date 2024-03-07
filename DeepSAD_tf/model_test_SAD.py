import tensorflow as tf
import numpy as np
from datasets.pipeline_audio import create_semisupervised_setting, load_tfdataset
from datasets.audioprocessing import audio_data_to_net_pro
from utils import load_yaml
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def normalize_list(original_list):
    min_value = min(original_list)
    max_value = max(original_list)

    normalized_list = [(x - min_value) / (max_value - min_value) for x in original_list]

    return normalized_list


def load_pipeline_audio(cfg):
    x_test, y_test = audio_data_to_net_pro(excel_dir='testnormalpro/testnormalpro.csv', target_samples=51000,
                                           target_sr=cfg['samples_sr'],
                                           root_dir='testnormalpro', hop_size=20000, n_mfcc=cfg['n_mfcc'])

    normal_classes = tuple(cfg['normal_class'])
    known_outlier_classes = tuple(cfg['known_outlier_class'])
    outlier_classes = list(range(0, 2))
    for i in normal_classes: outlier_classes.remove(i)  # 剔除正常类
    outlier_classes = tuple(outlier_classes)
    """
    上边代码功能为创建正常类别与异常类别
    """

    ratio_known_normal = cfg['ratio_known_normal']
    ratio_known_outlier = cfg['ratio_known_outlier']
    ratio_pollution = cfg['ratio_pollution']

    y_test = np.array([int(x in outlier_classes) for x in y_test])
    test_data = load_tfdataset(cfg, x_test, y_test, np.zeros_like(y_test))

    return test_data

if __name__ == '__main__':
    up_line = 0
    low_line = 0.3
    model = tf.saved_model.load('saved_models/SAD_model_03_07_6')
    c = np.load("saved_models/03_07_6c.npy")
    c = tf.constant(c)
    cfg_test = load_yaml('configs/model_tset.yaml')
    test_dataset = load_pipeline_audio(cfg_test)
    n = len(test_dataset)
    print('Starting Test')
    label_list = []
    score_list = []

    # for inputs, labels, semi_labels in self.train_dataset:
    for inputs, labels, semi_labels in test_dataset:
        outputs = model(inputs, training=False)
        dist = tf.reduce_sum((outputs - c) ** 2, 1)
        # print(dist)
        # print(labels)
        label_list.append(labels)
        score_list.append(dist)

    k = 0
    y_pred = np.zeros_like(label_list)
    thresh_hold = 0.1
    for i in range(n):
        score = float(score_list[i])
        print('序号：', i, "score:", score, 'label:', int(label_list[i]))

        if score >= thresh_hold:
            y_pred[i] = 1
        else: y_pred[i] = 0

    for i in range(n):
        if int(y_pred[i]) == int(label_list[i]):
            k = k + 1
    acc = k / n * 100

    print('测试精度为：', acc, '%')

    # score_list = normalize_list(score_list)
    labels = tf.concat(label_list, axis=0).numpy()
    scores = tf.concat(score_list, axis=0).numpy()
    test_auc = roc_auc_score(labels, scores)

    print('Test AUC: {:.2f}%'.format(100. * test_auc))


    # 计算ROC曲线的点
    fpr, tpr, thresholds = roc_curve(labels, scores)
    print("阈值:", thresholds)

    # 计算AUC值
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # 绘制整体散点图

    # 你的数据
    point_size_0 = 5
    point_size = 10
    # score = score_list
    # label = label_list

    # 将数据列表转换为 NumPy 数组
    score = np.array(score_list)
    label = np.array(label_list)

    # 获取红色点和蓝色点的索引
    red_indices = np.where(label == 0)
    blue_indices = np.where(label == 1)
    # 根据索引提取红色点和蓝色点的数据
    red_scores = score[red_indices]
    blue_scores = score[blue_indices]

    plt.figure(figsize=(8, 5))
    # 绘制红色点
    plt.scatter(red_indices[0], red_scores, s=point_size_0, color='red', label='Abnormal')
    # 绘制蓝色点
    plt.scatter(blue_indices[0], blue_scores, s=point_size_0, color='blue', label='Normal')
    # 设置图例
    plt.legend(loc='upper left', title='Label')
    # 设置横纵坐标轴标签
    plt.xlabel('Data Index')
    plt.ylabel('Score')
    plt.title('Scatter Plot of All Data Points')
    # 显示图形
    plt.show()
    # 绘制细节散点图
    red_indices = np.where(label == 0)
    blue_indices = np.where(label == 1)
    # 根据索引提取红色点和蓝色点的数据
    red_scores = score[red_indices]
    blue_scores = score[blue_indices]
    plt.figure(figsize=(8, 5))
    # 绘制红色点
    plt.scatter(red_indices[0], red_scores, s=point_size_0, color='red', label='Abnormal')
    # 绘制蓝色点
    plt.scatter(blue_indices[0], blue_scores, s=point_size_0, color='blue', label='Normal')
    # 设置图例
    plt.legend(loc='upper left', title='Label')
    # 设置横纵坐标轴标签
    plt.xlabel('Data Index')
    plt.ylabel('Score')
    plt.title('Scatter Plot of Data Points with Score=(0,0.3)')
    plt.ylim(up_line, low_line)  # 绘制细节位置
    # 显示图形
    plt.show()

