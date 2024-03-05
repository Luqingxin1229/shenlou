import numpy as np
import tensorflow as tf

def load_tfdataset(cfg, image, label, semi_label):
    dataset = tf.data.Dataset.from_tensor_slices((image, label, semi_label))
    dataset = dataset.cache()
    dataset = dataset.shuffle(4096)
    dataset = dataset.map(transform_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(cfg['batch_size'])
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset


def transform_data(img, label, semi_label):
    img = tf.cast(img, tf.float32) / 255.

    return img, label, semi_label



def create_semisupervised_setting(labels, normal_classes, outlier_classes, known_outlier_classes,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution):
    """
    创建半监督数据设置。
    :param labels: np.array 包含所有数据集样本的标签
    :param normal_classes: 带有正常类标签的元组
    :param outlier_classes: 带有异常类标签的元组
    :param known_outlier_classes: 具有已知（标记）异常类标签的元组
    :param ratio_known_normal: 已知（标记）正常样本的所需比例
    :param ratio_known_outlier: 已知（标记）异常样本的所需比例
    :param ratio_pollution: 具有未知（未标记）异常的未标记数据的期望污染率。
    :return: 包含样本索引列表、原始标签列表和半监督标签列表的元组
    """
    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()  # 获取所有标签中属于正常类标签的索引列表
    idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()  # 所有异常类的索引列表
    idx_known_outlier_candidates = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()  # 获取标记异常类的索引列表

    n_normal = len(idx_normal)  # 获取正常类样本个数

    # 求解线性方程组以获得相应的样本数
    a = np.array([[1, 1, 0, 0],
                  [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                  [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                  [0, -ratio_pollution, (1-ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)

    # 获取样本数
    n_known_normal = int(x[0])
    n_unlabeled_normal =  int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_known_outlier = int(x[3])

    # 样本指数
    perm_normal = np.random.permutation(n_normal)  # 对0-n_normal的数字进行随机排列
    perm_outlier = np.random.permutation(len(idx_outlier))
    perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

    idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()  # 已知正常类样本索引
    idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()  # 未知正常样本索引
    idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()  # 未知异常样本索引
    idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist()  # 已知异常样本索引

    # 获取原始类别标签
    labels_known_normal = labels[idx_known_normal].tolist()
    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_known_outlier = labels[idx_known_outlier].tolist()

    # 获取半监督设置标签
    # 重新设置标签,将已知正常样本设置为+1,未知样本设置为0,已知异常样本标签设置为-1
    semi_labels_known_normal = np.ones(n_known_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
    semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()

    # Create final lists
    # 所有数据样本索引
    list_idx = idx_known_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
    # 所有数据样本的原始标签
    list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
    # 所有数据样本的半监督标签
    list_semi_labels = (semi_labels_known_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier
                        + semi_labels_known_outlier)

    return list_idx, list_labels, list_semi_labels

