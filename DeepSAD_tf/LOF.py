import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from DeepSAD_tf.datasets.audioprocessing import audio_data_to_net_pro
import os

os.environ['CUDA_VISIBLE_DEVICES']='3'
# 1. 准备数据
x_train, y_train = audio_data_to_net_pro(excel_dir='aug_abnormal_total/aug_abnormal_2.csv', target_samples=51000,
                                         target_sr=22050,
                                         root_dir='aug_abnormal_total', hop_size=20000, n_mfcc=20)
x_test, y_test = audio_data_to_net_pro(excel_dir='testnormalpro/testnormalpro_ocsvm.csv', target_samples=51000,
                                       target_sr=22050,
                                       root_dir='testnormalpro', hop_size=20000, n_mfcc=20)

x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)

# 构建LOF模型
model = LocalOutlierFactor(n_neighbors=50, contamination=0.1, novelty=True)

# 训练模型
model.fit(x_train)

# 预测测试数据的局部异常因子分数
test_scores = model.decision_function(x_test)  # 通过取相反数得到局部异常因子分数

# 假设 y_test 包含真实的标签信息，0 表示正常，-1 表示异常
# y_test = np.zeros(len(x_test))  # 假设测试集中所有样本都是正常的

# 计算AUROC
auroc = roc_auc_score(y_test, test_scores)

print("AUROC:", auroc)
