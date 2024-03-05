import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from DeepSAD_tf.datasets.audioprocessing import audio_data_to_net_pro
import os
import random

os.environ['CUDA_VISIBLE_DEVICES']='3'
# 1. 准备数据
x_train, y_train = audio_data_to_net_pro(excel_dir='aug_abnormal_total/aug_abnormal_2.csv', target_samples=51000,
                                         target_sr=22050,
                                         root_dir='aug_abnormal_total', hop_size=20000, n_mfcc=20)
x_test, y_test = audio_data_to_net_pro(excel_dir='testnormalpro/testnormalpro_ocsvm.csv', target_samples=51000,
                                       target_sr=22050,
                                       root_dir='testnormalpro', hop_size=20000, n_mfcc=20)

x_train = x_train.reshape(len(x_train), -1)
random.shuffle(x_train)
x_test = x_test.reshape(len(x_test), -1)

ocsvm = OneClassSVM(kernel='rbf', nu=0.1)  # 根据需要调整参数

# 3. 训练 One-Class SVM 模型
ocsvm.fit(x_train)
# train_predictions = ocsvm.predict(x_train)
# 4. 进行测试集的异常检测
test_predictions = ocsvm.decision_function(x_test)
t = np.array(test_predictions)
# auroc_train = roc_auc_score(y_train, train_predictions)

# 5. 计算测试集的AUROC
auroc = roc_auc_score(y_test, (-1) * test_predictions)

# 6. 打印AUROC
print("测试集的AUROC:", auroc)

# nu=0.1
# 0.76
# 0.767
# 0.767

# nu=0.05
# 0.729

# 10/23 0.902715

# 12/28
# 0.9221
# 0.9177
# 0.9199.
# 0.9222
# 0.9258
# 0.9255
# 0.9177
# 0.9160
# 0.9178
# 0.9191
# 0.9190

