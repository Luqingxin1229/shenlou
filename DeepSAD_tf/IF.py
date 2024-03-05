import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from DeepSAD_tf.datasets.audioprocessing import audio_data_to_net_pro
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'

# 数据输入前处理
x_train, y_train = audio_data_to_net_pro(excel_dir='aug_abnormal_total/aug_abnormal_2.csv', target_samples=51000,
                                         target_sr=22050,
                                         root_dir='aug_abnormal_total', hop_size=20000, n_mfcc=20)
x_test, y_test = audio_data_to_net_pro(excel_dir='testnormalpro/testnormalpro_IF.csv', target_samples=51000,
                                       target_sr=22050,
                                       root_dir='testnormalpro', hop_size=20000, n_mfcc=20)

x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)

# 初始化孤立森林模型
model = IsolationForest(n_estimators=100,
                        max_samples='auto',
                        contamination=0.1)  # contamination 参数控制异常值的比例

# 训练模型
epochs = 1
for i in range(epochs):
    print('epoch:', i)
    model.fit(x_train)

# 预测训练集


# 预测测试集
test_predictions = model.decision_function(x_test)
t = np.array(test_predictions)
print(test_predictions)
y = model.predict(x_test)
y = np.array(y)

# 计算测试集的AUROC

# 计算AUROC
auroc = roc_auc_score(y_test, test_predictions)

# 打印AUROC
print("测试集的AUROC:", auroc)

# 200
# 0.8378
# 0.8224

# 100
# 0.8555
# 0.8157
# 0.7876
# 0.8555
# 0.8022
# 0.7281
# 0.8645
# 0.8022
# 0.8348
# 0.8024
# 0.8012
# 0.7995
# 0.8248
# 0.8044
# 0.8204
# 0.8464
# 0.8286
# 0.8354
# 0.7638
# 0.7937
# 0.7561
# 0.7700
# 0.8506
# 0.8405
# 0.8297
