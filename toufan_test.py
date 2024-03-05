import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
# 加载保存的 StandardScaler 归一化尺度
scaler_filename = 'scaler_3.pkl'
with open(scaler_filename, 'rb') as scaler_file:
    feature_scaler, target_scaler = pickle.load(scaler_file)

# 读取测试数据，注意需要根据实际情况准备测试数据
test_data = pd.read_excel('senor.xlsx', index_col='x', sheet_name=0)
test_features = test_data[['X1', 'X2', 'X3']]

# 对测试数据进行特征归一化
test_features = feature_scaler.transform(test_features)

# 加载已保存的模型
model = load_model('toufan_model_3.h5')

# 进行预测
y_pred_normalized = model.predict(test_features)

# 恢复预测结果到原始尺度
y_pred_original_scale = target_scaler.inverse_transform(y_pred_normalized)
# y_pred_original_scale = np.array(y_pred_original_scale)
print(y_pred_original_scale)

# 将预测结果添加到测试数据中
# test_data['Y_pred'] = y_pred_original_scale

test_labels = test_data['Y']

# 保存包含预测结果的数据到文件
# test_data.to_excel('predicted_results.xlsx')

# 绘制预测结果和标签
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_labels, label='label')
plt.plot(test_data.index, y_pred_original_scale, label='predict')
plt.xlabel('index')
plt.ylabel('Y')
plt.legend()
plt.title('test result')
plt.show()

# 保存包含预测结果的数据到文件
test_data.to_excel('predicted_results.xlsx')