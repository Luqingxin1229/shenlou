from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

hidden_units_size = 20
batch_size = 256
training_iterations = 60

def Train_1():

    inputfile = 'senor.xlsx'
    # modelfile = 'modelweight.model'
    data = pd.read_excel(inputfile, index_col='x', sheet_name=0)
    feature = ['X1', 'X2', 'X3']
    label = ['Y']
    data_train = data.loc[range(1, len(data))].copy()

    # 数据预处理和使用 StandardScaler 进行特征和目标变量的标准化
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    data_train[feature] = feature_scaler.fit_transform(data_train[feature])
    data_train[label] = target_scaler.fit_transform(data_train[label])  # 对目标变量进行归一化

    x = data_train[feature].values
    y = data_train[label].values

    # 分割数据集为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42, shuffle=True)

    # 保存 StandardScaler 的归一化尺度
    scaler_filename = 'scaler_4.pkl'
    with open(scaler_filename, 'wb') as scaler_file:
        pickle.dump((feature_scaler, target_scaler), scaler_file)

    # 模型
    Input = layers.Input((3,))
    x = layers.Dense(32, activation='relu', name='hidden_layer')(Input)
    x = layers.Dense(16, activation='relu', name='hidden_layer1')(x)
    # x = layers.Dense(256, activation='relu', name='hidden_layer3')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(128, activation='relu', name='hidden_layer4')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(64, activation='relu', name='hidden_layer2')(x)
    # x = layers.Dense(32, activation='relu')(x)
    Output = layers.Dense(1)(x)

    model = models.Model(inputs=Input, outputs=Output)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=training_iterations)

    # 评估模型性能
    test_loss = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}')
    model.save('toufan_model_4.h5')
    # model.save_weights(modelfile)


Train_1()
