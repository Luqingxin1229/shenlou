import pandas as pd
import numpy as np
from tensorflow.keras import layers, models,metrics
# read data
if __name__ == '__main__':
    df = pd.read_excel('ceshi.xlsx')
    # 行列互换
    df2 = df.stack()
    df3 = df2.unstack(0)
    #
    lst = df3.values.tolist()  # 转列表
    lst1, lst2 = lst[:-1], lst[-1] # ：-1除最后一项  -1最后一项
    data, target = lst1[0], lst2
    print(data)
    print(target)

# model
def train():
    result = {'正常': 0, '疑似': 1, '渗漏': 2}

    x = layers.Input
    x = layers.Dense

    model = models.Model(inputs=inputshape, outputs=output)
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics='acc')
    model.fit(x=data, y=target, epochs=50)