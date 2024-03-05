import tensorflow as tf
from tensorflow.keras import layers, models


class CustomModel_1(tf.keras.Model):
    def __init__(self, rep_dim=32):
        super(CustomModel_1, self).__init__()

        self.conv1 = layers.Conv2D(64, (3, 1), strides=(1, 1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.bn1 = layers.BatchNormalization(epsilon=1e-4)

        self.conv2 = layers.Conv2D(128, (3, 1), strides=(1, 1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.bn2 = layers.BatchNormalization(epsilon=1e-4)

        self.conv3 = layers.Conv2D(64, (5, 1), strides=(1, 1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.bn3 = layers.BatchNormalization(epsilon=1e-4)

        self.conv4 = layers.Conv2D(128, (5, 1), strides=(1, 1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.pool4 = layers.MaxPooling2D((2, 2))
        self.bn4 = layers.BatchNormalization(epsilon=1e-4)

        self.conv5 = layers.Conv2D(64, (7, 1), strides=(1, 1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.pool5 = layers.MaxPooling2D((2, 2))
        self.bn5 = layers.BatchNormalization(epsilon=1e-4)

        self.conv6 = layers.Conv2D(128, (7, 1), strides=(1, 1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.pool6 = layers.MaxPooling2D((2, 2))
        self.bn6 = layers.BatchNormalization(epsilon=1e-4)

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(rep_dim)

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.pool1(x1)
        x1 = self.bn1(x1)

        x1 = self.conv2(x1)
        x1 = self.pool2(x1)
        x1 = self.bn2(x1)

        x2 = self.conv3(inputs)
        x2 = self.pool3(x2)
        x2 = self.bn3(x2)

        x2 = self.conv4(x2)
        x2 = self.pool4(x2)
        x2 = self.bn4(x2)

        x3 = self.conv5(inputs)
        x3 = self.pool5(x3)
        x3 = self.bn5(x3)

        x3 = self.conv6(x3)
        x3 = self.pool6(x3)
        x3 = self.bn6(x3)

        x1 = self.global_avg_pool(x1)
        x2 = self.global_avg_pool(x2)
        x3 = self.global_avg_pool(x3)

        merged = layers.Concatenate(axis=-1)([x1, x2, x3])

        x = self.dense1(merged)
        x = self.dropout(x)
        x = self.dense2(x)

        return x


# 使用给定的输入形状和 H 来创建模型
# input_shape = (28, 28, 1)  # 假设输入图像尺寸为 28x28，通道数为 1
# H = 32  # 输出维度 H
# custom_model = CustomModel(input_shape, H)
