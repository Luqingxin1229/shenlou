import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

input_shape=(20, 100, 3)
# 最优
class multi_CNN_base(Model):
    def __init__(self, req_dim=32):
        super(multi_CNN_base, self).__init__()
        """
        :param rep_dim: 超平面维度  default：32
        """
        self.req_dim = req_dim
        self.conv1_1 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', input_shape=input_shape),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D()
        ])
        self.conv1_2 = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D()
        ])

        self.conv2_1 = tf.keras.Sequential([
            layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', input_shape=input_shape),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D()
        ])
        self.conv2_2 = tf.keras.Sequential([
            layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D()
        ])

        self.conv3_1 = tf.keras.Sequential([
            layers.Conv2D(64, (7, 7), strides=(1, 1), padding='same', input_shape=input_shape),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D()
        ])
        self.conv3_2 = tf.keras.Sequential([
            layers.Conv2D(128, (7, 7), strides=(1, 1), padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D()
        ])

        self.GAP = layers.GlobalAveragePooling2D()
        self.concatenate = layers.Concatenate(axis=-1)
        self.fc_1 = layers.Dense(128)
        self.leakyrelu = layers.LeakyReLU(alpha=0.2)
        self.Dropout = layers.Dropout(0.5)
        self.fc_2 = layers.Dense(req_dim)

    def call(self, x):
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)

        x3 = self.conv3_1(x)
        x3 = self.conv3_2(x3)

        x1 = self.GAP(x1)
        x2 = self.GAP(x2)
        x3 = self.GAP(x3)

        merged = self.concatenate([x1, x2, x3])
        merged = self.fc_1(merged)
        merged = self.leakyrelu(merged)
        merged = self.Dropout(merged)
        merged = self.fc_2(merged)
        return merged

class multi_CNN_Decoder_base(Model):
    def __init__(self, req_dim=32):
        super(multi_CNN_Decoder_base, self).__init__()

        self.req_dim = req_dim

        self.fc_1 = layers.Dense(128)
        self.fc_2 = layers.Dense(128*3)
        self.fc_3 = layers.Dense(128*5*25)
        self.deconv1_1 = tf.keras.Sequential([
            layers.Reshape((5, 25, 128)),
            layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.deconv1_2 = tf.keras.Sequential([
            layers.Reshape((5, 25, 128)),
            layers.Conv2DTranspose(128, (5, 5), strides=2, padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.deconv1_3 = tf.keras.Sequential([
            layers.Reshape((5, 25, 128)),
            layers.Conv2DTranspose(128, (7, 7), strides=2, padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2DTranspose(64, (7, 7), strides=2, padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.concatenate = layers.Concatenate(axis=-1)
        self.deconv2 = tf.keras.Sequential([
            layers.Conv2DTranspose(3, (1, 1), strides=1, padding='same')
        ])

    def call(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        x1 = self.deconv1_1(x)
        x2 = self.deconv1_2(x)
        x3 = self.deconv1_3(x)

        merged = self.concatenate([x1, x2, x3])
        merged = self.deconv2(merged)
        return merged