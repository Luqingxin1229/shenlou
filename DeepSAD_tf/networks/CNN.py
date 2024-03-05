import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

input_shape=(20, 100, 3)

class CNN_0(Model):
    def __init__(self, req_dim=32):
        super(CNN_0, self).__init__()
        """
        :param rep_dim: 超平面维度  default：32
        """
        self.req_dim = req_dim
        self.conv1_1 = tf.keras.Sequential([
            layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', input_shape=input_shape),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D()
        ])
        self.conv1_2 = tf.keras.Sequential([
            layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D()
        ])

        self.GAP = layers.GlobalAveragePooling2D()
        # self.fl = layers.Flatten()
        # self.fc_1 = layers.Dense(128)
        # self.fc_1 = layers.Dense(128)
        # self.leakyrelu = layers.LeakyReLU(alpha=0.2)
        # self.Dropout = layers.Dropout(0.5)
        self.fc_2 = layers.Dense(req_dim)

    def call(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)

        x = self.GAP(x)
        merged = self.fc_2(x)
        return merged

class CNN_Decoder(Model):
    def __init__(self, req_dim=32):
        super(CNN_Decoder, self).__init__()

        self.req_dim = req_dim

        self.fc_1 = layers.Dense(128)
        self.fc_2 = layers.Dense(128*5*25)
        self.deconv1_1 = tf.keras.Sequential([
            layers.Reshape((5, 25, 128)),
            layers.Conv2DTranspose(128, (5, 5), strides=2, padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same'),
            layers.BatchNormalization(epsilon=1e-4),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.deconv2 = tf.keras.Sequential([
            layers.Conv2DTranspose(3, (1, 1), strides=1, padding='same')
        ])

    def call(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.deconv1_1(x)

        merged = self.deconv2(x)
        return merged