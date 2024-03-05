import logging
import tensorflow as tf
from tensorflow.keras import layers

class BaseNet(tf.keras.Model):
    """所有神经网络的基类"""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the code layer or last layer

    def call(self, inputs, training=False):
        """
        前向传播逻辑
        :param inputs: 输入数据
        :param training: 布尔值，指示模型是否处于训练模式
        :return: 网络输出
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        trainable_params = sum([tf.reduce_prod(p.shape) for p in self.trainable_variables])
        self.logger.info(f'Trainable parameters: {trainable_params}')
        self.logger.info(self)

class CustomModel_1(BaseNet):
    def __init__(self, rep_dim=32):
        # super(CustomModel_1, self).__init__()
        super().__init__()

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

    def forward(self, inputs):
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

class CustomModel_1_Decoder(BaseNet):
    def __init__(self, input_shape):
        # super(CustomModel_1_Decoder, self).__init__()
        super().__init__()

        # Define the layers for the decoder
        self.dense1 = layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.2))

        # Reshape the output to match the shape of the last convolutional layer
        self.reshape = layers.Reshape((7, 7, 16))  # Adjust the shape according to your encoder

        # Add (3, 1) convolutional layer
        self.deconv1 = layers.Conv2DTranspose(64, (3, 1), strides=(2, 2), padding='same',
                                              activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.bn1 = layers.BatchNormalization(epsilon=1e-4)

        # Add (5, 1) convolutional layer
        self.deconv2 = layers.Conv2DTranspose(32, (5, 1), strides=(2, 2), padding='same',
                                              activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.bn2 = layers.BatchNormalization(epsilon=1e-4)

        # Add (7, 1) convolutional layer
        self.deconv3 = layers.Conv2DTranspose(16, (7, 1), strides=(2, 2), padding='same',
                                              activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.bn3 = layers.BatchNormalization(epsilon=1e-4)

        # Final convolutional layer to match the input shape
        self.deconv4 = layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same',
                                              activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)

        # Reshape the output
        x = self.reshape(x)

        # Apply (3, 1) convolutional layer
        x = self.deconv1(x)
        x = self.bn1(x)

        # Apply (5, 1) convolutional layer
        x = self.deconv2(x)
        x = self.bn2(x)

        # Apply (7, 1) convolutional layer
        x = self.deconv3(x)
        x = self.bn3(x)

        # Apply the final convolutional layer to match input shape
        x = self.deconv4(x)

        return x


class MNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = CustomModel_1(rep_dim=rep_dim)
        self.decoder = CustomModel_1_Decoder(rep_dim=rep_dim)

    def call(self, x, training=False):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
