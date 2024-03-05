import keras.models
from tensorflow.keras import models,metrics,layers,losses
import tensorflow as tf
import os
import numpy as np
from keras import regularizers

__all__ = ['VGGish', 'VGGish_1', 'VGGish_2']

# 自定义注意力层
class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W_q = self.add_weight("W_q", shape=(input_shape[-1], input_shape[-1]))
        self.W_k = self.add_weight("W_k", shape=(input_shape[-1], input_shape[-1]))
        self.W_v = self.add_weight("W_v", shape=(input_shape[-1], input_shape[-1]))
        self.b_q = self.add_weight("b_q", shape=(input_shape[-1],))
        self.b_k = self.add_weight("b_k", shape=(input_shape[-1],))
        self.b_v = self.add_weight("b_v", shape=(input_shape[-1],))

    def call(self, inputs):
        q = tf.matmul(inputs, self.W_q) + self.b_q
        k = tf.matmul(inputs, self.W_k) + self.b_k
        v = tf.matmul(inputs, self.W_v) + self.b_v

        attention_weights = tf.matmul(q, k, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        output = tf.matmul(attention_weights, v)
        return output

def VGGish(inputshape, H):
    # Input_shape = ()
    inputs = layers.Input(shape=inputshape)
    # convolution
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="conv1")(inputs)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool1')(x)
    x = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', name='conv3')(x)  # c1
    x = layers.Conv2D(filters=512, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', name='conv4')(x)  # c2
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool2')(x)
    # x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5')(x)
    # x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv6')(x)
    # x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool3')(x)
    # fully connection
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(H)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    # model.save('D_VGGish.h5')

    return model

# v = VGGish((3, 20, 45))

def VGGish_1(inputshape):
    # Input_shape = ()
    inputs = layers.Input(shape=inputshape)
    # convolution
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="conv1")(inputs)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool1')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4')(x)
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool2')(x)
    # x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5')(x)
    # x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv6')(x)
    # x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    # x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool3')(x)
    # fully connection
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(64)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    # model.save('D_VGGish.h5')

    return model

# v = VGGish((3, 20, 45))

def VGGish_2(inputshape):
    # Input_shape = ()
    inputs = layers.Input(shape=inputshape)
    # convolution
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="conv1")(inputs)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool1')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4')(x)
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool2')(x)
    # x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5')(x)
    # x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv6')(x)
    # x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    # x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool3')(x)
    # fully connection
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(64)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    # model.save('D_VGGish.h5')

    return model

# v = VGGish((20, 216, 3))
def VGGish_multi(inputshape, H):
    # Input_shape = ()
    inputs = layers.Input(shape=inputshape)
    # convolution
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                      name="conv1")(inputs)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                      name='conv2')(x)
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool1')(x)

    # Multi-scale convolution
    x_3x3 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                          name='conv3_3x3')(x)
    x_5x5 = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same',
                          name='conv4_5x5')(x)
    x_7x7 = layers.Conv2D(filters=256, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same',
                          name='conv5_7x7')(x)

    # Merge multi-scale convolution outputs
    x = layers.concatenate([x_3x3, x_5x5, x_7x7], axis=-1)
    x = layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool2')(x)
    # x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5')(x)
    # x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv6')(x)
    # x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool3')(x)
    # fully connection
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(H)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    # model.save('D_VGGish.h5')

    return model

def model_test(inputshape, H):
    inputs = layers.Input(shape=inputshape)

    # 3*3
    branch1 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)
    # 5*5
    branch2 = layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same')(inputs)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)
    # 7*7
    branch3 = layers.Conv2D(64, (7, 7), strides=(1, 1), activation='relu', padding='same')(inputs)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)

    merged = layers.Concatenate()([branch1, branch2, branch3])
    merged = layers.Conv2D(32, (1, 1), strides=(1, 1), activation='relu', padding='same')(merged)
    x1 = layers.Flatten()(merged)
    x2 = layers.Dense(2048, activation='relu')(x1)
    x3 = layers.Dropout(0.5)(x2)
    x3 = layers.Dense(1024, activation='relu')(x3)
    x3 = layers.Dropout(0.5)(x3)

    x = layers.Dense(128, activation='relu')(x3)
    x = layers.Dropout(0.5)(x)
    # x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(H)(x)   # kernel_initializer='random_uniform'
    # outputs = layers.Dense(32)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
# model_test((20, 98, 3))

def model_test_m(inputshape, H):
    inputs = layers.Input(shape=inputshape)

    # 3*3

    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)

    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)

    # 5*5
    branch2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)

    branch2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)
    # 7*7
    branch3 = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)

    branch3 = layers.Conv2D(64, (7, 7), activation='relu', padding='same')(branch3)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)

    merged = layers.Concatenate()([branch1, branch2, branch3])

    # x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merged)
    # x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.BatchNormalization(epsilon=1e-4)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merged)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization(epsilon=1e-4)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(H)(x)
    # outputs = layers.Dense(32)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def model_test_m1(inputshape, H):
    inputs = layers.Input(shape=inputshape)
    # 添加注意力模块
    # ainputs= AttentionLayer()(inputs)
    # 3*3

    branch1 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)
    branch1 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)

    branch1 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(branch1)
    branch1 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)



    # 5*5
    branch2 = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(inputs)
    branch2 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)

    branch2 = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same')(branch2)
    branch2 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)


    # 7*7
    branch3 = layers.Conv2D(64, (7, 7), strides=(1, 1), padding='same')(inputs)
    branch3 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch3)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)

    branch3 = layers.Conv2D(128, (7, 7), strides=(1, 1), padding='same')(branch3)
    branch3 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch3)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)


    x1 = layers.GlobalAveragePooling2D()(branch1)
    x2 = layers.GlobalAveragePooling2D()(branch2)
    x3 = layers.GlobalAveragePooling2D()(branch3)

    merged = layers.Concatenate(axis=-1)([x1, x2, x3])  # [:,np.newaxis]

    x = layers.Dense(128)(merged)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(H)(x)   # kernel_initializer='random_uniform'
    # outputs = layers.Dense(32)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model
# model = model_test_m1((20, 98, 3), H=32)

def model_test_m2(inputshape, H):
    inputs = layers.Input(shape=inputshape)

    # 3*3

    branch1 = layers.Conv2D(16, (1, 3), strides=(1, 1), padding='same')(inputs)
    branch1 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)

    branch1 = layers.Conv2D(32, (1, 3), strides=(1, 1), padding='same')(branch1)
    branch1 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)

    branch1 = layers.Conv2D(64, (1, 3), strides=(1, 1), padding='same')(branch1)
    branch1 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.BatchNormalization(epsilon=1e-4)(branch1)

    # 5*5
    branch2 = layers.Conv2D(16, (1, 5), strides=(1, 1), padding='same')(inputs)
    branch2 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)

    branch2 = layers.Conv2D(32, (1, 5), strides=(1, 1), padding='same')(branch2)
    branch2 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)

    branch2 = layers.Conv2D(64, (1, 5), strides=(1, 1), padding='same')(branch2)
    branch2 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.BatchNormalization(epsilon=1e-4)(branch2)
    # 7*7
    branch3 = layers.Conv2D(16, (1, 7), strides=(1, 1), padding='same')(inputs)
    branch3 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch3)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)

    branch3 = layers.Conv2D(32, (1, 7), strides=(1, 1), padding='same')(branch3)
    branch3 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch3)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)

    branch3 = layers.Conv2D(64, (1, 7), strides=(1, 1), padding='same')(branch3)
    branch3 = tf.keras.layers.LeakyReLU(alpha=0.2)(branch3)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.BatchNormalization(epsilon=1e-4)(branch3)

    x1 = layers.GlobalAveragePooling2D()(branch1)
    x2 = layers.GlobalAveragePooling2D()(branch2)
    x3 = layers.GlobalAveragePooling2D()(branch3)

    merged = layers.Concatenate(axis=-1)([x1, x2, x3])  # [:,np.newaxis]

    x = layers.Dense(128)(merged)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(H)(x)   # kernel_initializer='random_uniform'
    # outputs = layers.Dense(32)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model
"""
def VGGish(inputshape):
    model = keras.models.Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                            name="conv1", input_shape=inputshape))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='conv2'))
    model.add(layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool1'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='conv3'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='conv4'))
    model.add(layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool2'))
    #model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                            #name='conv5'))
    #model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                            #name='conv6'))
    #model.add(layers.BatchNormalization(epsilon=1e-4, trainable=False))
    #model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool3'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(4096, activation='relu'))
    #model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(128))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64))

    return model
"""