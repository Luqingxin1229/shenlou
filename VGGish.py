import tensorflow as tf
import numpy as np
from tensorflow.keras import models,metrics,layers,losses
from dataloader_preprocess import data_generator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))  # 启用设备的GPU进行训练

"""
路径及参数
"""
# root_dir = r'UrbanSound8k/UrbanSound8K/audio'  # 音频文件根目录
# excel_dir = r'UrbanSound8k/UrbanSound8K/metadata/UrbanSound8K.csv'  # csv文件目录

"""
data generator
"""
# train_data_loader = data_generator(root_dir=root_dir, excel_dir=excel_dir, target_sr=10_000, target_samples=50_000,
#                               batch_size=100, shuffle=True, spectrogram_dim=(128, 45), n_class=10)

"""
VGGish model architecture
"""

def VGGish_train(x_train, y_train,batch_size, epoch, name):
    # Input_shape = ()
    inputs = layers.Input(shape=(128, 98, 3))
    # convolution
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="conv1")(inputs)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool1')(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4')(x)
    x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool2')(x)
    """ x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv6')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool3')(x)"""
    # fully connection
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    # tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='sparse_categorical_crossentropy',
                    metrics=['sparse_categorical_accuracy'])
    model.summary()

    """model.fit_generator(generator=train_data_loader,
                        epochs=50)"""
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epoch)
    model.save(name)
    print('model saved successfully')

# VGGish_train()