from tensorflow.keras import layers, models
from VGGish import VGGish_train
from Deep_SVDD_data_pro import audio_data_to_net_pro
from SVDD_models import VGGish, VGGish_1, VGGish_2, VGGish_multi, model_test, model_test_m, model_test_m1
from sklearn.metrics import roc_auc_score
from Deep_SVDD_2 import DeepSVDD
import numpy as np
import tensorflow as tf



def create_conv_model():
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
    """x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv6')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool3')(x)"""
    model = models.Model(inputs=inputs, outputs=x)
    return model


def get_conv_output(input_data, model_path):
    conv_model = create_conv_model()
    # Load the weights from the pretrained VGGish model
    conv_model.load_weights(model_path, by_name=True)

    conv_output = conv_model.predict(input_data)
    return conv_output

if __name__ == '__main__':
    create_conv_model()
    samples_sr = 22050
    nmfcc = None
    deep_model_name = 'VGGish_SVDD_885.h5'
    # 数据预处理
    x_train, y_train = audio_data_to_net_pro(excel_dir='VGGish_SVDD/VGGish_SVDD_z0.csv',target_samples=50000,
                                             target_sr=samples_sr, root_dir='VGGish_SVDD', hop_size=20000, n_mfcc=nmfcc)
    x_train_svdd, y_train_svdd = audio_data_to_net_pro(excel_dir='normal_LD18/normal_LD18_2.csv',target_samples=50000,
                                             target_sr=samples_sr, root_dir='normal_LD18', hop_size=20000, n_mfcc=nmfcc)
    x_test, y_test = audio_data_to_net_pro(excel_dir='LD_18_test/LD18_test.csv',target_samples=50000,
                                             target_sr=samples_sr, root_dir='LD_18_test', hop_size=20000, n_mfcc=nmfcc)

    # VGGish训练及特征提取
    VGGish_train(x_train=x_train, y_train=y_train, batch_size=32, epoch=25, name=deep_model_name)
    conv_output = get_conv_output(x_train_svdd, deep_model_name)
    x_test_conv = get_conv_output(x_test, deep_model_name)

    # Deep SVDD训练
    model = model_test(inputshape=(32, 25, 256), H=32)
    svdd = DeepSVDD(deep_model=model, objective='one-class', batch_size=64, nu=0.1,
                    representation_dim=32)
    auc = 0
    svdd.fit(conv_output, x_test_conv, y_test, epochs=20, verbose=True)
    print(svdd.c)
    # test Deep SVDD
    score = svdd.predict(x_test_conv)
    auc = roc_auc_score(y_test, score)
    _ = svdd.predict_fcn(np.zeros((1, 32, 25, 256)))  # the same as model above
    tf.saved_model.save(svdd, 'deep_svdd_models/VGGish+deep_svdd_8_8_5')
    print(score)
    print('AUROC: %.3f' % auc)

