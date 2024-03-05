import keras.backend
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import task
from math import ceil
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
"""
Deep SVDD
Qingxin_Lu_Jiangnan University
2023/6/13
"""
"""
文件及参数说明：
deep_model: embedding tool
input_shape: the shape of the input data
objective:running mode
lr: learning rate
"""
class DeepSVDD:
    def __init__(self, deep_model, input_shape=(), objective='one_class',
                 nu=0.1, representation_dim=32, batch_size=128, lr=1e-3):
        self.representation_dim = representation_dim  # The dimensions of the embedded space
        self.deep_model = deep_model  # embedding tool
        self.objective = objective  # running mode
        self.nu = nu  # Hyperparameters: v*n   where n is the number of the data in the training set
        self.batch_size = batch_size
        self.R = tf.compat.v1.get_variable('R', [], dtype=tf.float32, trainable=False)  # create Tensor 'R'
        self.c = tf.compat.v1.get_variable('c', [self.representation_dim], dtype=tf.float32, trainable=False)
        # create Tensor 'c' and initialize
        self.warm_up_n_epochs = 10  # warm up phase

        with task('Build graph'):  # create graph
            self.x = tf.compat.v1.placeholder(tf.float32, [None] + list(input_shape))
            # The memory needed to build
            self.latent_op = self.deep_model(self.x)  # latent space
            self.dist_op = tf.reduce_sum(tf.square(self.latent_op - self.c), axis=-1)
            # 经过embedding后的点距超球面球心c的distance

            if self.objective == 'soft-boundary':
                self.score_op = self.dist_op - self.R ** 2  # score role
                penalty = tf.maximum(self.score_op, tf.zeros_like(self.score_op))
                # Penalties for points located outside the ball
                self.loss_op = self.R ** 2 + (1/self.nu) * penalty  # loss function

            else:  # one_class
                self.score_op = self.dist_op  # score role
                self.loss_op = self.score_op  # loss function

            opt = tf.compat.v1.train.AdamOptimizer(lr)  # Adam optimizer
            self.train_op = opt.minimize(self.loss_op)  # Optimize

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # 按需分配GPU内存
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())  # 创建一个会话，并初始化所有的全局变量。
        # 创建会话后可以使用该会话来执行TensorFlow图中的操作和计算

    def __del__(self):
        self.sess.close()  # 在该类实例被销毁前关闭会话

    def fit(self, X, X_test, y_test, epochs=10, verbose=True):
        # X:training data; X_test:testing data; y_test:testing labels
        N = X.shape[0]  # the number of the data in the training set
        BS = self.batch_size
        BN = int(ceil(N / BS))  # batch number

        self.sess.run(tf.compat.v1.global_variables_initializer())  # 初始化所有全局变量
        self._init_c(X)  # 初始化c值

        ops = {
            'train': self.train_op,
            'loss': tf.reduce_mean(self.loss_op),
            'dist': self.dist_op
        }
        keras.backend.set_learning_phase(True)  # training phrase

        for i_epoch in range(epochs):
            index = np.random.permutation(N)  # 对0-N的数字进行随机排序
            x_train = X[index]
            g_batch = tqdm(range(BN)) if verbose else range(BN)  # verbose=1时展示训练进度条
            for i_batch in g_batch:
                x_batch = x_train[i_batch * BS: (i_batch + 1) * BS]
                results = self.sess.run(ops, feed_dict={self.x: x_batch})  # 将x_batch传给占位符self.x
                # ops：训练过程中获取该字典中的值； feed_dict：喂入数据

                if self.objective == 'soft-boundary' and i_epoch >= self.warm_up_n_epochs:
                    self.sess.run(tf.compat.v1.assign(self.R, self._get_R(results['dist'], self.nu)))  # 更新R值

            else:
                if verbose:
                    pred = self.predict(X_test)
                    auc = roc_auc_score(y_test, -pred)  # AUC值
                    print('\rEpoch: %3d AUROC: %.3f' % (i_epoch, auc))  # 输出该epoch的预测结果

    def predict(self, X):
        N = X.shape[0]  # training data number
        BS = self.batch_size
        BN = int(ceil(N / BS))  # batch number
        scores = list()
        keras.backend.set_learning_phase(False)  # testing phrase

        for i_batch in range(BN):
            x_batch = X[i_batch * BS: (i_batch + 1) * BS]
            s_batch = self.sess.run(self.score_op, feed_dict={self.x: x_batch})  # 获取打分值
            scores.append(s_batch)

        return np.concatenate(scores)

    def _init_c(self, X, eps=1e-1):  # initialize 球心 c
        N = X.shape[0]  # number of training data
        BS = self.batch_size
        BN = int(ceil(N / BS))  # batch number
        keras.backend.set_learning_phase(False)  # F:test   T:train

        with task('1. Get output'):
            latent_sum = np.zeros(self.latent_op.shape[-1])
            for i_batch in range(BN):
                x_batch = X[i_batch * BS: (i_batch + 1) * BS]  # i-th batch
                latent_v = self.sess.run(self.latent_op, feed_dict={self.x: x_batch})
                latent_sum += latent_v.sum(axis=0)  # latent space中所有点相加

            c = latent_sum / N  # mean

        with task('2. Modify eps'):
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

        self.sess.run(tf.compat.v1.assign(self.c, c))  # 将c替换为新的c值

    def _get_R(self, dist, nu):
        return np.quantile(np.sqrt(dist), 1 - nu)  # 返回1-nu分位数
        # 这里取分位数是为了将正常数据与异常数据分开  nu是异常值的比例

svdd = DeepSVDD(1, input_shape=(28, 28, 1), representation_dim=32,
                    objective='soft-boundary')