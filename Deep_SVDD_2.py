import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
from math import ceil
from tqdm import tqdm

class DeepSVDD(tf.keras.Model):
    def __init__(self, deep_model, objective='one_class',
                 nu=0.1, representation_dim=32, batch_size=128, lr=1e-3):
        super(DeepSVDD, self).__init__()
        self.representation_dim = representation_dim
        self.deep_model = deep_model
        self.objective = objective
        self.nu = nu
        self.batch_size = batch_size
        self.R = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.c = tf.Variable(tf.zeros([self.representation_dim]), dtype=tf.float32, trainable=False)
        self.warm_up_n_epochs = 10

        self.optimizer = tf.keras.optimizers.Adam(lr)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    @tf.function
    def predict_fcn(self, x):
        return self.call(x)['score']

    def call(self, x):
        z = self.deep_model(x)
        dist = tf.reduce_sum(tf.square(z - self.c), axis=-1)
        if self.objective == 'soft-boundary':
            score = dist - self.R ** 2
            loss = self.R ** 2 + (1/self.nu) * tf.maximum(score, tf.zeros_like(score))
        else:
            score = dist
            loss = score
        return {'score': score, 'loss': loss, 'dist': dist}

    def train_step(self, x_batch):
        with tf.GradientTape() as tape:
            results = self.call(x_batch)
            loss = tf.reduce_mean(results['loss'])
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return results

    def fit(self, X, X_test, y_test, epochs=10, verbose=True):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))

        self._init_c(X)

        for i_epoch in range(epochs):
            index = np.random.permutation(N)
            x_train = X[index]
            g_batch = tqdm(range(BN)) if verbose else range(BN)
            for i_batch in g_batch:
                x_batch = x_train[i_batch * BS: (i_batch + 1) * BS]
                results = self.train_step(x_batch)
                if self.objective == 'soft-boundary' and i_epoch >= self.warm_up_n_epochs:
                    self.R.assign(self._get_R(results['dist'], self.nu))
            else:
                if verbose:
                    pred = self.predict(X_test)
                    auc = roc_auc_score(y_test, pred)
                    print('\rEpoch: %3d AUROC: %.3f' % (i_epoch, auc))


    def predict(self, X):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))
        scores = []

        for i_batch in range(BN):
            x_batch = X[i_batch * BS: (i_batch + 1) * BS]
            s_batch = self.call(x_batch)['score']
            scores.append(s_batch)

        return np.concatenate(scores)

    def _init_c(self, X, eps=1e-1):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))

        with tf.keras.backend.learning_phase_scope(0):
            latent_sum = np.zeros(self.representation_dim)
            for i_batch in range(BN):
                x_batch = X[i_batch * BS: (i_batch + 1) * BS]
                latent_v = self.deep_model(x_batch)
                latent_sum += np.sum(latent_v, axis=0)

            c = latent_sum / N

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.c.assign(c)

    def _get_R(self, dist, nu):
        return np.quantile(np.sqrt(dist), 1 - nu)


# 示例用法
"""
input_shape = (3, 128, 147)
representation_dim = 32
batch_size = 128
lr = 1e-3
deep_model = ...  # 定义你的深度模型
X = ...  # 训练数据
X_test = ...  # 测试数据
y_test = ...  # 测试标签
epochs = ...
nu = ...
verbose = True

deep_svdd = DeepSVDD(deep_model, input_shape, 'one_class', nu, representation_dim, batch_size, lr)
deep_svdd.fit(X, X_test, y_test, epochs, verbose)
predictions = deep_svdd.predict(X_test)
"""
