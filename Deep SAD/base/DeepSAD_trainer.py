import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam

import logging
import time
from base_trainer import BaseTrainer
from base_dataset import BaseADDataset
from base_net import BaseNet


class DeepSADTrainer(BaseTrainer):
    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = tf.convert_to_tensor(c, dtype=tf.float32) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None


    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # 获取训练数据加载器
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # 为网络设置设备
        net = net.to(self.device)

        # 设置优化器（目前使用 Adam 优化器）
        optimizer = Adam(learning_rate=self.lr, weight_decay=self.weight_decay)
        net.compile(optimizer=optimizer, loss='mean_squared_error')

        # 设置学习率调度器
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=len(train_loader) * 5,  # 假设每个 epoch 有 5 个步骤
            decay_rate=0.1
        )

        # 初始化超球中心 c（如果 c 未加载）
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # 训练
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()  # +
            if epoch in self.lr_milestones:
                # logger.info('  LR scheduler: new learning rate is %g' % float(self.lr * 0.1))
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
                optimizer.learning_rate.assign(self.lr * 0.1)

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # 清除网络参数的梯度
                with tf.GradientTape() as tape:
                    outputs = net(inputs)
                    dist = tf.reduce_sum((outputs - self.c) ** 2, axis=1)
                    losses = tf.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets))
                    loss = tf.reduce_mean(losses)

                grads = tape.gradient(loss, net.trainable_variables)
                optimizer.apply_gradients(zip(grads, net.trainable_variables))

                epoch_loss += loss.numpy()
                n_batches += 1

            # 记录每个 epoch 的统计信息
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # 获取测试数据加载器
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # 为网络设置设备
        net = net.to(self.device)

        # 测试
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        for data in test_loader:
            inputs, labels, semi_targets, idx = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            semi_targets = semi_targets.to(self.device)
            idx = idx.to(self.device)

            with tf.GradientTape() as tape:
                outputs = net(inputs)
                dist = tf.reduce_sum((outputs - self.c) ** 2, axis=1)
                losses = tf.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets))
                loss = tf.reduce_mean(losses)

            scores = dist

            idx_label_score += list(zip(idx.numpy().tolist(), labels.numpy().tolist(), scores.numpy().tolist()))

            epoch_loss += loss.numpy()
            n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score


        # 计算AUROC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # 记录结果
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')

    def init_center_c(self, train_loader, net, eps=0.1):
        """将超球中心 c 初始化为数据的初始前向传递的均值。"""
        n_samples = 0
        c = tf.zeros(net.rep_dim, dtype=tf.float32)

        net.compile(optimizer=Adam(learning_rate=0), loss='mean_squared_error')  # 使用 Adam 优化器进行前向传播
        for data in train_loader:
            inputs, _, _, _ = data
            inputs = inputs.to(self.device)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += tf.reduce_sum(outputs, axis=0)

        c /= n_samples

        # 如果 c_i 太接近于 0，则设置为+-eps。原因：零单元可以与零权重轻松匹配。
        c = tf.where(tf.abs(c) < eps, eps * tf.sign(c), c)

        return c.numpy()
