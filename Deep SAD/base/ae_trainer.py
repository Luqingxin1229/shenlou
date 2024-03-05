import logging
import time
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import roc_auc_score

class AETrainer():
    def __init__(self, optimizer_name='adam', lr=0.001, n_epochs=150, lr_milestones=(), batch_size=128,
                 weight_decay=1e-6, device='cuda', n_jobs_dataloader=0):
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None

    def train(self, dataset, ae_net):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = MeanSquaredError()

        # Set device
        ae_net = ae_net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = Adam(learning_rate=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        def lr_scheduler(epoch):
            if epoch in self.lr_milestones:
                return self.lr * 0.1
            return self.lr

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.compile(optimizer=optimizer, loss=criterion)
        ae_net.fit(train_loader, epochs=self.n_epochs, callbacks=[lr_callback], verbose=0)

        self.train_time = time.time() - start_time
        logger.info('Pretraining Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset, ae_net):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = MeanSquaredError()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Testing
        logger.info('Testing autoencoder...')
        start_time = time.time()
        idx_label_score = []
        ae_net.evaluate(test_loader, verbose=0)

        epoch_loss = ae_net.test_on_batch(test_loader)
        n_batches = len(test_loader)

        with tf.device('cpu'):
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs = inputs.to(self.device)
                rec = ae_net(inputs, training=False)
                rec_loss = criterion(inputs, rec)
                scores = tf.math.reduce_mean(rec_loss, axis=tuple(range(1, rec.ndim)))

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.numpy().tolist(), labels.numpy().tolist(), scores.numpy().tolist()))

        self.test_time = time.time() - start_time

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing autoencoder.')
