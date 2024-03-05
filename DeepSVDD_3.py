import tensorflow as tf
from tensorflow.keras import layers, models, initializers, regularizers, optimizers
from sklearn.metrics import roc_auc_score
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from Deep_SVDD_data_pro import audio_data_to_net_pro



class DeepSVDD:
    def __init__(self, input_shape, latent_dim, nu=0.1, mode='one-class'):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.nu = nu
        self.mode = mode
        self.model = self.build_model()

    def build_model(self):
        # Input_shape = ()
        inputs = layers.Input(shape=self.input_shape)
        # convolution
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                          name="conv1")(inputs)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                          name='conv2')(x)
        x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
        x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool1')(x)
        x = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same',
                          name='conv3')(x)  # c1
        x = layers.Conv2D(filters=512, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same',
                          name='conv4')(x)  # c2
        x = layers.BatchNormalization(epsilon=1e-4, trainable=False)(x)
        x = layers.MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool2')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.latent_dim)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, data, epochs=50, batch_size=64):
        if self.mode == 'one-class':
            self.model.compile(optimizer=optimizers.Adam(), loss='mse')
            self.model.fit(data, data, batch_size=batch_size, epochs=epochs)
        elif self.mode == 'soft-boundary':
            self.model.compile(optimizer=optimizers.Adam(), loss=self.get_hinge_loss)
            self.model.fit(data, np.ones(len(data)), batch_size=batch_size, epochs=epochs)

    def get_hinge_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.nn.relu(1.0 - y_true * y_pred))

    def get_center(self, data, batch_size=64):
        return np.mean(self.model.predict(data, batch_size=batch_size), axis=0)

    def predict(self, data, batch_size=64):
        distances = np.linalg.norm(data - self.get_center(data, batch_size=batch_size), axis=1)
        if self.mode == 'one-class':
            threshold = np.percentile(distances, 100 * (1 - self.nu))
            return distances, distances > threshold
        elif self.mode == 'soft-boundary':
            scores = distances - np.percentile(distances, 100 * (1 - self.nu))
            return distances, scores

    def compute_auc(self, data, labels, batch_size=64):
        _, scores = self.predict(data, batch_size=batch_size)
        return roc_auc_score(labels, scores)


if __name__ == "__main__":
    # Generate some example data
    data = np.random.rand(1000, 50)

    # Labels for the data (1 for normal, 0 for anomaly)
    labels = np.random.choice([0, 1], size=(1000,), p=[0.1, 0.9])

    # Split the data into training and test sets
    # data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    data_train, labels_train = audio_data_to_net_pro(excel_dir='air_c.csv', target_samples=50000, target_sr=22050,
                          root_dir='UrbanSound8k/UrbanSound8K/audio', hop_size=20000, n_mfcc=20)  # 50_000, 10_000 ub
    x_test, y_test= audio_data_to_net_pro(excel_dir='air_test.csv', target_samples=50000, target_sr=22050,
                               root_dir='UrbanSound8k/UrbanSound8K/audio', hop_size=20000, n_mfcc=20)
    # Create and train Deep SVDD model in one-class mode
    deep_svdd_one_class = DeepSVDD(input_shape=(50,), latent_dim=20, nu=0.1, mode='one-class')

    epochs = 50
    batch_size = 64

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Shuffle the training data and labels before each epoch
        idx = np.arange(len(data_train))
        np.random.shuffle(idx)
        data_train = data_train[idx]
        labels_train = labels_train[idx]

        num_batches = len(data_train) // batch_size

        for batch_idx in tqdm(range(num_batches)):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size

            batch_data = data_train[start:end]
            batch_labels = labels_train[start:end]

            deep_svdd_one_class.train(batch_data, epochs=1, batch_size=batch_size)

        # Compute AUC for test set
        auc_test = deep_svdd_one_class.compute_auc(x_test, y_test)
        print("Test AUC:", auc_test)