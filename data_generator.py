import numpy as np
import keras
from keras.layers import Dense


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, train_data, test_data, batch_size=32, n_channels=1, shuffle=True):
        'Initialization'
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.train_data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        # X, y = self.__data_generation(train_data_temp)

        X = self.train_data[indexes]
        y = self.test_data[indexes]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def gen_model():

    model = keras.models.Sequential()

    model.add(Dense(units=1, use_bias=False, input_shape=(1,)))  # 仅有的1个权重在这里

    return model


if __name__ == '__main__':

    # 数据比较简单，用 CPU 即可
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Parameters
    params = {'batch_size': 10,
              'n_channels': 1,
              'shuffle': False}

    # Datasets
    # partition =  # IDs
    # labels =  # Labels
    x = {}
    x['train'] = np.arange(100, dtype='int32')
    x['validation'] = np.arange(100, 120, dtype='int32')
    y = {}
    y['train'] = -x['train']
    y['test'] = -x['train']


    # Generators
    training_generator = DataGenerator(x['train'], y['train'], **params)
    # validation_generator = DataGenerator(x['validation'], y['test'], **params)

    # Design model
    model = gen_model()
    model.compile(loss='mse', optimizer='adam')

    # model.fit(x['train'], y['train'], epochs=1000, batch_size=10, verbose=2)

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        # validation_data=validation_generator,
                        epochs=2000,
                        verbose=2,
                        workers=6,
                        )

    print(model.layers[0].get_weights())