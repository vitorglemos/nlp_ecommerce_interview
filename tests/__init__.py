import numpy as np
import pandas as pd

import tensorflow as tf

from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import InputLayer
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split


class AgeModel:
    def __init__(self):
        self.models = None
        self.history = None
        self.X = None
        self.y = None

    def create_features(self, dataset_path: str):
        dataframe = pd.read_csv(dataset_path)
        dataframe['pixels'] = dataframe['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))

        self.X = np.array(dataframe['pixels'].tolist())
        self.X = self.X.reshape(self.X.shape[0], 48, 48, 1)
        self.y = dataframe['age']

    def fit_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.60, random_state=37)
        return  X_train, X_test, y_train, y_test
        #self.history = self.models.fit(
            #X_train, y_train, epochs=20, validation_split=0.1, batch_size=128,

       # )

    def fit_transform(self, data):
        data = np.array(data.tolist())
        data = data.reshape(data.shape[0], 48, 48, 1)
        print(data)

    def save_model(self):
        pass

    def load_model(self, path_model: str):
        self.models = tf.keras.models.load_model(path_model)

    def build_model(self):
        self.models = tf.keras.Sequential([
            InputLayer(input_shape=(48, 48, 1)),
            Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(rate=0.5),
            Dense(1, activation='relu')
        ])

        optimizer_sgd = tf.keras.optimizers.SGD(momentum=0.9)
        self.models.compile(optimizer=optimizer_sgd,
                            loss='mean_squared_error',
                            metrics=['mae'])

    def predict(self, data):
        #data = self.fit_transform(data)
        data = data.reshape(48, 48, 1)
        print(np.argmax(self.models.predict(data), axis=1))


def plot_samples():
    plt.figure(figsize=(16, 16))
    for i in range(1500, 1520):
        plt.subplot(5, 5, (i % 25) + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data['pixels'].iloc[i].reshape(48, 48))
        plt.xlabel(
            "Age:" + str(data['age'].iloc[i]) +
            "  Ethnicity:" + str(data['ethnicity'].iloc[i]) +
            "  Gender:" + str(data['gender'].iloc[i])
        )
    plt.show()


if __name__ == '__main__':
    model_manager = AgeModel()
    model_manager.create_features('C:\\Users\\vgama\\Downloads\\age_gender.csv')
    # model_manager.build_model()
    X_train, X_test, y_train, y_test = model_manager.fit_model()
    model_manager.load_model('C:\\Users\\vgama\\Downloads\\age_training_v1.h5')
    #dataframe = pd.read_csv('C:\\Users\\vgama\\Downloads\\age_gender.csv')
    #dataframe['pixels'] = dataframe['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))

   # X = np.array(dataframe['pixels'].tolist())
    #X = X.reshape(X.shape[0], 48, 48, 1)
   # print(X[0])
    model_manager.predict(X_test[4])
    if False:
        data = pd.read_csv('C:\\Users\\vgama\\Downloads\\age_gender.csv')
        data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))

        print('Total rows: {}'.format(len(data)))
        print('Total columns: {}'.format(len(data.columns)))

        X = np.array(data['pixels'].tolist())

        ## Converting pixels from 1D to 3D
        X = X.reshape(X.shape[0], 48, 48, 1)

        y = data['age']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=37
                                                            )

        model = tf.keras.Sequential([
            L.InputLayer(input_shape=(48, 48, 1)),
            L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            L.BatchNormalization(),
            L.MaxPooling2D((2, 2)),
            L.Conv2D(64, (3, 3), activation='relu'),
            L.MaxPooling2D((2, 2)),
            L.Conv2D(128, (3, 3), activation='relu'),
            L.MaxPooling2D((2, 2)),
            L.Flatten(),
            L.Dense(64, activation='relu'),
            L.Dropout(rate=0.5),
            L.Dense(1, activation='relu')
        ])

        sgd = tf.keras.optimizers.SGD(momentum=0.9)

        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mae'])


        ## Stop training when validation loss reach 110
        class myCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('val_loss') < 110):
                    print("\nReached 110 val_loss so cancelling training!")
                    self.model.stop_training = True


        callback = myCallback()
        model.summary()

        history = model.fit(
            X_train, y_train, epochs=20, validation_split=0.1, batch_size=128,
        )

        fig = px.line(
            history.history, y=['loss', 'val_loss'],
            labels={'index': 'epoch', 'value': 'loss'},
            title='Training History')
        fig.show()
