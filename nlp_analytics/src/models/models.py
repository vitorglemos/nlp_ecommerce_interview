from keras.optimizers import Adam
from keras.optimizers import SGD

from keras.metrics import Precision, Recall
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import BatchNormalization

from keras.models import Sequential


def build_model_lstm_v3(input_len):
    max_words = 10000
    embed_dim = 128
    model = Sequential()

    model.add(Embedding(max_words, embed_dim, input_length=input_len))
    model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    model.add(Flatten())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_model_lstm_v6(input_len):
    learning_rate = 0.1
    decay_rate = learning_rate / 1
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    vocab_size = 5000
    embedding_size = 32
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=input_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy', Precision(), Recall()])
    return model


def build_model_lstm_v2_2(input_len):
    max_words = 10000
    embed_dim = 128
    model = Sequential()
    model.add(Embedding(max_words, embed_dim, input_length=input_len))
    model.add(LSTM(64))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy', Precision(), Recall()])
    return model

def build_model_lstm_v2(input_len):
    max_words = 10000
    embed_dim = 128
    model = Sequential()
    model.add(Embedding(max_words, embed_dim, input_length=input_len))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy', Precision(), Recall()])
    return model


def build_model_lstm_v1(input_len):
    embed_dim = 128
    max_words = 10000

    model = Sequential()
    model.add(Embedding(max_words, embed_dim, input_length=input_len))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])
    return model


def build_model_lstm_v0(input_len):
    vocab_size = 5000
    embedding_size = 32
    epochs = 20
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    # Build model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=input_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy', Precision(), Recall()])

    return model
