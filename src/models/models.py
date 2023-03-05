
from keras.optimizers import Adam

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import BatchNormalization

from keras.models import Sequential


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
