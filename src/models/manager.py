import pickle
import logging

import numpy as np
import pandas as pd

from src.models.models import build_model_lstm_v1

from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

logging.basicConfig(filename="app.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, dataframe, models=None, max_features=1000):
        self.models = models
        self.tokenizer = Tokenizer(num_words=max_features, split=" ")
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        self.tokenizer.fit_on_texts(dataframe["review_text"].values)
        self.X = pad_sequences(self.tokenizer.texts_to_sequences(dataframe["review_text"].values))
        self.y = pd.get_dummies(dataframe["overall_rating"]).values

        if self.models is None:
            self.models = build_model_lstm_v1(self.X.shape[1])

    def load_model(self, file_path_model: str, file_path_tokenizer: str):
        """
        Load final machine learning model
        """
        try:
            self.models.load_weights(file_path_model)
            with open(file_path_tokenizer, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            logging.info("The model is loaded!")
        except FileNotFoundError as error:
            logging.error(error)

    def save_model(self, file_path_model: str, file_path_tokenizer: str):
        """
        Save final machine learning model after training process
        """
        try:
            self.models.save_weights(file_path_model)
            with open(file_path_tokenizer, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("The model is saved!")
        except FileNotFoundError as error:
            logging.error(error)

    def fit(self):
        """
        Train model using processed dataset
        """
        try:
            batch_size = 32
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
            history = self.models.fit(X_train, y_train, epochs=7, batch_size=batch_size,
                                      verbose=1, validation_data=(X_test, y_test),
                                      callbacks=[self.early_stopping])

            score, acc = self.models.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
            print(score, acc)
            logging.info(f"Final evaluate model: {score} - {acc}")
            return history
        except ValueError as error:
            logging.error(error)

    def fit_transform(self, review_text) -> object:
        """
        Transform data in model format
        :param review_text: review text send by users
        """
        try:
            text_transform = pad_sequences(self.tokenizer.texts_to_sequences([review_text]))
            return text_transform
        except ValueError as error:
            logging.error(error)

    def predict(self, review_text) -> float:
        """
        Get inference using review text (0 in positive case and 1 in negative case)
        :param review_text: review text send by users
        """
        return np.argmax(self.models.predict(self.fit_transform(review_text)))

    def predict_proba(self, review_text: str) -> float:
        """
        Get inference using review text (using probability return)
        :param review_text: review text send by users
        """
        return self.models.predict_prob(review_text)
