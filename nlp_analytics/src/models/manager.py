import spacy
import pickle
import logging
import tensorflow as tf
import numpy as np
import pandas as pd

nlp = spacy.load("pt_core_news_md")

from nlp_analytics.src.features import dataprocessing
from nlp_analytics.src.models.models import build_model_lstm_v2
from nlp_analytics.src.models.models import build_model_lstm_v0
from nlp_analytics.src.models.models import build_model_lstm_v2_2

from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

logging.basicConfig(filename="app.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, dataframe=None, shape=628, models=None, max_features=10000,
                 to_categorical=False):
        self.models = models
        self.max_len = shape
        self.tokenizer = Tokenizer(num_words=max_features, split=" ")
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        self.sentiment_classes = ['positive', 'neutral', 'negative']
        if dataframe is not None:
            self.tokenizer.fit_on_texts(dataframe["review_text"])
            self.X = pad_sequences(self.tokenizer.texts_to_sequences(dataframe["review_text"]),
                                   padding='post', maxlen=self.max_len)
            if to_categorical:
                self.y = dataframe["overall_rating"].map({"positive": 0, "neutral": 1, "negative": 2})
                self.y = tf.keras.utils.to_categorical(self.y, 3)
            else:
                self.y = dataframe["overall_rating"].values

        if self.models is None:
            self.models = build_model_lstm_v2(self.max_len)

    @staticmethod
    def get_word_key(review_text: str) -> list:
        """
        Get word key in review text
        """
        adjective = list()
        doc = nlp(review_text)
        for token in doc:
            if token.pos_ == "ADJ":
                adjective.append(token.text)
        return adjective

    def tokenizer_analytics(self):
        """
        Get tokenizer word index
        """
        file_handle = open("tokenizer_word_index.txt", "wt")
        for items in self.tokenizer.word_index:
            file_handle.write(f"{items}\n")
        file_handle.close()

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
            batch_size = 64
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            history = self.models.fit(X_train, y_train, epochs=10, batch_size=batch_size,
                                      verbose=1, validation_data=(X_test, y_test),
                                      callbacks=[self.early_stopping])

            results = self.models.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
            print(results)
            logging.info(f"Final evaluate model: {results}")
            return history
        except ValueError as error:
            logging.error(error)

    def fit_transform(self, review_text) -> object:
        """
        Transform data in model format
        :param review_text: review text send by users
        """
        try:
            text_transform = self.tokenizer.texts_to_sequences(review_text)
            text_transform = pad_sequences(text_transform, padding='post', maxlen=self.max_len)
            return text_transform
        except ValueError as error:
            logging.error(error)

    @staticmethod
    def data_prepare(review_text: str) -> str:
        """
        Get cleaned data text
        :param review_text: reviewed text send by users
        """
        text = dataprocessing.remove_word_punctuation(review_text.lower())
        text = dataprocessing.remove_numbers_and_special_symbols(text)
        text = dataprocessing.remove_word_accent(text)
        text = dataprocessing.remove_stop_words(text)
        return text

    def predict_class(self, review_text: str, key_words: bool = False):
        """
         Get inference using reviewed text using loaded model
         :param review_text: reviewed text send by users
         :param key_words: show list of adjectives find in review text
        """
        keys_adj = []
        if key_words:
            keys_adj = self.get_word_key(review_text)

        review_tex = self.fit_transform(self.data_prepare(review_text))
        review_predict = self.models.predict(review_tex).argmax(axis=1)
        review_class = self.sentiment_classes[review_predict[0]]
        print('The predicted sentiment is', review_class)

        predicted = {"text": review_text, "sentiment": review_class, "adjectives": keys_adj}
        return predicted
