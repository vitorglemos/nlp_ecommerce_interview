import numpy as np
import pandas as pd


class ModelManager:
    def __init__(self, models=None):
        self.models = models

    def load_model(self, file_path):
        """
        Load final machine learning model
        """
        pass

    def save_model(self, file_path):
        """
        Save final machine learning model after training process
        """
        pass

    def fit(self):
        """
        Train model using processed dataset
        """
        pass

    def fit_transform(self, review_text) -> object:
        """
        Transform data in model format
        :param review_text: review text send by users
        """
        pass

    def predict(self, review_text) -> float:
        """
        Get inference using review text (0 in positive case and 1 in negative case)
        :param review_text: review text send by users
        """
        return self.models.predict(review_text)

    def predict_proba(self, review_text: str) -> float:
        """
        Get inference using review text (using probability return)
        :param review_text: review text send by users
        """
        return self.models.predict_prob(review_text)
