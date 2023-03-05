import unittest
from src.features import dataprocessing

validation_english_str = "Hi, this is only a test."
validation_portuguese_str = "Olá, este é apenas um teste."


class TestPreprocessing(unittest.TestCase):

    def test_remove_word_punctuation(self):
        validation_return_en = dataprocessing.remove_word_punctuation(validation_english_str)
        validation_return_pt = dataprocessing.remove_word_punctuation(validation_portuguese_str)
        self.assertTrue(validation_return_en == "Hi this is only a test")
        self.assertTrue(validation_return_pt == "Olá este é apenas um teste")

    def test_remove_word_accent(self):
        validation_return_pt = dataprocessing.remove_word_accent(validation_portuguese_str)
        self.assertTrue(validation_return_pt == "Ola, este e apenas um teste.")

    def test_remove_numbers_and_special_symbols(self):
        validation_return = dataprocessing.remove_numbers_and_special_symbols("@#$59590Ol4á")
        self.assertTrue(validation_return == "Olá")

    def test_remove_stop_words(self):
        validation_return_pt = dataprocessing.remove_stop_words(validation_portuguese_str)
        self.assertTrue(validation_return_pt == "Olá, apenas teste.")
