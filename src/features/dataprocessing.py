import re
import nltk
import string
import logging
import unicodedata
from nltk.corpus import stopwords

logging.basicConfig(filename="app.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)

nltk.download('stopwords')
stopwords_pt = set(stopwords.words('portuguese'))


def remove_stop_words(text: str) -> str:
    """
    Remove all stop words in text to reduce dimension problem
    :param text: original string
    :return: text after cleanup
    """
    try:
        filtered_words = [word for word in text.split(" ") if word.lower() not in stopwords_pt]
        return ' '.join(filtered_words)
    except ValueError as error:
        logging.error(f"Value error: {error}")


def remove_word_punctuation(text: str) -> str:
    """
    Remove all punctuation in any text
    :param text: original string
    :return: text after cleanup
    """
    try:
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    except ValueError as error:
        logging.error(f"Value error: {error}")


def remove_word_accent(text: str) -> str:
    """
    Remove all accent in portuguese text
    :param text: original string
    :return: text after cleanup
    """
    try:
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        return text
    except ValueError as error:
        logging.error(f"Value error: {error}")


def remove_numbers_and_special_symbols(text: str) -> str:
    """
    Remove all numbers and also special characters in text
    :param text: original string
    :return: text after cleanup
    """
    try:
        text = re.sub(r'[^a-zA-zÀ-ú\s]', '', text)
        return text
    except ValueError as error:
        logging.error(f"Value error: {error}")
