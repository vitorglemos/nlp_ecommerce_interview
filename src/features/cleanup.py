import spacy
import logging

logging.basicConfig(filename="app.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)

nlp = spacy.load("pt_core_news_md")


def remove_short_texts(dataframe):
    """
    Remove outliers in dataframe as short text without any context
    :param dataframe: original dataframe    :return:  after cleanup
    """
    try:
        dataframe = dataframe[dataframe['review_text'].apply(lambda x: len(x.split()) >= 3)]
        return dataframe
    except ValueError as error:
        logging.error(f"Value error: {error}")


def filter_text_by_language_rule(dataframe):
    """
    Remove outliers in dataframe as text without phrase structure
    :param dataframe: original dataframe    :return:  after cleanup
    """
    try:
        df_clean = dataframe.copy()
        for idx, full_text in zip(dataframe.index, dataframe['review_text']):
            tokens_count = {
                "ADJ": 0,
                "VERB": 0,
                "ADV": 0,
                "NOUN": 0
            }
            doc = nlp(full_text)
            tk = []
            for token in doc:
                if token.pos_ in tokens_count:
                    tokens_count[token.pos_] += 1
                tk.append(token.pos_)
            full_text_len = len(full_text.split(" "))
            full_text_bool = False
            if full_text_len <= 4:
                if tokens_count["ADJ"] >= 1:
                    full_text_bool = True
            elif full_text_len >= 5:
                if tokens_count["VERB"] >= 1 or tokens_count["NOUN"] >= 1:
                    if tokens_count["ADJ"] >= 1 or tokens_count["ADV"] >= 1:
                        full_text_bool = True
                    elif tokens_count["NOUN"] >= 1:
                        full_text_bool = True
                elif tokens_count["ADJ"] >= 1:
                    full_text_bool = True
            if not full_text_bool:
                df_clean.drop(index=idx)
        return df_clean
    except ValueError as error:
        logging.error(f"Value error: {error}")