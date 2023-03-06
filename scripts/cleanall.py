import pandas as pd

from nlp_analytics.src.features import cleanup, dataprocessing

map_sentiment = {1: 0,
                 2: 0,
                 3: 2,
                 4: 1,
                 5: 1}

if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\vgama\\Projects\\nlp_ecommerce_interview\\nlp_analytics\\data\\raw\\B2W-Reviews01.csv", low_memory=False)
    df["overall_rating"] = df["overall_rating"].map(map_sentiment)
    df = df.dropna(subset=['review_text'])
    df = df.dropna(subset=['overall_rating'])
    df["review_text"] = df["review_text"].astype(str)
    df["review_text"] = df["review_text"].str.lower()
    df.drop_duplicates(subset=["review_text"], keep="first", inplace=True)
    df["review_text"] = df["review_text"].apply(dataprocessing.remove_word_punctuation)
    df["review_text"] = df["review_text"].apply(dataprocessing.remove_numbers_and_special_symbols)
    df = cleanup.filter_text_by_language_rule(df)
    df["review_text"] = df["review_text"].apply(dataprocessing.remove_word_accent)
    df["review_text"] = df["review_text"].apply(dataprocessing.remove_stop_words)
    df.to_csv('B2W-Processed02.csv', index=False)
