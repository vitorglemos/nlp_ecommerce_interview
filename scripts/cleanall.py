import pandas as pd

from src.features import cleanup
from src.features import dataprocessing

map_sentiment = {1: "negative",
                 2: "negative",
                 3: "neutral",
                 4: "positive",
                 5: "positive"}

if __name__ == "__main__":
    df = pd.read_csv("./data/raw/B2W-Reviews01.csv", low_memory=False)
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
    df.to_csv('B2W-Processed01.csv', index=False)
