import pandas as pd

from src.models.manager import ModelManager

if __name__ == "__main__":
    df = pd.read_csv("B2W-Processed01.csv", low_memory=False)
    df["review_text"] = df["review_text"].astype(str)
    model_manager = ModelManager(dataframe=df)
    model_manager.fit()
    model_manager.save_model(file_path_model="model_v2.pickle",
                             file_path_tokenizer="tokenizer_v2.pickle")