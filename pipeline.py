import pandas as pd

from nlp_analytics.src.models.manager import ModelManager

if __name__ == "__main__":
    training_model = False
    if training_model:
        df = pd.read_csv("./nlp_analytics/data/processed/B2W-Processed03.csv", low_memory=False)
        df["review_text"] = df["review_text"].astype(str)

        model_manager = ModelManager(dataframe=df, to_categorical=True)
        model_manager.fit()
        model_manager.save_model(file_path_model="./nlp_analytics/data/output/model_v8.h5",
                                 file_path_tokenizer="./nlp_analytics/data/output/tokenizer_v8.pickle")

        print(model_manager.predict_class("NÃO GOSTEI. Produto ruim!"))
    else:
        model_manager = ModelManager()
        model_manager.load_model(file_path_model="./nlp_analytics/data/output/model_v7.h5",
                                 file_path_tokenizer="./nlp_analytics/data/output/tokenizer_v7.pickle")
        print(model_manager.predict_class("Cadeira confortável, a cadeira é rosa, muito bonita. Mas parece frágil!", key_words=True))
