## Introduction 

This project aims to address a problem related to natural language processing and text classification in the context of user reviews of e-commerce products. This is technical documentation, so there are extra details not covered in the README.md summary of this repository. Here you can find specific solution details, including instructions for creating your own templates for this problem. The main problem involves the analysis of sentiments in texts provided by users. The text can be classified into three main classes, being positive, neutral or negative. The goal of a NLP model is to enable computers to understand, interpret, and generate human language. This involves developing algorithms and models that can analyze and process large amounts of natural language data, such as text, speech, and images. The classification of texts in Portuguese is a great challenge, since the databases for this language are more limited. In addition, dealing with user comments also involves understanding slang, contractions and typos.

### The architecture of the project

The project was built as a Python installation package. In this way, it is possible for you to install the package and use your own machine learning models. In addition, it is also possible to use this package to extend your creation, such as using it in APIs. The project was conceived as a manager, with all the necessary functions for creating and using your own machine learning model. In addition, this facilitates software maintenance and the addition of new features and project versions. The project was divided into several folders, the main one being ./src. The src folder contains all tested models and the manager structure. The manager.py file has all the treatment done and the functions to train, test, predict, load and save new models.

### The save tokenization and models

To represent the text for the model, the tokenization function was used, a process in which the text was divided into smaller units (the tokens). With that, not only the model needs to be saved, but also the tokenization function to avoid that the representations are different in the inference. For the model, the keras save function was used, which allows saving only the final weights after training the machine learning model. In this case, two files are important, tokenization.pickle and model.pickle.

### As use the inference model

To use the inference function after installing the packages, it is possible to load the default model already trained with the data. Below is an example of how an inference can be created using the predict class function. Remembering that data processing is done exclusively by functions within the project, so it is not necessary to insert normalized text, as this is already handled by the preprocessing module.

```python
from nlp_analytics.src.models.manager import ModelManager

if __name__ == "__main__":
    model_manager = ModelManager(dataframe=None, to_categorical=False)
    model_manager.load_model(file_path_model="./nlp_analytics/data/output/model_v7.h5",
                             file_path_tokenizer="./nlp_analytics/data/output/tokenizer_v7.pickle")


    print(model_manager.predict_class("NÃO GOSTEI. Produto Ruim!"))
```

---
## Natural Language Processing

### The machine learning model

The neural network chosen was a bidirectional LSTM. The traditional LSTM model is unidirectional, it processes data sequences in only one direction. On the other hand, the Bidirectional LSTM model processes the sequences in both competitions simultaneously. This difference in the Bidirectional LSTM network has some advantages, such as:
- Allows you to capture the textual context in a bidirectional way. The model can learn relationships between words in earlier and later contexts in the sequence.
- Allows you to reduce the impact on missing data, as it uses information bidirectionally, improving accuracy in case of missing data or words.

---
### Avoiding overfitting

Some treatments were necessary to avoid overfitting. Initially, a simpler LSTM (Long Short-Term Memory) model was trained to test all the functions of the project. In this way, some needs arose to treat the overfitting of the model, some methods used for this were:
- Improve dataset pre-processing: The dataset has a lot of noisy data, texts without context, numbers, random symbols and nonexistent words. Therefore, all these items were treated by the pre-processing module and inserted in the final version of the project.
- Early stopping: this technique was used to prevent the model from continuing to be trained after its performance decayed during training. Furthermore, as the base is unbalanced, metrics such as loss, precision and recall were used to monitor the performance of the model.

---
### Reducing the size of the problem

For this work, some data treatments were useful to reducing the dimensionality of the problem, mainly
in removing outliers. Among the methods available, the following were used:

- Removal of stopwords: Common words that usually don't add much to the meaning of a sentence or document.
- Removal of punctuation and special characters: punctuation and special characters did not comply in any way with the model, on the contrary, they make the problem increase.
- Word normalization: Words have been normalized to prevent multiple tokens from being created unnecessarily. Furthermore,
any number in the text has been removed.
- Context: a special method was created to handle some outliers. In this case, for texts with only one word, only
considered sentences that have at least one adjective. Adjectives are important in classifying feelings, as they can indicate positive or negative feelings. In addition, the presence of verbs and nouns was also considered.
to ensure that the phrases have context and that they are nothing more than random typing as in some cases in the dataset.

---

### Evaluation of the last trained model:
Below is the list of results obtained based on the last trained model. The last version of this project was trained with the Bidirectional LSTM model and with the B2W-Processed03.csv. From the B2W-Processed03.csv base, some noisy comments were excluded, in addition to meaningless texts with only one word.

| Model        | Last loss | Precision | Recall | Acc |
|--------------|-----------|-----------|------|-----|
| BI-LSTM      | 0.4665    |   0.8486     |   0.8176   |0.8316  |

| Model        | File name   | Tokenization        | 
|--------------|-------------|---------------------|
| BI-LSTM      | model_v7.h5 | tokenizer_v7.pickle |  

---
### Sample output

```json
{'text': 'Não gostei de nada, muito ruim', 'sentiment': 'negative', 'adjectives': ['ruim']}
{'text': 'A cor é feia, mas o produto é bom', 'sentiment': 'neutral', 'adjectives': ['feia', 'bom']}
{'text': 'Estou muito satisfeito, amei o produto. Ele é lindo!', 'sentiment': 'positive', 'adjectives': ['satisfeito']}
{'text': 'Cadeira confortável, a cadeira é rosa, muito bonita. Mas parece frágil!', 'sentiment': 'neutral', 'adjectives': ['confortável', 'bonita', 'frágil']}
```

---
### Future works

In this project, an LSTM network was used to classify sentiments in product evaluation texts written by users. One of the advantages of the LSTM network is that it can retain information in the long term, which allows for good applicability in text classification. Another machine learning model that has an advantage over this is the BERT. For this project, the LSTM network was chosen due to the shorter computing time to process and shorter inference time, but for future work, I would still like to compare the BERT models to this type of data.

