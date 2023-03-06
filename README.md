# NLP Ecommerce Interview

## 📍 Introduction

This project aims to address a problem related to natural language processing and text classification in the context of user reviews of e-commerce products. The goal is to classify user comments on e-commerce platforms as positive or negative, while also identifying keywords corresponding to the product. The original dataset includes several metadata fields such as reviewer text, overall rating, reviewer gender, title, and others. However, the focus of this project will be on the review comments and overall ratings. The other fields will be used only for exploratory analysis and to help understand the database and outliers
## 🏎💨 Getting Started

### Prerequisites

Ensure that you have the following prerequisites installed:

- Python 3.9 or higher
- Tensorflow 2.11.0
- Keras 2.11.0
- Spacy 3.5.0
  
All these prerequisites are in requirements.txt file, you can install its using the Python command:
```bash
pip3 install -r requirements.txt
```

It is also important install the spacy portuguese model. If you haven't downloaded pt-core-news-md yet, you can use the following command.
```bash
python -m spacy download pt_core_news_md   
```

### GitHub Repository

You can also contribute to the project, first clone the repository and push it to a branch.
```bash
git clone git@github.com:vitorglemos/nlp_ecommerce_interview.git
```

### Installation 

You can install this package through setup.py. To do this, run the command inside the project:
```bash
python3 setup.py install
```

### Project Tree
```bash
my_project/
├── README.md
├── LICENSE
├── requirements.txt
├── pipeline.py
├── src/
│   ├── __init__.py
│   ├── __version__.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── cleanup.py
│   │   └── dataprocessing.py
│   └──  models/
│         ├── manager.py
│         └── models.py
│   
├── scripts/
│       ├── __init__.py
│       └── cleanall.py
├── notebooks/
│      └── __init__.py
└── data/
    ├── processed/
    │       └── B2W-Processed01.csv
    └── raw/
        └── B2W-Reviews01.csv
```

---
## Natural Language Processing
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

### Future works

In this project, an LSTM network was used to classify sentiments in product evaluation texts written by users. One of the advantages of the LSTM network is that it can retain information in the long term, which allows for good applicability in text classification. Another machine learning model that has an advantage over this is the BERT. For this project, the LSTM network was chosen due to the shorter computing time to process and shorter inference time, but for future work, I would still like to compare the BERT models to this type of data.

---
## License

This project is licensed under the MIT License.

---
## Contact me

If you have any questions or feedback about this project, please feel free to reach out.
