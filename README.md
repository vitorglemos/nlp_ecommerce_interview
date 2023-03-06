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
---

### Example of use

After installation, it is possible to instantiate in Python code with the model already preloaded. Below is an example of using the application using inference from the last trained model.

```python
from nlp_analytics.src.models.manager import ModelManager

if __name__ == '__main__':

    model_manager = ModelManager()
    model_manager.load_model(file_path_model="./nlp_analytics/data/output/model_v7.h5",
                             file_path_tokenizer="./nlp_analytics/data/output/tokenizer_v7.pickle")
    print(model_manager.predict_class("Cadeira confortável, a cadeira é rosa, muito bonita. Mas parece frágil!",
                                      key_words=True))
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
## License

This project is licensed under the MIT License.

---
## Contact me

If you have any questions or feedback about this project, please feel free to reach out.
