# ParsBERT-Sentiment-Analysis
Using ParsBERT monolingual BERT-based model for persian language sentiment analysis
ParsBERT is a monolingual pretrained model using fine-tuning and is used for text classification, sentiment analysis and named entity recognition.
The authors have gathered large persian text corpora and evaluated the three mentioned NLP downstream tasks on them and compared them with baselines.
ParsBERT is available for public use as well as most of the datasets that are mentioned in the article.
I have included the code for sentiment analysis using tensorflow, pytorch or scripts in this repository.
Paper presenting ParsBERT: DOI: 10.1007/s11063-021-10528-4
ParsBERT original repository: https://github.com/hooshvare/parsbert
## ParsBERT Sentiment Analysis

![ParsBERT Sentiment Analysis Banner Image](banner.png)

**ParsBERT** is a monolingual Persian language model that is fine-tuned and used for text classification, sentiment analysis, and named entity recognition.

This repository contains code for sentiment analysis using ParsBERT, implemented in TensorFlow, PyTorch, and Python scripts.

**Features:**
* Sentiment analysis of Persian text
* Easy to use code
* Well-documented examples

**Installation:**

```python
pip install parsbert

import parsbert

# Load the ParsBERT model
model = parsbert.load_model()

# Perform sentiment analysis on a piece of text
text = "این یک متن فارسی مثبت است."
sentiment = model.predict(text)

# Print the sentiment label
print(sentiment)
'''

**Examples:**

See the `examples` directory for examples of how to use the ParsBERT code in TensorFlow, PyTorch, and Python scripts.

**Contributing:**

We welcome contributions to this repository. Please see the `CONTRIBUTING.md` file for guidelines on how to contribute.
