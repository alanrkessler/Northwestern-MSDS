"""Keyword Extraction using TFIDF based on script from class."""

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_doc(doc):
    """Return processed tokens for a given document."""
    # Split into "words"
    tokens = doc.split()
    # Remove punctuation
    re_punc = re.compile(f"[{re.escape(string.punctuation)}]")
    tokens = [re_punc.sub('', word) for word in tokens]
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Remove short tokens
    tokens = [word for word in tokens if len(word) > 4]
    # Make tokens lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# Read in entire corpus
text_data = pd.read_csv('./Extract Engine Comparison/corpus.csv')

text_titles = text_data['Doc_Title'].tolist()
text_bodies = text_data['Text'].tolist()

# Create list of tokens for the corpus
processed_text = []
for document in text_bodies:
    processed_text.append(clean_doc(document))

# Stitch the strings back together for entire document
final_processed_text = []
for i, _document in enumerate(processed_text):
    final_processed_text.append(" ".join(processed_text[i]))

# Run results through tfidf
tfidf = TfidfVectorizer(ngram_range=(1, 3))
tfidf_matrix = tfidf.fit_transform(final_processed_text)
matrix = pd.DataFrame(tfidf_matrix.toarray(),
                      columns=tfidf.get_feature_names(),
                      index=text_titles).T
matrix.to_csv('./Extract Engine Comparison/tfidf_matrix.csv')
