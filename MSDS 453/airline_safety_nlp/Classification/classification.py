"""Classification using TFIDF."""

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


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
    # Lemmatization to account for things like plurals
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(token) for token in tokens]
    return tokens


def tfidf_analysis(text, text_titles):
    """Generate sorted TFIDF output for text."""
    # Run results through tfidf (prevalent - require 25% of docs)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=0.25)
    tfidf.fit(text)
    tfidf_matrix = tfidf.transform(text)
    matrix = pd.DataFrame(tfidf_matrix.toarray(),
                          columns=tfidf.get_feature_names(),
                          index=text_titles).T
    # Measure importance by the 80th percentile
    matrix['imp_value'] = matrix.quantile(0.8, axis=1)
    matrix.sort_values(by=['imp_value'], axis=0, inplace=True, ascending=False)
    return matrix, tfidf


# Read in entire corpus
text_data = pd.read_csv('./corpus.csv')
text_data.sort_values(by=['Doc_Title'], inplace=True)
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

# Generate tfidf results - ranked by the percentile
# Select features only using training data
tfidf_matrix, model_tfidf = tfidf_analysis(final_processed_text[:-2],
                                           text_titles[:-2])
# Generate list of preliminary terms (top n)
top_n_tfidf = tfidf_matrix.iloc[0:10, :].index.tolist()

# Generate tfidf values
tfidf_values = model_tfidf.transform(final_processed_text)
tfidf_values = pd.DataFrame(tfidf_values.toarray(),
                            columns=model_tfidf.get_feature_names(),
                            index=text_titles)
tfidf_values.sort_index(inplace=True)
# Filter to the top n terms identified
tfidf_values = tfidf_values[top_n_tfidf]

# Import labels (manually defined classes)
labels = pd.read_csv('labels.csv')
labels['software'] = np.where(labels['class'] == "software", 1, 0)
y_values = labels[['software']].values.ravel()

# Model specification & fit on training data
logistic = LogisticRegression(solver='lbfgs')
logistic.fit(tfidf_values.iloc[:-2, :].values, y_values[:-2])

# Generate predictions for training data
scores = logistic.predict_proba(tfidf_values.values)

# Feature analysis
features = pd.DataFrame(logistic.coef_, columns=top_n_tfidf).T
features['rel_imp'] = np.abs(features[0])
features['rel_imp'] = features['rel_imp'] / np.max(features['rel_imp'])
features.sort_values(by=['rel_imp'], ascending=False, inplace=True)
features.rename(columns={0: 'coeff'}, inplace=True)
