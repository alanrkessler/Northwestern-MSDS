"""Vectorized Representation using TFIDF and Word2Vec."""

import pandas as pd
import numpy as np
import re
import glob
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec


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
    tfidf_matrix = tfidf.fit_transform(text)
    matrix = pd.DataFrame(tfidf_matrix.toarray(),
                          columns=tfidf.get_feature_names(),
                          index=text_titles).T
    # Measure importance by the 80th percentile
    matrix['imp_value'] = matrix.quantile(0.8, axis=1)
    matrix.sort_values(by=['imp_value'], axis=0, inplace=True, ascending=False)
    return matrix


def retrieve_document(file_path):
    """Return text as a string for a given txt file path."""
    with open(file_path, 'r', encoding='utf-8') as document:
        lines = document.readlines()
    text = ''.join(lines)
    return text


def count_terms(text, term):
    """Return the number of times a term occurs in text."""
    count = len(re.findall(term, text))
    return count


def w2v_term_search(input_path, data):
    """Return dataframe of term occurrences."""
    # Load each text file and search for each term
    path_list = glob.glob(input_path)
    terms_list = data.index.tolist()
    for i, path in enumerate(path_list):
        text = retrieve_document(path)
        count_list = []
        for term in terms_list:
            count_list.append(count_terms(text, term))
        data[f"Doc_{i}"] = count_list
    # Add the number of documents that the term appears
    data['prev'] = data.iloc[:, -1*len(path_list):].astype(bool).sum(axis=1)
    # Drop vectors themselves
    data.drop(data.iloc[:, 0:100], axis=1, inplace=True)
    return data


# Read in entire corpus
text_data = pd.read_csv('./corpus.csv')

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

# Generate tfidf results
tfidf_matrix = tfidf_analysis(final_processed_text, text_titles)

tfidf_matrix.to_csv('tfidf_output.csv')
# Correlation of TFIDF values for the top terms
correlation = tfidf_matrix[0:4].T.corr()
correlation.to_csv('correlation.csv')

# Word2Vec Model
model_w2v = Word2Vec(processed_text,
                     size=100,
                     window=5,
                     min_count=1,
                     workers=4)

# Flatten the list of terms
terms_w2v = [term for doc in processed_text for term in doc]
# Return unique terms
terms_w2v = list(set(terms_w2v))

# Gather weight vectors for each word
vectors_w2v = {}
for i in terms_w2v:
    vectors_w2v[i] = model_w2v.wv[i]

# Represent vectors as a dataframe
w2v_df = pd.DataFrame(vectors_w2v).T

# Cluster terms from Word2Vec output
km = KMeans(n_clusters=10, random_state=89)
distances = km.fit_transform(w2v_df)
# Add cluster assignments and distance from center
w2v_df['Cluster'] = km.labels_.tolist()
w2v_df['Distance'] = np.min(distances, axis=1)
w2v_df.sort_values(by=['Cluster', 'Distance'], inplace=True)
# Add prevalence of each term
w2v_df = w2v_term_search('./raw_data/*.txt', w2v_df)
w2v_df[w2v_df['prev'] > 3].to_csv('w2v_output.csv')
