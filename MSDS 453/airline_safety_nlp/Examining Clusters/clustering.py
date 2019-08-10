"""Clustering using TFIDF, k-means, and LDA."""

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


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
    # Run results through tfidf (prevalent - require a % of docs)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=0.15)
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
text_data.sort_values(by=['DSI_Title'], inplace=True)
text_titles = text_data['DSI_Title'].tolist()
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
tfidf_matrix, model_tfidf = tfidf_analysis(final_processed_text, text_titles)
# Generate list of preliminary terms (top n)
top_n_tfidf = tfidf_matrix.iloc[0:30, :].index.tolist()
# Generate tfidf values
tfidf_values = model_tfidf.transform(final_processed_text)
tfidf_values = pd.DataFrame(tfidf_values.toarray(),
                            columns=model_tfidf.get_feature_names(),
                            index=text_titles)
tfidf_values.sort_index(inplace=True)
# Filter to the top n terms identified
tfidf_values = tfidf_values[top_n_tfidf]

# Cluster terms from tfidf output
km = KMeans(n_clusters=4, random_state=89)
distances = km.fit_transform(tfidf_values)
cluster_assignment_df = text_data.copy()
cluster_assignment_df['Cluster'] = km.labels_.tolist()
cluster_assignment_df.sort_values(by=['Cluster'], inplace=True)
cluster_assignment_df.to_csv('cluster_assignment.csv')

# Analyze most/least important terms in each
tfidf_clusters = tfidf_values.copy()
tfidf_clusters['cluster'] = km.labels_.tolist()
avg_tfidf = tfidf_clusters.groupby('cluster').mean()
avg_tfidf = avg_tfidf.T
top_vals = {}
bot_vals = {}
for column in avg_tfidf.columns:
    tfidf_sorted = avg_tfidf.sort_values([column], ascending=False)
    top_vals[column] = tfidf_sorted.head(5).index.tolist()
    bot_vals[column] = tfidf_sorted.tail(5).index.tolist()
pd.DataFrame(top_vals).to_csv('top_vals.csv')
pd.DataFrame(bot_vals).to_csv('bot_vals.csv')
