"""Create data where lemmatization happens earlier with no negations."""

import pandas as pd
from pathlib import Path
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from timeit import default_timer as timer
import dask.dataframe as dd


def lemm_pos(word):
    """Return POS for lemmatizing (default - noun)."""
    tag = nltk.pos_tag([word])[0][1][0]
    tag_dict = {"V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def clean(comment):
    """Return processed tokens for a given comment."""
    # Split into "words"
    tokens = comment.split()
    # Remove punctuation
    re_punc = re.compile(f"[{re.escape(string.punctuation)}]")
    tokens = [re_punc.sub('', word) for word in tokens]
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Move lemmatization up before removing stop words
    # Lemmatization with POS to account for things like plurals
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(token, lemm_pos(token)) for token in tokens]
    # Remove short tokens (making sure not to remove four letter words)
    tokens = [word for word in tokens if len(word) > 2]
    # Remove long tokens (possibly URLs)
    tokens = [word for word in tokens if len(word) < 20]
    # Make tokens lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Fill with blank if a comment is totally removed by processing
    if len(tokens) == 0:
        tokens = [" "]
    else:
        pass
    # Remove "talk" typical at the end of certain wiki comments
    if tokens[-1] == 'talk':
        clean_comment = " ".join(tokens[:-1])
    else:
        clean_comment = " ".join(tokens)
    return clean_comment


# Load training data
data_directory = Path('./data')
train = pd.read_csv(data_directory / 'train.csv')

# Use dask to partition apply cleaning function to rows of training
start = timer()
train_dd = dd.from_pandas(train, npartitions=6)
train_dd['comment_clean'] = train_dd['comment_text'].map(clean).compute()
train_dd.compute().to_csv(data_directory / 'train_order.csv')
end = timer()

# Save processing time for training data
output_directory = Path('./reports/output')
with open(output_directory / 'order_processing.txt', 'w') as f:
    print("Data Set Processing", file=f)
    print(f"Training Processing Complete in {end - start: ,.2f} seconds",
          file=f)

# Load testing data
test = pd.read_csv(data_directory / 'test.csv')

# Use dask to partition apply cleaning function to rows of testing
start = timer()
test_dd = dd.from_pandas(test, npartitions=6)
test_dd['comment_clean'] = test_dd['comment_text'].map(clean).compute()
test_dd.compute().to_csv(data_directory / 'test_order.csv')
end = timer()

# Save processing time
with open(output_directory / 'order_processing.txt', 'a') as f:
    print(f"Testing Processing Complete in {end - start: ,.2f} seconds",
          file=f)
