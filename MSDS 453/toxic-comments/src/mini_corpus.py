"""Clean a small part the training data."""

from pathlib import Path
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.util import mark_negation


def lemm_pos(word):
    """Return POS for lemmatizing (default - noun)."""
    tag = nltk.pos_tag([word])[0][1][0]
    tag_dict = {"V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def clean_doc(doc):
    """Return processed tokens for a given document."""
    # Split into "words"
    tokens = doc.split()
    # Remove punctuation
    re_punc = re.compile(f"[{re.escape(string.punctuation)}]")
    tokens = [re_punc.sub('', word) for word in tokens]
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Remove short tokens (making sure not to remove four letter words)
    tokens = [word for word in tokens if len(word) > 2]
    # Remove long tokens (possibly URLs)
    tokens = [word for word in tokens if len(word) < 20]
    # Make tokens lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization to account for things like plurals
    # Takes in part of speech (POS)
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(token, lemm_pos(token)) for token in tokens]
    # Add negations
    tokens = mark_negation(tokens)
    # Return tokens
    clean_comment = " ".join(tokens)
    return clean_comment


def rem_talk(doc):
    """Remove "talk" if it is the last word in a clean comment."""
    if doc.split()[-1] == 'talk':
        clean_comment = ' '.join(doc.split()[:-1])
    else:
        clean_comment = doc
    return clean_comment


# Load data
data_directory = Path('./data')
train = pd.read_csv(data_directory / 'train.csv')

# Same to create a micro corpus to work on in more detail
train_micro = train.sample(n=100, random_state=10)

# Save examples
output_directory = Path('./reports/output')
with open(output_directory / 'examples.txt', 'w') as f:
    # Comment examples
    print("\nRaw Comment Examples", file=f)
    print("\nClean Comment Example:", file=f)
    print(repr(train_micro.iloc[1, 1]), file=f)
    print("\nToxic Comment Example:", file=f)
    print(repr(train_micro.iloc[17, 1]), file=f)

    # Interesting format examples
    print("\nInteresting Comment Format Examples:", file=f)
    print("IP Address", file=f)
    print(repr(train_micro.iloc[4, 1]), file=f)
    print("\nTalk", file=f)
    print(repr(train_micro.iloc[20, 1]), file=f)

    # Processed comment examples
    print("\nProcessed Examples", file=f)
    print("\nClean Comment Example:", file=f)
    print(repr(clean_doc(train_micro.iloc[1, 1])), file=f)
    print("\nToxic Comment Example:", file=f)
    print(repr(clean_doc(train_micro.iloc[17, 1])), file=f)

    # Processed interesting format examples
    print("\nProcessed Interesting Comment Format Examples:", file=f)
    print("IP Address", file=f)
    print(repr(clean_doc(train_micro.iloc[4, 1])), file=f)
    print("\nTalk", file=f)
    print(repr(clean_doc(train_micro.iloc[20, 1])), file=f)
    print("\nFixed to Remove Talk", file=f)
    print(repr(rem_talk(clean_doc(train_micro.iloc[20, 1]))), file=f)
