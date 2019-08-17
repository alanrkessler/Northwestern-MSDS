"""Train a model based on V5 in a classification chain."""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class comment_classifier:
    """Baseline classifier."""

    def __init__(self, train_df, test_df, train_features, test_features):
        """Initiatize class with data to use."""
        self.train_df = train_df
        self.test_df = test_df
        self.train_features = train_features
        self.test_features = test_features
        self.model = None
        self.target_values = None

    def full_fit(self, target):
        """Specify and train a classifier."""
        # Get y values from training data
        target_values = self.train_df[target]
        # Specify the model
        # For logistic regression on large data, the sag solver is faster
        # Since there is a large number of features, regularization is needed
        # L2 is Ridge which is the default
        # A lower C increases the magnitude of regularization
        model = LogisticRegression(solver='sag', C=0.2)
        # Fit to the entire training data
        model.fit(self.train_features, target_values)
        # Set values after full fit
        self.model = model
        self.target_values = target_values
        return self

    def submitter(self):
        """Generate predictions on test data."""
        return self.model.predict_proba(self.test_features)[:, 1]

    def add_chain(self, target):
        """Add predicted values for the classification chain."""
        pred = self.model.predict_proba(self.train_features)[:, 1]
        self.train_features[target] = pred
        self.test_features[target] = self.submitter()
        return self


# Load data - processing earlier does still make NAs possible
data_directory = Path('./data')
train = pd.read_csv(data_directory / 'train_order.csv').fillna(" ")
test = pd.read_csv(data_directory / 'test_order.csv').fillna(" ")

# Combine prior to TF-IDF to ensure all features work when scoring
train_clean = train['comment_clean']
test_clean = test['comment_clean']
full_corpus = pd.concat([train_clean, test_clean])

# Specify vectorizer using single word terms for baseline.
# Set a max number of features to avoid performance issues in training.
# Sublinear_tf used to avoid giving term frequency linear weight.
tfidf_vect = TfidfVectorizer(max_features=3000, sublinear_tf=True)

# Fit the vectorizer on the entire corpus
tfidf_vect.fit(full_corpus)

# Generate features for training and testing data
train_tfidf = pd.DataFrame(tfidf_vect.transform(train_clean).toarray(),
                           columns=tfidf_vect.get_feature_names(),
                           index=train['id'].values)
test_tfidf = pd.DataFrame(tfidf_vect.transform(test_clean).toarray(),
                          columns=tfidf_vect.get_feature_names(),
                          index=test['id'].values)

# Specify output directories
submissions_directory = Path('./reports/submissions')

# Train models
chain_model = comment_classifier(train, test, train_tfidf, test_tfidf)
# Place to store submission
submission = pd.DataFrame.from_dict({'id': test['id']})
# Classes to loop through
y_names = ['toxic',
           'severe_toxic',
           'obscene',
           'threat',
           'insult',
           'identity_hate']
for y in y_names:
    # Fit on entire data set
    chain_model.full_fit(y)
    # Generate submission
    submission[y] = chain_model.submitter()
    # Add feature for chain
    chain_model.add_chain(y)

# Save entire submission
submission.to_csv(submissions_directory / 'v7_submission.csv', index=False)
