"""Train final classifier keeping only what is necessary for submission."""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class comment_classifier:
    """Comment classifier model training and scoring."""

    def __init__(self, train_df, test_df, train_features, test_features):
        """Initiatize class with data to use."""
        self.train_df = train_df
        self.test_df = test_df
        self.train_features = train_features
        self.test_features = test_features
        self.model = None
        self.target_values = None

    def fit(self, target):
        """Specify and train a classifier."""
        self.target_values = self.train_df[target]
        # L2 with sag solver is efficient on large data
        self.model = LogisticRegression(solver='sag', C=0.2)
        self.model.fit(self.train_features, self.target_values)
        return self

    def submit(self):
        """Generate predictions on test data."""
        return self.model.predict_proba(self.test_features)[:, 1]


# Load data - earlier processing makes NAs possible
data_directory = Path('./data')
train = pd.read_csv(data_directory / 'train_order.csv').fillna(" ")
test = pd.read_csv(data_directory / 'test_order.csv').fillna(" ")

# Combine prior to TF-IDF to ensure all features exist when scoring
train_clean = train['comment_clean']
test_clean = test['comment_clean']
full_corpus = pd.concat([train_clean, test_clean])

# Specify vectorizer using single word terms.
# Set a max number of features to avoid performance issues in training.
# Sublinear_tf used to avoid giving term frequency linear weight.
tfidf_vect = TfidfVectorizer(max_features=10000, sublinear_tf=True)

# Fit the vectorizer on the entire corpus
tfidf_vect.fit(full_corpus)

# Generate features for training and testing data
train_tfidf = pd.DataFrame(tfidf_vect.transform(train_clean).toarray(),
                           columns=tfidf_vect.get_feature_names(),
                           index=train['id'].values)
test_tfidf = pd.DataFrame(tfidf_vect.transform(test_clean).toarray(),
                          columns=tfidf_vect.get_feature_names(),
                          index=test['id'].values)

# Specify output directory and submission location
submissions_directory = Path('./reports/submissions')
submission = pd.DataFrame.from_dict({'id': test['id']})

final_model = comment_classifier(train, test, train_tfidf, test_tfidf)
# Labels to iterate over
y_labels = ['toxic',
            'severe_toxic',
            'obscene',
            'threat',
            'insult',
            'identity_hate']
for y in y_labels:
    # Train and score model for a single class independently
    final_model.fit(y)
    submission[y] = final_model.submit()

# Save entire submission
submission.to_csv(submissions_directory / 'final_submission.csv', index=False)
