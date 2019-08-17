"""Train a model based on V5 with additional features."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


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

    def cross_val(self, folds=5):
        """Apply cross validation on a specified model."""
        # Generate cross-validation results
        cv_avg = np.mean(cross_val_score(self.model,
                                         self.train_features,
                                         self.target_values,
                                         scoring='roc_auc', cv=folds))
        return cv_avg

    def submitter(self):
        """Generate predictions on test data."""
        return self.model.predict_proba(self.test_features)[:, 1]

    def logistic_importance(self):
        """Generate feature importance from logistic regression."""
        features = pd.DataFrame(self.model.coef_,
                                columns=self.train_features.columns).T
        # Calculate relative importance
        features['rel_imp'] = np.abs(features[0])
        features['rel_imp'] = features['rel_imp'] / np.max(features['rel_imp'])
        features.sort_values(by=['rel_imp'], ascending=False, inplace=True)
        features.rename(columns={0: 'coeff'}, inplace=True)
        return features


def plt_importance(feature_imp, topn=20, output=None):
    """Plot feature importance for a logistic regression."""
    plt.figure(figsize=(6.5, 3))
    plt.style.use('default')
    feat_filtered = feature_imp.head(topn)
    direction = np.where(np.sign(feat_filtered['coeff'].values) > 0,
                         'More Likely', 'Less Likely')
    label_bar = sns.barplot(x=feat_filtered.index,
                            y=feat_filtered['rel_imp'].values,
                            hue=direction,
                            dodge=False)
    xlabels = label_bar.get_xticklabels()
    label_bar.set_xticklabels(xlabels, rotation=70)
    plt.title("Feature Importance")
    plt.xlabel("TF-IDF Variable")
    plt.ylabel("Relative Variable Importance")
    plt.tight_layout()
    if output is not None:
        plt.savefig(output)
    else:
        pass


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

# Specify output directories
plot_directory = Path('./reports/figures/v9')
submissions_directory = Path('./reports/submissions')

# Train models
baseline_model = comment_classifier(train, test, train_tfidf, test_tfidf)
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
    baseline_model.full_fit(y)
    # Output feature importance
    feature_imp = baseline_model.logistic_importance()
    # Plot feature importance
    plt_importance(feature_imp,
                   output=plot_directory / f"importance_{y}.png")
    submission[y] = baseline_model.submitter()

# Save entire submission
submission.to_csv(submissions_directory / 'v9_submission.csv', index=False)
