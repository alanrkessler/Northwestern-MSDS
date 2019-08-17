# toxic-comments
Toxic Comment Classification Challenge for MSDS 453

**Disclaimer: The comments analyzed contain strong language. They are exactly as downloaded from [Kaggle.com](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).**

## Steps Completed:
-   EDA on training data labels - `label_eda.py`.
-   Developed processing steps for a sample of the training data - `mini_corpus.py`.
-   Applied processing steps to training and testing data - `baseline_processing.py`.
-   Training a logistic regression to TF-IDF features `baseline_classification.py`. The feature importances themselves are ignored in the repository because the feature names are offensive themselves.
-   Attempted to improve on the baseline model through a variety of different adjustments. These are all recorded on the my [leaderboard](src/iterations/README.md). This resulted in a new processing steps in `final_processing.py` and a cleaned up final model in `final_classification.py`.
-   Produced reports for the [assignment](./reports)

## Steps Potentially in Future Projects:
-   Consider using character ngrams as terms may be parts of larger words.
-   Consider the use of pre-trained embeddings like GLOVE.