# Week 4 - Classification

For Week 4, I am using TF-IDF to classify documents as either about a software issue or with aircraft safety rules. 

### Scripts
-   `corpus_creation.py` - Creates a CSV corpus following the examples for the class selecting all documents from the `raw_data` directory. The result is `corpus.csv`. Additionally, two test cases have been added to see how the model performs. The labels for each document are in `labels.csv` in the same order as the corpus when sorted by document title.
-   `classification.py` - Creates features and classifies the documents.