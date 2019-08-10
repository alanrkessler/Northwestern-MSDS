# Week 3 - Vector Representation

For Week 3, I am using TFIDF and Word2Vec to represent the most important and prevalent terms and topics. 

### Scripts
-   `corpus_creation.py` - Creates a CSV corpus following the examples for the class selecting all documents from the `raw_data` directory. The result is `corpus.csv`.
-   `manual_count.py` - Counts the number of times a given term is found in each document for the qualitative analysis. It imports the list of terms from `Manual_Terms.csv` and writes the counts to `manual_term_occ.csv`.
-   `vectorized_rep.py` - Uses scikit-learn's TF-IDF vectorizer and Word2Vec for extraction. This script also processes the text data prior to analysis. TF-IDF output is written to `tfidf_output.csv` and correlation among top terms is written to `correlation.csv`. Word2Vec output is written to `w2v_output.csv`.