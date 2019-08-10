# Week 2 - Term Extraction

The assignment is to take the first two documents in the corpus, extract term extractions using multiple approaches, and compare/evaluate the results. 

-   `corpus_creation.py` - Creates a CSV corpus following the examples for the class selecting only the first two documents. The result is `corpus.csv`.
-   `extract_format.py` - Formats the occurrence and output information from manual extraction and the two automated extraction engines: TerMine and FiveFilter. The result is `formatted_extracts.csv`.
-   `tfidf_extract.py` - Uses scikit-learn's TF-IDF vectorizer to do this extraction within Python based on code provided to the class. The result is `tfidf_matrix.csv`. 
