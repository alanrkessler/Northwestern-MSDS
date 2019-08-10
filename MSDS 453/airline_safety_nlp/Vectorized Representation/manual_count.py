"""Check occurrence rates of manually selected terms."""

import glob
import pandas as pd
import re
import corpus_creation


def load_manual_terms(manual_terms_path):
    """Return list of manual terms from CSV."""
    terms_df = pd.read_csv(manual_terms_path, header=None)
    manual_terms = terms_df[0].tolist()
    return manual_terms


def retrieve_document(file_path):
    """Return text as a string for a given txt file path."""
    with open(file_path, 'r', encoding='utf-8') as document:
        lines = document.readlines()
    text = ''.join(lines)
    return text


def count_terms(text, term):
    """Return the number of times a term occurs in text."""
    count = len(re.findall(term, text))
    return count


def manual_term_search(input_path, manual_terms_path):
    """Return dataframe of term occurrences."""
    # Load list of manual terms
    manual_terms = load_manual_terms(manual_terms_path)
    # Save terms and occurrences in dictionary
    manual_dict = {}
    manual_dict['Term'] = manual_terms
    # Load each text file and search for each term
    path_list = glob.glob(input_path)
    for i, path in enumerate(path_list):
        text = retrieve_document(path)
        count_list = []
        for term in manual_terms:
            count_list.append(count_terms(text, term))
        manual_dict[i] = count_list
    # Covert output into a dataframe
    df = pd.DataFrame.from_dict(manual_dict)
    df.set_index('Term', inplace=True)
    return df


def main():
    """Format extract results which is pretty manual for this assignment."""
    # Create the corpus
    corpus_creation.main()
    # Search for manual terms
    manual = manual_term_search('./raw_data/*.txt', 'Manual_Terms.csv')
    manual.to_csv('manual_term_occ.csv')
    return manual


if __name__ == '__main__':
    main()
