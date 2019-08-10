"""Format extraction results for Week 2 Discussion."""

import os
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
    # Set working directory for this specific week
    owd = os.getcwd()
    os.chdir('./Extract Engine Comparison')
    # Create the corpus
    corpus_creation.main()
    # Search for manual terms
    manual = manual_term_search('./raw_data/*.txt', 'Manual_Terms.csv')
    # Import online output
    Doc1_FF = pd.read_csv('./online_terms/Doc1_FF.csv', index_col=0)
    Doc2_FF = pd.read_csv('./online_terms/Doc2_FF.csv', index_col=0)
    Doc1_TM = pd.read_csv('./online_terms/Doc1_TM.csv', index_col=1)
    Doc2_TM = pd.read_csv('./online_terms/Doc2_TM.csv', index_col=1)
    # Join results together
    Doc1 = Doc1_TM.join(Doc1_FF, how='outer')
    Doc2 = Doc2_TM.join(Doc2_FF, how='outer')
    Doc = Doc1.join(Doc2, how='outer', lsuffix='_Doc1', rsuffix='_Doc2')
    final = manual.join(Doc, how='outer')
    final.columns = ['Manual Count Doc 1', 'Manual Count Doc 2',
                     'TerMine Rank Doc 1', 'TerMine Score Doc 1',
                     'FiveFilters Occurrence Doc 1',
                     'FiveFilters Word Count Doc 1',
                     'TerMine Rank Doc 2', 'TerMine Score Doc 2',
                     'FiveFilters Occurrence Doc 2',
                     'FiveFilters Word Count Doc 2']
    final.to_csv('formatted_extracts.csv')
    os.chdir(owd)
    return final


if __name__ == '__main__':
    main()
