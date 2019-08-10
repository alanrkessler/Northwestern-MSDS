"""Create corpus from plain text files as first step."""
import os
import glob
import pandas as pd


def retrieve_document(file_path):
    """Return text as a string for a given txt file path."""
    with open(file_path, 'r', encoding='utf-8') as document:
        lines = document.readlines()
    text = ''.join(lines)
    return text


def main():
    """Create corpus as a CSV file."""
    # Generate list of text file paths
    path_list = glob.glob('./raw_data/*.txt')
    # Generate list of text file names
    file_list = [os.path.basename(i) for i in path_list]

    # Generate list of text file content
    text_list = []
    for text_file in path_list:
        text_list.append(retrieve_document(text_file))

    # Create dictionary for corpus
    corpus = {'Doc_Title': file_list, 'Text': text_list}
    # Generate a CSV of the corpus
    pd.DataFrame(corpus).to_csv('./corpus.csv', index=file_list)


if __name__ == "__main__":
    main()
