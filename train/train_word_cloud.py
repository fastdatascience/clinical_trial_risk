import bz2
import operator
import os
import pickle as pkl
import re
import sys
from collections import Counter

import numpy as np

sys.path.append("../front_end/")

from util.page_tokeniser import tokenise_pages
from util.word_cloud_generator import word_cloud_token_regex

INPUT_FOLDER = "../data/preprocessed_tika/"
OUTPUT_FOLDER = "../front_end/models/"
OUTPUT_FILE = OUTPUT_FOLDER + "/idfs_for_word_cloud.pkl.bz2"
DIAGNOSTICS_FOLDER = "diagnostics"
DIAGNOSTICS_FILE = DIAGNOSTICS_FOLDER + "/word_cloud_idf_diagnostics.txt"

file_to_text = {}
for root, folder, files in os.walk(INPUT_FOLDER):
    for file_name in files:
        if not file_name.endswith("pkl"):
            continue
        pdf_file = re.sub(".pkl", "", file_name)

        full_file = INPUT_FOLDER + "/" + file_name
        with open(full_file, 'rb') as f:
            text = pkl.load(f)
        file_to_text[pdf_file] = text

if len(file_to_text) == 0:
    print ("ERROR! NO TRAINING FILES WERE FOUND.\nHave you downloaded the training data to the data folder?\nPlease go to data/raw_protocols and run download_raw_protocols.sh, then go to folder data/ and run the file preprocess.py, then return to this file and run it and you can train the word cloud.")
    exit()

# Get the document frequency of tokens
dfs = Counter()

for file, pages in file_to_text.items():
    tokens = set()
    for page_tokens in tokenise_pages(pages):
        for token in page_tokens:
            if word_cloud_token_regex.match((token)):
                tokens.add(token.lower())
    for token in tokens:
        dfs[token] += 1

# Calculate Inverse Document Frequency (IDF)
idf = dict()
for token, df in dfs.items():
    if df > 1:
        idf[token] = np.log(len(file_to_text) / df)

idf[""] = np.log(len(file_to_text) / 1)
print(f"Saving IDFS to {OUTPUT_FILE}")
with bz2.open(OUTPUT_FILE, "wb") as f:
    pkl.dump(idf, f)

print(f"Saving diagnostics information to {DIAGNOSTICS_FILE}")

with open(DIAGNOSTICS_FILE, "w", encoding="utf-8") as f:
    for term, value in sorted(idf.items(), key=operator.itemgetter(1)):
        f.write(f"{term}\t{value}\n")
