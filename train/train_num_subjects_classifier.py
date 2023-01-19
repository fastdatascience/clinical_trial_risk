import bz2
import datetime
import os
import pickle as pkl
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../front_end")

from num_subjects_classifier_annotations import annotations
from processors.num_subjects_extractor import FEATURE_NAMES, extract_features
from util.page_tokeniser import tokenise_pages
from sklearn.metrics import accuracy_score

INPUT_FOLDER = "../data/preprocessed_tika/"
OUTPUT_FOLDER = "../front_end/models/"
OUTPUT_FILE = OUTPUT_FOLDER + "/num_subjects_classifier.pkl.bz2"
DIAGNOSTICS_FOLDER = "diagnostics"
DIAGNOSTICS_FILE_FEATURE_IMPORTANCES = DIAGNOSTICS_FOLDER + "/num_subjects_feature_importances.png"
DIAGNOSTICS_FILE_POSITIVE_EXAMPLES = DIAGNOSTICS_FOLDER + "/num_subjects_positive_examples.txt"
DIAGNOSTICS_FILE_N_GRAMS = DIAGNOSTICS_FOLDER + "/num_subjects_ngram_contexts.txt"
DIAGNOSTICS_FILE_EXCEL = DIAGNOSTICS_FOLDER + "/num_subjects_classifier_results.xlsx"
SUMMARY_FILE = "diagnostics/summary.txt"

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

print(f"Loaded {len(file_to_text)} files")

if len(file_to_text) == 0:
    print(
        "ERROR! NO TRAINING FILES WERE FOUND.\nHave you downloaded the training data to the data folder?\nPlease go to data/raw_protocols and run download_raw_protocols.sh, then go to folder data/ and run the file preprocess.py, then return to this file and run it and you can train the number of subjects classifier.")
    exit()

with open("../data/ctgov/protocols.pkl.gz", "rb") as f:
    file_to_pages_ctgov = pkl.load(f)

num_ctgov_files_loaded = 0
for annot in annotations:
    if "NCT" in annot:
        if annot not in file_to_pages_ctgov:
            print("missing protocol text for " + annot)
        else:
            file_to_text[annot] = file_to_pages_ctgov[annot]
            num_ctgov_files_loaded += 1

del file_to_pages_ctgov

print(f"Loaded {num_ctgov_files_loaded} ClinicalTrials.gov training files. Now {len(file_to_text)} files are loaded.")

df = pd.DataFrame()
df["file_name"] = list([a for a in annotations if annotations[a] is not None and a in file_to_text])
df["ground_truth"] = df.file_name.map(annotations).apply(lambda x: re.sub(r'\D.+', '', x))

print(f"Loaded {len(df)} annotations")

all_feature_sets = []
for file_name, ground_truth in annotations.items():
    if ground_truth is None:
        continue
    if file_name not in file_to_text:
        print(
            f"WARNING! Missing training file {file_name}.  The finished model will be more accurate if you can supply all training data.")
        continue
    raw_texts = file_to_text[file_name]
    tokenised_pages = list(tokenise_pages(raw_texts))

    if any(i.isdigit() for i in ground_truth):
        ground_truth = re.sub(r'\D.+', '', str(ground_truth))
    df_instances, _, _ = extract_features(tokenised_pages)
    df_instances["ground_truth"] = (df_instances["candidate"] == ground_truth).apply(int)
    df_instances["file_name"] = file_name

    if df_instances.ground_truth.sum() == 0:
        print("No matches found for", file_name, ground_truth)
        print ("\tskipping...")
        continue

    all_feature_sets.append(df_instances)

df_instances = pd.concat(all_feature_sets)

m = RandomForestClassifier()

y_pred = []
y_pred_proba = []
for test_file in df.file_name:
    print(f"Test file is {test_file}")
    df_train = df_instances[df_instances.file_name != test_file]
    m.fit(df_train[FEATURE_NAMES], df_train["ground_truth"])

    df_test = df_instances[df_instances.file_name == test_file]
    if len(df_test) > 0:
        probas = m.predict_proba(df_test[FEATURE_NAMES])

        winning_index = np.argmax(probas[:, 1])

        winner = df_test.candidate.iloc[winning_index]
        p = probas[winning_index]
    else:
        winner = "0"
        p = 0
    y_pred.append(winner)
    y_pred_proba.append(p)

df["y_pred"] = y_pred
df["is_correct"] = df["y_pred"] == df["ground_truth"]

acc = accuracy_score(df.ground_truth, df.y_pred)
print(f"Accuracy using {len(df)}-fold cross-validation is {acc * 100:.2f}%")

df["is_correct_within_10%_margin"] = (df.y_pred.apply(int) >= df.ground_truth.apply(int) * 0.9) & (
        df.y_pred.apply(int) <= df.ground_truth.apply(int) * 1.1)
acc_10pc = df['is_correct_within_10%_margin'].mean()
print(
    f"Proportion correct within 10% margin using {len(df)}-fold cross-validation is {acc_10pc * 100:.2f}%")

df.to_excel(DIAGNOSTICS_FILE_EXCEL, index=False)

with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
    f.write(
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\tNumber of subjects\t{acc * 100:.1f}%\t{acc_10pc * 100:.2f}%\tAccuracy within 10% margin\n")

print("Retraining model on all data")
m.fit(df_instances[FEATURE_NAMES], df_instances["ground_truth"])

print(f"Writing model to {OUTPUT_FILE}")

with bz2.open(OUTPUT_FILE, "wb") as f:
    pkl.dump(m, f)

print(f"\tSaved model to {OUTPUT_FILE}")

print(f"Saving feature importances graph to {DIAGNOSTICS_FILE_FEATURE_IMPORTANCES}")

df_fi = pd.DataFrame({"fi": m.feature_importances_, "name": FEATURE_NAMES})

figure(figsize=(8, 6), dpi=80)
sns.barplot(x="fi", y="name", data=df_fi)
plt.title("Weights of random forest classification model for sample size")
plt.savefig(DIAGNOSTICS_FILE_FEATURE_IMPORTANCES, dpi=300, bbox_inches = "tight")
