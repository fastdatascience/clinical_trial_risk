import bz2
import operator
import os
import pickle as pkl
import re
import sys
from collections import Counter

import datetime
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pycountry
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

sys.path.append("../front_end")

from effect_estimate_classifier_annotations import annotations
from util.page_tokeniser import tokenise_pages, tokenise_text_and_lowercase, iterate_tokens

INPUT_FOLDER = "../data/preprocessed_tika/"
OUTPUT_FOLDER = "../front_end/models/"
OUTPUT_FILE = OUTPUT_FOLDER + "/effect_estimate_classifier.pkl.bz2"
DIAGNOSTICS_FOLDER = "diagnostics"
DIAGNOSTICS_FILE = DIAGNOSTICS_FOLDER + "/effect_estimate_classifier_diagnostics.txt"
DIAGNOSTICS_FILE_POSITIVE_TRAINING_EXAMPLES = DIAGNOSTICS_FOLDER + "/effect_estimate_positive_training_examples.txt"
DIAGNOSTICS_FILE_TOKENS_AND_BIGRAMS = DIAGNOSTICS_FOLDER + "/effect_estimate_common_tokens_and_bigrams.txt"
SUMMARY_FILE = "diagnostics/summary.txt"
NUM_FEATURES = 500

import sys

sys.path.append("../front_end")
# Function for converting the page-level probabilities of SAP into a document-level probability score.
from processors.effect_estimate_extractor import NUMBERS_REGEX, NUMBERS_IN_WORDS, transform_tokens, \
    get_context

tok = RegexpTokenizer(r'[a-zA-Z=≤≥<>]+')

stops = set(stopwords.words('english')).union(set(stopwords.words('french')))

for c in pycountry.countries:
    for t in tok.tokenize(c.name.lower()):
        stops.add(t)

with open("effect_estimate_classifier_stopwords.txt", encoding="utf-8") as f:
    for l in f:
        for token in tok.tokenize(l.strip().lower()):
            stops.add(token)

# Clean up the stopwords - there are some stopwords which are indicative of it being the effect estimate.
stops.remove("more")
stops.remove("to")
stops.remove("about")
stops.remove("than")
stops.remove("during")
stops.remove("if")
stops.remove("same")
stops.remove("between")
stops.remove("n")
stops.remove("above")
stops.remove("below")
stops.remove("we")
stops.remove("after")
stops.remove("until")
print(f"Loaded {len(stops)} stopwords")

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
    print ("ERROR! NO TRAINING FILES WERE FOUND.\nHave you downloaded the training data to the data folder?\nPlease go to data/raw_protocols and run download_raw_protocols.sh, then go to folder data/ and run the file preprocess.py, then return to this file and run it and you can train the effect estimate classifier.")
    exit()

annotated_files = list(sorted(annotations.keys()))

train_files, test_files = train_test_split(list(annotations), random_state=42)

print("Training files:")
print("Matching pages\tFile name")
for f in train_files:
    print(f"\t{len([x for x in annotations[f] if type(x) is int])}\t{f}\t")
print("Test files:")
print("Matching pages\tFile name")
for f in test_files:
    print(f"\t{len([x for x in annotations[f] if type(x) is int])}\t{f}\t")

instances = []
labels = []
contexts = []
file_sources = []
for test_file, annots in annotations.items():
    if test_file not in file_to_text:
        print (f"WARNING! Missing training file {test_file}. The finished model will be more accurate if you can supply all training data.")
        continue
    tokenised_pages = tokenise_pages(file_to_text[test_file])
    all_tokens = list(iterate_tokens(tokenised_pages))

    all_page_nos_that_should_match = set()
    all_page_nos_that_did_match = set()

    first_token_annots = [re.split(r"-| ", x)[0] for x in annots if type(x) is str and ("-" in x or " " in x)]

    for idx, (page_no, token_no, token) in enumerate(all_tokens):
        if page_no in annots:
            all_page_nos_that_should_match.add(page_no)
        if NUMBERS_REGEX.match(token) or token in NUMBERS_IN_WORDS:
            label = 0
            if page_no in annots and (token in annots or token in first_token_annots):
                all_page_nos_that_did_match.add(page_no)
                label = 1
            context = get_context(all_tokens, idx)
            if context:
                instances.append(token)
                labels.append(label)
                contexts.append(context)
                file_sources.append(test_file)

    for p in all_page_nos_that_should_match:
        if not p in all_page_nos_that_did_match:
            print("Missing match", test_file, p, annots)

ctr = Counter()
for l in labels:
    ctr[l] += 1

print("Breakdown of training data classes:")
print(ctr)

print(f"Writing positive training examples to {DIAGNOSTICS_FILE_POSITIVE_TRAINING_EXAMPLES}")

with open(DIAGNOSTICS_FILE_POSITIVE_TRAINING_EXAMPLES, "w", encoding="utf-8") as f:
    f.write("All the positive training examples:\n")
    ctr = Counter()
    for file, instance, label, context in zip(file_sources, instances, labels, contexts):
        if label == 1:
            f.write(file + "\t" + " ".join(context) + "\n")
            for w in context:
                ctr[w] += 1

print(f"Writing common tokens and bigrams to {DIAGNOSTICS_FILE_TOKENS_AND_BIGRAMS}")
with open(DIAGNOSTICS_FILE_TOKENS_AND_BIGRAMS, "w", encoding="utf-8") as f:
    f.write("-- MOST COMMON TOKENS --\n")

    for token, frequency in sorted(ctr.items(), key=operator.itemgetter(1), reverse=True)[:200]:
        f.write(f"{token}\t{frequency}\n")

    #  Find out commonest bigrams
    tokens = []
    for instance, label, context in zip(instances, labels, contexts):
        if label == 1:
            for token in context:
                tokens.append(token)
    bgs = nltk.bigrams(tokens)

    f.write("\n\n-- MOST COMMON BIGRAMS --\n")

    fdist = nltk.FreqDist(bgs)
    for (token1, token2), frequency in sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)[:200]:
        f.write(f"{token1}\t{token2}\t{frequency}\n")

vectoriser = CountVectorizer(lowercase=True, stop_words=stops, min_df=1, max_features=NUM_FEATURES,
                             analyzer=tokenise_text_and_lowercase
                             )
transformer = TfidfTransformer()

nb = MultinomialNB()
model = make_pipeline(vectoriser, transformer, nb)

df_all_data = pd.DataFrame({"tokens": contexts, "label": labels, "file": file_sources})

df_all_data["text"] = df_all_data["tokens"].apply(lambda t: " ".join(t))

df_train = df_all_data[df_all_data.file.isin(train_files)].copy()
df_test = df_all_data[df_all_data.file.isin(test_files)].copy()

print(f"There are {len(df_train)} training instances and {len(df_test)} test instances")
print("Class breakdowns:")
print("Train:")
print(df_train.label.value_counts())

print("Test:")
print(df_test.label.value_counts())


def upsample(df):
    extra_df = df[df.label == 1]
    rows_to_upsample = [df]
    for i in range(20):
        rows_to_upsample.append(extra_df)
    return pd.concat(rows_to_upsample)


df_train = upsample(df_train)
df_all_upsampled = upsample(df_all_data)

print("Class distribution after upsampling:")
print(df_train.label.value_counts())

# Try simple TF*IDF + Naive Bayes

model.fit(df_train.text, df_train.label)
y_pred = model.predict(df_test.text)
y_pred_proba = model.predict_proba(df_test.text)
df_test["y_pred"] = y_pred
df_test["y_pred_proba"] = y_pred_proba[:, 1]


def validate(graph_title, filename, df_test, is_write_summary = False):
    y, y_pred, y_pred_proba = df_test.label, df_test.y_pred, df_test.y_pred_proba
    print("Testing model on validation data...")

    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc * 100:.1f}%")

    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    print(f"\tAUC is {auc(fpr, tpr):.3f}")

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(graph_title)
    full_file = DIAGNOSTICS_FOLDER + "/effect_estimate_confusion_matrix_" + filename + ".png"
    print(f"Writing confusion matrix to {full_file}")
    plt.savefig(full_file)

    print("Calculating document level accuracy")
    doc_level_ground_truths_and_predictions = df_test.groupby("file").agg(
        {"label": "any", "y_pred": "any", "y_pred_proba": "max"})
    num_docs = len(set(df_test.file))
    doc_level_ground_truths_and_predictions[
        "is_correct"] = doc_level_ground_truths_and_predictions.label == doc_level_ground_truths_and_predictions.y_pred
    num_docs_correct = doc_level_ground_truths_and_predictions["is_correct"].sum()
    print("Document level ground truths:",
          " ".join(doc_level_ground_truths_and_predictions.label.apply(lambda x: f"{x:.3f}")))
    print("Document level predictions:  ",
          " ".join(doc_level_ground_truths_and_predictions.y_pred.apply(lambda x: f"{x:.3f}")))
    print("Document level predictions:  ",
          " ".join(doc_level_ground_truths_and_predictions.y_pred_proba.apply(lambda x: f"{x:.3f}")))
    print(
        f"Number of documents correctly classified as having/not having effect estimate: {num_docs_correct} of {num_docs}. Accuracy: {num_docs_correct / num_docs * 100:.1f}%")
    fpr, tpr, thresholds = roc_curve(doc_level_ground_truths_and_predictions.label,
                                     doc_level_ground_truths_and_predictions.y_pred_proba)
    dauc = auc(fpr, tpr)
    print(f"\tDocument level AUC is {dauc:.3f}")

    if is_write_summary:
        with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\tEffect Estimate\t{acc * 100:.1f}%\t{dauc:.3f}\n")


validate("Confusion matrix for effect estimate model\nToken level,\nsimple TF*IDF + Naive Bayes on window of 20 tokens",
         "01_simple_tf_idf_naive_bayes", df_test)

fake_document = " ".join(vectoriser.vocabulary_)
vectorised_document = vectoriser.transform([fake_document])
transformed_document = transformer.transform(vectorised_document)
probas = np.zeros((transformed_document.shape[1]))

with open(DIAGNOSTICS_FILE, "w", encoding="utf-8") as f:
    for prediction_idx in [1]:
        f.write(f"Strongest predictors for class {prediction_idx}\n")
        for i in range(transformed_document.shape[1]):
            zeros = np.zeros(transformed_document.shape)
            zeros[0, i] = transformed_document[0, i]
            proba = nb.predict_log_proba(zeros)
            probas[i] = proba[0, prediction_idx]

        for ctr, j in enumerate(np.argsort(-probas)):
            for w, i in vectoriser.vocabulary_.items():
                if i == j:
                    f.write(f"{ctr}\t{w}\n")

# Now convert to the weighted model


X_train = transform_tokens(list(df_train.tokens), vectoriser)

shorter_pipeline = make_pipeline(transformer, nb)
shorter_pipeline.fit(X_train, df_train.label)

X_test = transform_tokens(list(df_test.tokens), vectoriser)
X_all = transform_tokens(list(df_all_upsampled.tokens), vectoriser)

y_pred = shorter_pipeline.predict(X_test)
y_pred_proba = shorter_pipeline.predict_proba(X_test)
df_test["y_pred"] = y_pred
df_test["y_pred_proba"] = y_pred_proba[:, 1]

validate(
    "Confusion matrix for effect estimate model\nToken level, simple TF*IDF + Naive Bayes\non window of 20 tokens with\nhigher weights for tokens close to candidate",
    "02_proximity_weighted_tf_idf_naive_bayes", df_test, True)

print("Refitting model on all data (train+val)...")
shorter_pipeline.fit(X_all, df_all_upsampled.label)

y_pred = shorter_pipeline.predict(X_test)
y_pred_proba = shorter_pipeline.predict_proba(X_test)
df_test["y_pred"] = y_pred
df_test["y_pred_proba"] = y_pred_proba[:, 1]

validate(
    "Confusion matrix for effect estimate model\nToken level, simple TF*IDF + Naive Bayes\non window of 20 tokens with\nhigher weights for tokens\nclose to candidate\nTrained and validated on all data",
    "03_proximity_weighted_tf_idf_naive_bayes_trained_on_all_data", df_test)

print(f"Writing model to {OUTPUT_FILE}")

with bz2.open(OUTPUT_FILE, "wb") as f:
    pkl.dump(model, f)

print(f"\tSaved model to {OUTPUT_FILE}")

print(f"Saving diagnostics information to {DIAGNOSTICS_FILE}")

fake_document = " ".join(vectoriser.vocabulary_)
vectorised_document = vectoriser.transform([fake_document])
transformed_document = transformer.transform(vectorised_document)
probas = np.zeros((transformed_document.shape[1]))

with open(DIAGNOSTICS_FILE, "w", encoding="utf-8") as f:
    for prediction_idx in [1]:
        f.write(f"Strongest predictors for class {prediction_idx}\n")
        for i in range(transformed_document.shape[1]):
            zeros = np.zeros(transformed_document.shape)
            zeros[0, i] = transformed_document[0, i]
            proba = nb.predict_log_proba(zeros)
            probas[i] = proba[0, prediction_idx]

        for ctr, j in enumerate(np.argsort(-probas)):
            for w, i in vectoriser.vocabulary_.items():
                if i == j:
                    f.write(f"{ctr}\t{w}\n")
