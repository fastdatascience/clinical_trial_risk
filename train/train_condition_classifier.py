import bz2
import os
import pickle as pkl
import re
from collections import Counter

import numpy as np
import pycountry
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import datetime

INPUT_FOLDER = "../data/preprocessed_tika/"
OUTPUT_FOLDER = "../front_end/models/"
OUTPUT_FILE = OUTPUT_FOLDER + "/condition_classifier.pkl.bz2"
DIAGNOSTICS_FILE = "diagnostics/condition_classifier_diagnostics.txt"
SUMMARY_FILE = "diagnostics/summary.txt"
NUM_FEATURES = 1000

tok = RegexpTokenizer(r'[a-zA-Z=≤≥<>]+')

stops = set(stopwords.words('english')).union(set(stopwords.words('french')))

for c in pycountry.countries:
    for t in tok.tokenize(c.name.lower()):
        stops.add(t)

with open("condition_classifier_stopwords.txt", encoding="utf-8") as f:
    for l in f:
        for token in tok.tokenize(l.strip().lower()):
            stops.add(token)

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
    print ("ERROR! NO TRAINING FILES WERE FOUND.\nHave you downloaded the training data to the data folder?\nPlease go to data/raw_protocols and run download_raw_protocols.sh, then go to folder data/ and run the file preprocess.py, then return to this file and run it and you can train the condition (pathology) classifier.")
    exit()

words = []
categories = []

for file, pages in file_to_text.items():
    cat = 0
    if "HIV" in file:
        cat = 1
    elif "TB" in file:
        cat = 2
    # TODO: if "Abrams" in file cat is both.
    # Make classes not mutually exclusive.

    words.append(" ".join(pages))
    categories.append(cat
                      )

print(f"Breakdown of categories in training and validation data: {Counter(categories)}")

features_train, features_test, labels_train, labels_test = train_test_split(words, categories, test_size=0.1,
                                                                            random_state=10)


def upsample(features, labels):
    original_number_of_training_examples = len(labels)
    upsampled_features = []
    upsampled_labels = []
    for idx in range(original_number_of_training_examples):
        label = labels[idx]
        feature = features[idx]
        repetitions = 1
        if label == 1:
            repetitions = 2
        elif label == 2:
            repetitions = 4
        for r in range(repetitions):
            upsampled_features.append(feature)
            upsampled_labels.append(label)
    return upsampled_features, upsampled_labels


print("Running upsampling")
upsampled_features_train, upsampled_labels_train = upsample(features_train, labels_train)
print(f"Breakdown of categories in training data after upsampling: {Counter(upsampled_labels_train)}")

upsampled_features_all, upsampled_labels_all = upsample(words, categories)

print(f"Breakdown of categories in all data after upsampling: {Counter(upsampled_labels_all)}")

# vectoriser = TfidfVectorizer(stop_words=stops, min_df=5, max_features=NUM_FEATURES,
#                              token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
#                              )

vectoriser = CountVectorizer(lowercase=True, stop_words=stops, min_df=5, max_features=NUM_FEATURES,
                             token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b')
transformer = TfidfTransformer()

nb = MultinomialNB()
model = make_pipeline(vectoriser, transformer, nb)

print(f"Number of training instances: {len(upsampled_features_train)}")

print("Fitting model...")

model.fit(upsampled_features_train, upsampled_labels_train)


def validate_model(is_write_summary = False):
    print("Testing model on validation data...")
    y_pred = model.predict(features_test)
    acc = accuracy_score(labels_test, y_pred)
    print(f"Accuracy: {acc * 100:.1f}%")

    print("\tGround truth:\t", labels_test)
    print("\tPrediction:\t", y_pred)

    y_pred_proba = model.predict_proba(features_test)
    aucs = []
    for category in range(3):
        fpr, tpr, thresholds = roc_curve(labels_test, y_pred_proba[:, category], pos_label=category)
        this_auc = auc(fpr, tpr)
        print(f"\tAUC for class {category} is {this_auc:.2f}")
        aucs.append(this_auc)

    if is_write_summary:
        with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\tCondition\t{acc * 100:.1f}%\t{np.mean(aucs)}\n")

validate_model(True)

print("Refitting model on all data (train+val)...")
model.fit(upsampled_features_all, upsampled_labels_all)

validate_model()

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
    for prediction_idx in range(3):
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
