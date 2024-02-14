import bz2
import os
import pickle as pkl
import re
import sys
from collections import Counter

import numpy as np
import pycountry
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import datetime

from sap_classifier_annotations import annotations

sys.path.append("../front_end")
# Function for converting the page-level probabilities of SAP into a document-level probability score.
from processors.sap_extractor import derive_feature

INPUT_FOLDER = "../data/preprocessed_tika/"
OUTPUT_FOLDER = "../front_end/models/"
OUTPUT_FILE = OUTPUT_FOLDER + "/sap_classifier.pkl.bz2"
DIAGNOSTICS_FILE = "diagnostics/sap_classifier_diagnostics.txt"
NUM_FEATURES = 500
SUMMARY_FILE = "diagnostics/summary.txt"

tok = RegexpTokenizer(r'[a-zA-Z]+')

stops = set(stopwords.words('english')).union(set(stopwords.words('french')))

for c in pycountry.countries:
    for t in tok.tokenize(c.name.lower()):
        if t not in ("monte", "carlo"):
            stops.add(t)

with open("sap_classifier_stopwords.txt", encoding="utf-8") as f:
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
    print ("ERROR! NO TRAINING FILES WERE FOUND.\nHave you downloaded the training data to the data folder?\nPlease go to data/raw_protocols and run download_raw_protocols.sh, then go to folder data/ and run the file preprocess.py, then return to this file and run it and you can train the SAP classifier.")
    exit()

def upsample(features, labels):
    original_number_of_training_examples = len(labels)
    upsampled_features = []
    upsampled_labels = []
    for idx in range(original_number_of_training_examples):
        label = labels[idx]
        feature = features[idx]
        repetitions = 1
        if label == 1:
            repetitions = 6
        for r in range(repetitions):
            upsampled_features.append(feature)
            upsampled_labels.append(label)
    return upsampled_features, upsampled_labels


for test_file in list(annotations):
    if test_file not in file_to_text:
        print (f"WARNING! Missing training file {test_file}. The finished model will be more accurate if you can supply all training data.")
        
annotated_files = list([a for a in annotations.keys() if a in file_to_text])


# Specially engineered regex to include 95%, 95%ci, etc
vectoriser = CountVectorizer(lowercase=True, stop_words=list(stops), min_df=5, max_features=NUM_FEATURES,
                             token_pattern=r'[59][05]%?(?:ci)?|[a-z][a-z]+')
transformer = TfidfTransformer()

nb = MultinomialNB()
model = make_pipeline(vectoriser, transformer, nb)

ground_truths = []
y_pred = []
y_pred_proba = []

doc_ground_truths = []
doc_features = []
doc_files = []



for test_file in annotated_files + [None]:
    if test_file:
        print(f"Testing model on {test_file} and training on all other documents.")
    else:
        print("Training model on all data.")

    features_train = []
    labels_train = []
    features_test = []
    labels_test = []
    for file, annotations_for_this_file in annotations.items():
        if file not in file_to_text:
            continue
        if file == test_file:
            for page_no, page_text in enumerate(file_to_text[file]):
                if page_no in annotations_for_this_file:
                    cat = 1
                else:
                    cat = 0
                features_test.append(page_text)
                labels_test.append(cat)
        else:
            for page_no, page_text in enumerate(file_to_text[file]):
                if page_no in annotations_for_this_file:
                    cat = 1
                else:
                    cat = 0
                features_train.append(page_text)
                labels_train.append(cat)

    print("Running upsampling")
    upsampled_features_train, upsampled_labels_train = upsample(features_train, labels_train)
    print(f"Breakdown of categories in training data after upsampling: {Counter(upsampled_labels_train)}")

    print(f"Number of training instances: {len(upsampled_features_train)}")

    print("Fitting model...")

    model.fit(upsampled_features_train, upsampled_labels_train)

    if not test_file:
        break

    ground_truths.extend(labels_test)
    page_level_probas = model.predict_proba(features_test)[:, 1]
    y_pred_proba.extend(page_level_probas)
    y_pred.extend(model.predict(features_test))

    doc_files.append(test_file)
    doc_ground_truths.append(int(1 in labels_test))
    doc_features.append(page_level_probas)

print(f"Testing page level models on validation data ({len(ground_truths)}-fold cross-validation)")
print(f"Accuracy: {accuracy_score(ground_truths, y_pred) * 100:.1f}%")

fpr, tpr, thresholds = roc_curve(ground_truths, y_pred_proba)
print(f"\tAUC is {auc(fpr, tpr):.2f}")

doc_derived_features = np.asarray([derive_feature(f) for f in doc_features])

m = RandomForestClassifier()

cv_scores = cross_validate(m, doc_derived_features, doc_ground_truths, scoring=["roc_auc", "accuracy"], cv=10)

print(f"Cross-validation AUC: {cv_scores['test_roc_auc'].mean()}, accuracy: {cv_scores['test_accuracy'].mean()}")


with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
    f.write(
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\tSAP\t{cv_scores['test_accuracy'].mean() * 100}%\t{cv_scores['test_roc_auc'].mean()}\n")


print("Refitting secondary model on all data")

m.fit(doc_derived_features, doc_ground_truths)

print("Accuracy is now", accuracy_score(doc_ground_truths, m.predict(doc_derived_features)))

fpr, tpr, thresholds = roc_curve(doc_ground_truths, m.predict_proba(doc_derived_features)[:, 1])
print(f"\tAUC is {auc(fpr, tpr):.2f}")

print(f"Writing both models to {OUTPUT_FILE}")

with bz2.open(OUTPUT_FILE, "wb") as f:
    pkl.dump([model, m], f)

print(f"\tSaved models to {OUTPUT_FILE}")

print(f"Saving diagnostics information to {DIAGNOSTICS_FILE}")

fake_document = " ".join(vectoriser.vocabulary_)
vectorised_document = vectoriser.transform([fake_document])
transformed_document = transformer.transform(vectorised_document)
probas = np.zeros((transformed_document.shape[1]))

with open(DIAGNOSTICS_FILE, "w", encoding="utf-8") as f:
    for prediction_idx in range(2):
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
