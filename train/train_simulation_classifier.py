import bz2
import os
import pickle as pkl
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score
import datetime

sys.path.append("../front_end")

from simulation_classifier_annotations import annotations
from util.page_tokeniser import tokenise_text_and_lowercase, iterate_tokens
from processors.simulation_extractor import FEATURE_NAMES, extract_features

INPUT_FOLDER = "../data/preprocessed_tika/"
OUTPUT_FOLDER = "../front_end/models/"
OUTPUT_FILE = OUTPUT_FOLDER + "/simulation_classifier.pkl.bz2"
DIAGNOSTICS_FOLDER = "diagnostics"
DIAGNOSTICS_FILE = DIAGNOSTICS_FOLDER + "/simulation_classifier_diagnostics.txt"
DIAGNOSTICS_FILE_FEATURE_IMPORTANCES = DIAGNOSTICS_FOLDER + "/simulation_feature_importances.png"
DIAGNOSTICS_FILE_MISCLASSIFIED = DIAGNOSTICS_FOLDER + "/simulation_misclassified.xlsx"
DIAGNOSTICS_FILE_POSITIVE_TRAINING_EXAMPLES = DIAGNOSTICS_FOLDER + "/simulation_positive_training_examples.xlsx"
DIAGNOSTICS_FILE_TOKENS_AND_BIGRAMS = DIAGNOSTICS_FOLDER + "/simulation_common_tokens_and_bigrams.txt"
DIAGNOSTICS_FILE_PAGE_LEVEL_CONFUSION_MATRIX_IMAGE = DIAGNOSTICS_FOLDER + "/simulation_classifier_confusion_matrix_01_page_level_tf_idf_naive_bayes.png"
DIAGNOSTICS_FILE_DOC_LEVEL_CONFUSION_MATRIX_IMAGE = DIAGNOSTICS_FOLDER + "/simulation_classifier_confusion_matrix_02_document_level_tf_idf_naive_bayes.png"
DIAGNOSTICS_FILE_FEATURE_ENGINEERED_CONFUSION_MATRIX_IMAGE = DIAGNOSTICS_FOLDER + "/simulation_classifier_confusion_matrix_03_document_level_feature_engineered.png"
DIAGNOSTICS_FILE_WORDCLOUD = DIAGNOSTICS_FOLDER + "/simulation_classifier_wordcloud.png"
SUMMARY_FILE = "diagnostics/summary.txt"

NUM_FEATURES = 1500

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
    print ("ERROR! NO TRAINING FILES WERE FOUND.\nHave you downloaded the training data to the data folder?\nPlease go to data/raw_protocols and run download_raw_protocols.sh, then go to folder data/ and run the file preprocess.py, then return to this file and run it and you can train the simulation classifier.")
    exit()

if False:
    # Alternative code branch which trains a page-level Naive Bayes classifier
    # and generates a word cloud.
    # This is useful for diagnostics.

    tok = RegexpTokenizer(r'[a-zA-Z]+')

    stops = set(stopwords.words('english')).union(set(stopwords.words('french')))

    for c in pycountry.countries:
        for t in tok.tokenize(c.name.lower()):
            if t not in ("monte", "carlo"):
                stops.add(t)

    with open("simulation_classifier_stopwords.txt", encoding="utf-8") as f:
        for l in f:
            for token in tok.tokenize(l.strip().lower()):
                stops.add(token)

    # Clean up the stopwords - there are some stopwords which are indicative of it being the effect estimate.
    print(f"Loaded {len(stops)} stopwords")

    annotated_files = list(sorted(annotations.keys()))

    texts = []
    labels = []
    file_sources = []
    page_nos = []
    for file_name, annots in annotations.items():
        for page_no, page_text in enumerate(file_to_text[file_name]):
            label = int(page_no in annots)
            texts.append(page_text)
            labels.append(label)
            file_sources.append(file_name)
            page_nos.append(page_no)

    df_all_data = pd.DataFrame({"text": texts, "label": labels, "file": file_sources, "page_no": page_nos})

    print(f"There are {len(df_all_data)} instances")
    print("Class breakdowns (page level):")
    print(df_all_data.label.value_counts())

    print("Class breakdowns (document level):")
    df_all_documents = df_all_data.groupby("file")[["label"]].any()
    df_all_documents["label"] = df_all_documents["label"].apply(int)
    print(df_all_documents.label.value_counts())

    print(f"Writing common tokens and bigrams to {DIAGNOSTICS_FILE_TOKENS_AND_BIGRAMS}")
    with open(DIAGNOSTICS_FILE_TOKENS_AND_BIGRAMS, "w", encoding="utf-8") as f:
        f.write("-- MOST COMMON TOKENS --\n")

        all_tokens = []
        tokenised_docs = df_all_data[df_all_data.label == 1].text.apply(tokenise_text_and_lowercase)
        for tokens in tokenised_docs:
            all_tokens.extend(tokens)
        ctr = Counter()
        for token in all_tokens:
            ctr[token] += 1

        for token, frequency in sorted(ctr.items(), key=operator.itemgetter(1), reverse=True)[:200]:
            f.write(f"{token}\t{frequency}\n")

        #  Find out commonest bigrams
        bgs = nltk.bigrams(all_tokens)

        f.write("\n\n-- MOST COMMON BIGRAMS --\n")

        fdist = nltk.FreqDist(bgs)
        for (token1, token2), frequency in sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)[:200]:
            f.write(f"{token1}\t{token2}\t{frequency}\n")

    print(f"Saving positive examples to {DIAGNOSTICS_FILE_POSITIVE_TRAINING_EXAMPLES}")
    df_all_data[df_all_data.label == 1].to_excel(DIAGNOSTICS_FILE_POSITIVE_TRAINING_EXAMPLES)


    def upsample(df):
        extra_df = df[df.label == 1]
        rows_to_upsample = [df]
        for i in range(10):
            rows_to_upsample.append(extra_df)
        return pd.concat(rows_to_upsample)


    vectoriser = CountVectorizer(lowercase=True, stop_words=stops, max_features=NUM_FEATURES,
                                 # analyzer=tokenise_text_and_lowercase
                                 token_pattern=r'[a-zA-Z][a-zA-Z]+', min_df=15
                                 )
    transformer = TfidfTransformer()

    nb = MultinomialNB()
    model = make_pipeline(vectoriser, transformer, nb)

    ground_truths = []
    y_pred = []
    y_pred_proba = []

    doc_ground_truths = []
    doc_pred = []
    doc_y_pred_proba = []
    doc_files = []

    for test_file in annotated_files + [None]:
        if test_file:
            print(f"Testing model on {test_file} and training on all other documents.")
        else:
            print("Training model on all data.")

        df_train = df_all_data[df_all_data.file != test_file]
        df_test = df_all_data[(df_all_data.file == test_file) | (test_file is None)]
        print(f"\tThere are {len(df_train)} training instances and {len(df_test)} test instances")

        print(f"Breakdown of categories in training data before upsampling:")
        print(df_train.label.value_counts())
        print("Running upsampling")
        upsampled_df_train = upsample(df_train)
        print(f"Breakdown of categories in training data after upsampling:")
        print(upsampled_df_train.label.value_counts())

        print(f"Number of upsampled training instances: {len(upsampled_df_train)}")

        print("Fitting model...")

        model.fit(upsampled_df_train.text, upsampled_df_train.label)

        if not test_file:
            break

        ground_truths.extend(df_test.label)
        page_level_probas = model.predict_proba(df_test.text)[:, 1]
        y_pred_proba.extend(page_level_probas)
        prediction = model.predict(df_test.text)
        y_pred.extend(prediction)

        doc_files.append(test_file)
        doc_ground_truths.append(int(df_test.label.any()))
        doc_pred.append(int(max(prediction)))
        doc_y_pred_proba.append(np.max(page_level_probas))

    print(f"Testing page level models on validation data ({len(doc_ground_truths)}-fold cross-validation)")
    print(f"Accuracy: {accuracy_score(ground_truths, y_pred) * 100:.1f}%")

    print(f"Writing page level confusion matrix to {DIAGNOSTICS_FILE_PAGE_LEVEL_CONFUSION_MATRIX_IMAGE}")
    cm = confusion_matrix(ground_truths, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(DIAGNOSTICS_FILE_PAGE_LEVEL_CONFUSION_MATRIX_IMAGE)

    fpr, tpr, thresholds = roc_curve(ground_truths, y_pred_proba)
    print(f"\tPage level AUC is {auc(fpr, tpr):.2f}")

    print(f"Testing document level models on validation data ({len(doc_ground_truths)}-fold cross-validation)")
    print(f"Accuracy: {accuracy_score(doc_ground_truths, doc_pred) * 100:.1f}%")

    fpr, tpr, thresholds = roc_curve(doc_ground_truths, doc_y_pred_proba)
    print(f"\tDocument level AUC is {auc(fpr, tpr):.2f}")

    print(f"Writing document level confusion matrix to {DIAGNOSTICS_FILE_DOC_LEVEL_CONFUSION_MATRIX_IMAGE}")

    cm = confusion_matrix(doc_ground_truths, doc_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(DIAGNOSTICS_FILE_DOC_LEVEL_CONFUSION_MATRIX_IMAGE)

    print(f"Saving diagnostics information to {DIAGNOSTICS_FILE}")

    fake_document = " ".join(vectoriser.vocabulary_)
    vectorised_document = vectoriser.transform([fake_document])
    transformed_document = transformer.transform(vectorised_document)
    probas = np.zeros((transformed_document.shape[1]))

    with open(DIAGNOSTICS_FILE, "w", encoding="utf-8") as f:
        for prediction_idx in range(1, 2):
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

    # Create word cloud
    print(f"Saving word cloud to {DIAGNOSTICS_FILE_WORDCLOUD}")

    freqs = {}
    for ctr, j in enumerate(np.argsort(probas)):
        for w, i in vectoriser.vocabulary_.items():
            if i == j:
                freqs[w] = ctr

    wordcloud = WordCloud(width=1920, height=1080,
                          background_color='white',
                          min_font_size=10).generate_from_frequencies(freqs)

    wordcloud.to_file(DIAGNOSTICS_FILE_WORDCLOUD)

# Begin training linear regression model
print(f"Extracting features for linear regression model")
texts = []
gts = []
features = []
files = []
for file_name, annots in annotations.items():
    if file_name not in file_to_text:
        print (f"WARNING! Missing training file {file_name}. The finished model will be more accurate if you can supply all training data.")
        continue
    tokenised_pages = [tokenise_text_and_lowercase(text) for text in file_to_text[file_name]]
    all_tokens = list(iterate_tokens(tokenised_pages))

    feat, simulation_to_pages, contexts, _ = extract_features(all_tokens)

    gts.append(int(len(annots) > 0))
    features.append(feat)
    files.append(file_name)

print("Building training data frame")
df = pd.DataFrame()
df["gt"] = gts
df["file"] = files
for feature_idx, feature_name in enumerate(FEATURE_NAMES):
    df[feature_name] = [feat[feature_idx] for feat in features]

m = RandomForestClassifier()
m.fit(features, gts)

y_pred = []
y_pred_proba = []
for idx_to_test in range(len(df)):
    df_train = df.drop(index=idx_to_test)
    df_test = df[df.index == idx_to_test]

    m.fit(df_train[FEATURE_NAMES], df_train["gt"])
    pred = m.predict(df_test[FEATURE_NAMES])
    y_pred.append(pred[0])
    y_pred_proba.append(m.predict_proba(df_test[FEATURE_NAMES])[0][1])
df["y_pred"] = y_pred
df["y_pred_proba"] = y_pred_proba

fpr, tpr, thresholds = roc_curve(df["gt"], df.y_pred_proba)
print(f"\tDocument level AUC for feature engineered model is {auc(fpr, tpr):.2f}")

acc = accuracy_score(df["gt"], df.y_pred)

with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
    f.write(
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\tSimulation\t{acc* 100:.1f}%\t{auc(fpr, tpr):.2f}\n")


print(f"Writing document level confusion matrix to {DIAGNOSTICS_FILE_FEATURE_ENGINEERED_CONFUSION_MATRIX_IMAGE}")

cm = confusion_matrix(df["gt"], df.y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig(DIAGNOSTICS_FILE_FEATURE_ENGINEERED_CONFUSION_MATRIX_IMAGE)

print(f"Saving all documents which were missclassified to {DIAGNOSTICS_FILE_MISCLASSIFIED}")
df[df["y_pred"] != df["gt"]].to_excel(DIAGNOSTICS_FILE_MISCLASSIFIED, index=False)

df_fi = pd.DataFrame({"fi": m.feature_importances_, "name": FEATURE_NAMES})
df_fi["name"] = df_fi["name"].apply(lambda x: re.sub("-", "\nvs\n", x))

figure(figsize=(8, 6), dpi=80)
sns.barplot(x="fi", y="name", data=df_fi)
plt.title("Feature importances of classification model for simulation")
plt.savefig(DIAGNOSTICS_FILE_FEATURE_IMPORTANCES)

print("Retraining model on all data including training and validation data")
m.fit(df[FEATURE_NAMES], df["gt"])

print(f"Writing model to {OUTPUT_FILE}")

with bz2.open(OUTPUT_FILE, "wb") as f:
    pkl.dump(m, f)

print(f"\tSaved model to {OUTPUT_FILE}")
