import bz2
import pickle as pkl
from os.path import exists

import numpy as np


class ConditionExtractor:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD CONDITION CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.model = None
            return
        with bz2.open(path_to_classifier, "rb") as f:
            self.model = pkl.load(f)
        self.vectoriser = self.model.named_steps['countvectorizer']
        self.transformer = self.model.named_steps['tfidftransformer']
        self.nb = self.model.named_steps['multinomialnb']

        self.vocabulary = {v: k for k, v in self.vectoriser.vocabulary_.items()}

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify the pathology of the trial.
        Just HIV and TB supported currently.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from condition to the pages it's mentioned in.
        """
        if self.model is None:
            print("Warning! Condition classifier not loaded.")
            return {"prediction": "Error"}

        token_counts = np.zeros((1, len(self.vectoriser.vocabulary_)))
        for tokens in tokenised_pages:
            for token in tokens:
                token_lower = token.lower()
                if token_lower in self.vectoriser.vocabulary_:
                    token_counts[0, self.vectoriser.vocabulary_[token_lower]] += 1

        transformed_document = self.transformer.transform(token_counts)

        prediction_probas = self.nb.predict_proba(transformed_document)[0]
        prediction_idx = int(np.argmax(prediction_probas))

        # prediction_idx = self.nb.predict(transformed_document)[0]
        if prediction_idx == 0:
            prediction = "Other"
        elif prediction_idx == 1:
            prediction = "HIV"
        else:
            prediction = "TB"

        probas = np.zeros((transformed_document.shape[1]))
        for i in range(transformed_document.shape[1]):
            zeros = np.zeros(transformed_document.shape)
            zeros[0, i] = transformed_document[0, i]
            proba = self.nb.predict_log_proba(zeros)
            probas[i] = proba[0, prediction_idx]
        """
        Identify the pathology of the trial.
        Just HIV and TB supported currently.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from condition to the pages it's mentioned in.
        """

        token_counts = np.zeros((1, len(self.vectoriser.vocabulary_)))
        for tokens in tokenised_pages:
            for token in tokens:
                token_lower = token.lower()
                if token_lower in self.vectoriser.vocabulary_:
                    token_counts[0, self.vectoriser.vocabulary_[token_lower]] += 1

        transformed_document = self.transformer.transform(token_counts)

        prediction_probas = self.nb.predict_proba(transformed_document)[0]
        prediction_idx = int(np.argmax(prediction_probas))

        # prediction_idx = self.nb.predict(transformed_document)[0]
        if prediction_idx == 0:
            prediction = "Other"
        elif prediction_idx == 1:
            prediction = "HIV"
        else:
            prediction = "TB"

        probas = np.zeros((transformed_document.shape[1]))
        for i in range(transformed_document.shape[1]):
            zeros = np.zeros(transformed_document.shape)
            zeros[0, i] = transformed_document[0, i]
            proba = self.nb.predict_log_proba(zeros)
            probas[i] = proba[0, prediction_idx]

        condition_to_pages = {}
        for vocab_idx in np.argsort(-probas):
            condition_to_pages[self.vocabulary[vocab_idx]] = []
            if len(condition_to_pages) > 20:
                break

        informative_terms = {}
        for vocab_idx in np.argsort(-probas):
            informative_terms[self.vocabulary[vocab_idx]] = 1
            if len(informative_terms) > 50:
                break

        for page_no, tokens in enumerate(tokenised_pages):
            for token in tokens:
                if token.lower() in condition_to_pages:
                    condition_to_pages[token.lower()].append(page_no)

        return {"prediction": prediction, "pages": condition_to_pages, "score": prediction_probas[prediction_idx],
                "probas": list(prediction_probas), "terms": informative_terms}
