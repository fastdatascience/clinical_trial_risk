import bz2
import pickle as pkl
from os.path import exists

import numpy as np


# Best model: Model 9

class NumSubjectsExtractorNaiveBayes:

    def __init__(self, path_to_classifier):
        print("Initialising Num Subjects classifier", path_to_classifier)
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD NUM SUBJECTS CLASSIFIER {path_to_classifier}. You need to run the training script.")
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
        Identify whether the trial takes place in multiple countries.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from condition to the pages it's mentioned in.
        """
        if self.model is None:
            print("Warning! Num subjects classifier not loaded.")
            return {"prediction": "Error"}

        token_counts = np.zeros((1, len(self.vectoriser.vocabulary_)))
        for page_no, tokens in enumerate(tokenised_pages):
            if page_no >= 30:
                break
            for token_idx, token in enumerate(tokens):
                token_lower = token.lower()
                if token_lower in self.vectoriser.vocabulary_:
                    token_counts[0, self.vectoriser.vocabulary_[token_lower]] += 1

                if token_idx < len(tokens) - 1:
                    tokens_lower = (token + " " + tokens[token_idx + 1]).lower()
                    if tokens_lower in self.vectoriser.vocabulary_:
                        token_counts[0, self.vectoriser.vocabulary_[tokens_lower]] += 1
        transformed_document = self.transformer.transform(token_counts)
        prediction = self.nb.predict(transformed_document)[0]

        prediction_proba = self.nb.predict_proba(transformed_document)[0]

        subjects_to_score = {}
        for idx, class_name in enumerate(self.nb.classes_):
            subjects_to_score[class_name] = prediction_proba[idx]

        return {"prediction": prediction, "pages": {},
                "proba": subjects_to_score}
