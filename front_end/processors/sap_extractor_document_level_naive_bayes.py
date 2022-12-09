import bz2
import pickle as pkl
from os.path import exists

import numpy as np


# Best model: Model 3

class SapExtractorDocumentLevel:

    def __init__(self, path_to_classifier):
        print("Initialising SAP document level classifier", path_to_classifier)
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD SAP DOCUMENT LEVEL CLASSIFIER {path_to_classifier}. You need to run the training script.")
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
        Identify whether the trial has a SAP.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from condition to the pages it's mentioned in.
        """
        if self.model is None:
            print("Warning! SAP document level classifier not loaded.")
            return {"prediction": "Error"}

        token_counts = np.zeros((1, len(self.vectoriser.vocabulary_)))
        for page_no, tokens in enumerate(tokenised_pages):
            for token_idx, token in enumerate(tokens):
                token_lower = token.lower()
                if token_lower in self.vectoriser.vocabulary_:
                    token_counts[0, self.vectoriser.vocabulary_[token_lower]] += 1
        transformed_document = self.transformer.transform(token_counts)
        prediction_proba = self.nb.predict_proba(transformed_document)[0][1]

        is_sap_pred = int(prediction_proba > 0.5)

        return {"prediction": is_sap_pred, "pages": {}, "score": prediction_proba}
