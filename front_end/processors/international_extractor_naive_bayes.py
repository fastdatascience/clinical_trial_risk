import bz2
import pickle as pkl
from os.path import exists

import numpy as np


# Best model: Model 9

class InternationalExtractorNaiveBayes:

    def __init__(self, path_to_classifier):
        print("Initialising int classifier", path_to_classifier)
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD INTERNATIONAL CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.model = None
            return
        with bz2.open(path_to_classifier, "rb") as f:
            self.model = pkl.load(f)
        self.vectoriser = self.model.named_steps['countvectorizer']
        self.transformer = self.model.named_steps['tfidftransformer']
        self.nb = self.model.named_steps['bernoullinb']

        self.vocabulary = {v: k for k, v in self.vectoriser.vocabulary_.items()}

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify whether the trial takes place in multiple countries.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from condition to the pages it's mentioned in.
        """
        if self.model is None:
            print("Warning! International classifier not loaded.")
            return {"prediction": "Error"}

        # print ("toks", tokenised_pages[0][:100])

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
        prediction_proba = self.nb.predict_proba(transformed_document)[0][1]

        is_international_pred = int(prediction_proba > 0.5)

        return {"prediction": is_international_pred, "pages": {}, "score": prediction_proba}
