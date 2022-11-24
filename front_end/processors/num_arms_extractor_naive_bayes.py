import bz2
import pickle as pkl
import re
from os.path import exists

import numpy as np

is_number_regex = re.compile(r'^\d+$')


# Best model: Model 9

class NumArmsExtractorNaiveBayes:

    def __init__(self, path_to_classifier):
        print("Initialising Num Arms classifier", path_to_classifier)
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD NUM ARMS CLASSIFIER {path_to_classifier}. You need to run the training script.")
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
            print("Warning! Num arms classifier not loaded.")
            return {"prediction": "Error"}

        token_counts = np.zeros((1, len(self.vectoriser.vocabulary_)))
        for page_no, tokens in enumerate(tokenised_pages):
            # if page_no >= 30:
            #     break
            for token_idx, token in enumerate(tokens):
                token_lower = token.lower()
                # exclude single digits on their own - we only use these as a feature if part of bigram.
                if token_lower in self.vectoriser.vocabulary_ and not is_number_regex.match(token):
                    token_counts[0, self.vectoriser.vocabulary_[token_lower]] += 1

                if token_idx < len(tokens) - 1:
                    tokens_lower = (token + " " + tokens[token_idx + 1]).lower()
                    if tokens_lower in self.vectoriser.vocabulary_:
                        token_counts[0, self.vectoriser.vocabulary_[tokens_lower]] += 1
        transformed_document = self.transformer.transform(token_counts)
        prediction = self.nb.predict(transformed_document)[0]

        prediction_proba = self.nb.predict_proba(transformed_document)[0]
        prediction_idx = int(np.argmax(prediction_proba))

        arms_to_score = {}
        for idx, class_name in enumerate(self.nb.classes_):
            arms_to_score[class_name] = prediction_proba[idx]

        probas = np.zeros((transformed_document.shape[1]))
        for i in range(transformed_document.shape[1]):
            zeros = np.zeros(transformed_document.shape)
            zeros[0, i] = transformed_document[0, i]
            proba = self.nb.predict_log_proba(zeros)
            probas[i] = proba[0, prediction_idx]

        arms_to_pages = {}
        for vocab_idx in np.argsort(-probas):
            arms_to_pages[self.vocabulary[vocab_idx]] = []
            if len(arms_to_pages) > 20:
                break

        informative_terms = {}
        for vocab_idx in np.argsort(-probas):
            informative_terms[self.vocabulary[vocab_idx]] = 1
            if len(informative_terms) > 50:
                break

        for page_no, tokens in enumerate(tokenised_pages):
            for token in tokens:
                if token.lower() in arms_to_pages:
                    arms_to_pages[token.lower()].append(page_no)

        # Remove any stopwords which accidentally got in there.
        for w in ["to"]:
            if w in informative_terms:
                del informative_terms[w]
            if w in arms_to_pages:
                del arms_to_pages[w]

        return {"prediction": prediction, "pages": {},
                "proba": arms_to_score,
                "pages": arms_to_pages, "score": prediction_proba[prediction_idx],
                "terms": informative_terms
                }
