import bz2
import pickle as pkl
from os.path import exists

import numpy as np


# Shared between training and inference code
def derive_feature(f):
    return [max(f), len([x for x in f if x > 0.5]), sum([x for x in f if x > 0.5]), sum(f), len(f), sum(f[-10:]),
            sum(f[-20:])]


class SapExtractor:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD SAP CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.model = None
            return
        with bz2.open(path_to_classifier, "rb") as f:
            self.model = pkl.load(f)
        self.vectoriser = self.model[0].named_steps['countvectorizer']
        self.transformer = self.model[0].named_steps['tfidftransformer']
        self.nb = self.model[0].named_steps['multinomialnb']
        self.model2 = self.model[1]

        self.vocabulary = {v: k for k, v in self.vectoriser.vocabulary_.items()}

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify whether the trial has a completed SAP.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from condition to the pages it's mentioned in.
        """

        if self.model is None:
            print("Warning! SAP classifier not loaded.")
            return {"prediction": -1}

        page_to_probas = []

        for tokens in tokenised_pages:
            token_counts = np.zeros((1, len(self.vectoriser.vocabulary_)))
            for token in tokens:
                token_lower = token.lower()
                if token_lower in self.vectoriser.vocabulary_:
                    token_counts[0, self.vectoriser.vocabulary_[token_lower]] += 1

            transformed_document = self.transformer.transform(token_counts)

            prediction_probas = self.nb.predict_proba(transformed_document)[0]
            page_to_probas.append(prediction_probas[1])

        '''
        import pandas as pd
        import plotly.express as px
        from plotly import graph_objects as go

        df = pd.DataFrame({"page":list(range(len(page_to_probas))), "proba":page_to_probas})

        fig = px.bar(df, x="page", y="proba")
        layout = go.Layout(
            title="Word counts of each page",
            # margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.update_layout(layout)
        fig.show()
        '''

        doc_derived_features = np.asarray([derive_feature(page_to_probas)])

        prediction_idx = self.model2.predict(doc_derived_features)[0]

        probability_of_sap = self.model2.predict_proba(doc_derived_features)[0]

        # Now try to find out which pages contributed most to the decision and what the informative words on those pages were.

        # Find out which pages are in the top quartile of SAP probabilities.
        # Make a pseudo-document of these candidate SAP pages only and run it through the classifier.
        top_quartile_probas = np.quantile(page_to_probas, 0.75)

        token_counts = np.zeros((1, len(self.vectoriser.vocabulary_)))

        for page_no, tokens in enumerate(tokenised_pages):
            if page_to_probas[page_no] >= top_quartile_probas:
                for token in tokens:
                    token_lower = token.lower()
                    if token_lower in self.vectoriser.vocabulary_:
                        token_counts[0, self.vectoriser.vocabulary_[token_lower]] += 1

        transformed_document = self.transformer.transform(token_counts)

        probas = np.zeros((transformed_document.shape[1]))
        for i in range(transformed_document.shape[1]):
            zeros = np.zeros(transformed_document.shape)
            zeros[0, i] = transformed_document[0, i]
            proba = self.nb.predict_log_proba(zeros)
            probas[i] = proba[0, 1]

        sap_to_pages = {}
        for vocab_idx in np.argsort(-probas):
            sap_to_pages[self.vocabulary[vocab_idx]] = []
            if len(sap_to_pages) > 20:
                break

        for page_no, tokens in enumerate(tokenised_pages):
            for token in tokens:
                if token.lower() in sap_to_pages:
                    sap_to_pages[token.lower()].append(page_no)

        # prediction_idx = int(np.argmax(prediction_probas))
        #
        # print ("prediction_idx is", prediction_idx)
        #
        # probas = np.zeros((transformed_document.shape[1]))
        # for i in range(transformed_document.shape[1]):
        #     zeros = np.zeros(transformed_document.shape)
        #     zeros[0, i] = transformed_document[0, i]
        #     proba = self.nb.predict_log_proba(zeros)
        #     probas[i] = proba[0, 1]
        #
        # sap_to_pages = {}
        # for vocab_idx in np.argsort(-probas):
        #     sap_to_pages[self.vocabulary[vocab_idx]] = []
        #     if len(sap_to_pages) > 15:
        #         break
        #
        # for page_no, tokens in enumerate(tokenised_pages):
        #     for token in tokens:
        #         if token.lower() in sap_to_pages:
        #             sap_to_pages[token.lower()].append(page_no)

        return {"prediction": prediction_idx, "pages": sap_to_pages, "page_scores": page_to_probas,
                "score": probability_of_sap[1]}
