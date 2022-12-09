from os.path import exists

import spacy


# Current best model: Expt16
class CountryGroupExtractor:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD COUNTRY GROUP CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.nlp = None
            return
        self.nlp = spacy.load(path_to_classifier)

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify whether the trial takes place in US/Canada, LMIC countries, or other.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str): "USCA", "HIGH_INCOME", "LMIC"
        """
        if self.nlp is None:
            print("Warning! Country group classifier not loaded.")
            return {"prediction": "Error"}

        text = ""
        for page_no, tokens in enumerate(tokenised_pages):
            if page_no >= 10:
                break
            text += " ".join(tokens) + " "
        doc = self.nlp(text)
        prediction_proba = dict(doc.cats)

        prediction = max(prediction_proba, key=prediction_proba.get)

        return {"prediction": prediction, "pages": {}, "probas": prediction_proba}
