from os.path import exists

import spacy


# Current best model: Expt11
class InternationalExtractorSpacy:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD INTERNATIONAL CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.nlp = None
            return
        self.nlp = spacy.load(path_to_classifier)

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify whether the trial takes place in multiple countries.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from condition to the pages it's mentioned in.
        """
        if self.nlp is None:
            print("Warning! International classifier not loaded.")
            return {"prediction": "Error"}

        text = ""
        for page_no, tokens in enumerate(tokenised_pages):
            if page_no >= 3:
                break
            text += " ".join(tokens) + " "
        doc = self.nlp(text)
        prediction_proba = doc.cats["1"]

        is_international_pred = int(prediction_proba > 0.5)

        return {"prediction": is_international_pred, "pages": {}, "score": prediction_proba}
