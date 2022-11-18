from os.path import exists

import spacy
import re

phase_map = {'Early Phase 1':0.5,
 'Not Applicable':0,
 'Phase 1':1,
 'Phase 1/Phase 2':1.5,
 'Phase 2':2,
 'Phase 2/Phase 3':2.5,
 'Phase 3':3,
 'Phase 4':4}

# Current best model: Expt4
class PhaseExtractorSpacy:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD PHASE CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.nlp = None
            return
        self.nlp = spacy.load(path_to_classifier)

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify phase.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and scores.
        """
        if self.nlp is None:
            print("Warning! Phase classifier not loaded.")
            return {"prediction": "Error"}

        text = ""
        for page_no, tokens in enumerate(tokenised_pages):
            if page_no >= 3:
                break
            text += " ".join(tokens) + " "
        doc = self.nlp(text)

        prediction_proba = {}
        for phase_str, phase_float in phase_map.items():
            clean_name = "Phase " + str(phase_float)
            prediction_proba[clean_name] = doc.cats[phase_str]



        prediction = max(prediction_proba, key=prediction_proba.get)

        prediction = float(re.sub(r'Phase ', '', prediction))

        return {"prediction": prediction, "probas": prediction_proba}
