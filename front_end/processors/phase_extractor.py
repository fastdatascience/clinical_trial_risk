import bz2
import pickle as pkl
import re
from os.path import exists

import numpy as np
import spacy
from spacy.matcher import PhraseMatcher

from util.spacy_wrapper import nlp

phase_synonyms = {0: ['phase 0', 'pilot study'],
                  0.5: ['early phase i', 'early phase 1', 'early phase 1.0', 'phase 0.5'],
                  1: ['phase i', 'phase 1 b', 'phase 1', 'phase 1.0'],
                  1.5: ['phase i ii', 'phase 1 2', 'phase 1 2 a'],
                  2: ['phase ii', 'phase 2 b', 'phase 2', 'phase 2.0'],
                  2.5: ['phase 2.5', 'phase ii iii'],
                  3: ['phase iii', 'phase 3', 'phase 3.0'],
                  4: ['phase iv', 'phase 4', 'phase 4.0', 'phase 4 a', 'phase 4 b'],
                  }

phrase_matcher = PhraseMatcher(nlp.vocab)

for phase_number, synonyms in phase_synonyms.items():
    phases = [nlp.make_doc(text) for text in synonyms]

    phrase_matcher.add(f"Phase {phase_number}", None, *phases)


class PhaseExtractor:

    def __init__(self, path_to_classifier):
        print("Initialising Phase Random Forest classifier", path_to_classifier)
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD PHASE RANDOM FOREST CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.model = None
            return
        with bz2.open(path_to_classifier, "rb") as f:
            self.model = pkl.load(f)

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify the trial phase.
        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from phase to the pages it's mentioned in.
        """

        tokenised_pages = [[string.lower() for string in sublist] for sublist in tokenised_pages]

        phase_to_pages = {}

        contexts = {}

        for page_number, page_tokens in enumerate(tokenised_pages):
            doc = spacy.tokens.doc.Doc(
                nlp.vocab, words=page_tokens)
            phrase_matches = phrase_matcher(doc)
            for word, start, end in phrase_matches:
                phase_number = nlp.vocab.strings[word]
                if phase_number not in phase_to_pages:
                    phase_to_pages[phase_number] = []
                phase_to_pages[phase_number].append(page_number)

                if phase_number not in contexts:
                    contexts[phase_number] = ""
                start = max(0, start - 15)
                end = min(len(page_tokens) - 1, end + 15)
                contexts[phase_number] = (
                        contexts[phase_number] + " " + f"Page {page_number + 1}: " + " ".join(
                    tokenised_pages[page_number][start:end])).strip()

        phase_to_pages = sorted(phase_to_pages.items(), key=lambda v: len(v[1]), reverse=True)

        X = []
        for idx, (phase, pages) in enumerate(phase_to_pages):
            phase_float = float(re.sub(r'\D', '', phase))
            feat_vect = [phase_float, len(pages), min(pages), max(pages), idx]

            X.append(feat_vect)

        X = np.asarray(X)

        if len(X) > 0:

            probas = self.model.predict_proba(X)[:, 1]

            scores = {}
            for (phase, pages), score in zip(phase_to_pages, probas):
                scores[phase] = score

            prediction = max(scores, key=scores.get)
            prediction = float(re.sub(r'[A-Za-z /]', '', prediction))
        else:
            prediction = 0.0
            scores = {}

        return {"prediction": prediction, "pages": dict(phase_to_pages), "probas": scores, "context": contexts}
