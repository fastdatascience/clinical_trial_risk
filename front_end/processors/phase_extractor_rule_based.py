import bz2
import pickle as pkl
import re
from os.path import exists

import numpy as np
import spacy
from spacy.matcher import PhraseMatcher

from util.spacy_wrapper import nlp

phase_synonyms = {0: ['phase 0'],  # , 'pilot study'],
                  0.5: [  # 'early phase i', 'early phase 1', 'early phase 1.0',
                      'phase 0.5'],
                  1: ['phase i', 'phase 1 b', 'phase 1', 'phase 1.0', 'phase ib', 'phase ia'],
                  1.5: ['phase i ii', 'phase 1 2', 'phase 1 2 a', 'phase 1 - 2', 'phase 1-2', 'phase i iia',
                        'phase 1 2a', 'phase 1/2a', 'phase ib/ii', 'phase ib ii', 'phase ib / ii'],
                  2: ['phase ii', 'phase 2 b', 'phase 2', 'phase 2.0', 'phase iib', 'phase iia', 'phase 2 a',
                      'phase 2a'],
                  2.5: ['phase 2.5', 'phase ii iii', 'phase 2 3', 'phase 2 - 3', 'phase 2-3', 'phase 2 3a',
                        'phase 2 3b', 'phase 2/3a', 'phase 2/3b', 'phase 2 - 3a', 'phase 2 - 3b'],
                  3: ['phase iii', 'phase 3', 'phase 3.0', 'phase 3b', 'phase 3a', 'phase iiia', 'phase iiib'],
                  4: ['phase iv', 'phase 4', 'phase 4.0', 'phase 4 a', 'phase 4 b', 'phase 4a', 'phase 4b', 'phase iva',
                      'phase ivb'],
                  }

phrase_matcher = PhraseMatcher(nlp.vocab)

for phase_number, synonyms in phase_synonyms.items():
    phases = []
    for text in synonyms:
        phases.append(nlp.make_doc(text))
        phases.append(nlp.make_doc(re.sub(r'phase', 'phase:', text)))
        phases.append(nlp.make_doc(re.sub(r'phase', 'phase :', text)))

    phrase_matcher.add(f"Phase {phase_number}", None, *phases)

patterns = dict()

patterns["interventional"] = ['interventional']
patterns["this study"] = ['this study', 'the study', 'this trial', 'this protocol']
patterns["clinical trial"] = ['clinical trial']
patterns["study title"] = ['study title', 'protocol title', 'official title', 'title of the study', 'title of study',
                           'title of protocol', 'title of the protocol']
patterns["title"] = ["title"]
patterns["this"] = ['this']
patterns["this phase"] = ['this phase']
patterns["this is a"] = ['this is a', 'this will be a', 'this is an', 'this will be an']
patterns["this X phase"] = ['this proposed phase', 'this planned phase', 'this prospective phase',
                            'this exploratory phase', 'this ongoing phase', 'this randomized phase',
                            'this randomised phase',
                            'this multicenter phase', 'this multi-center phase', 'this multi center phase',
                            'this multicentre phase', 'this multi-centre phase', 'this multi centre phase',
                            'this pivotal phase',
                            'this combined phase', 'this single phase']
patterns["aim"] = ['aim', 'objective', 'purpose']
patterns["aims"] = ['aim']
patterns["this"] = ['clinical']
patterns["placebo"] = ['placebo']
patterns["randomised"] = ['randomised', 'randomized']
patterns["multinational"] = ["multinational", "international", "multi-national"]
patterns["multicenter"] = ["multicenter", "multi-center", "multicentre", "multi-centre", "multisite", "multi-site"]


patterns["neg journals"] = ['lancet', 'nature', 'doi']
patterns["neg in a"] = ["in a", "was a"]
patterns["neg reported"] = ['reported', 'conducted', 'demonstrated', 'showed', 'mentioned', 'conducted', 'noted']
patterns["neg ongoing"] = ['ongoing', 'pivotal', 'influential']
patterns["neg studies_plural"] = ['studies', 'trials']
patterns["neg et al"] = ["et al"]

context_matcher = PhraseMatcher(nlp.vocab)

for feature_name, feature_patterns in patterns.items():
    context_patterns = [nlp.make_doc(text) for text in feature_patterns]
    context_matcher.add(feature_name, None, *context_patterns)

# If any of these words appear adjacent to the phase, it's discounted
exclude_adjacent_matcher = PhraseMatcher(nlp.vocab)

exclusions = [nlp.make_doc(text) for text in ["viral", "decay", "trials", "part", "studies", "clinical studies"]]
exclude_adjacent_matcher.add(f"exclude_on_right", None, *exclusions)

exclusions = [nlp.make_doc(text) for text in
              ["two", "three", "four", "several", "recent", "chronic", "subsequent", "exponential", "initial"]]
exclude_adjacent_matcher.add(f"exclude_on_left", None, *exclusions)

FEATURE_NAMES = ["phase", "num_pages", "min_page", "max_page", "phase_idx", "span_over_doc",
                 "num_title_case_occurrences", "num_upper_case_occurrences"]
FEATURE_NAMES.extend([p + "_left" for p in sorted(patterns)])
FEATURE_NAMES.extend([p + "_right" for p in sorted(patterns)])


def extract_features(orig_tokenised_pages):
    # orig_tokenised_pages stores the original case information
    tokenised_pages = [[string.lower() for string in sublist] for sublist in orig_tokenised_pages]

    phase_to_pages = {}

    phase_to_features = {}
    phase_to_title_case = {}
    phase_to_upper_case = {}

    contexts = {}

    for page_number, page_tokens in enumerate(tokenised_pages):
        doc = spacy.tokens.doc.Doc(
            nlp.vocab, words=page_tokens)
        phrase_matches = phrase_matcher(doc)
        exclusion_matches = exclude_adjacent_matcher(doc)
        excluded_indices = set()
        for word, start, end in exclusion_matches:
            exclusion_pattern_id = nlp.vocab.strings[word]
            if exclusion_pattern_id == "exclude_on_right":
                excluded_indices.add(start)
            elif exclusion_pattern_id == "exclude_on_left":
                excluded_indices.add(end)

        context_matches = context_matcher(doc)

        for word, start, end in sorted(phrase_matches, key=lambda t: t[2] - t[1], reverse=True):
            if start in excluded_indices or start - 1 in excluded_indices or end in excluded_indices or end + 1 in excluded_indices:
                continue
            for i in range(start, end + 1):
                excluded_indices.add(i)

            orig_token = orig_tokenised_pages[page_number][start]

            phase_number = nlp.vocab.strings[word]
            if phase_number not in phase_to_pages:
                phase_to_pages[phase_number] = []
            phase_to_pages[phase_number].append(page_number)

            if orig_token.startswith("PHASE"):
                if phase_number not in phase_to_upper_case:
                    phase_to_upper_case[phase_number] = 0
                phase_to_upper_case[phase_number] = phase_to_upper_case[phase_number] + 1
            elif orig_token.startswith("P"):
                if phase_number not in phase_to_title_case:
                    phase_to_title_case[phase_number] = 0
                phase_to_title_case[phase_number] = phase_to_title_case[phase_number] + 1

            if phase_number not in phase_to_features:
                phase_to_features[phase_number] = {}
            for side in ["left", "right"]:
                for context_word_id, context_start, context_end in context_matches:
                    if (side == "left" and context_start > start) or (side == "right" and context_end < end):
                        continue
                    context_word = nlp.vocab.strings[context_word_id] + "_" + side
                    distances = []
                    if context_word in phase_to_features[phase_number]:
                        distances.append(phase_to_features[phase_number][context_word])
                    distances.extend([abs(context_start - end), abs(start - context_end)])

                    # negative features are things which are expected to reduce the probability of
                    # it being the trial phase. For these features we take the max distance
                    # from the candidate phase in question. e.g. et al - if the phase only occurs
                    # in close proximity to et al, it's not a good candidate.
                    if "neg" in context_word:
                        min_dist = max(distances)
                    else:
                        min_dist = min(distances)
                    phase_to_features[phase_number][context_word] = min_dist

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
        feat_vect = [phase_float, len(pages), min(pages), max(pages), idx,
                     (max(pages) - min(pages)) / len(tokenised_pages),
                     phase_to_title_case.get(phase, 0), phase_to_upper_case.get(phase, 0)]

        num_non_nlp_features = len(feat_vect)

        for pattern in FEATURE_NAMES[num_non_nlp_features:]:
            feat_vect.append(phase_to_features[phase].get(pattern, 9999))

        X.append(feat_vect)

    X = np.asarray(X)

    return X, phase_to_pages, contexts


class PhaseExtractorRuleBased:

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

        X, phase_to_pages, contexts = extract_features(tokenised_pages)

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
