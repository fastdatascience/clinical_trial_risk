import bz2
import pickle as pkl
import re
from collections import Counter
from os.path import exists

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher

from util.demonym_finder import demonym_to_country_code
from util.page_tokeniser import iterate_tokens
from util.spacy_wrapper import nlp

patterns = dict()

patterns["sample size"] = ['sample size is #', 'sample size #', 'sample size of #', 'sample size to #',
                           'sample size increase to #']
patterns["sample"] = ['sample #', 'sample of #', 'sampling #']
patterns["enroll"] = ['enroll #', 'enrol #', 'enrolling #', 'enroll up to #', 'enrolling up to #', 'enrol up to #',
                      'enrolment #', 'enrollment #', 'enrolment of #', 'enrollment of #']
patterns["will_enroll"] = ["will enroll #", "will enrol #", "aim to enroll #", "aim to enrol #"]
patterns["recruit"] = ['recruit #', 'recruiting #', 'recruitment #', 'recruitment of']
patterns["will_recruit"] = ["will recruit #", "aim to recruit #"]
patterns["total"] = ['total #', 'total of #']
patterns["target"] = ['target #', 'target of #', 'targeting #', 'target is #']
patterns["accrual"] = ['accrual #', 'accrual of #', 'accruing #', 'accrue #']
patterns["will_target"] = ["will target #"]
patterns["total"] = ["total #", "total of #"]
patterns["total_after"] = ["# total", "# overall"]
patterns["n ="] = ['n = #', 'n > #', 'n ≥ #']
patterns["participants"] = ['# total participants', 'number of participants #', '# participants',
                            'participants up to #']
patterns["subjects"] = ['# total subjects', 'number of subjects #', '# subjects', 'subjects up to #']
patterns["misc_personal_noun"] = ['# people', '# persons', '# residents', '# mother infant pairs',
                                  '# mother child pairs', '# mother - child pairs',
                                  '# mother - infant pairs', '# individuals', "# sexually active", "# patients",
                                  "# pts", "# cases", "# * cases", '# * patients', '# * pts',
                                  '# outpatients', '# * outpatients', '# * subjects']
patterns["gender"] = ['# male', '# males', '# female', '# females', '# women', '# men', '# mothers', '# pregnant']
patterns["age"] = ['# infants', '# adult', '# adults', '# adolescents', '# babies', '# children']
patterns["disease_state"] = ['# healthy', '# hiv infected', '# hiv positive', '# hiv negative', '# hiv - infected',
                             '# hiv - positive', '# hiv - negative', '# evaluable',
                             '# evaluable', '# efficacy-evaluable', '# efficacy - evaluable', '# activated']
patterns["selection"] = ['selection #', 'selection of #', ]
patterns["demonym"] = ["# " + demonym.lower() for demonym in demonym_to_country_code]
patterns["approximately"] = ["approximately #", "up to #"]
patterns["to achieve"] = ["# to achieve"]
patterns["optional_colon"] = ["number of subjects : #", "planned subjects : #", "subjects planned : #", "enrolment : #",
                              "enrollment : #", "sample size : #",
                              "number of subjects #", "planned subjects #", "subjects planned #", "enrolment #",
                              "enrollment #", "sample size #"]

# These patterns do not contain a number. The feature generated from them is just the shortest distance from a candidate number to the nearest occurrence of these words.
patterns["distance to sample size no number"] = ["sample size"]
patterns["distance to power no number"] = ["power", "powered"]
patterns["distance to num subjects no number"] = ["number of subjects", "number of participants", "number of patients",
                                                  "number of pts"]
patterns["distance to subjects no number"] = ["subjects", "participants", "patients", "pts"]
patterns["distance to cases no number"] = ["cases"]
# These are anticipated to be negative features.
patterns["distance to per no number"] = ["per"]
patterns["distance to arm/group no number"] = ["arm", "group"]

patterns_without_number = set([x for x in patterns if "no number" in x])

FEATURE_NAMES = list(patterns.keys())
FEATURE_NAMES.append("first_page_no")
FEATURE_NAMES.append("last_page_no")
FEATURE_NAMES.append("num_occurrences")
FEATURE_NAMES.append("magnitude")

matcher = Matcher(nlp.vocab)

num_regex = re.compile(r'^[1-9]\d*,?\d+$')

ABSOLUTE_MINIMUM = 35
ABSOLUTE_MAXIMUM = 10000

for feature_name, feature_patterns in patterns.items():
    patterns = []
    for feature_pattern in feature_patterns:
        for is_range in [0, 1, 2, 3]:
            pattern = []
            for word in feature_pattern.split(" "):
                if word == "#":
                    if is_range == 2:
                        pattern.append({"LOWER": {"IN": ["=", ">", "≥", "approx", "approximately", "planned"]}})
                    elif is_range == 3:
                        pattern.append({"LOWER": "up"})
                        pattern.append({"LOWER": "to"})
                    pattern.append({"LIKE_NUM": True})
                    if is_range == 1:
                        pattern.append({"LOWER": {"IN": ["-", "–", "to"]}})
                        pattern.append({"LIKE_NUM": True})
                elif word == "*":  # wildcard
                    pattern.append({"LIKE_NUM": False})
                else:
                    pattern.append({"LOWER": word})
            patterns.append(pattern)
    matcher.add(feature_name, patterns)

# Exclude things that are clearly not sample size, e.g. 50 ml
negative_matcher = Matcher(nlp.vocab)
negative_patterns = []
negative_patterns.append([{"LIKE_NUM": True}, {"LOWER": {
    "IN": ["mg", "kg", "ml", "l", "g", "kg", "mg", "s", "days", "months", "years", "hours", "seconds", "minutes", "sec",
           "min", "mcg",
           "mol", "mmol", "mi", "h", "s", "m", "km", "lb", "oz", "moles", "mole", "wk", "wks", "week", "weeks",
           "lot", "cells", "appointments"]}}])
negative_matcher.add("MASK", negative_patterns)


def extract_features(tokenised_pages: list):
    features = {}
    num_subjects_to_pages = {}

    contexts = {}

    all_tokens = list(iterate_tokens(tokenised_pages))

    tokens = [item[2] for item in all_tokens]

    doc = spacy.tokens.doc.Doc(
        nlp.vocab, words=tokens)
    matches = matcher(doc)

    token_indexes = {}

    tokens_to_exclude = set()
    negative_matches = negative_matcher(doc)
    for phrase_match in negative_matches:
        for i in range(phrase_match[1], phrase_match[2] + 1):
            tokens_to_exclude.add(i)

    for phrase_match in matches:

        if phrase_match[1] in tokens_to_exclude or phrase_match[2] in tokens_to_exclude:
            # print ("skipping match at", tokens[phrase_match[1]], tokens[phrase_match[2]])
            continue

        value = None
        matcher_name = nlp.vocab.strings[phrase_match[0]]

        if matcher_name in patterns_without_number:
            if matcher_name not in token_indexes:
                token_indexes[matcher_name] = set()
            token_indexes[matcher_name].add(phrase_match[1])
            token_indexes[matcher_name].add(phrase_match[2])
            continue

        for token_idx in range(phrase_match[1], phrase_match[2]):
            page_no, token_no, token = all_tokens[token_idx]
            if num_regex.match(token):
                value = re.sub(r',', '', token)
                if value not in token_indexes:
                    token_indexes[value] = set()
                token_indexes[value].add(token_idx)
        if value:
            if value not in features:
                features[value] = Counter()
                features[value]["first_page_no"] = page_no
                features[value]["last_page_no"] = page_no
            features[value][matcher_name] += 1
            features[value]["last_page_no"] = page_no
            features[value]["num_occurrences"] += 1
            if value not in num_subjects_to_pages:
                num_subjects_to_pages[value] = []
            num_subjects_to_pages[value].append(page_no)

            if value not in contexts:
                contexts[value] = ""
            start = max(0, phrase_match[1] - 15)
            end = min(len(tokens) - 1, phrase_match[2] + 15)

            contexts[value] = (
                    contexts[value] + " " + f"Page {page_no + 1}: " + " ".join(tokens[start:end])).strip()

    for candidate in features:
        for distance_feature in patterns_without_number:
            min_dist = -1
            for i in token_indexes[candidate]:
                for j in token_indexes.get(distance_feature, set()):
                    if min_dist == -1 or abs(i - j) < min_dist:
                        min_dist = abs(i - j)
            if min_dist == -1 or min_dist > 1000:
                min_dist = 1000
            features[candidate][distance_feature] = min_dist
        features[candidate]["magnitude"] = min(int(re.sub(r'\D.+$', '', candidate)), 50)

    candidates = []
    feature_vectors = []
    for cand, features in features.items():
        if features["magnitude"] < ABSOLUTE_MINIMUM or features["magnitude"] > ABSOLUTE_MAXIMUM:
            continue
        feature_vector = []
        for feature_name in FEATURE_NAMES:
            feature_vector.append(features.get(feature_name, 0))
        candidates.append(str(cand))
        feature_vectors.append(feature_vector)

    df_instances = pd.DataFrame({"candidate": candidates})
    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        df_instances[feature_name] = [fv[feature_idx] for fv in feature_vectors]

    return df_instances, num_subjects_to_pages, contexts


class NumSubjectsExtractor:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD NUMBER OF SUBJECTS CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.model = None
            return
        with bz2.open(path_to_classifier, "rb") as f:
            self.model = pkl.load(f)

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify the number of subjects in the trial.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (int) and a map from numbers to the pages it's mentioned in.
        """

        df_instances, num_subjects_to_pages, contexts = extract_features(tokenised_pages)

        if len(df_instances) == 0:
            return {"prediction": 0, "pages": {}, "context": [],
                    "score": 0, "comment": "No possible subject numbers found.", "proba": {}, "is_per_arm": {}}

        probas = self.model.predict_proba(df_instances[FEATURE_NAMES])[:, 1]
        winning_index = np.argmax(probas)
        score = np.max(probas)

        top_indices = list(np.argsort(-probas))
        if len(top_indices) > 5:
            top_indices = top_indices[:5]

        value_to_score = {}
        possible_candidates = []
        for idx in top_indices:
            if probas[idx] > score * 0.1 and len(possible_candidates) < 3:
                possible_candidates.append(df_instances.candidate.iloc[idx])
                value_to_score[df_instances.candidate.iloc[idx]] = probas[idx]
        possible_candidates = "Possible sample sizes found: " + ", ".join(possible_candidates)

        num_subjects = df_instances.candidate.iloc[winning_index]

        top_values = set()
        for top_index in top_indices:
            top_values.add(df_instances.candidate.iloc[top_index])
        for k in list(num_subjects_to_pages):
            if k not in top_values:
                del num_subjects_to_pages[k]
        for k in list(contexts):
            if k not in top_values:
                del contexts[k]

        is_per_arm = []
        for k, v in contexts.items():
            if "per arm" in v:
                is_per_arm.append(k)

        return {"prediction": int(num_subjects), "pages": num_subjects_to_pages, "context": contexts, "score": score,
                "comment": possible_candidates, "is_per_arm": is_per_arm, "proba": value_to_score}
