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
                           'sample size increase to #', 'sample size will be #']
patterns["sample"] = ['sample #', 'sample of #', 'sampling #']
patterns["enroll"] = ['enroll #', 'enrol #', 'enrolling #', 'enroll up to #', 'enrolling up to #', 'enrol up to #',
                      'enrolment #', 'enrollment #', 'enrolment of #', 'enrollment of #', 'enrolment goal #',
                      'enrolment goal : #']
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
                            'participants up to #', '# consecutive patients', '# consecutive subjects',
                            '# consecutive participants']
patterns["subjects"] = ['# total subjects', 'number of subjects #', '# subjects', 'subjects up to #']
patterns["misc_personal_noun"] = ['# people', '# persons', '# residents', '# mother infant pairs',
                                  '# mother child pairs', '# mother - child pairs',
                                  '# mother - infant pairs', '# individuals', "# sexually active", "# patients",
                                  "# pts", "# cases", "# * cases", '# * patients', '# * pts',
                                  '# outpatients', '# * outpatients', '# * subjects', "# volunteers",
                                  "# high risk", "# high - risk",
                                  "# neonates", "# vaccine recipients", "# recipients"]
patterns["gender"] = ['# male', '# males', '# female', '# females', '# women', '# men', '# mothers', '# pregnant']
patterns["age"] = ['# infants', '# adult', '# adults', '# adolescents', '# babies', '# children']
patterns["disease_state"] = ['# healthy', '# hiv infected', '# hiv positive', '# hiv negative', '# hiv - infected',
                             '# hiv - positive', '# hiv - negative', '# evaluable',
                             '# evaluable', '# efficacy-evaluable', '# efficacy - evaluable', '# activated',
                             "analyzable #", "analysable #"]
patterns["selection"] = ['selection #', 'selection of #', ]
patterns["demonym"] = ["# " + demonym.lower() for demonym in demonym_to_country_code]
patterns["approximately"] = ["approximately #", "up to #"]
patterns["to achieve"] = ["# to achieve"]
patterns["study population"] = ["study population #", "study population of #"]
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
patterns["distance to total no number"] = ["total"]
# These are anticipated to be negative features.
patterns["distance to per no number"] = ["per"]
patterns["distance to each no number"] = ["each", "every"]
patterns["distance to site no number"] = ["site"]
patterns["distance to arm/group no number"] = ["arm", "group"]
patterns["distance to future indicator no number"] = ["will"]
patterns["distance to past indicator no number"] = ["was", "were", "to date", "literature"]
patterns["distance to contents no number"] = ["table of contents", "index"]
patterns["distance to et al no number"] = ["et al"]
patterns["distance to conclude no number"] = ["conclude"]
patterns["distance to plan or plans no number"] = ["plan", "plans", "planned", "planning"]
patterns["distance to propose or proposes no number"] = ["propose", "proposes", "proposed", "proposing"]
patterns["distance to target or targets no number"] = ["target", "targets"]
patterns["distance to page no number"] = ["page"]
patterns["distance to percent no number"] = ["%"]
patterns["distance to vaccine no number"] = ["vaccine", "vaccinated"]

patterns_without_number = set([x for x in patterns if "no number" in x])

FEATURE_NAMES = list(patterns.keys())
FEATURE_NAMES.append("first_page_no")
FEATURE_NAMES.append("last_page_no")
FEATURE_NAMES.append("num_occurrences")
FEATURE_NAMES.append("magnitude")

matcher = Matcher(nlp.vocab)

num_regex = re.compile(r'(?i)^(?:[1-9]\d*,?\d+|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)$')

num_lookup = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
              "ninety": 90}

ABSOLUTE_MINIMUM = 8
ABSOLUTE_MAXIMUM = 1000000

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
                elif word == "%":  # percentage
                    pattern.append({"TEXT": {"REGEX": r"^\d+%$"}})
                elif word == "*":  # wildcard
                    pattern.append({"LIKE_NUM": False})
                else:
                    pattern.append({"LOWER": word})
            patterns.append(pattern)
    matcher.add(feature_name, patterns)

# Exclude things that are clearly not sample size, e.g. 50 ml
# We must be careful with the negative matcher as this is very hard to debug.
# Anything excluded with this pattern is excluded right at the beginning of the process. So SI units are a good example as they give us 100% confidence it's not the sample size under discussion.
negative_matcher = Matcher(nlp.vocab)
negative_patterns = []
negative_patterns.append([{"LIKE_NUM": True}, {"LOWER": {
    "IN": ["fold", "gy", "cycles", "doses", "mci", "ci", "mg", "kg", "ml", "l", "g", "kg", "mg", "s", "days", "months",
           "years", "hours", "seconds", "minutes", "sec",
           "min", "mcg", "cc", "ng", "kcal", "cal", "events",
           "mol", "mmol", "mi", "h", "hr", "hrs", "s", "m", "km", "lb", "oz", "moles", "mole", "wk", "wks", "week",
           "weeks", "µm", "cases", "progression", "death", "adverse", "yrs",
           "cells", "appointments", "µg", "episodes", "incidents", "sites", "locations", "countries", "centres",
           "centers", "liters", "litres", "milliliters", "millilitres", "centiliters", "centilitres",
           "effect", "visits", "revolutions", "cgy", "mm3", "mm", "cm", "cm3", "sec", "pages", "mcg", "µl", "c", "°C",
           "°", "platelets", "dl", "pg", "mmhg", "hg", "gl", "msec", "ms", "µs", "cohorts"]}}])
negative_patterns.append([{"LIKE_NUM": True}, {"LOWER": {
    "IN": ["investigational", "investigative", "experimental", "clinical", "study"]}}, {"LOWER": {
    "IN": ["sites", "centers", "centres", "locations", "studies", "trials"]}}])

negative_patterns.append([{"LIKE_NUM": True}, {"LOWER": {
    "IN": ["new"]}}, {"LOWER": {
    "IN": ["cases"]}}])

negative_patterns.append([{"LOWER": {
    "IN": ["incidence", "prevalence"]}}, {"LOWER": {
    "IN": ["of"]}}, {"LIKE_NUM": True}])

negative_patterns.append([{"LOWER": {
    "IN": ["et"]}}, {"LOWER": {
    "IN": ["al"]}}, {"LIKE_NUM": True}])

# Exclude dates
negative_patterns.append([{"LOWER": {
    "IN": ["january", "jan", "february", "feb", "march", "mar", "april", "apr", "june", "jun", "july", "jul", "august",
           "aug", "september", "sep", "sept", "october", "oct", "november", "nov", "december", "dec"]}},
    {"LIKE_NUM": True}])
negative_patterns.append([{"LIKE_NUM": True}, {"LOWER": {
    "IN": ["january", "jan", "february", "feb", "march", "mar", "april", "apr", "june", "jun", "july", "jul", "august",
           "aug", "september", "sep", "sept", "october", "oct", "november", "nov", "december", "dec"]}}])

negative_patterns.append([{"LOWER": {
    "IN": ["serving"]}}, {"LOWER": {
    "IN": ["more"]}}, {"LOWER": {
    "IN": ["than"]}}, {"LIKE_NUM": True}])  # serving more than 1000 patients

negative_patterns.append([{"LOWER": {
    "IN": ["per", "additional", "remaining"]}}, {"LIKE_NUM": True}])

negative_patterns.append([{"LIKE_NUM": True}, {"LOWER": {"IN": ["additional"]}}])

# Exclude contents page
negative_patterns.append([{"LIKE_NUM": True}, {"TEXT": {"REGEX": r"^\d+\.\d+$"}}])

# The first 31 participants
negative_patterns.append([{"LOWER": "the"}, {"LOWER": "first"}, {"LIKE_NUM": True}])

negative_patterns.append([{"TEXT": "Week"}, {"LIKE_NUM": True}])  # Week 16

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

        is_ignore = False

        for i in range(phrase_match[1], phrase_match[2]):
            if i in tokens_to_exclude:
                is_ignore = True
                break
        if is_ignore:
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

                parsed = num_lookup.get(value.lower())
                if not parsed:
                    parsed = int(value)
                if parsed < ABSOLUTE_MINIMUM or parsed > ABSOLUTE_MAXIMUM:
                    value = None
                    continue

                if value not in token_indexes:
                    token_indexes[value] = set()
                token_indexes[value].add(token_idx)
                continue

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
            if page_no not in num_subjects_to_pages[value]:
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
        n = num_lookup.get(candidate.lower())
        if not n:
            n = int(re.sub(r'\D.+$', '', candidate))
        features[candidate]["magnitude"] = min(n, 50)

    candidates = []
    feature_vectors = []
    for cand, features in features.items():

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
            if probas[idx] > score * 0.1 and len(possible_candidates) < 5:
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

        is_low_confidence = int(sum([p for p in probas if p > 0.5]) != 1)

        is_per_arm = []
        for k, v in contexts.items():
            v = v.lower()
            if "per arm" in v or "in each arm" in v or "per cohort" in v or "in each cohort" in v or "per group" in v or "in each group" in v or "in each of the cohorts" in v or "in each of the arms" in v:
                is_per_arm.append(k)

        return {"prediction": int(num_subjects), "pages": num_subjects_to_pages, "context": contexts, "score": score,
                "comment": possible_candidates, "is_per_arm": is_per_arm, "proba": value_to_score,
                "is_low_confidence": is_low_confidence}
