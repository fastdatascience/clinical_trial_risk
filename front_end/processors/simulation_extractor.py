import bz2
import pickle as pkl
from os.path import exists
import pandas as pd
import numpy as np

from util.page_tokeniser import iterate_tokens

FEATURE_NAMES = ["simulate-power", "scenarios-power", "simulate-sample", "scenarios-sample", "simulate-sample size",
                 "scenarios-sample size"]


def extract_features(all_tokens):
    key_words = [
        ["simulate", "simulates", "simulated", "simulating", "simulation", "simulations"],
        ["scenarios"],
        ["power", "powers", "powered", "powering"],
        ["sample", "sampled", "samples", "sampling"],
        ["sample size", ("sample", "size"), ("sample", "sizes")]
    ]

    token_indexes = {}
    simulation_to_pages = {}
    for key_word_list in key_words:
        canonical = key_word_list[0]
        synonyms = set(key_word_list)
        token_indexes[canonical] = set()
        simulation_to_pages[canonical] = []
        for idx, (page_no, token_no, token) in enumerate(all_tokens):
            if token in synonyms:
                token_indexes[canonical].add(idx)
                simulation_to_pages[canonical].append(page_no)
            if type(key_word_list[-1]) is tuple and idx < len(all_tokens) - 1 and \
                    (token, all_tokens[idx + 1][2]) in synonyms:
                token_indexes[canonical].add(idx)
                simulation_to_pages[canonical].append(page_no)

    feat = []

    contexts = {}

    feature_pages = []

    for feature in FEATURE_NAMES:
        feat1, feat2 = feature.split("-")

        min_dist = -1
        winning_i = None
        winning_j = None
        for i in token_indexes[feat1]:
            for j in token_indexes[feat2]:
                if min_dist == -1 or abs(i - j) < min_dist:
                    min_dist = abs(i - j)
                    winning_i = i
                    winning_j = j

        if min_dist == -1 or min_dist > 1000:
            min_dist = 1000
            page_no = None
        else:
            start_context = min([winning_i, winning_j])
            end_context = max([winning_i, winning_j])
            start_context = max([0, start_context - 10])
            end_context = min([len(all_tokens) - 1, end_context + 10])
            page_no = all_tokens[winning_i][0]
            contexts[
                f"Occurrence of “{feat1}” within {min_dist} tokens from “{feat2}”"] = f"Page {page_no + 1}: " + " ".join(
                [t[-1] for t in all_tokens[start_context: end_context + 1]])
        feat.append(min_dist)
        feature_pages.append(page_no)

    return feat, simulation_to_pages, contexts, feature_pages


class SimulationExtractor:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD SIMULATION CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.model = None
            return
        with bz2.open(path_to_classifier, "rb") as f:
            self.model = pkl.load(f)

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify whether the trial uses simulation (e.g. Monte Carlo).

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (int) and a map to the pages it's mentioned in.
        """

        if self.model is None:
            print("Warning! Simulation classifier not loaded.")
            return {"prediction": -1}

        lc_tokenised_pages = [[tok.lower() for tok in tokenised_page] for tokenised_page in tokenised_pages]

        all_tokens = list(iterate_tokens(lc_tokenised_pages))

        feat, simulation_to_pages, contexts, feature_pages = extract_features(all_tokens)

        df = pd.DataFrame()
        for feature_idx, feature_name in enumerate(FEATURE_NAMES):
            lst = [feat[feature_idx]]
            for j in range(len(FEATURE_NAMES)):
                if j == feature_idx:
                    lst.append(1000)
                else:
                    lst.append(feat[feature_idx])
            df[feature_name] = lst

        prediction_idx = self.model.predict(df.iloc[0:1])[0]

        probabilities = [p[1] for p in self.model.predict_proba(df)]

        probability_of_simulation = probabilities[0]

        # most_informative_feature = np.argmin(probabilities) - 1
        # print ("most_informative_feature", most_informative_feature)

        page_to_probas = [0] * len(tokenised_pages)
        for idx, proba in enumerate(probabilities[1:]):
            if feature_pages[idx] is not None:
                page_to_probas[feature_pages[idx]] = probabilities[0] - proba


        return {"prediction": prediction_idx, "pages": simulation_to_pages,  "page_scores": page_to_probas,
                "score": probability_of_simulation, "context": contexts

                }
