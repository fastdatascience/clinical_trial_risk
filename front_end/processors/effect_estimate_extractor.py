import bz2
import pickle as pkl
import re
from os.path import exists

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

from util.page_tokeniser import iterate_tokens

# Should be shared with training code

NUMBERS_REGEX = re.compile(r'-?(?:\d+-)?\d+(?:\.\d+)?%?(?:sd|pc)?')
PURE_NUMBERS_REGEX = re.compile(r'^\d+$')

NUMBERS_IN_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
for n in list(NUMBERS_IN_WORDS):
    NUMBERS_IN_WORDS.add(f"{n}-fold")
    NUMBERS_IN_WORDS.add(f"{n}fold")
for n in range(1, 101):
    NUMBERS_IN_WORDS.add(f"{n}-fold")

WINDOW_SIZE = 20


def transform_tokens(list_of_lists_of_tokens, vectoriser):
    token_counts = np.zeros((len(list_of_lists_of_tokens), len(vectoriser.vocabulary_)))
    for doc_no in range(len(list_of_lists_of_tokens)):
        for i in range(WINDOW_SIZE * 2):
            weight = WINDOW_SIZE + 1 - abs(i - WINDOW_SIZE)
            token_lower = list_of_lists_of_tokens[doc_no][i].lower()
            if token_lower in vectoriser.vocabulary_:
                token_counts[doc_no, vectoriser.vocabulary_[token_lower]] += weight
    return token_counts


def get_context(all_tokens, idx):
    context = []
    start = idx - WINDOW_SIZE
    end = idx + WINDOW_SIZE
    for j in range(start, end):
        if j < 0 or j >= len(all_tokens):
            w = ""  # pad
        else:
            w = all_tokens[j][-1]
            if w.endswith("%"):
                w = "DIGITPERCENT"
            elif NUMBERS_REGEX.match(w):
                if "." in w:
                    w = "DECIMALNUMBER"
                elif PURE_NUMBERS_REGEX.match(w):
                    w = f"{len(w)}DIGITS"
                else:
                    w = "OTHERNUMBER"
            elif w in NUMBERS_IN_WORDS:
                w = "NUMBERWORD"
        context.append(w)
    return context


# end stuff that should be shared

class EffectEstimateExtractor:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD EFFECT ESTIMATE CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.model = None
            return
        with bz2.open(path_to_classifier, "rb") as f:
            self.model = pkl.load(f)
        self.vectoriser = self.model.named_steps['countvectorizer']
        self.transformer = self.model.named_steps['tfidftransformer']
        self.nb = self.model.named_steps['multinomialnb']

        self.vocabulary = {v: k for k, v in self.vectoriser.vocabulary_.items()}

        self.shorter_pipeline = make_pipeline(self.transformer, self.nb)

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify the effect estimate in the document.

        Key words:

        * Cohen’s d
        * Cohens d
        * Cohen's d
        * RR
        * relative risk
        * odds ratio
        * hazard ratio
        * risk ratio
        * Cohen’s h
        * prevention efficacy (e.g. 90%)
        * prevention effectiveness
        * effect estimator

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from effect estimate to the pages it's mentioned in.
        """
        if self.model is None:
            print("Warning! Effect estimate classifier not loaded.")
            return {"prediction": -1}

        all_tokens = list(iterate_tokens(tokenised_pages))

        instances = []
        contexts = []
        page_nos = []
        token_idxs = []
        for idx, (page_no, token_no, token) in enumerate(all_tokens):
            if NUMBERS_REGEX.match(token) or token in NUMBERS_IN_WORDS:
                if idx < len(all_tokens) - 1:
                    next_token = all_tokens[idx + 1][2]
                    # Hard coded rule to weed out some common false positives
                    if next_token.lower() in {"sided", "power", "significance", "clusters", "children", "adults",
                                              "sites", "centers", "centres", "confidence"}:
                        continue
                if idx > 2:
                    prev_prev_token = all_tokens[idx - 2][2]
                    prev_token = all_tokens[idx - 1][2]
                    if prev_token.lower() in {"of", "is"} and prev_prev_token.lower() in {"power", "confidence"}:
                        continue

                context = get_context(all_tokens, idx)
                if context:
                    instances.append(token)
                    contexts.append(context)
                    page_nos.append(page_no)
                    token_idxs.append(idx)

        page_to_probas = [0] * len(tokenised_pages)
        if len(contexts) > 0:
            X_test = transform_tokens(contexts, self.vectoriser)

            y_pred = self.shorter_pipeline.predict(X_test)

            y_pred_proba = self.shorter_pipeline.predict_proba(X_test)[:, 1]

            df_result = pd.DataFrame(
                {"token_idx": token_idxs, "token": instances, "page_no": page_nos, "y_pred": y_pred,
                 "y_pred_proba": y_pred_proba})

            # hack: make it more lenient because there were very few positive examples in the training set.
            df_result["y_pred"] = df_result["y_pred_proba"] > 0.2

            top_score = df_result.y_pred_proba.max()
            page_to_max_proba = df_result.groupby("page_no")["y_pred"].max()
            for idx in range(len(tokenised_pages)):
                if idx in page_to_max_proba.index:
                    page_to_probas[idx] = page_to_max_proba[idx]

            top_results = df_result[df_result.y_pred == 1]
        else:
            top_score = 0
            top_results = []

        effect_estimate_to_pages = {"effect size": [], "effect estimate": [], "reduction": [], "detect": [],
                                    "odds/hazard/risk ratio": [], "relative risk/RR": [],
                                    "prevention efficacy/effectiveness": [], "Cohen's d/h": []}
        is_effect_estimate = 0
        context = {}
        score = 0
        if len(top_results) > 0:
            is_effect_estimate = 1
            for idx in range(len(top_results)):
                token = top_results.token.iloc[idx]
                if token not in effect_estimate_to_pages:
                    effect_estimate_to_pages[token] = []
                effect_estimate_to_pages[token].append(top_results.page_no.iloc[idx])

            top_results.sort_values("y_pred_proba", ascending=False, inplace=True)
            score = top_results.y_pred_proba.iloc[0]
            top_results.drop_duplicates("token", inplace=True)
            for idx in range(len(top_results)):
                token_idx = top_results.token_idx.iloc[idx]
                token = top_results.token.iloc[idx]
                if token not in context:
                    context[token] = ""
                context[token] = (context[token] + " " + f"Page {top_results.page_no.iloc[0] + 1}: " + " ".join(
                    [token for page_no, token_no, token in all_tokens[token_idx - 20:token_idx + 20]])).strip()

        # Add in some hard coded keywords for display purposes only
        # The presence of these keywords alone does not affect the classifier's output, as they are meaningless without a numerical value.
        last_token = None
        for page_no, tokens in enumerate(tokenised_pages):
            for token_idx, token in enumerate(tokens):
                token_orig = token
                token = token.lower()
                if last_token == "effect" and token in {"size", "sizes"}:
                    effect_estimate_to_pages["effect size"].append(page_no)
                elif last_token == "effect" and token in {"estimate", "estimates", "estimator"}:
                    effect_estimate_to_pages["effect estimate"].append(page_no)
                elif token in {"reduction", "reductions"}:
                    effect_estimate_to_pages["reduction"].append(page_no)
                elif token in {"detect"}:
                    effect_estimate_to_pages["detect"].append(page_no)
                elif last_token in {"odds", "hazard", "risk"} and token in {"ratio", "ratios"}:
                    effect_estimate_to_pages["odds/hazard/risk ratio"].append(page_no)
                elif (last_token in {"relative"} and token in {"risk", "risks"}) or token_orig == "RR":
                    effect_estimate_to_pages["relative risk/RR"].append(page_no)
                elif last_token in {"prevention"} and token in {"efficacy", "effectiveness"}:
                    effect_estimate_to_pages["prevention efficacy/effectiveness"].append(page_no)
                elif last_token in {"cohens", "cohen's", "cohen’s"} and token in {"d", "h"}:
                    effect_estimate_to_pages["Cohen's d/h"].append(page_no)
                last_token = token

        ret = {"prediction": is_effect_estimate, "pages": effect_estimate_to_pages, "context": context,
               "page_scores": page_to_probas,
               "score": top_score}
        if score > 0:
            ret["score"] = score

        return ret
