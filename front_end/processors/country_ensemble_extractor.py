import bz2
import pickle as pkl
from os.path import exists
import numpy as np
import pandas as pd
import operator

from processors.country_extractor import LMIC_COUNTRIES

FEATURES = ["num_mentions", "first_mention", "last_mention",
            "num_mentions_country", "first_mention_country", "last_mention_country",
            "num_mentions_demonym", "first_mention_demonym", "last_mention_demonym",
            "num_mentions_phone", "first_mention_phone", "last_mention_phone",
            "num_mentions_email", "first_mention_email", "last_mention_email",
            "num_mentions_url", "first_mention_url", "last_mention_url",
            "is_lmic", "is_us_ca",
           "is_protocol_high_income", "is_protocol_lmic", "is_protocol_usca", "is_protocol_international", "is_protocol_international_2"]

def make_feature_vector(rule_based_countries, country_group_probas,
    international_score_spacy, international_score_nb):
    """
    Create a feature vector from the previous 4 models, which can go into the ensemble model.

    :param rule_based_countries:
    :param country_group_probas:
    :param international_score_spacy:
    :param international_score_nb:
    :return:
    """
    X = []
    country_identities = []

    fv2 = [
        country_group_probas["HIGH_INCOME"],
        country_group_probas["LMIC"],
        country_group_probas["USCA"],
        international_score_spacy,
        international_score_nb
    ]

    for k, v in rule_based_countries.items():
        all_mentions = [c[0] for c in v]
        country_mentions = [c[0] for c in v if c[1] == "country"]
        demonym_mentions = [c[0] for c in v if c[1] == "demonym"]
        phone_mentions = [c[0] for c in v if c[1] == "phone"]
        email_mentions = [c[0] for c in v if c[1] == "email"]
        url_mentions = [c[0] for c in v if c[1] == "url"]

        fv = [len(all_mentions), min(all_mentions), max(all_mentions),
              len(country_mentions), min(country_mentions) if len(country_mentions) > 0 else 0,
              max(country_mentions) if len(country_mentions) > 0 else 0,
              len(demonym_mentions), min(demonym_mentions) if len(demonym_mentions) > 0 else 0,
              max(demonym_mentions) if len(demonym_mentions) > 0 else 0,
              len(phone_mentions), min(phone_mentions) if len(phone_mentions) > 0 else 0,
              max(phone_mentions) if len(phone_mentions) > 0 else 0,
              len(email_mentions), min(email_mentions) if len(email_mentions) > 0 else 0,
              max(phone_mentions) if len(phone_mentions) > 0 else 0,
              len(url_mentions), min(url_mentions) if len(url_mentions) > 0 else 0,
              max(phone_mentions) if len(phone_mentions) > 0 else 0,
              k in LMIC_COUNTRIES, k in {"US", "CA"}]
        fv.extend(fv2)
        fv = np.asarray(fv)
        X.append(fv)
        country_identities.append(k)

        fv = [0] * 20
        fv.extend(fv2)
        fv = np.asarray(fv)
        X.append(fv)
        country_identities.append("XX")

        if "US" not in rule_based_countries and "CA" not in rule_based_countries:
            fv = [0] * 20
            fv[-1] = 1
            fv.extend(fv2)
            fv = np.asarray(fv)
            X.append(fv)
            country_identities.append("US")

    X = np.asarray(X)
    country_identities = np.asarray(country_identities)

    return X, country_identities


class CountryEnsembleExtractor:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD COUNTRY ENSEMBLE MODEL {path_to_classifier}. You need to run the training script.")
            self.model = None
            return
        with bz2.open(path_to_classifier, "rb") as f:
            self.model = pkl.load(f)

    def process(self,rule_based_countries, country_group_probas,
    international_score_spacy, international_score_nb) -> tuple:
        """
        Identify the countries the trial takes place in.

        :param pages: List of string content of each page.
        :return: The prediction (list of strings of Alpha-2 codes) and a map from each country code to the pages it's mentioned in.
        """
        if self.model is None:
            print("Warning! Country ensemble model not loaded.")
            return {"prediction": -1}

        mean_international_score = np.mean([international_score_nb, international_score_spacy])

        X, country_identities = make_feature_vector(rule_based_countries, country_group_probas,
            international_score_spacy, international_score_nb)

        probas = self.model.predict_proba(X)[:,1]

        scores = {}
        for i in range(len(X)):
            scores[country_identities[i]] = probas[i]

        country_predictions = []

        if mean_international_score > 0.5:
            threshold = 0.1
        else:
            threshold = 0.5

        for country, score in sorted(scores.items(), key=operator.itemgetter(1), reverse=True):
            if score > threshold or len(country_predictions) == 0 or \
                (len(country_predictions) == 1 and mean_international_score > 0.5):
                country_predictions.append(country)

        return {"prediction": country_predictions,
         "score": scores}