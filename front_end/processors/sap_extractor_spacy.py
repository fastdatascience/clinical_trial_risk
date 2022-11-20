import operator
from os.path import exists

import spacy

stats_vocab = {'pe',
               'sap',
               'tabulated',
               'hazard',
               'inferiority',
               'categorical',
               'meddra',
               'residual',
               'itt',
               'continuous',
               'summarised',
               'variables',
               'brv',
               'variable',
               'cox',
               'listings',
               'laz',
               'statistics',
               'descriptive',
               'regression',
               'proportional',
               'hazards',
               'sided',
               'deviation',
               'presented',
               'corresponding',
               'percentage',
               'calculated',
               'analysed',
               'censored',
               'derived',
               'coding',
               'cumulative',
               'seroconversion',
               'soc',
               'tables',
               'proportion',
               'covariates',
               'survival',
               'deviations',
               'pv',
               'interim',
               'class',
               'hypothesis',
               'sensitivity',
               'power',
               'ratio',
               'summarized',
               'median',
               'measurements',
               'model',
               'endpoint',
               'exploratory',
               'statistical',
               'plan',
               'confidence',
               'log',
               'ipm',
               'estimate',
               'planned',
               'demographic',
               'classified',
               '95%',
               'solicited',
               'iu',
               'significance',
               'adjusted',
               'pq',
               'randomisation',
               'ci',
               'values',
               'differences',
               'measures',
               'signed',
               'outcomes',
               'assigned',
               'analyses',
               'overview',
               'intervals',
               'randomised',
               'measurement',
               '90%',
               'versus',
               'analysis',
               'point',
               'adherence',
               'interval',
               'secondary',
               'value',
               'groups',
               'estimated',
               'mean',
               'relative',
               'treat',
               'frequency',
               'outcome',
               'detect',
               'discharge',
               'distribution',
               'baseline',
               'endpoints',
               'dsmb',
               'objectives',
               'efficacy',
               'method',
               'negative',
               'concentration',
               'assessed',
               'confirmed',
               'table',
               'fluid',
               'ae',
               'rate',
               'death',
               'stopping',
               'cd',
               'self',
               'otherwise',
               'parameters',
               'maximum',
               'compared',
               'multiple',
               'crf',
               'type',
               'general',
               'incidence',
               'defined',
               'function',
               '50%',
               'scheduled',
               'considerations',
               'methods',
               'reported',
               'expected',
               'period',
               'meeting',
               'details',
               'first',
               'aes',
               'occurring',
               'doses',
               'rates',
               'count',
               'received',
               'completed',
               'relationship',
               'status',
               'occurrence',
               'placebo',
               'specified'}


# Current best model: Expt01
class SapExtractorSpacy:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD SAP CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.nlp = None
            return
        self.nlp = spacy.load(path_to_classifier)

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify whether the trial has a SAP.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and probability information.
        """
        if self.nlp is None:
            print("Warning! SAP classifier not loaded.")
            return {"prediction": "Error"}

        page_to_num_matches = {}
        pages_to_save = []
        for page_no, tokens in enumerate(tokenised_pages):
            num_matches = 0
            for token in tokens:
                if token.lower() in stats_vocab:
                    num_matches += 1
            page_to_num_matches[page_no] = num_matches

        for page, _ in sorted(page_to_num_matches.items(), key=operator.itemgetter(1), reverse=True):
            pages_to_save.append(page)
            if len(pages_to_save) >= 10:
                break

        text = " ".join([" ".join(tokenised_pages[page]) for page in pages_to_save])

        doc = self.nlp(text)
        prediction_proba = doc.cats["1"]

        is_sap_pred = int(prediction_proba > 0.5)

        return {"prediction": is_sap_pred, "pages": {}, "probas": prediction_proba}
