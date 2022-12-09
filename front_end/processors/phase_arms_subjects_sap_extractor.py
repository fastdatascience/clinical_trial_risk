import re
from os.path import exists

import spacy

word2num = {'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            'ten': 10,
            'eleven': 11,
            'twelve': 12,
            'thirteen': 13,
            'fourteen': 14,
            'fifteen': 15,
            'sixteen': 16,
            'seventeen': 17,
            'eighteen': 18,
            'nineteen': 19,
            'both': 2,
            'single': 2,
            'twenty': 20,
            'thirty': 30,
            'forty': 40,
            'fifty': 50,
            'sixty': 60,
            'seventy': 70,
            'eighty': 80,
            'ninety': 90,
            'hundred': 100,
            'thousand': 1000}

is_number = re.compile(r'^\d+\.?\d*$')

INTERESTING_TERMS_ANY_CONTEXT = {'accrual', 'accrue', 'accrued', 'accruing', 'achieve', 'approximately', 'arm', 'armed',
                                 'arms', 'cohort', 'cohorts', 'enrol', 'enroll', 'enrolled', 'enrolling', 'enrolment',
                                 'enrols', 'group', 'groups', 'n', 'overall', 'phase', 'phases', 'power', 'powered',
                                 'recruit', 'recruited', 'recruiting', 'recruitment', 'recruits', 'sample', 'sampled',
                                 'samples', 'sampling', 'scenarios', 'select', 'selection', 'simulate', 'simulated',
                                 'simulates', 'simulating', 'simulation', 'simulations', 'target', 'total'}

INTERESTING_TERMS_MUST_BE_PRECEDED_BY_NUMBER = {'cases', 'female', 'females', 'healthy', 'individuals', 'infected',
                                                'male', 'males', 'men', 'pairs', 'participants',
                                                'patients', 'people', 'persons', 'pts', 'subjects', 'women'}

ALL_INTERESTING_TERMS = INTERESTING_TERMS_ANY_CONTEXT.union(INTERESTING_TERMS_MUST_BE_PRECEDED_BY_NUMBER)
ARM_TERMS = {'arm', 'armed',
             'arms', 'cohort', 'cohorts', 'group', 'groups'}

phase_lookup = {'0': 0, '1': 0.5, '2': 1, '3': 1.5, '4': 2, '5': 2.5, '6': 3, '7': 4}
arms_lookup = {'8': 1, '9': 2, '10': 3, '11': 4, '12': 5}

num_subjects_lookup = {'13': '1-24', '14': '25-49', '15': '50-99', '16': '100-199', '17': '200-499', '18': '500-999',
                       '19': '1000-9999', '20': '10000-'}


# Current best model: Expt11
class PhaseArmsSubjectsSAPMultiExtractor:

    def __init__(self, path_to_classifier):
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD MULTI CLASSIFIER {path_to_classifier}. You need to run the training script.")
            self.nlp = None
            return
        self.nlp = spacy.load(path_to_classifier)

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify multiple parameters of the trial.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from condition to the pages it's mentioned in.
        """
        if self.nlp is None:
            print("Warning! Multi classifier not loaded.")
            return {"prediction": "Error"}

        phase_to_pages = {"Phase": []}
        num_arms_to_pages = {}
        num_subjects_to_pages = {}
        sap_to_pages = {}
        text = ""
        for page_no, tokens in enumerate(tokenised_pages):
            is_include = [False] * len(tokens)
            for idx, tok in enumerate(tokens):
                lc_tok = tok.lower()
                next_tok = None
                if idx < len(tokens) - 1:
                    next_tok = tokens[idx + 1]
                prev_tok = None
                if idx > 0:
                    prev_tok = tokens[idx - 1]
                antepenultimate_tok = None
                if idx > 1:
                    antepenultimate_tok = tokens[idx - 2]

                if lc_tok in ALL_INTERESTING_TERMS:
                    if lc_tok in {"phase", "phases"}:
                        phase_to_pages["Phase"].append(page_no)
                    elif lc_tok in ARM_TERMS:
                        if lc_tok not in num_arms_to_pages:
                            num_arms_to_pages[lc_tok] = []
                        num_arms_to_pages[lc_tok].append(page_no)
                    else:
                        if lc_tok not in num_subjects_to_pages:
                            num_subjects_to_pages[lc_tok] = []
                        num_subjects_to_pages[lc_tok].append(page_no)
                    to_include = True
                    # Override the "interesting terms" list by using some context dependent information.
                    if lc_tok == "n" and next_tok is not None and next_tok not in {"=", ">", "<", "≥"}:
                        to_include = False
                    if idx > 1 and lc_tok in INTERESTING_TERMS_MUST_BE_PRECEDED_BY_NUMBER and not (
                            is_number.match(prev_tok) or is_number.match(
                        antepenultimate_tok) or prev_tok.lower() in word2num or antepenultimate_tok.lower() in word2num):
                        to_include = False

                    if to_include:
                        for token_index in range(idx - 15, idx + 15):
                            if token_index >= 0 and token_index < len(tokens):
                                is_include[token_index] = True

            for idx, tok in enumerate(tokens):
                if is_include[idx]:
                    text += tok + " "

        doc = self.nlp(text)

        # Spacy will tag everything together as one dict. We split it up and get the best phase, arms etc
        phases_dict = {}
        for output_idx, phase in phase_lookup.items():
            phases_dict[phase] = doc.cats[output_idx]

        phase = max(phases_dict, key=phases_dict.get)
        phase_proba = phases_dict[phase]

        arms_dict = {}
        for output_idx, arms in arms_lookup.items():
            arms_dict[arms] = doc.cats[output_idx]

        num_arms = max(arms_dict, key=arms_dict.get)
        num_arms_proba = arms_dict[num_arms]

        num_subjects_dict = {}
        for output_idx, subjects in num_subjects_lookup.items():
            num_subjects_dict[subjects] = doc.cats[output_idx]

        num_subjects = max(num_subjects_dict, key=num_subjects_dict.get)
        num_subjects_proba = num_subjects_dict[num_subjects]

        sap_proba = doc.cats["21"]
        has_sap = int(sap_proba > 0.5)

        return {"prediction": [phase, num_arms, num_subjects, has_sap],
                "pages": [phase_to_pages, num_arms_to_pages, num_subjects_to_pages, sap_to_pages],
                "score": [phase_proba, num_arms_proba, num_subjects_proba, sap_proba]}
