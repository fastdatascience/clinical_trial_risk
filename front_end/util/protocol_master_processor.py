import re
import traceback

import numpy as np

from processors.condition_extractor import ConditionExtractor
from processors.country_extractor import CountryExtractor, allowed_countries
from processors.country_group_extractor import CountryGroupExtractor
from processors.duration_extractor import DurationExtractor
from processors.effect_estimate_extractor import EffectEstimateExtractor
from processors.international_extractor_spacy import InternationalExtractorSpacy
from processors.num_arms_extractor import NumArmsExtractor
from processors.num_arms_extractor_naive_bayes import NumArmsExtractorNaiveBayes
from processors.num_arms_extractor_spacy import NumArmsExtractorSpacy
from processors.num_endpoints_extractor import NumEndpointsExtractor
from processors.num_sites_extractor import NumSitesExtractor
from processors.num_subjects_extractor import NumSubjectsExtractor
from processors.num_subjects_extractor_naive_bayes import NumSubjectsExtractorNaiveBayes
from processors.phase_extractor import PhaseExtractor
from processors.phase_extractor_spacy import PhaseExtractorSpacy
from processors.sap_extractor import SapExtractor
from processors.sap_extractor_document_level_naive_bayes import SapExtractorDocumentLevel
from processors.simulation_extractor import SimulationExtractor
from util import page_tokeniser

is_number_regex = re.compile(r'^\d+,?\d+$')


class MasterProcessor:

    def __init__(self, condition_extractor_model_file: str, phase_extractor_model_file: str,
                 phase_extractor_model_file_spacy: str, sap_extractor_model_file_document_level: str,
                 sap_extractor_model_file: str,
                 effect_estimator_extractor_model_file: str, num_subjects_extractor_model_file: str,
                 num_subjects_extractor_nb_model_file: str,
                 num_arms_extractor_model_file: str, num_arms_extractor_model_file_spacy: str,
                 international_extractor_model_file: str, country_group_extractor_model_file: str,
                 simulation_extractor_model_file: str):
        self.condition_extractor = ConditionExtractor(condition_extractor_model_file)
        self.phase_extractor = PhaseExtractor(phase_extractor_model_file)
        self.phase_extractor_spacy = PhaseExtractorSpacy(phase_extractor_model_file_spacy)
        self.sap_extractor_document_level = SapExtractorDocumentLevel(sap_extractor_model_file_document_level)
        self.sap_extractor = SapExtractor(sap_extractor_model_file)
        self.effect_estimate_extractor = EffectEstimateExtractor(effect_estimator_extractor_model_file)
        self.duration_extractor = DurationExtractor()
        self.num_endpoints_extractor = NumEndpointsExtractor()
        self.num_sites_extractor = NumSitesExtractor()
        self.num_subjects_extractor = NumSubjectsExtractor(num_subjects_extractor_model_file)
        self.num_subjects_extractor_nb = NumSubjectsExtractorNaiveBayes(num_subjects_extractor_nb_model_file)
        self.num_arms_extractor_nb = NumArmsExtractorNaiveBayes(num_arms_extractor_model_file)
        self.num_arms_extractor_spacy = NumArmsExtractorSpacy(num_arms_extractor_model_file_spacy)
        self.num_arms_extractor = NumArmsExtractor()
        self.country_extractor = CountryExtractor()
        self.country_group_extractor = CountryGroupExtractor(country_group_extractor_model_file)
        self.international_extractor = InternationalExtractorSpacy(international_extractor_model_file)
        self.simulation_extractor = SimulationExtractor(simulation_extractor_model_file)

    def process_protocol(self, pages: list, report_progress=print, disable: set = {}) -> tuple:
        """
        Apply the various NLP components to the raw text of the protocol to identify the key parameters such as phase, condition, etc.
        :param pages: List of strings: full text of page 1, text of page 2, etc...
        :param report_progress: A function which can be called with a single string argument to log a human readable description of progress, defaulting to print.
        :return: A tuple of the tokenised pages, and then the various parameters and their page numbers which were extracted.
        """
        report_progress("Splitting the document into words (tokens)...")

        tokenised_pages = page_tokeniser.tokenise_pages(pages)

        total_words = sum([len(p) for p in tokenised_pages])
        report_progress(f"There were {total_words} words in the document.\n")

        report_progress("Searching for a likely pathology...")

        if "condition" in disable:
            condition_to_pages = {"prediction": "Error"}
        else:
            condition_to_pages = self.condition_extractor.process(tokenised_pages)
            if condition_to_pages["prediction"] == "Error":
                report_progress(
                    "The machine learning model which detects the pathology was not loaded.\n")
            elif condition_to_pages["prediction"] != "Other":
                if condition_to_pages["prediction"] == "HIV":
                    n = "n"
                else:
                    n = ""
                report_progress(f"This looks like a{n} {condition_to_pages['prediction']} trial.\n")
            else:
                report_progress("This does not look like either an HIV or TB protocol.\n")
            if total_words < 500 or len(pages) < 5:
                report_progress(f"Warning! The document appears to be short. Perhaps it has a lot of image data?")
                condition_to_pages['prediction'] = "Error"
                condition_to_pages['error'] = f"The document is too short ({len(pages)} pages, {total_words} words)"

        if "phase" in disable:
            phase_to_pages = {"prediction": 0}
        else:
            report_progress("Searching for a phase...")
            try:
                phase_to_pages = self.phase_extractor.process(tokenised_pages)
                report_progress(f"This looks like a Phase {phase_to_pages['prediction']} trial.\n")
            except:
                report_progress("The tool was unable to identify a trial phase. An error occurred.\n")
                report_progress(traceback.format_exc())
                print(traceback.format_exc())
                phase_to_pages = {'prediction': 0}

            try:
                phase_to_pages_spacy = self.phase_extractor_spacy.process(tokenised_pages)
                report_progress(f"Neural network thought it was a Phase {phase_to_pages_spacy['prediction']} trial.\n")

                combined_scores = {}
                for phase, score in phase_to_pages_spacy["probas"].items():
                    orig_score = phase_to_pages["probas"].get(phase)
                    if orig_score is not None:
                        combined_scores[phase] = np.mean([float(score), float(orig_score)])
                if len(combined_scores) > 0:
                    phase_to_pages["prediction"] = float(
                        re.sub(r'Phase ', '', max(combined_scores, key=combined_scores.get)))
            except:
                report_progress("Error running neural network model.\n")

        if "sap" in disable:
            sap_to_pages = {"prediction": -1}
        else:
            report_progress("Searching for a statistical analysis plan...")
            try:
                sap_to_pages = self.sap_extractor.process(tokenised_pages)
                if sap_to_pages['prediction'] == 1:
                    report_progress(
                        "It looks like the authors have included their statistical analysis plan in the protocol.\n")
                elif sap_to_pages['prediction'] == -1:
                    report_progress(
                        "The machine learning model which detects SAPs was not loaded.\n")
                else:
                    report_progress("It does not look like the protocol contains a statistical analysis plan.\n")

                report_progress(
                    "Testing top pages for SAP with document level SAP Naive Bayes model to refine SAP prediction.\n")
                sap_to_pages_document_level = self.sap_extractor_document_level.process(tokenised_pages)
                report_progress(
                    "Document level Naive Bayes model found SAP score " + str(
                        sap_to_pages_document_level["prediction"]) + " with score " + str(
                        sap_to_pages_document_level["score"]) + ".\n")
                sap_to_pages["prediction"] = sap_to_pages_document_level["prediction"]
                sap_to_pages["score"] = sap_to_pages_document_level["score"]
            except:
                print(traceback.format_exc())
                report_progress("The tool was unable to identify an SAP. An error occurred.\n")
                sap_to_pages = {'prediction': 0}

        if "effect_estimate" in disable:
            effect_estimate_to_pages = {"prediction": -1}
        else:
            report_progress("Searching for an effect estimate...")
            try:
                effect_estimate_to_pages = self.effect_estimate_extractor.process(tokenised_pages)
                if effect_estimate_to_pages['prediction'] == 1:
                    report_progress("Identified probable effect estimate.\n")
                elif effect_estimate_to_pages['prediction'] == -1:
                    report_progress(
                        "The machine learning model which detects the effect estimate was not loaded.\n")
                else:
                    report_progress("It does not look like the protocol contains an effect estimate.\n")
            except:
                report_progress("Error extracting effect estimate!\n")
                effect_estimate_to_pages = {"prediction": 0}
                print(traceback.format_exc())

        if "num_arms" in disable:
            num_arms_to_pages = {"prediction": 0}
        else:
            try:
                num_arms_to_pages_nb = self.num_arms_extractor_nb.process(tokenised_pages)
                report_progress(f"Naive Bayes arms prediction probabilities: {num_arms_to_pages_nb['proba']}.\n")
            except:
                report_progress("Error extracting number of arms!\n")
                num_arms_to_pages_nb = {"prediction": "2"}
                print(traceback.format_exc())

            try:
                num_arms_to_pages_spacy = self.num_arms_extractor_spacy.process(tokenised_pages)
                report_progress(f"Spacy arms prediction probabilities: {num_arms_to_pages_spacy['proba']}.\n")
            except:
                report_progress("Error extracting number of arms!\n")
                num_arms_to_pages_spacy = {"prediction": "2"}
                print(traceback.format_exc())

            combined_arms_probabilities = {}
            for num_arms in ["1", "2", "3+"]:
                combined_arms_probabilities[num_arms] = (num_arms_to_pages_spacy["proba"][num_arms] +
                                                         num_arms_to_pages_nb["proba"][num_arms]) / 2
            most_likely_arms = max(combined_arms_probabilities, key=combined_arms_probabilities.get)

            report_progress("Searching for a number of arms...")
            try:
                num_arms_to_pages = self.num_arms_extractor.process(tokenised_pages)
                if num_arms_to_pages['prediction'] is not None:
                    report_progress(f"It looks like the trial has {num_arms_to_pages['prediction']} arm(s).\n")

                    if most_likely_arms in ("1", "2") and num_arms_to_pages['prediction'] == int(
                            re.sub(r'\+', '', num_arms_to_pages_nb["prediction"])):
                        report_progress("The NB prediction and the rule based prediction match!")
                    elif most_likely_arms == "3+" and num_arms_to_pages['prediction'] >= 3:
                        report_progress("The NB prediction and the rule based prediction match by range!")
                else:
                    report_progress(f"No explicit mention of arms found.\n")
                    num_arms_to_pages["prediction"] = int(re.sub(r'\+', '', most_likely_arms))
            except:
                report_progress("Error extracting number of arms!\n")
                num_arms_to_pages = {"prediction": 0}
                print(traceback.format_exc())

            num_arms_to_pages["pages"] = num_arms_to_pages["pages"] | num_arms_to_pages_nb["pages"]

        if "num_subjects" in disable:
            num_subjects_to_pages = {"prediction": 0}
        else:
            report_progress("Running Naive Bayes classifier for number of subjects...")
            num_subjects_to_pages_nb = self.num_subjects_extractor_nb.process(tokenised_pages)

            report_progress("Searching for a number of subjects...")
            try:
                num_subjects_to_pages = self.num_subjects_extractor.process(tokenised_pages)
                report_progress(f"It looks like the trial has {num_subjects_to_pages['prediction']} participants.\n")

                '''
                # This part is actually reducing the accuracy
                def get_num_subjects_clean(num):
                    if num >= 134:
                        return "134+"
                    if num >= 34:
                        return "34-133"
                    return "1-33"

                combined_probas = {}
                for i, p in num_subjects_to_pages["proba"].items():
                    # Multiply by number of arms if applicable
                    if i in num_subjects_to_pages["is_per_arm"]:
                        i *= num_arms_to_pages["prediction"]
                    cat = get_num_subjects_clean(int(i))
                    combined_probas[i] = (p + num_subjects_to_pages_nb["proba"][cat]) / 2

                num_subjects_to_pages["proba"] = combined_probas
                num_subjects_to_pages["prediction"] = max(combined_probas, key=combined_probas.get)
                '''
            except:
                report_progress("Error extracting number of subjects!\n")
                num_subjects_to_pages = {"prediction": 0}
                print(traceback.format_exc())

        if "country" in disable:
            country_to_pages = {"prediction": []}
        else:
            report_progress("Searching for the countries of investigation...")
            country_to_pages = self.country_extractor.process(pages)
            if len(country_to_pages['prediction']) > 1:
                country_ies = "countries"
            else:
                country_ies = "country"
            if len(country_to_pages['prediction']) == 0:
                report_progress("No country was found.")
            else:
                report_progress(
                    f"It looks like the trial takes place in {len(country_to_pages['prediction'])} {country_ies}: {','.join(country_to_pages['prediction'])}\n")

            country_group_to_pages = self.country_group_extractor.process(tokenised_pages)

            report_progress(
                f"Neural network found that trial country is likely to be {country_group_to_pages['prediction']}.\n")

            if country_group_to_pages["prediction"] == "USCA":
                report_progress(
                    f"Neural network found that trial is likely to be US/Canada only.\n")
                if len(country_to_pages["prediction"]) > 1:
                    country_to_pages["prediction"] = [c for c in country_to_pages["prediction"] if c in ("US", "CA")]
                    report_progress(
                        f"Overriding countries found. Setting to. " + str(country_to_pages["prediction"]) + ".\n")

            is_international_to_pages = self.international_extractor.process(tokenised_pages)

            if is_international_to_pages["prediction"] == 0:
                report_progress(
                    f"Neural network found that trial is likely to be a single-country trial.\n")
                if len(country_to_pages["prediction"]) > 1:
                    report_progress(
                        f"Overriding countries found. Taking the highest-scoring country.\n")
                    country_to_pages["prediction"] = country_to_pages["prediction"][:1]
            else:
                report_progress(
                    f"Neural network found that trial is likely to be international.\n")
                if country_group_to_pages["prediction"] != "USCA":
                    if len(country_to_pages["prediction"]) <= 1:
                        report_progress(
                            f"Overriding countries found. Taking all countries.\n")
                        country_to_pages["prediction"] = list(sorted(country_to_pages["pages"]))
                    if country_group_to_pages["prediction"] == "HIGH INCOME":
                        country_to_pages["prediction"] = list(
                            [x for x in sorted(country_to_pages["pages"]) if x not in allowed_countries])
                    if len(country_to_pages["prediction"]) <= 1:
                        country_to_pages["prediction"].append("XX")

        if "simulation" in disable:
            simulation_to_pages = {"prediction": -1}
        else:
            report_progress("Searching for any mentions of simulation...")
            simulation_to_pages = self.simulation_extractor.process(tokenised_pages)
            if simulation_to_pages['prediction'] == 1:
                report_progress("The authors probably used simulation for sample size.\n")
            elif simulation_to_pages['prediction'] == -1:
                report_progress(
                    "The machine learning model which detects the simulation_to_pages was not loaded.\n")
            else:
                report_progress("It does not look like the authors used simulation for sample size.\n")

        return tokenised_pages, condition_to_pages, phase_to_pages, sap_to_pages, \
               effect_estimate_to_pages, num_subjects_to_pages, num_arms_to_pages, country_to_pages, simulation_to_pages
