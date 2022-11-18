import json
import re
import traceback

import numpy as np

from processors.condition_extractor import ConditionExtractor
from processors.country_extractor import CountryExtractor
from processors.duration_extractor import DurationExtractor
from processors.effect_estimate_extractor import EffectEstimateExtractor
from processors.international_extractor_spacy import InternationalExtractorSpacy
from processors.num_arms_extractor import NumArmsExtractor
from processors.num_endpoints_extractor import NumEndpointsExtractor
from processors.num_sites_extractor import NumSitesExtractor
from processors.num_subjects_extractor import NumSubjectsExtractor
from processors.phase_arms_subjects_sap_extractor_keras import PhaseArmsSubjectsSAPMultiExtractorKeras
from processors.phase_extractor import PhaseExtractor
from processors.phase_extractor_spacy import PhaseExtractorSpacy
from processors.sap_extractor import SapExtractor
from processors.simulation_extractor import SimulationExtractor
from util import page_tokeniser

is_number_regex = re.compile(r'^\d+,?\d+$')


class MasterProcessor:

    def __init__(self, condition_extractor_model_file: str,phase_extractor_model_file:str,  phase_extractor_model_file_spacy: str, sap_extractor_model_file: str,
                 effect_estimator_extractor_model_file: str, num_subjects_extractor_model_file: str,
                 international_extractor_model_file: str, simulation_extractor_model_file: str,
                 phase_arms_subjects_sap_multi_extractor_file: str):
        self.condition_extractor = ConditionExtractor(condition_extractor_model_file)
        self.phase_extractor = PhaseExtractor(phase_extractor_model_file)
        self.phase_extractor_spacy = PhaseExtractorSpacy(phase_extractor_model_file_spacy)
        self.sap_extractor = SapExtractor(sap_extractor_model_file)
        self.effect_estimate_extractor = EffectEstimateExtractor(effect_estimator_extractor_model_file)
        self.duration_extractor = DurationExtractor()
        self.num_endpoints_extractor = NumEndpointsExtractor()
        self.num_sites_extractor = NumSitesExtractor()
        self.num_subjects_extractor = NumSubjectsExtractor(num_subjects_extractor_model_file)
        self.num_arms_extractor = NumArmsExtractor()
        self.country_extractor = CountryExtractor()
        self.international_extractor = InternationalExtractorSpacy(international_extractor_model_file)
        self.simulation_extractor = SimulationExtractor(simulation_extractor_model_file)
        self.spacy_phase_arms_subjects_sap_multi_extractor = PhaseArmsSubjectsSAPMultiExtractorKeras(
            phase_arms_subjects_sap_multi_extractor_file)

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
                    phase_to_pages["prediction"] = float(re.sub(r'Phase ', '', max(combined_scores, key=combined_scores.get)))
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

        if "num_subjects" in disable:
            num_subjects_to_pages = {"prediction": 0}
        else:
            report_progress("Searching for a number of subjects...")
            try:
                num_subjects_to_pages = self.num_subjects_extractor.process(tokenised_pages)
                report_progress(f"It looks like the trial has {num_subjects_to_pages['prediction']} participants.\n")
            except:
                report_progress("Error extracting number of subjects!\n")
                num_subjects_to_pages = {"prediction": 0}
                print(traceback.format_exc())

        if "num_arms" in disable:
            num_subjects_to_pages = {"prediction": 0}
        else:
            report_progress("Searching for a number of arms...")
            try:
                num_arms_to_pages = self.num_arms_extractor.process(tokenised_pages)
                if num_arms_to_pages['prediction'] is not None:
                    report_progress(f"It looks like the trial has {num_arms_to_pages['prediction']} arm(s).\n")
                else:
                    report_progress(f"No explicit mention of arms found.\n")
            except:
                report_progress("Error extracting number of arms!\n")
                num_arms_to_pages = {"prediction": 0}
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

            is_international_to_pages = self.international_extractor.process(tokenised_pages)

            if is_international_to_pages["prediction"] == 0:
                report_progress(
                    f"Neural network found that trial is likely to be a single-country trial.")
                if len(country_to_pages["prediction"]) > 1:
                    report_progress(
                        f"Overriding countries found. Taking the highest-scoring country.")
                    country_to_pages["prediction"] = country_to_pages["prediction"][:1]
            else:
                report_progress(
                    f"Neural network found that trial is likely to be international.")
                if len(country_to_pages["prediction"]) <= 1:
                    report_progress(
                        f"Overriding countries found. Taking all countries.")
                    country_to_pages["prediction"] = list(sorted(country_to_pages["pages"]))
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

        '''
        if "multi" not in disable:
            # Override or modify some of the predictions of the earlier rule-based components.
            # This is using a neural network model which does a lot of heavy lifting but which is computationally expensive.
            report_progress("Running neural network multi-label model to refine predictions...\n")
            try:
                multi_to_pages = self.spacy_phase_arms_subjects_sap_multi_extractor.process(tokenised_pages)
                report_progress("Neural network model output is " + json.dumps(multi_to_pages["prediction"]) + ".\n")

                report_progress("Phase is likely to be: " + str(multi_to_pages["prediction"][0]) + ".\n")
                phase_to_pages["prediction"] = multi_to_pages["prediction"][0]
                phase_to_pages["pages"] = phase_to_pages["pages"] | multi_to_pages["pages"][0]

                report_progress("Number of arms is likely to be: " + str(multi_to_pages["prediction"][1]) + ".\n")
                num_arms_to_pages["prediction"] = multi_to_pages["prediction"][1]
                num_arms_to_pages["pages"] = num_arms_to_pages["pages"] | multi_to_pages["pages"][1]

                report_progress("Number of subjects is likely in range: " + multi_to_pages["prediction"][2] + ".\n")
                num_subjects_range_str = multi_to_pages["prediction"][2]
                num_subjects_to_pages["pages"] = num_subjects_to_pages["pages"] | multi_to_pages["pages"][2]
                num_subjects_to_pages["comment"] = num_subjects_to_pages["comment"] + " / " + multi_to_pages["comment"][2]

                sap_to_pages["prediction"] = multi_to_pages["prediction"][3]
                sap_to_pages["pages"] = sap_to_pages["pages"] | multi_to_pages["pages"][3]

                # Logic to put number of subjects in correct range
                num_subjects_range = num_subjects_range_str.split("-")
                num_subjects_range_lower = int(num_subjects_range[0])
                num_subjects_range_upper = None
                if num_subjects_range[1] != "":
                    num_subjects_range_upper = int(num_subjects_range[1])
                if num_subjects_to_pages["prediction"] >= num_subjects_range_lower and (
                        num_subjects_range_upper is None or num_subjects_to_pages[
                    "prediction"] <= num_subjects_range_upper):
                    report_progress(
                        f"Num subjects {num_subjects_to_pages['prediction']} is already within acceptable range {num_subjects_range_str}. No change needed.\n")
                else:
                    report_progress(
                        f"Num subjects {num_subjects_to_pages['prediction']} is not already within acceptable range {num_subjects_range_str}. Updating.\n")
                    is_found_new_num_subjects = False
                    for i in num_subjects_to_pages["pages"]:
                        if is_number_regex.match(i) and int(i) >= num_subjects_range_lower and (
                                num_subjects_range_upper is None or int(i) <= num_subjects_range_upper):
                            report_progress(f"Overriding with {i}")
                            num_subjects_to_pages['prediction'] = i
                            is_found_new_num_subjects = True
                            break
                    if not is_found_new_num_subjects:
                        if num_subjects_range_upper is not None:
                            mid_point_of_range = np.mean([num_subjects_range_lower, num_subjects_range_upper])
                        else:
                            mid_point_of_range = num_subjects_range_lower
                        report_progress(f"Mid point of the range of probable sample sizes is {mid_point_of_range}. Looking for the number occurring in the text which is closest to this midpoint.")
                        candidates = set()
                        for i in num_subjects_to_pages["pages"]:
                            if is_number_regex.match(i):
                                candidates.add(int(i))
                        if len(candidates) > 0:
                            num_subjects_to_pages['prediction'] = min(candidates, key = lambda i : abs(i - mid_point_of_range))
                            report_progress(f"Overridden with {num_subjects_to_pages['prediction']}")

            except:
                report_progress("Error running Spacy multi-label model!\n")
                print(traceback.format_exc())
        '''

        return tokenised_pages, condition_to_pages, phase_to_pages, sap_to_pages, \
               effect_estimate_to_pages, num_subjects_to_pages, num_arms_to_pages, country_to_pages, simulation_to_pages
