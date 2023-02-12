import numpy as np
import pandas as pd

import time

from util.score_to_risk_level_converter import get_excel_formula_for_risk_level, get_risk_level_and_traffic_light, \
    get_human_readable_risk_levels


def calculate_risk_level(file_name: str, condition: str, phase: float, sap: int, effect_estimate: int,
                         num_subjects_and_tertile: list,num_arms:int, is_international: int, simulation: int,

high_risk_threshold : int,
    low_risk_threshold : int,
    weight_number_of_arms : float,
    weight_phase : float,
    weight_sap : float,
    weight_effect_estimate : float,
    weight_num_subjects : float,
    weight_international : float,
    weight_simulation : float,
    weight_bias : float
                         ) -> tuple:
    """
    Calculate the risk of a trial given the parameters that have been extracted about it by the NLP components.
    :param file_name:
    :param condition:
    :param phase:
    :param sap:
    :param effect_estimate:
    :num_subjects_and_tertile num_subjects:
    :param is_international:
    :param simulation:
    :return: The total score of the protocol, a dataframe containing the Excel formulae, and a human-readable description of the risk calculation.
    """

    start_time = time.time()
    num_subjects, num_subjects_tertile, lower_tertile, upper_tertile = num_subjects_and_tertile
    if phase is None:
        phase, lower_tertile, upper_tertile, num_subjects_tertile, num_subjects_tertile_name = None, None, None, None, ""
    else:
        if num_subjects_tertile == 0:
            num_subjects_tertile_name = "small"
        elif num_subjects_tertile == 1:
            num_subjects_tertile_name = "medium"
        else:
            num_subjects_tertile_name = "large"
    df = pd.DataFrame()
    df["Parameter"] = [file_name,
                       "Trial is for condition",
                       "Number of subjects",
                       "Lower tertile number of subjects for phase and pathology",
                       "Upper tertile number of subjects for phase and pathology",
                       "Number of arms",
                       "Trial phase",
                       "SAP completed?",
                       "Effect Estimate disclosed?",
                       "Number of subjects low/medium/high",
                       "International?",
                       "Simulation?",
                       "Constant"
                       ]
    df["reason"] = [file_name,
                    "",
                    "",
                    "",
                    "",
                    f"because the trial has {num_arms} arm{'s'[:num_arms^1]}",
                    f"because the trial is Phase {phase}",
                    "because the trial included a Statistical Analysis Plan (SAP)",
                    "because the authors disclosed an effect estimate",
                    f"because the sample size is {num_subjects_tertile_name}",
                    "because the trial takes place in multiple countries",
                    "because the authors used sample size simulation",
                    "CONSTANT"
                    ]
    df["Value"] = [None,
                   condition,
                   num_subjects,
                   lower_tertile,
                   upper_tertile,
                   num_arms,
                   max((phase, 0)),
                   max((sap, 0)),
                   max((effect_estimate, 0)),
                   num_subjects_tertile,
                   is_international,
                   max((simulation, 0)),
                   1
                   ]
    df["Weight"] = [None,
                    None,
                    None,
                    None,
                    None,
                    weight_number_of_arms,
                    weight_phase,
                    weight_sap,
                    weight_effect_estimate,
                    weight_num_subjects,
                    weight_international,
                    weight_simulation,
                    weight_bias,
                    ]
    scores = df.Value * df.Weight
    df["Score"] = scores

    description = []
    description.append("Calculating the score:")
    for idx in range(-1, len(df) - 1):
        if df.Score.iloc[idx] is not None and (df.Score.iloc[idx] > 0 or df.Score.iloc[idx] < 0):
            if df.reason.iloc[idx] == "CONSTANT":
                description.append(f"Start at {df.Score.iloc[idx]} points.")
            else:
                description.append(f"Add {df.Score.iloc[idx]} points {df.reason.iloc[idx]}.")

    df["Excel Formula"] = [
        f"=B{r}*C{r}" if r > 6 else None for r in range(2, len(df) + 2)
    ]

    risk_level = None
    if phase is None or phase == 0:
        total_score = None

        description.append("Cannot calculate a total score without knowing the phase.")
    else:
        total_score = int(min((100, np.round(df.Score.sum()))))

        description.append(f"Total score is {total_score}.")
        description.append(
            get_human_readable_risk_levels(high_risk_threshold,low_risk_threshold))

        risk_level, _ = get_risk_level_and_traffic_light(total_score,high_risk_threshold,low_risk_threshold)
        description.append(f"Risk is therefore {risk_level}")

    df = df.append(
        pd.DataFrame(
            {"Parameter": [
                f"Total score ({low_risk_threshold}-100=low risk, 0-{high_risk_threshold}=high risk)"],
                "Score": [total_score], "Excel Formula": [f"=MIN(100,SUM(D7:D{len(df) + 1}))"]})
    )

    if risk_level:
        df = df.append(
            pd.DataFrame(
                {"Parameter": [
                    f"Risk level"],
                    "Score": [risk_level.upper()], "Excel Formula": [get_excel_formula_for_risk_level(f"D{len(df) + 1}",high_risk_threshold,low_risk_threshold)]})
        )

    errors = []
    if phase is None or phase == 0 or phase == -1:
        errors.append("PHASE")
    if sap == -1:
        errors.append("SAP")
    if effect_estimate == -1:
        errors.append("EFFECT ESTIMATE")
    if simulation == -1:
        errors.append("SIMULATION")
    if len(errors) > 0:
        total_score = "ERROR FINDING " + "/".join(errors) + " IN FILE"

    end_time = time.time()

    description.append(f"Score calculated in {end_time-start_time:.2f} seconds.")
    return total_score, df.drop(columns=["reason"]), description
