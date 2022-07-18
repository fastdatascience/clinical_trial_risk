from util.constants import SCORE_UPPER_TERTILE, SCORE_LOWER_TERTILE

'''
There are several points in the code where the risk level is calculated from the total score.

They are all unified in this file.

The logic of the Excel formula should be equivalent to the logic of the Python formula.
'''


def get_excel_formula_for_risk_level(total_score_cell_id):
    return f"=IF({total_score_cell_id}<{SCORE_LOWER_TERTILE},\"HIGH\",IF({total_score_cell_id}<{SCORE_UPPER_TERTILE},\"MEDIUM\",\"LOW\"))"


def get_risk_level_and_traffic_light(total_score):
    if total_score < SCORE_LOWER_TERTILE:
        return "HIGH", "#ff0000"
    elif total_score < SCORE_UPPER_TERTILE:
        return "MEDIUM", "#FFA500"
    else:
        return "LOW", "#00aa00"


def get_human_readable_risk_levels():
    return f"Scores between {SCORE_UPPER_TERTILE} and 100 are low risk, scores between {SCORE_LOWER_TERTILE} and {SCORE_UPPER_TERTILE} are medium risk, and scores between 0 and {SCORE_LOWER_TERTILE} are high risk."
