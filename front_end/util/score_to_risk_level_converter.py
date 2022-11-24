'''
There are several points in the code where the risk level is calculated from the total score.

They are all unified in this file.

The logic of the Excel formula should be equivalent to the logic of the Python formula.
'''


def get_excel_formula_for_risk_level(total_score_cell_id:int,high_risk_threshold:int,low_risk_threshold:int):
    return f"=IF({total_score_cell_id}<{high_risk_threshold},\"HIGH\",IF({total_score_cell_id}<{low_risk_threshold},\"MEDIUM\",\"LOW\"))"


def get_risk_level_and_traffic_light(total_score:float,high_risk_threshold:int,low_risk_threshold:int):
    if total_score < high_risk_threshold:
        return "HIGH", "#ff0000"
    elif total_score < low_risk_threshold:
        return "MEDIUM", "#FFA500"
    else:
        return "LOW", "#00aa00"


def get_human_readable_risk_levels(high_risk_threshold:int,low_risk_threshold:int):
    return f"Scores between {low_risk_threshold} and 100 are low risk, scores between {high_risk_threshold} and {low_risk_threshold} are medium risk, and scores between 0 and {high_risk_threshold} are high risk."
