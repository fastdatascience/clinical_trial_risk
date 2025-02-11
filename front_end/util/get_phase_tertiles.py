def get_tertile(tertile_provider, pathology: str, phase: float) -> list:
    """
    Return the lower and upper tertile of the sample size for a given pathology and phase.

    :param pathology: "HIV" or "TB"
    :param phase: the phase of the trial (1.0, 1.5, 2.0, 2.5, 3.0).
    :return:
    """
    f = tertile_provider.df.Phase == phase
    row = tertile_provider.df[
        f
    ][[pathology + " lower tertile", pathology + " upper tertile"]]
    return list(row.mean())
