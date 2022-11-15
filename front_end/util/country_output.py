import pycountry


def pretty_print_countries(countries: list, show_flags = True) -> str:
    """
    Output the list of countries found in the document in a human readable phone optionally showing flags as Unicode characters, which may not be displayed on all platforms

    :param countries: The list of countries as 2-letter country codes, which will be used to retrieve PyCountry objects
    :param show_flags: whether flags of the country should be displayed or not
    :return: A human readable string separated by commas which can be used for natural language generation (NLG).
    """
    if len(countries) == 0:
        return "no countries of sufficient confidence"
    human_readable_prediction = ""
    for idx, country_code in enumerate(countries):
        if idx > 0:
            human_readable_prediction += ","
        human_readable_prediction += " "
        if country_code == "XX":
            human_readable_prediction += "one or more unspecified countries"
        else:
            if show_flags:
                human_readable_prediction += pycountry.countries.lookup(country_code).flag
            human_readable_prediction += pycountry.countries.lookup(
                    country_code).name

    return human_readable_prediction