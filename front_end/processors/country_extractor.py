import re

from country_named_entity_recognition.country_finder import find_countries

from util.demonym_finder import find_demonyms

# List of low to medium income countries.
allowed_countries = {'AF',
                     'AL',
                     'AM',
                     'AO',
                     'AR',
                     'AS',
                     'AZ',
                     'BA',
                     'BD',
                     'BF',
                     'BG',
                     'BI',
                     'BJ',
                     'BO',
                     'BR',
                     'BT',
                     'BW',
                     'BY',
                     'BZ',
                     'CF',
                     'CI',
                     'CM',
                     'CN',
                     'CO',
                     'CR',
                     'CU',
                     'CV',
                     'DJ',
                     'DM',
                     'DO',
                     'DZ',
                     'EC',
                     'ER',
                     'ET',
                     'FJ',
                     'GA',
                     'GD',
                     'GE',
                     'GH',
                     'GN',
                     'GQ',
                     'GT',
                     'GW',
                     'GY',
                     'HN',
                     'HT',
                     'ID',
                     'IN',
                     'IQ',
                     'JM',
                     'JO',
                     'KE',
                     'KH',
                     'KI',
                     'KM',
                     'KZ',
                     'LB',
                     'LK',
                     'LR',
                     'LS',
                     'LY',
                     'MA',
                     'MD',
                     'ME',
                     'MG',
                     'MH',
                     'MK',
                     'ML',
                     'MM',
                     'MN',
                     'MR',
                     'MU',
                     'MV',
                     'MW',
                     'MX',
                     'MY',
                     'MZ',
                     'NA',
                     'NE',
                     'NG',
                     'NI',
                     'NP',
                     'PA',
                     'PE',
                     'PG',
                     'PH',
                     'PK',
                     'PY',
                     'RO',
                     'RS',
                     'RU',
                     'RW',
                     'SB',
                     'SD',
                     'SL',
                     'SN',
                     'SO',
                     'SR',
                     'SS',
                     'ST',
                     'SV',
                     'SY',
                     'SZ',
                     'TD',
                     'TG',
                     'TH',
                     'TJ',
                     'TL',
                     'TM',
                     'TN',
                     'TO',
                     'TR',
                     'TV',
                     'TZ',
                     'UA',
                     'UG',
                     'UZ',
                     'VN',
                     'VU',
                     'WS',
                     'ZA',
                     'ZM',
                     'ZW'}

international_regex = re.compile(r"(?i)\b(?:(?:glob|internation|multination)al(?:ly)?|worldwide)\b")


class CountryExtractor:

    def process(self, pages: list) -> tuple:
        """
        Identify the countries the trial takes place in.

        :param tokenised_pages: List of string content of each page.
        :return: The prediction (list of strings of Alpha-2 codes) and a map from each country code to the pages it's mentioned in.
        """
        country_to_pages = {}
        terms_to_pages = {}

        contexts = {}

        for page_no, page_text in enumerate(pages):
            countries = find_countries(page_text)

            demonyms = find_demonyms(page_text)

            countries.extend(demonyms)

            for country, match in countries:
                if True or country.alpha_2 in allowed_countries:
                    if country.alpha_2 not in country_to_pages:
                        country_to_pages[country.alpha_2] = []
                        start = match.start() - 100
                        end = match.end() + 100
                        if start < 0:
                            start = 0
                        if end > len(page_text) - 1:
                            end = len(page_text)

                        if country.flag + country.name not in contexts:
                            contexts[country.flag + country.name] = ""
                        contexts[country.flag + country.name] = (
                                contexts[country.flag + country.name] + " " + f"Page {page_no + 1}: " + re.sub(
                            r'\w+$', '',
                            re.sub(r'^\w+', '',
                                   page_text[
                                   start:end])).strip()).strip()
                    country_to_pages[country.alpha_2].append(page_no)

            international_matches = list(international_regex.finditer(page_text))
            if len(international_matches) > 0:
                for match in international_matches:
                    match_text = re.sub(r'ly$', '', match.group().lower())
                    if match_text not in terms_to_pages:
                        terms_to_pages[match_text] = []
                    terms_to_pages[match_text].append(page_no)
                    if match_text not in contexts:
                        contexts[match_text] = ""
                    start = match.start() - 20
                    end = match.end() + 20
                    if start < 0:
                        start = 0
                    if end > len(page_text) - 1:
                        end = len(page_text)
                    contexts[match_text] = (
                            contexts[match_text] + " " + f"Page {page_no + 1}: " + re.sub(
                        r'\w+$', '',
                        re.sub(r'^\w+', '',
                               page_text[
                               start:end])).strip()).strip()

        prediction = set()

        if len(country_to_pages) > 0:
            earliest_page = sorted([min(p) for p in country_to_pages.values()])[0]
            for candidate, pages in country_to_pages.items():
                # Any country mentioned on 30+ pages is an investigation country
                unique_pages = set(pages)
                if len(unique_pages) > 30:
                    prediction.add(candidate)
                # Any country mentioned on the first contentful page is also a candidate
                if 0 in unique_pages or earliest_page in unique_pages:
                    prediction.add(candidate)

            first_mentioned_countries = sorted(country_to_pages.items(), key=lambda a: min(a[1]))
            prediction.add(first_mentioned_countries[0][0])

        # country_to_pages = country_to_pages | terms_to_pages

        return {"prediction": list(prediction), "pages": country_to_pages, "context": contexts}
