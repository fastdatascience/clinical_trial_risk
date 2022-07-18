class NumSitesExtractor:

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify the number of sites (centres) that the trial takes place in.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (int) and a map from numbers to the pages it's mentioned in.
        """
        return {"prediction": 10, "pages": []}
