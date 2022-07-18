class NumEndpointsExtractor:

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify the number of primary endpoints of the trial.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (int) and a map from numbers to the pages they are mentioned in.
        """
        return {"prediction": 1, "pages": []}
