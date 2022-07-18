class DurationExtractor:

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify the primary duration of the trial in months.

        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (int) and a map to the mentions by page.
        """
        return {"prediction": 12, "pages": []}
