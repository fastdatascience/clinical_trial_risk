import pandas as pd


class DefaultParameterProvider:
    def __init__(self, csv_file):
        """
        Load the table of parameter values from CSV to populate in the UI when it is loaded.

        :param csv_file:
        """
        self.df = pd.read_csv(csv_file, encoding="utf-8", sep="\t")
        self.DEFAULT_WEIGHTS_DATA = list([dict(d) for _, d in self.df.iterrows()])
        self.DF_PARAMETERS_COLUMNS_FOR_DASH = [{"name": i, "id": i} for i in self.df.columns]