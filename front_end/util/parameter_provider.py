import pandas as pd


class DefaultSampleSizeTertileProvider:
    def __init__(self, csv_file):
        """
        Load the table of sample size tertiles from CSV to populate in the UI when it is loaded.

        :param csv_file:
        """
        self.df = pd.read_csv(csv_file, encoding="utf-8", sep="\t")
        self.DF_TERTILES_DATA_FOR_DASH = list([dict(d) for _, d in self.df.iterrows()])
        self.DF_TERTILES_COLUMNS_FOR_DASH = [{"name": i, "id": i} for i in self.df.columns]