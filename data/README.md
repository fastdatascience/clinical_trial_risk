# Data for developing the Clinical Trial Risk Tool

You only need data in this folder if you are planning on training any further models.

There are two datasets:

## 1. Manual dataset

This is a set of between 100 and 300 protocols which have been read through individually and annotated with key parameters such as the sample size. The number annotated per parameter varied between 100 and 300.

## 2. ClinicalTrials.gov dataset

This is a much larger dataset of 11925 protocols downloaded from ClinicalTrials.gov. These came together with NCT ID, phase, pathology, SAP, number of arms and number of subjects, but the data was voluntarily provided by the researchers and in many cases is out of date or inaccurate.

By combining the two datasets, it has been possible to obtain some of the advantages of a large dataset and some of the advantages of a smaller, more accurate dataset.

# Downloading the manual dataset

1. Start Apache Tika (https://tika.apache.org/) running for PDF extraction.

2. Go into `raw_protocols`.

3. Run `download_raw_protocols.sh`

4. Run `preprocess.py`.

# Downloading the ClinicalTrials.gov dataset

1. Start Apache Tika (https://tika.apache.org/) running for PDF extraction.

2. Go into `ctgov/raw_protocols`.

3. Run `download_raw_protocols.sh`

4. Run `02_parse_all_PDFs_to_json.ipynb`.

# Working further with the ClinicalTrials.gov dataset using the Postgres Database Dump

Follow the instructions in `ctgov/README.md` to download/extract the database dump.
