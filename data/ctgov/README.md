# read_ctgov_data

Code for downloading data from ClinicalTrials.gov

These notebooks download clinical trial protocol PDFs and associated structured data.

This was made following instructions from:
https://aact.ctti-clinicaltrials.org/install_postgres

First I downloaded the database dump on 4 Nov 2022.

Then I ran the following commands:

```
sudo -u postgres psql postgres
GRANT ALL PRIVILEGES ON DATABASE aact TO PUBLIC;
CREATE ROLE ctti;
ALTER ROLE ctti WITH LOGIN;

CREATE ROLE read_only;
ALTER ROLE read_only WITH LOGIN;

pg_restore --host "localhost" --port "5432" --username "postgres"  --no-password --role "ctti" --dbname "aact"  --verbose "postgres_data.dmp"
```

To run the Python code I had to install

```
pip install psycopg2-binary
pip install tika
```

and download the Tika JAR file and run it.

Then I was able to run the Jupyter notebooks.
