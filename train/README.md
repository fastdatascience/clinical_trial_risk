# Training scripts

This folder contains Python scripts and Jupyter notebooks for training the various models that make up the Clinical Trial Risk Tool.

The scripts that are operating on the manual datasets are in the format `train_*.py`.

The code that is working on the much larger ClinicalTrials.gov dataset is generally in Jupyter format.

There is also an associated folder `experiments_ctgov` which contains the models which were developed and trained on the ClinicalTrials.gov dataset. Most models did not make it into the final design, but the information about the experiments conducted is included here for completeness and reproducibility.

You can check the performance of these candidate models in the leaderboard in [experiments_ctgov/README.md](experiments_ctgov/README.md).
