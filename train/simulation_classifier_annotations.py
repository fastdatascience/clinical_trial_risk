#  Annotations are page number (zero-indexed)
annotations = {
    'TB-TB-2018-DAR-PIA.pdf': [],
    'TB-TB-2018-M-P-A.pdf': [],
    'HIV-HIV-2016-Ayles-A Study of the HIV prevention Trials Network.pdf': [80],
    'TB-TB-2019-T-A-b.pdf': [],
    'HIV-HIV-2016-Laufer-Trimethoprim-Sulfamethoxazole-Chloroquine.pdf': [],
    'NTD-Filariasis-2019-Weil-DOLF IDA Papua New Guinea.pdf': [],
    'HIV-HIV-2017-Abrams-IMPAACT P1101.pdf': [],
    'NUT-A-2019-M-T.pdf': [],
    'HIV-HIV-2018-IMPAACT-P1112.pdf': [],  # Uses simulation/Monte Carlo for PK not for sample size
    'EDD-rotavirus-2017-Isanaka-ROSE.pdf': [],
    'NTD-Y-2019-M-Y.pdf': [],
    'NTD-Chagas-2015-Morillo-BENEFIT.pdf': [],
    'MAT-V-2018-R-M.pdf': [],
    # page 31 (30 zero-indexed) has a graph of detectable z-score vs sample size but this does not look like it is simulation-derived.
    'HIV-HIV-2018-GSK1265744.pdf': [],  # Uses simulation for dose but not sample size
    "POL-P-2018-O-A-L.pdf": [],
    'MAT-Hypothermia-2018-Hansen-Electricity-free Infant Warmer.pdf': [],
    # light   bulbs to simulate an infant’s metabolism
    'EDD-T-2019-S-T.pdf': [],
    'NTD-Hookwork-2018-Keiser-Efficacy and safety of a single.pdf': [],
    'TB-TB-2018-Piazza-A Phase 1b, Randomized, Double-blind.pdf': [],
    'MAL-Malaria-2019-Llanos-Cuentas-Tafenoquine vs. Primaquine to.pdf': [],
    'HIV-HIV-2017-Corey-HVTN703-HPTN081.pdf': [45, 47, 51],
    # These power calculations are conducted using the open source R package seqDesign developed by the protocol statisticians [117], which computes power based on simulating many thousands of efficacy trials, applying  the sequential monitoring procedures to each trial, and computing power as the  fraction of the trials where the 1-sided 0.025-level Wald test rejects the null  hypothesis. The calculations are based on a large number of simulated efficacy  trials.
    'MAL-M-2019-A-S.pdf': [63],
    # least 80% power using 10,000 simulations with the data from Sumba
    'NTD-E-2017-A-A.pdf': [],
    'HIV-HIV-2017-Bekker-HVTN100.pdf': [52, 53],  # omputer simulations (details provided below)
    'MAL-Malaria-2019-Tinto-Extension to study MALARIA-055 PRI.pdf': [],
    'NUT-NUT-2019-A-I.pdf': [],
    # 'HIV-HIV-2018-M-A.pdf': [], # image based PDF - exclude from dataset
    'HIV-HIV-2017-L-I.pdf': [],
    'TB-TB-2019-M-A.pdf': [],
    'VAC-G-2016-VAC-G-2016-V.pdf': [],
    'HIV-HIV-2016-N-S.pdf': [],  # Uses simulation but not for sample size
    'VAC-Shigellosis-2019-Raqib-Shigella WRSS1 Vaccine trial in Bangladesh.pdf': [],
    'HIV-HIV-2017-Evolocumab-AMG145.pdf': [],
    'TB-TB-2015-D-A.pdf': [],  # Using the sample size calculation formula
    'POL-P-2016-G-T.pdf': [],
    'TB-TB-2019-Suliman-A Phase I_IIa Double-Blind.pdf': [],
    # 'MAT-Anemia-2015-Etheredge-Prenatal Iron Supplements Safety and.pdf': [],
    # 'HIV-HIV-2016-R-E.pdf': [],
    'HIV-HIV-2019-H-S.pdf': [57, 58, 63, 136, 138, 144, 169, 170, 172, 175, 176, 180, 181, 182, 183, 184, 186,
                                       189, 192, 193, 194, 199, 206, 209, 212, 213, 215, 238],
    # a lot of this seems to deal with simulation for sample size
    # 'NTD-Filariasis-2019-Weil-DOLF IDA Haiti.pdf': [],
    # 'MAT-MAT-2018-G-F.pdf': [],
    # 'EDD-R-2017-G-V.pdf': [],
    'MAT-HIV-2016-Fowler-PROMISE.pdf': [22, 118, 119, 351, 539, 776],  # Not sure - need to revisit
    'TB-TB-2018-H-A.pdf': [54, 137],
    'HIV-HIV-2016-HVTN114.pdf': [33],
    'HIV-HIV-2018-Long-acting Cabotegravir Plus Long-acting Rilpivirine.pdf': [],  # Not sure
    'NUT-S-2018-R-S.pdf': [95],  # Unsure - see p 95 (zero indexed)
    # 'VAC-Tdap-2019-Sancovski-A Post-marketing, Observational, Retrospective.pdf': [],
    'NTD-t-2017-M-E.pdf': [],
    # 'HIV-HIV-2018-Lockman-Dolutegravir-Containing versus Efavirenz-Containing Antiretroviral Therapy.pdf': [],
    # 'MAL-M-2018-D-C.pdf': [],
    # 'NTD-D-2017-S-P.pdf': [],
    'HIV-HIV-2016-Garrett-HVTN108.pdf': [52],
    # These simulations are based on 10,000 simulated datasets and use an exchangeable correlation structure between responses with pairwise     correlations of 0.3
    # 'HIV-HIV-2018-Labhardt-CASCADE.pdf': [],
    # 'TB-TB-2019-Dawson-A Phase 2 Open‐Label Partially .pdf': [],
    # 'MAL-MAL-2018-Dorsey-PROMOTE Birth Cohort 1.pdf': [],
    # 'PNE-P-2019-C-P.pdf': [],
    # 'HIV-HIV-2019-Venter-ADVANCE.pdf': [],
    # 'HIV-HIV-2019-K-C.pdf': [],
    'PNE-Pneumonia-2019-Alexander-LEAP2.pdf': [],
    # Monte Carlo simulation were utilized - but I don't think for sample size.
    # 'MAT-Cervical Disease-2019-Greene-LEEP.pdf': [],
    'TB-TB-2018-Churchyard-Bedaquiline-Delamanid.pdf': [73],
    # 'MAL-Malaria-2019-Foy-RIMDAMAL.pdf': [],
    # 'TB-TB-2019-N-S.pdf': [],
    # 'NUT-Stunting-2019-Humphrey-SHINE.pdf': [],
    'PNE-Pneumonia-2018-Keenan-Mortality Reduction After Oral Azithromycin.pdf': [],
    # Use Monte Carlo but only for analysis, not for sample size planning
    'EDD-Cholera-2016-Qadri-Shanchol.pdf': [],  # Kolmogorov used but not for sample size - just for Bayesian p-values.
    # 'TB-TB-2020-John-Stewart-iTIPS.pdf': [],
    # 'VAC-Ecoli-2019-Qadri-ETVAX.pdf': [],
    'HIV-HIV-2018-Havlir-SEARCH.pdf': [65, 66, 73],
    # We will also using Monte Carlo multi-variable simulations to estimate
    # 'HIV-HIV-2016-Kurth-High-yield HIV testing, facilitated linkage to care.pdf': []
    'HIV-HIV-2017-Corey-HVTN704-HPTN085.pdf': [42, 43, 48],
    # 'HIV-HIV-2016_Tebas-A5354.pdf': [],
    # 'HIV-HIV-2016-Weinberg-NICHP1091.pdf': [],
    # 'MAL-Malaria-2016-Valencia-Phase 1 and 2a Clinical.pdf': [],
    # 'NTD-Filariasis-2019-Weil-DOLF IDA Indonesia.pdf': [],
    # 'VAC-M-2015-L-W.pdf': []
}
