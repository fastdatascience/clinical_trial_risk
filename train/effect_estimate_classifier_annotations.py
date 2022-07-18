# Annotations are page number (zero-indexed), and the actual value. Comments include the text.
annotations = {
    'TB-TB-2018-DAR-PIA.pdf': [7, 54, "50%"],  # will provide 80% power to  detect vaccine efficacy of 50%.
    'TB-TB-2018-M-P-A.pdf': [49, "50%"],
    # of TB IRIS in the placebo arm and a 50% reduction in TB IRIS in the corticosteroid arm ie to 17.5% and requiring 80% power to test for the difference in TB IRIS incidence at a two sided significance level
    'HIV-HIV-2016-Ayles-A Study of the HIV prevention Trials Network.pdf': [],
    'TB-TB-2019-T-A-b.pdf': [],
    'HIV-HIV-2016-Laufer-Trimethoprim-Sulfamethoxazole-Chloroquine.pdf': [61, "35%"],
    # For estimating the power of the study, the statistical null hypothesis is that the effect of TS prophylaxis in preventing the first occurrence of a severe event (WHO stage 3 or 4 event or death), relative to no prophylaxis, is at least a 35% reduction in the hazard rate for first occurrences over the study period.
    'NTD-Filariasis-2019-Weil-DOLF IDA Papua New Guinea.pdf': [35, "30%"],
    # 80% power for detecting an effect size of 30%
    'HIV-HIV-2017-Abrams-IMPAACT P1101.pdf': [],  # None
    'NUT-A-2019-M-T.pdf': [126, "5%"],
    # requires 3162 SA cases, assuming 50% are complicated and 50% uncomplicated, to detect a 5% absolute reduction from average control mortality of 25%
    # 'HIV-HIV-2018-IMPAACT-P1112.pdf': [], # tricky - has a table of adverse event rates, could be effect size?
    'EDD-rotavirus-2017-Isanaka-ROSE.pdf': [42, 110, "20%"],
    # we will assess immunogenicity in 420 children to have 90%  # power to detect a 20% difference in the proportion of children that sero-convert
    'NTD-Y-2019-M-Y.pdf': [],
    # talks about odds ratio but does not provide figure
    'NTD-Chagas-2015-Morillo-BENEFIT.pdf': [17, 45, 63, "26.5%", "20%"],
    # all drop-out  rate of 20%, to detect a 20% risk reduction the tria
    'MAT-V-2018-R-M.pdf': [11, 30, 78, 97, "0.44"],
    # At 1 year of age, infants in the vitamin D group had LAZ that were, on average, 0.44 z-score units
    'HIV-HIV-2018-GSK1265744.pdf': [],  # doesn't seem to have effect estimate
    "POL-P-2018-O-A-L.pdf": [37, "0.90", "0.33"],
    # The power to declare superiority of GMTs is 90% for types 1 and 3 if the true difference in type-specific GMTs is 0.33 logs or greater.?
    'MAT-Hypothermia-2018-Hansen-Electricity-free Infant Warmer.pdf': [10, "80%"],
    # We aim to have 90% power to detect if the percent achieving the minimum temperature is 80% or less.
    'EDD-T-2019-S-T.pdf': [32, 75, "five-fold"],
    # this study has 80% power to detect a five-fold increase in this SAE.
    'NTD-Hookwork-2018-Keiser-Efficacy and safety of a single.pdf': [26, "20%", "40%"],
    # Based on the published literature with special attention to studies conducted on Pemba Island, we assume that the cure rate of single dose mebendazole against hookworm infections is 20% compared to 40% in the triple treatment regimen - effect size would be 40%-20% = 20% I think.
    'TB-TB-2018-Piazza-A Phase 1b, Randomized, Double-blind.pdf': [38, "3 percent"],
    # at least one adverse event which occurs at a rate of 3 percent
    'MAL-Malaria-2019-Llanos-Cuentas-Tafenoquine vs. Primaquine to.pdf': [394, 397, 401, "1.45"],
    # efficacies (based on an odds ratio of 1.45).  	9.2. How the
    'HIV-HIV-2017-Corey-HVTN703-HPTN081.pdf': [16, "60%"],
    # 60%:   The trial sample size is designed to provide 90% power to detect a prevention efficacy (PE) of 60%
    'MAL-M-2019-A-S.pdf': [9, 62, 63, "50%"],
    # The trial is designed to detect 50% reduction in any malaria infection at delivery
    'NTD-E-2017-A-A.pdf': [44, "60%", "40%"],
    # 60% for patients treated with the low dose regimen. We hypothesise that the high dose regimen will reduce this rate to 40%
    'HIV-HIV-2017-Bekker-HVTN100.pdf': [51, "4%"],
    # As a reference, in HVTN vaccine trials from December 2000 through September 2010, about 4% of participants who received placebos experienced an SAE.
    'MAL-Malaria-2019-Tinto-Extension to study MALARIA-055 PRI.pdf': [64, 94, "2-fold"],
    # sample size will allow to detect a 2-fold increase in  the
    'NUT-NUT-2019-A-I.pdf': [27, 42, 61, 68, "0.4"],
    # To observe a difference of 0.4 in mean LAZ
    # 'HIV-HIV-2018-M-A.pdf': [], # File is image based???
    'HIV-HIV-2017-L-I.pdf': [28, "10%"],
    # a non-inferiority margin of 10% - this is a non-inferiority trial so perhaps not applicable?
    'TB-TB-2019-M-A.pdf': [10, 96, 97, "0.70"],  # ould  have 90% power to detect a prevalence ratio of 0.70
    'VAC-G-2016-VAC-G-2016-V.pdf': [4, 36, "8%", "6.5%"],
    'HIV-HIV-2016-N-S.pdf': [18, 63, "4%"],  # power to detect a difference of 4% or larger
    'VAC-Shigellosis-2019-Raqib-Shigella WRSS1 Vaccine trial in Bangladesh.pdf': [50, "10%"],
    # will allow for the recognition of unacceptable toxicity rates (SAEs) occurring at a frequency of 10% or higher.
    'HIV-HIV-2017-Evolocumab-AMG145.pdf': [53, "61.98%"],
    # the treatment effect of evolocumab 420 mg QM compared to placebo and the corresponding 95% confidence interval at week 12 was -61.98%
    'TB-TB-2015-D-A.pdf': [88, "40%"],
    # We hope to show a 40% reduction in morbidity at M30 among patients
    'POL-P-2016-G-T.pdf': [10, 47, "60%"],
    # an average 60% seroconversion after administration of mOPV3 - I am not too sure about this
    'TB-TB-2019-Suliman-A Phase I_IIa Double-Blind.pdf': [46, "12.5%", "10.0%"],
    # g at least 1 specified event which occurs at a rate of 12.5% and 10.0%
    'MAT-Anemia-2015-Etheredge-Prenatal Iron Supplements Safety and.pdf': [28, 29, "35", "30%", "40%", "8.5", "5%",
                                                                           "10%", "2%", "5%"],
    # ave  good power to detect an effect size of 8.5%, and excellent power
    # 'HIV-HIV-2016-R-E.pdf': [],
    'HIV-HIV-2019-H-S.pdf': [183, "35%"],
    # we assumed there was a constant relative effect size of 35%. GRAPH: Effect size (in percent reduction) that we are powered to detect at 80%
    # 'NTD-Filariasis-2019-Weil-DOLF IDA Haiti.pdf': [],
    'MAT-MAT-2018-G-F.pdf': [25, 26, "20%", "20%", "25%", "15%"],
    # de 79% to 89% power to detect as little as a 20%   reducti
    'EDD-R-2017-G-V.pdf': [25, 85, "≥30", "15", "20"],
    # gned to provide ≥99% power to detect ≥30 percentage point differen
    'MAT-HIV-2016-Fowler-PROMISE.pdf': [181, 182, 183, 351, 353, 588, 591, 777, 779, "4%"],  # detect a difference of 4%
    'TB-TB-2018-H-A.pdf': [54, 55, 137, 138, "50%"],
    # the trial is designed to distinguish a QFT-GIT conversion rate reduction of 50% compared to placebo for each vaccine
    'HIV-HIV-2016-HVTN114.pdf': [],
    # p33 (32 zero-indexed) has a list of powers for different effect sizes... but no *effect estimate*!!!!
    # 'HIV-HIV-2018-Long-acting Cabotegravir Plus Long-acting Rilpivirine.pdf': [],
    'NUT-S-2018-R-S.pdf': [43, 98, '0.325'],
    # The estimated effect size is therefore set to 0.325 HAZ
    'VAC-Tdap-2019-Sancovski-A Post-marketing, Observational, Retrospective.pdf': [],
    # 'NTD-t-2017-M-E.pdf': [],
    'HIV-HIV-2018-Lockman-Dolutegravir-Containing versus Efavirenz-Containing Antiretroviral Therapy.pdf': [99, 102,
                                                                                                            "10%",
                                                                                                            "one half a standard deviation"],
    # 86% power to detect a 10%  difference
    'MAL-M-2018-D-C.pdf': [55, "50%"],
    # size of 80 episodes in each trimester will have 82% power to detect difference between trimesters if there is a 50% risk of placental malaria or any other adverse outcome in 1 trimester compared to a 25% risk in the
    'NTD-D-2017-S-P.pdf': [],  # could be under censored data on page 37
    'HIV-HIV-2016-Garrett-HVTN108.pdf': [51, 52, 53, "30%", "40%"],
    # 'HIV-HIV-2018-Labhardt-CASCADE.pdf': [],
    'TB-TB-2019-Dawson-A Phase 2 Open‐Label Partially .pdf': [89, "0.00524"],
    # the study will have 90% statistical power to detect  a mean group difference in BA TTP (0‐56) of 0.00524
    'MAL-MAL-2018-Dorsey-PROMOTE Birth Cohort 1.pdf': [41, "33%", "41-70%", "18-23%", "22-28%", "16-21%"],
    # we will be powered to detect a 33% relative difference in
    # 'PNE-P-2019-C-P.pdf': [],
    # 'HIV-HIV-2019-Venter-ADVANCE.pdf': [],
    'HIV-HIV-2019-K-C.pdf': [24, "30%"],  # Prevention effectiveness... 30% reduction in HIV-1 incidence
    # 'PNE-Pneumonia-2019-Alexander-LEAP2.pdf': [],
    # 'MAT-Cervical Disease-2019-Greene-LEEP.pdf': [],
    # 'TB-TB-2018-Churchyard-Bedaquiline-Delamanid.pdf': [],
    'MAL-Malaria-2019-Foy-RIMDAMAL.pdf': [26, "40%"],  # enrolled per cluster to detect a conservative 40% reduction
    # 'TB-TB-2019-N-S.pdf': [],
    'NUT-Stunting-2019-Humphrey-SHINE.pdf': [25, "25%", "0.25 SD"],
    # would provide >80% power to detect a shift of 0.25 SD or greater / we will be able to detect effect sizes on the order of 25% of one standard deviation
    'PNE-Pneumonia-2018-Keenan-Mortality Reduction After Oral Azithromycin.pdf': [160, 161, 163, 164, 176, 177, 178,
                                                                                  "24%", "35%", "25%", "0.15", "15%",
                                                                                  "0.10", "14%"],
    # We anticipate more than 80% power to # detect an effect size of approximately 0.10, or ten per-cent.
    # 'EDD-Cholera-2016-Qadri-Shanchol.pdf': [],
    # 'TB-TB-2020-John-Stewart-iTIPS.pdf': [],
    # 'VAC-Ecoli-2019-Qadri-ETVAX.pdf': [],
    'HIV-HIV-2018-Havlir-SEARCH.pdf': [67, "40%"],  # least 80% power to detect a 40% reduction in 3 year
    'HIV-HIV-2016-Kurth-High-yield HIV testing, facilitated linkage to care.pdf': [62, 63, "1.67", "2.32", "2.2"],
    # uses odds ratio 2.32, hazard ratio 2.2.  power is 80% to detect an odds ratio of 1.67. power is 80% to detect an odds ratio of 1.85.
    'HIV-HIV-2017-Corey-HVTN704-HPTN085.pdf': [15, 70, "60%"],
    # Overall PE = 60%. Three subgroups of patients with different PEs: PE=15%, 60%, 85%. Effect Size 4
    # 'HIV-HIV-2016_Tebas-A5354.pdf': [],
    'HIV-HIV-2016-Weinberg-NICHP1091.pdf': [55, "20 percent"],
    # ill  have 80 percent power to detect ≥ 20 percent difference in pr
    # 'MAL-Malaria-2016-Valencia-Phase 1 and 2a Clinical.pdf': [],
    'NTD-Filariasis-2019-Weil-DOLF IDA Indonesia.pdf': [31, "30%"],  # 80% power for detecting an effect size of 30%)
    # 'VAC-M-2015-L-W.pdf': []
}
