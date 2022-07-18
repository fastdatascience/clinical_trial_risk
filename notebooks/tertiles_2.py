import psycopg2
import pandas as pd
from sqlalchemy import create_engine


#conn = psycopg2.connect("dbname=ct user=thomas password=mypass")

cargs = {'database': "aact"}
alchemyEngine   = create_engine('postgresql+psycopg2://thomas:mypass@127.0.0.1', pool_recycle=3600, connect_args=cargs)

dbConnection    = alchemyEngine.connect();

df_hiv_trials = pd.read_sql("""SELECT nct_id
	FROM ctgov.browse_conditions
	where mesh_term like 'Acquired Immunodeficiency Syndrome';""", dbConnection)
	
df_phase_1_trials = pd.read_sql("""SELECT nct_id
	FROM ctgov.studies
	where phase= 'Phase 1';""", dbConnection)
	
for phase in ["Phase 1", "Phase 2", "Phase 3"]:
    for condition in ['Acquired Immunodeficiency Syndrome', 'Tuberculosis']:
        df_phase_and_condition = pd.read_sql("""SELECT phase, enrollment, mesh_term
	FROM ctgov.studies as s
	LEFT JOIN ctgov.browse_conditions as c
	ON c.nct_id = s.nct_id
	WHERE phase = '""" + phase + """' AND mesh_term = '""" + condition + """'
	;""", dbConnection)
        print (f"{phase} {condition} {len(df_phase_and_condition)} rows")
        print ("\tLower tertile", df_phase_and_condition.enrollment.quantile(0.33))
        print ("\tUpper tertile", df_phase_and_condition.enrollment.quantile(0.67))


'''
Phase 1 Acquired Immunodeficiency Syndrome 257 rows
	Lower tertile 20.0
	Upper tertile 40.0
Phase 1 Tuberculosis 141 rows
	Lower tertile 23.74000000000001
	Upper tertile 40.0
Phase 2 Acquired Immunodeficiency Syndrome 200 rows
	Lower tertile 48.02000000000001
	Upper tertile 111.99000000000001
Phase 2 Tuberculosis 169 rows
	Lower tertile 60.0
	Upper tertile 150.56
Phase 3 Acquired Immunodeficiency Syndrome 174 rows
	Lower tertile 196.09000000000006
	Upper tertile 517.99
Phase 3 Tuberculosis 106 rows
	Lower tertile 315.59999999999997
	Upper tertile 909.1000000000003
'''
