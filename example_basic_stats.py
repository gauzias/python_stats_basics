import matplotlib
matplotlib.use('TkAgg')
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np
if __name__ == "__main__":
    #http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
    # load the tables that will be used in this tutorial
    df_morphometry = pd.read_csv('data/ABIDE_morpho_stats_sulci_left_Z_10.csv', sep=',')
    df_phenotype = pd.read_excel('data/ABIDE_phenotypic_V1_0b.xls', 'Feuille1')

    # we can concatenate the tables while correspondence is given by a specified column
    df_pheno_morpho = df_phenotype.merge(df_morphometry, left_on='SUB_ID', right_on='subjects', how='inner')

    # IMPORTANT WARNING ######################################################
    # '.' should be avoided in columns name since in python '.' is used to access class fields
    # use the following command to replace '.' by '_' in columns name
    df_pheno_morpho.rename(columns=lambda x: x.replace('.', '_'), inplace=True)

    sns.catplot(x='SITE_ID', y='surface_S_C_left', kind="violin", inner="stick",
            palette="pastel", data=df_pheno_morpho)

    sns.lmplot(x="AGE_AT_SCAN", y="surface_S_C_left", data=df_pheno_morpho)

    sns.lmplot(x="AGE_AT_SCAN", y="surface_S_C_left", hue='SITE_ID', data=df_pheno_morpho)
    plt.show()

    ols_model = smf.ols('surface_S_C_left ~ C(SITE_ID)', data=df_pheno_morpho).fit()
    print(ols_model.summary())

    ols_model = smf.ols('surface_S_C_left ~ AGE_AT_SCAN', data=df_pheno_morpho).fit()
    print(ols_model.summary())
    y2 = df_pheno_morpho['surface_S_C_left']
    x1 = df_pheno_morpho['AGE_AT_SCAN']
    X = sm.add_constant(x1)
    resrlm = sm.RLM(y2, X).fit()

    prstd, iv_l, iv_u = wls_prediction_std(ols_model)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x1, y2, 'o', label="data")
    ax.plot(x1, ols_model.fittedvalues, 'r-', label="OLS")
    ax.plot(x1, iv_u, 'r--')
    ax.plot(x1, iv_l, 'r--')
    ax.plot(x1, resrlm.fittedvalues, 'g.-', label="RLM")
    legend = ax.legend(loc="best")

    infl = ols_model.get_influence()
    student = infl.summary_frame()['student_resid']
    print(student.loc[np.abs(student) > 2])

    sidak = ols_model.outlier_test('sidak')
    sidak.sort_values('unadj_p', inplace=True)
    print(sidak)


    results = smf.ols('surface_S_C_left ~ AGE_AT_SCAN + C(SITE_ID)', data=df_pheno_morpho).fit()
    print(results.summary())

    results = smf.ols('surface_S_C_left ~  + C(SITE_ID)+ AGE_AT_SCAN * C(SITE_ID)', data=df_pheno_morpho).fit()
    print(results.summary())

    # https://www.statsmodels.org/stable/index.html
    # https://www.statsmodels.org/stable/examples/notebooks/generated/interactions_anova.html