import matplotlib
matplotlib.use('TkAgg')
import statsmodels.formula.api as smf
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np

if __name__ == "__main__":
    '''
    The present script is a simple tutorial on basic statistics using statsmodels.
    extensive tutorials and doc are available on the website: 
    https://www.statsmodels.org/stable/index.html
    https://www.statsmodels.org/stable/examples/notebooks/generated/interactions_anova.html
    https://www.statsmodels.org/stable/examples/

    Note on the data
    The data are phenotype and morphometry tables from the ABIDE open dataset:
    http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
    The morphometry data has been computed using great open source tools:
    -freesurfer https://surfer.nmr.mgh.harvard.edu/
    -BrainVISA http://brainvisa.info/web/index.html
    '''

    # load the tables that will be used in this tutorial
    df_morphometry = pd.read_csv('data/ABIDE_morpho_stats_sulci_left_Z_10.csv',
                                 sep=',')
    df_phenotype = pd.read_excel('data/ABIDE_phenotypic_V1_0b.xls', 'Feuille1')

    # concatenate the tables with correspondence given by a specified column
    df_pheno_morpho = df_phenotype.merge(df_morphometry, left_on='SUB_ID',
                                         right_on='subjects', how='inner')

    # IMPORTANT WARNING ######################################################
    # '.' should be avoided in columns name since in python '.' is used to
    # access class fields
    # use the following command to replace '.' by '_' in columns name
    df_pheno_morpho.rename(columns=lambda x: x.replace('.', '_'), inplace=True)

    # example 1: effect of categorical factor 'SITE_ID' on surface_S_C_left
    # let us plot the data
    sns.catplot(x='SITE_ID', y='surface_S_C_left', kind="violin", inner="stick",
                palette="pastel", data=df_pheno_morpho)
    plt.show()
    # and assess whether the effect is significant
    ols_SITE = smf.ols('surface_S_C_left ~ C(SITE_ID)', data=df_pheno_morpho).fit()
    print(ols_SITE.summary())

    # example 2: effect of continuous factor 'AGE_AT_SCAN' on surface_S_C_left
    # let us plot the data
    sns.lmplot(x="AGE_AT_SCAN", y="surface_S_C_left", data=df_pheno_morpho)
    plt.show()
    # and assess whether the effect is significant
    ols_AGE = smf.ols('surface_S_C_left ~ AGE_AT_SCAN', data=df_pheno_morpho).fit()
    print(ols_AGE.summary())

    # example 3: effect of the interaction between continuous factor 'AGE_AT_SCAN'
    # and categorical factor 'SITE_ID' on surface_S_C_left
    # let us plot the data
    sns.lmplot(x="AGE_AT_SCAN", y="surface_S_C_left", hue='SITE_ID',
               data=df_pheno_morpho)
    plt.show()
    # and assess whether the effects of each factor
    results = smf.ols('surface_S_C_left ~ AGE_AT_SCAN + C(SITE_ID)+ AGE_AT_SCAN * C(SITE_ID)', data=df_pheno_morpho).fit()
    print(results.summary())


    # comparison between OLS and RLM
    y2 = df_pheno_morpho['surface_S_C_left']
    x1 = df_pheno_morpho['AGE_AT_SCAN']
    X = sm.add_constant(x1)
    ols_model = sm.OLS(y2, X).fit()
    print(ols_model.summary())
    rlm_model = sm.RLM(y2, X).fit()
    print(rlm_model.summary())

    # nice figure with confidence intervals
    prstd, iv_l, iv_u = wls_prediction_std(ols_model)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x1, y2, 'o', label="data")
    ax.plot(x1, ols_model.fittedvalues, 'r-', label="OLS")
    ax.plot(x1, iv_u, 'r--')
    ax.plot(x1, iv_l, 'r--')
    ax.plot(x1, rlm_model.fittedvalues, 'g.-', label="RLM")
    legend = ax.legend(loc="best")
    plt.show()

    # influence analysis for outliers detection
    infl = ols_model.get_influence()
    student = infl.summary_frame()['student_resid']
    suspect_data_index = student.loc[np.abs(student) > 2]
    print(suspect_data_index)

    sidak = ols_model.outlier_test('sidak')
    sidak.sort_values('unadj_p', inplace=True)
    print(sidak)

    ax.plot(x1[suspect_data_index.index], y2[suspect_data_index.index], 'ko', label="suscpect data")
    ax.plot(x1[sidak.index[0]], y2[sidak.index[0]], 'ro', label="sidak min p")
    legend = ax.legend(loc="best")
    plt.show()