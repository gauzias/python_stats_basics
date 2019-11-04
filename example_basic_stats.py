import matplotlib
matplotlib.use('TkAgg')
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    #http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
    # load the tables that will be used in this tutorial
    df_morphometry = pd.read_csv('/mnt/data/owncloud/TP_meca_retraite/carry_test/morpho_stats_abide_full_sulci_left_all_features_Z_10.csv', sep=',')
    df_phenotype = pd.read_excel('/mnt/data/owncloud/TP_meca_retraite/carry_test/Phenotypic_V1_0b_traitements_visual_check_GA.xls', 'Feuille1')

    # we can concatenate the tables while correspondence is given by a specified column
    df_pheno_morpho = df_phenotype.merge(df_morphometry, left_on='SUB_ID', right_on='subjects', how='inner')

    # replace '.' by '_' in columns name
    df_pheno_morpho.rename(columns=lambda x: x.replace('.', '_'), inplace=True)

    sns.catplot(x='SITE_ID', y='surface_S_C_left', kind="violin", inner="stick",
            palette="pastel", data=df_pheno_morpho)

    sns.lmplot(x="AGE_AT_SCAN", y="surface_S_C_left", data=df_pheno_morpho)

    sns.lmplot(x="AGE_AT_SCAN", y="surface_S_C_left", hue='SITE_ID', data=df_pheno_morpho)
    plt.show()

    results = smf.ols('surface_S_C_left ~ C(SITE_ID)', data=df_pheno_morpho).fit()
    print(results.summary())

    results = smf.ols('surface_S_C_left ~ AGE_AT_SCAN', data=df_pheno_morpho).fit()
    print(results.summary())

    results = smf.ols('surface_S_C_left ~ AGE_AT_SCAN + C(SITE_ID)', data=df_pheno_morpho).fit()
    print(results.summary())

    results = smf.ols('surface_S_C_left ~ AGE_AT_SCAN * C(SITE_ID)', data=df_pheno_morpho).fit()
    print(results.summary())
