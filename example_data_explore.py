import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


if __name__ == "__main__":
    # load the tables that will be used in this tutorial
    df_morphometry = pd.read_csv('/mnt/data/owncloud/TP_meca_retraite/morpho_stats_abide_full_sulci_left_all_features_Z_10.csv', sep=',')
    df_phenotype = pd.read_excel('/mnt/data/owncloud/TP_meca_retraite/Phenotypic_V1_0b_traitements_visual_check_GA.xls', 'Feuille1')

    # view what is in the loaded table
    print(df_phenotype.head())
    print(df_phenotype.describe())
    # I recommand to systematically check the type of each column.
    # here, categorical columns where not loaded appropriately and have the type 'object'
    # seaborn infers the type of columns but be careful when computing stats!
    print(df_phenotype.dtypes)
    # It is preferable to correct columns type when necessary
    df_phenotype['SITE_ID'] = df_phenotype['SITE_ID'].astype('category')
    df_phenotype['DX_GROUP'] = df_phenotype['DX_GROUP'].astype('category')
    print(df_phenotype.dtypes)
    # then you can use specific functions:
    print('here are the levels for the SITE_ID factor:', df_phenotype['SITE_ID'].unique())
    print(df_phenotype['SITE_ID'].describe())
    print(df_phenotype['SITE_ID'].value_counts())


    # extract a subset
    df_phenotype_sub = df_phenotype[(df_phenotype.SITE_ID=='NYU') | (df_phenotype.SITE_ID=='KKI') | (df_phenotype.SITE_ID=='PITT')].copy()

    # usefull plots with categorical factors
    f,axes = plt.subplots(1,3)
    sns.catplot(x='SITE_ID', y='AGE_AT_SCAN', kind="swarm", data=df_phenotype_sub, ax=axes[0])
    sns.catplot(x='SITE_ID', y='AGE_AT_SCAN', kind="box", hue='DX_GROUP', data=df_phenotype_sub, ax=axes[1])
    sns.catplot(x='DX_GROUP', y='AGE_AT_SCAN', hue='SEX', kind="violin", inner="stick", split=True,
            palette="pastel", data=df_phenotype_sub, ax=axes[2])
    plt.show()

    # add a new categorical column that splits the data between young and old
    df_phenotype_sub['young_or_old'] = df_phenotype_sub['AGE_AT_SCAN'] < df_phenotype_sub['AGE_AT_SCAN'].median()
    sns.catplot(x='SITE_ID', y='AGE_AT_SCAN', hue='DX_GROUP', col='young_or_old', aspect=.6,
            kind="swarm", data=df_phenotype_sub)
    plt.show()


    # now lets have a look at the other tables
    print(df_morphometry.head())
    print(df_morphometry.columns)
    print(df_morphometry.describe())
    print(df_morphometry.dtypes)


    # we can concatenate the tables while correspondence is given by a specified column
    df_pheno_morpho = df_phenotype.merge(df_morphometry, left_on='SUB_ID', right_on='subjects', how='inner')

    # some nice bivariate plots
    sns.jointplot(x="AGE_AT_SCAN", y="dMean_S.C.left", data=df_pheno_morpho)
    with sns.axes_style("white"):
        sns.jointplot(x="AGE_AT_SCAN", y="dMean_S.C.left", kind="hex", color="k", data=df_pheno_morpho)
    sns.relplot(x="AGE_AT_SCAN", y="dMean_S.C.left", hue='SITE_ID', data=df_pheno_morpho)
    plt.show()

    # linear models
    sns.lmplot(x="dMean_S.C.left", y="surface_S.C.left", data=df_pheno_morpho)
    sns.lmplot(x="dMean_S.C.left", y="surface_S.C.left", order=2, data=df_pheno_morpho)
    sns.lmplot(x="dMean_S.C.left", y="surface_S.C.left", lowess=True, data=df_pheno_morpho)
    plt.show()

    sns.lmplot(x="AGE_AT_SCAN", y="dMean_S.C.left", hue='DX_GROUP', data=df_pheno_morpho)
    plt.show()

    df_pheno_morpho_sub = df_pheno_morpho[['SITE_ID','surface_S.C.left','dMean_S.C.left','hullJ_S.C.left','Mdepth_S.C.left']].copy()
    sns.pairplot(df_pheno_morpho_sub)
    g = sns.pairplot(df_pheno_morpho_sub, hue="SITE_ID", palette="Set2", diag_kind="kde", height=2.5)

    df_pheno_morpho_sub = df_pheno_morpho[['DX_GROUP','surface_S.C.left','dMean_S.C.left','hullJ_S.C.left','Mdepth_S.C.left']].copy()
    df_pheno_morpho_sub['DX_GROUP'] = df_pheno_morpho_sub['DX_GROUP'].astype('category')
    g = sns.pairplot(df_pheno_morpho_sub, hue="DX_GROUP", palette="Set2", diag_kind="kde", height=2.5)
    plt.show()


    # IMPORTANT WARNING ######################################################
    # '.' should be avoided in columns name since in python '.' is used to access class fields
    # use the following command to replace '.' by '_' in columns name
    df_pheno_morpho.rename(columns=lambda x: x.replace('.', '_'), inplace=True)