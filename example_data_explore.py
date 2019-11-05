import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

if __name__ == "__main__":
    '''
    The present script is a simple tutorial on data exploration using seaborn and pandas.
    extensive tutorials for these packages are available on their respective website: 
    seaborn
    https://seaborn.pydata.org/tutorial.html
    pandas
    https://pandas.pydata.org/pandas-docs/stable/

    Note on the data
    The data are phenotype and morphometry tables from the ABIDE open dataset:
    http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
    The morphometry data has been computed using great open source tools:
    -freesurfer https://surfer.nmr.mgh.harvard.edu/
    -BrainVISA http://brainvisa.info/web/index.html
    '''

    # load the tables that will be used in this tutorial
    # a simple csv file for the morphometry data
    df_morphometry = pd.read_csv('data/ABIDE_FS_global_measures.csv', sep=',')
    # an excell file for the phenotype data
    df_phenotype = pd.read_excel('data/ABIDE_phenotypic_V1_0b.xls', 'Feuille1')

    # view what is in the loaded table
    print(df_phenotype.head())
    print(df_phenotype.describe())
    # It is highly recommanded to systematically check the type of each column.
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
    # NOTE THE BUG in pandas deep copy!
    df_phenotype_sub = df_phenotype[(df_phenotype.SITE_ID=='NYU') | (df_phenotype.SITE_ID=='KKI') | (df_phenotype.SITE_ID=='PITT')].copy()
    df_phenotype_sub['SITE_ID'] = df_phenotype_sub['SITE_ID'].astype('category')
    print('here are the levels for the SITE_ID factor:', df_phenotype_sub['SITE_ID'].unique())
    print(df_phenotype_sub['SITE_ID'].value_counts())
    # workaround through saving the table to disc.....
    df_phenotype_sub.to_csv('data/tmp.csv', sep=';', index=False)
    df_phenotype_sub = pd.read_csv('data/tmp.csv', sep=';')

    # usefull plots with categorical factors
    sns.catplot(x='SITE_ID', y='AGE_AT_SCAN', kind="swarm", data=df_phenotype_sub)
    sns.catplot(x='SITE_ID', y='AGE_AT_SCAN', kind="box", hue='DX_GROUP', data=df_phenotype_sub)
    sns.catplot(x='DX_GROUP', y='AGE_AT_SCAN', hue='SEX', kind="violin", inner="stick", split=True,
            palette="pastel", data=df_phenotype_sub)
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
    df_pheno_morpho = df_phenotype_sub.merge(df_morphometry, left_on='SUB_ID', right_on='subject', how='inner')

    # some nice bivariate plots
    sns.jointplot(x="AGE_AT_SCAN", y="MeanThickness", data=df_pheno_morpho)
    with sns.axes_style("white"):
        sns.jointplot(x="AGE_AT_SCAN", y="MeanThickness", kind="hex", color="k", data=df_pheno_morpho)
    sns.relplot(x="AGE_AT_SCAN", y="MeanThickness", hue='SITE_ID', data=df_pheno_morpho)
    plt.show()

    # linear models and curve fitting
    sns.lmplot(x="MeanThickness", y="AGE_AT_SCAN", data=df_pheno_morpho)
    sns.lmplot(x="MeanThickness", y="AGE_AT_SCAN", order=2, data=df_pheno_morpho)
    sns.lmplot(x="MeanThickness", y="AGE_AT_SCAN", lowess=True, data=df_pheno_morpho)
    plt.show()

    # regression with a categorical factor
    sns.lmplot(x="MeanThickness", y="AGE_AT_SCAN", hue='DX_GROUP', data=df_pheno_morpho)
    plt.show()

    df_pheno_morpho_sub = df_pheno_morpho[['SITE_ID', 'WhiteSurfArea', 'MeanThickness', 'CortexVol', 'CorticalWhiteMatterVol', 'SubCortGrayVol', 'IntraCranialVol']].copy()
    sns.pairplot(df_pheno_morpho_sub)
    plt.show()
    sns.pairplot(df_pheno_morpho_sub, hue="SITE_ID", palette="Set2", diag_kind="kde", height=2.5)
    plt.show()
    df_pheno_morpho_sub = df_pheno_morpho[['DX_GROUP', 'WhiteSurfArea', 'MeanThickness', 'CortexVol', 'CorticalWhiteMatterVol', 'SubCortGrayVol', 'IntraCranialVol']].copy()
    df_pheno_morpho_sub['DX_GROUP'] = df_pheno_morpho_sub['DX_GROUP'].astype('category')
    sns.pairplot(df_pheno_morpho_sub, hue="DX_GROUP", palette="Set2", diag_kind="kde", height=2.5)
    plt.show()
