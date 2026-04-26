Example
=======

This page illustrates how to use the **MIset** library by an example.

First, lets load in the **Breast Cancer Wisconsin (Diagnostic)** dataset.

.. code-block:: python

    >>> from ucimlrepo import fetch_ucirepo

    >>> breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 

    >>> X = breast_cancer_wisconsin_diagnostic.data.features 
    >>> y = breast_cancer_wisconsin_diagnostic.data.targets


Let us combine both the core data and the target variable into a single dataframe:

.. code-block:: python

    >>> import pandas as pd
    >>> pd.set_option('display.max_columns',None)

    >>> df=pd.concat([X,y],axis=1)
    >>> df.head()

.. code-block:: text

        radius1  texture1  perimeter1   area1  smoothness1  compactness1  concavity1  concave_points1  symmetry1  fractal_dimension1  radius2  texture2  perimeter2   area2  smoothness2  compactness2  concavity2  concave_points2  symmetry2  fractal_dimension2  radius3  texture3  perimeter3   area3  smoothness3  compactness3  concavity3  concave_points3  symmetry3  fractal_dimension3 Diagnosis
    0    17.99     10.38      122.80  1001.0      0.11840       0.27760      0.3001          0.14710     0.2419             0.07871   1.0950    0.9053       8.589  153.40     0.006399       0.04904     0.05373          0.01587    0.03003            0.006193    25.38     17.33      184.60  2019.0       0.1622        0.6656      0.7119           0.2654     0.4601             0.11890         M
    1    20.57     17.77      132.90  1326.0      0.08474       0.07864      0.0869          0.07017     0.1812             0.05667   0.5435    0.7339       3.398   74.08     0.005225       0.01308     0.01860          0.01340    0.01389            0.003532    24.99     23.41      158.80  1956.0       0.1238        0.1866      0.2416           0.1860     0.2750             0.08902         M
    2    19.69     21.25      130.00  1203.0      0.10960       0.15990      0.1974          0.12790     0.2069             0.05999   0.7456    0.7869       4.585   94.03     0.006150       0.04006     0.03832          0.02058    0.02250            0.004571    23.57     25.53      152.50  1709.0       0.1444        0.4245      0.4504           0.2430     0.3613             0.08758         M
    3    11.42     20.38       77.58   386.1      0.14250       0.28390      0.2414          0.10520     0.2597             0.09744   0.4956    1.1560       3.445   27.23     0.009110       0.07458     0.05661          0.01867    0.05963            0.009208    14.91     26.50       98.87   567.7       0.2098        0.8663      0.6869           0.2575     0.6638             0.17300         M
    4    20.29     14.34      135.10  1297.0      0.10030       0.13280      0.1980          0.10430     0.1809             0.05883   0.7572    0.7813       5.438   94.44     0.011490       0.02461     0.05688          0.01885    0.01756            0.005115    22.54     16.67      152.20  1575.0       0.1374        0.2050      0.4000           0.1625     0.2364             0.07678         M

This dataset contains many columns having continuous values. Before we implement our feature selection algorithm, let us first perform binning on these features.

.. code-block:: python

    >>> from sklearn.preprocessing import KBinsDiscretizer

    >>> kbd=KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='uniform')

    >>> cont_columns=list(df.columns)
    >>> cont_columns.remove('Diagnosis') # This is our target variable
    >>> df[cont_columns]=kbd.fit_transform(df[cont_columns])


Now, let us perform feature selection. We select the Joint Mutual Information Maximization algorithm to perform feature selection and retrieve the top 3 most relevant features out of the entire dataset.

We initialize the class object and call the :py:meth:`MIset.fit` method to fit the feature selection algorithm on our dataset.

.. code-block:: python

    >>> import MIset

    >>> miset=MIset(max_features=3,variant='jmim',verbose=False,n_jobs=-1)
    >>> miset.fit(df=df,feature_list=cont_columns,class_feature_name='Diagnosis')

Now that feature selection is done, we use :py:meth:`MIset.feature_ranking` method to get the top 3 relevant features in the entire dataset.

'concave_points3' was deemed as the most relevant feature by the selected feature selection algorithm.


.. code-block:: python

    >>> miset.feature_ranking()
    {1: 'concave_points3', 2: 'radius3', 3: 'texture3'}
