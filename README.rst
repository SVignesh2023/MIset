MIset
=====

**'MIset'** stands for '(M)utual (I)nformation (SET) of feature selection techniques'.

This is a library that provides a python implementation of the mutual information based feature selection techniques outlined in the following research papers:

1. **'Joint Mutual Information Maximization'** method as described here: `https://doi.org/10.1016/j.eswa.2015.07.007 <https://doi.org/10.1016/j.eswa.2015.07.007>`_.
2. **'Normalized Joint Mutual Information Maximization'** method as described here: `https://doi.org/10.1016/j.eswa.2015.07.007 <https://doi.org/10.1016/j.eswa.2015.07.007>`_.
3. **'Joint Mutual Information with Class Relevance'** method as described here: `https://doi.org/10.1016/j.jcmds.2023.100075 <https://doi.org/10.1016/j.jcmds.2023.100075>`_.


Installation
------------

To install use:

.. code-block:: console

    $ pip install miset


Note
----

It is **generally recommended** to apply binning to both continuous and discrete variables before using this feature selection technique, as this was the approach taken by its authors.
If binning is ignored on discrete or continuous variables, the MIset package will treat each distinct value of these variables as its own seperate category by default.


Requirements
------------

* pandas
* numpy
* joblib

Read the documentation at: `https://miset.readthedocs.io/en/latest/index.html <https://miset.readthedocs.io/en/latest/index.html>`_
