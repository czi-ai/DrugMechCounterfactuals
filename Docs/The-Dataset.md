# The Drug Mechanisms Queries Dataset

All the queries are available as JSON files in `$PROJDIR/Data/Counterfactuals`. A brief overview can be found in [this README file](../Data/Counterfactuals/README.txt), and a description of the dataset and how it was constructed is in our accompanying paper.


## Prerequisites

Depending on your usage, you will need to download some supporting data files and build the corresponding databases under the `$PROJDIR/Data/` directory:

* `DrugMechDB/`: [DrugMechDB](https://github.com/SuLab/DrugMechDB). Needed for Add-Link queries and all Closed-World queries. Also needed for factual queries. The function `drugmechcf.data.drugmechdb.load_drugmechdb()` loads the default cache, and builds it if not present.

* `PrimeKG/`: [PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/). Needed if exploring additional factual queries. The function `drugmechcf.data.primekg.load_primekg()` loads the default cache, and builds it if not present.

* `MONDO/`: [MONDO](https://mondo.monarchinitiative.org). Needed with PrimeKG. The function `drugmechcf.data.mondo.load_mondo()` loads the default cache, and builds it if not present.

See [this README file](../Data/README.txt) for the raw files needed to build these databases.


## Data Format

There is a separate file for each Query-type (Add-Link, Invert-Link, Delete-Link and factuals), for counterfactual depth (surface or deep, for the counterfactuals), and for positive or negative samples. See [this README file](../Data/Counterfactuals/README.txt) for the file naming convention.

Each file is a JSON file containing a list of dict's, each dict describing the data for one query sample. For Add-Link queries, each dict is a serialization of the `drugmechcf.llmx.test_addlink.AddLinkTask` class, and for Invert-Link and Delete-Link queries a serialization of the `drugmechcf.llmx.test_editlink.EditLinkTask` class.

See the function `drugmechcf.data.cfdata.pp_samples()` for how to read and interpret this data.  To view the samples in a more readable format, use the python package `drugmechcf.data.cfdata`. For example:

```
(dmcf) $ cd $PROJDIR/src
(dmcf) $ python -m drugmechcf.data.cfdata examples -h
usage: cfdata.py examples [-h] [-s START] [-c COUNT] cf_samples_file

positional arguments:
  cf_samples_file       Path to query samples JSON file

options:
  -h, --help            show this help message and exit
  -s START, --start START
                        Index of first example.
  -c COUNT, --count COUNT
                        Max nbr of examples.

(dmcf) $ python -m drugmechcf.data.cfdata examples ../Data/Counterfactuals/AddLink_pos_dpi_r1k.json
```

Please note that the prerequisite data described in the previous section are needed to run this script.
