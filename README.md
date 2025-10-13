# DrugMechCounterfactuals: The Drug Mechanisms Counterfactuals Dataset

This is a release of the Drug Mechanisms Counterfactuals Dataset, developed for evaluating Large Language Models on their ability to recall and reason about drug mechanisms.

The dataset is described in our accompanying paper published in FLLM 2025, and also available as a [preprint](...).

The dataset and related code is available for use under the [MIT license](LICENSE).


## Overview

### The Dataset

The counterfactual queries, and the related factual queries, are available as JSON files in the `Data/Counterfactuals` sub-directory. See [this README file](Data/Counterfactuals/README.txt).

The counterfactual queries are derived from [DrugMechDB](https://github.com/SuLab/DrugMechDB); using the Add-Link queries, and any Closed World queries (where the prompt includes information about relevant drug mechanisms), requires downloading DrugMechDB. See [this README file](Data/README.txt).

[DrugMechDB](https://github.com/SuLab/DrugMechDB) is also required for the factual queries used in the accompanying paper. The code also allows you to create your own factual queries, for which both [PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/) and [MONDO](https://mondo.monarchinitiative.org) are also required. See [this README file](Data/README.txt) for the needed files.


### Setup for the Code

Python code for the evaluation framework can be found under the `src/` sub-directory. Requirements for using the code are listed in `requirements.txt`. The code has been tested with Python version 3.11.5, but should also work with more recent versions.

Shown below is an example setup for using the code. For convenience, we will use the environment variable `PROJDIR` to refer to the path where DrugMechCounterfactuals has been cloned, e.g. `PROJDIR=/Users/Me/Projects/DrugMechCounterfactuals`. Note that this environment variable is not needed by the code or the setup process.

1. Create a virtual environment for the project, e.g. 

	```
	$ cd $PROJDIR
	$ uv venv --python 3.12 --seed --prompt dmcf
	$ source .venv/bin/activate
	(dmcf) $
	```
	
	This creates a virtual environment in `$PROJDIR/.venv`.
	
2. Install the requirements, e.g.

	```
	(dmcf) $ uv pip install -r requirements.txt
	```

3. Invoke the appropriate Python scripts from the `src` sub-directory. For example, to pretty-print samples from the Add-Link positives surface-counterfactuals query set:

	```
	(dmcf) $ cd $PROJDIR/src
	(dmcf) $ python -m drugmechcf.data.cfdata examples ../Data/Counterfactuals/AddLink_pos_dpi_r1k.json
	```

### For More Information

Please consult the following documents for a more detailed description of the Dataset and the Evaluation Framework:

* [The dataset](Docs/The-Dataset.md)
* [The Evaluation Framework](Docs/Evaluation-Framework.md)
* [How to test new models on the dataset](Docs/Testing-New-LLMs.md)


## How to cite

```
@inproceedings{DrugMechCounterfactuals,
  author = {Sunil Mohan and Theofanis Karaletsos},
  title = {How Well Does {ChatGPT} Understand Drug Mechanisms? A Knowledge + Reasoning Evaluation Dataset},
  booktitle = {2025 3rd International Conference on Foundation and Large Language Models (FLLM)},
  year = {2025},
  pages = {},
  doi = {},
  publisher = {IEEE},
}
```

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
