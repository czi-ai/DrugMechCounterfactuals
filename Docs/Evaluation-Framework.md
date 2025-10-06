# The Counterfactuals Evaluation Framework

See also:

* [Description of the dataset](The-Dataset.md)
* [How to evaluate other LLMs](Testing-New-LLMs.md)


There are two main ways to invoke the evaluation framework, for the experiments described in out accompanying paper:

* Test a batch of queries from the dataset
* Compute mean accuracy and confidence interval across a number of runs, for LLMs that display some stochastic variability in their responses.

These are described below.


## Running Counterfactual Queries

### Add-Link Queries

#### Batch mode

The function `drugmechcf.llmx.test_addlink.test_addlink_batch()` provides a convenient entry point for running a batch of positive or negative queries against one LLM model. This function can also be invoked from the command line by running the `drugmechcf.llmx.test_addlink` python package, e.g. the following command will print a help message:

```shell
(dmcf) $ cd $PROJDIR/src
(dmcf) $ python -m drugmechcf.llmx.test_addlink -h
```

and here is an example call:

```
$ python -m drugmechcf.llmx.test_addlink batch -m o4-mini \
			 ../Data/Counterfactuals/AddLink_pos_dpi_r1k.json  \
           ../Data/Sessions/ModelRuns/o4-mini/addlink_pos_dpi.json 2>&1  \
           | tee ../Data/Sessions/ModelRuns/o4-mini/addlink_pos_dpi_log.txt
```

which runs the set of positive surface counterfactuals from the dataset file `AddLink_pos_dpi_r1k.json` against the `o4-mini` model using the open-world setting.


#### Computing mean Accuracy

The Python package `drugmechcf.exp.cfvariances` contains the code to compute mean accuracy and confidence interval for all counterfactual queries. For example, the following command will run Invert-Link queries using the parameters specified in an options file `opt_add_open.json`:

```
(dmcf) $ cd $PROJDIR/src
(dmcf) $ python -m drugmechcf.exp.cfvariances add_link ../Data/Sessions/Variances/opt_add_open.json    \
               ../Data/Sessions/Variances/runs_add_open.json  \
               2>&1 | tee ../Data/Sessions/Variances/log_add_open.txt
```


### Invert-Link and Delete-Link Queries

#### Batch mode

The function `drugmechcf.llmx.test_editlink.test_editlink_batch()` provides a convenient entry point for running a batch of positive or negative queries of either query type against one LLM model. This function can also be invoked from the command line by running the `drugmechcf.llmx.ttest_editlink` python package, e.g. the following command will print a help message:

```shell
(dmcf) $ cd $PROJDIR/src
(dmcf) $ python -m drugmechcf.llmx.test_editlink -h
```

and here is an example call:

```
$ python -m drugmechcf.llmx.test_editlink batch -m o4-mini -k \
			 ../Data/Counterfactuals/change_pos_dpi_r250.json  \
           ../Data/Sessions/ModelRuns/o4-mini/change_pos_dpi_r250-k.json  \
           2>&1 | tee ../Data/Sessions/ModelRuns/o4-mini/change_pos_dpi_r250-k_log.txt
```

which runs the set of positive surface counterfactuals from the dataset file `change_pos_dpi_r250.json` (Edit-Link queries) against the `o4-mini` model using the closed-world setting.


#### Computing mean Accuracy

As for Add-Link queries, the Python package `drugmechcf.exp.cfvariances` is used to compute mean accuracy and confidence interval across a specified number of runs. For example:

```
(dmcf) $ cd $PROJDIR/src
(dmcf) $ python -m drugmechcf.exp.cfvariances edit_link ../Data/Sessions/Variances/opt_delete_closed.json    \
               ../Data/Sessions/Variances/runs_delete_closed.json  \
               2>&1 | tee ../Data/Sessions/Variances/log_delete_closed.txt
```

This command computes mean accuracy for Delete-Link queries in the closed world setting.


### Computing Grouped Accuracies

Table VI in our FLLM-25 paper reports mean accuracies across groups of queries, with variance computed using stratified bootstrap, from previously run sessions. These metrics are computed using the function `drugmechcf.exp.cfvariances.summarize_counterfactual_metrics_bs()`, invoked from the command line as:

```
(dmcf) $ python -m drugmechcf.exp.cfvariances summarize ../Data/Sessions/ModelRuns
```

Grouped metrics are computed from previously saved single sessions (i.e. their output JSON files) for all the query types and modes, as described in the **Batch mode** sections above.

The single argument (e.g. `../Data/Sessions/ModelRuns`) points to the dir under which there is a subdir for each model, and within that dir are the JSON session files for each query type. The name of the session file is the same as that of the input samples file (from `"Data/Counterfactuals/"`), except it ends in `"-k.json"` if the query mode was "Closed World".


## Running Factual Queries

### Batch mode

The batch mode entry points for factual queries are:

* The function `drugmechcf.llm.test_dmdb.test_positives_batch()`, or `python -m drugmechcf.llm.test_dmdb positives ...` from the command line, for positive queries, and
* the function `drugmechcf.llm.test_dmdb.test_negatives_batch()`, or `python -m drugmechcf.llm.test_dmdb negs ...` from the command line, for negative queries.

The following shell-based commands test the default LLM model (a 4o model), using factual queries from the dataset in `Data/Counterfactuals/factuals*.json`:

```
(dmcf) $ python -m drugmechcf.llm.test_dmdb positives -c 1000 -p 2 \
           -f ../Data/Counterfactuals/factuals_pos_r1k.json \
           -j ../Data/Sessions/Factuals/pos_p2_f1k.json \
           2>&1 | tee ../Data/Sessions/Factuals/pos_p2_f1k.txt

(dmcf) $ python -m drugmechcf.llm.test_dmdb negs -c 1000 -p 2 --contras \
              -f ../Data/Counterfactuals/factuals_neg_r1k.json \
              -j ../Data/Sessions/Factuals/neg_p2_f1k_contras.json \
           2>&1 | tee ../Data/Sessions/Factuals/neg_p2_f1k_contras.txt
```


#### Computing mean Accuracy

Mean accuracies across multiple LLM runs (5 runs by default) are computed by the function `drugmechcf.exp.tvariances.multirun_stats_known_moa()`, which can be invoked from the command line as:

```
(dmcf) $ python -m drugmechcf.exp.tvariances known_moa
```

This computes the metrics for both positive and negative samples, for all three modes: *Named-Disease*, *Anonymized-Disease*, and *Named Disease with Associations*.


## Some Components of the Evaluation Framework

### Prompts for the Queries

The prompts used for Add-Link queries can be found in the Python file [drugmechcf/llmx/prompts_addlink.py](../src/drugmechcf/llmx/prompts_addlink.py), and for Delete-Link and Invert-Link queries in the Python file [drugmechcf/llmx/prompts_editlink.py](../src/drugmechcf/llmx/prompts_editlink.py). Both these files contain a `PromptBuilder` class that is invoked with various arguments to produce tthe appropriate prompt for each query.

The prompts for factual queries are in the Python file [src/drugmechcf/llm/prmpt_instructions_basic.py](../src/src/drugmechcf/llm/prmpt_instructions_basic.py). The corresponding prompt-builder class is `drugmechcf.llm.drugmechdb_prompt_builder.DrugMechPromptBuilder`.


### Automated parsing of LLM Multiple-Choice response: Delete-Link and Invert-Link

These queries require the LLM to respond with the nature of the effect of the counterfactual on the efficacy of the Drug on the Disease. Multiple choices are provided for the LLM to choose one from. The results of these queries are parsed by the Python class `drugmechcf.text.optsmatcher.OptionsMatcher`.


### Automated parsing of LLM-provided MoA: Add-Link and Factuals

The counterfactual in positive Add-Link queries is a new interaction, and the LLM is asked whether a MoA is now effective from a specified Drug to a specified Disease. The LLM can respond with a "Yes" or "No", and if the response is "Yes", then it is asked to provide the new MoA in a specified format.

For positive queries, the LLM is expected to respond with a MoA, which is first parsed by the function `drugmechcf.llm.test_common.get_moa_from_llm_response()` into a graph (class `drugmechcf.data.text2graph.MoaFromText`), and then matched against the expected MoA using the class `drugmechcf.graphmatch.graphmatcher.BasicGraphMatcher`. 

Positive factual queries also ask the LLM to provide an MoA, which is parsed into a graph and matched against the reference as described above.


### Regression on Counterfactual Link Depth

The method `drugmechcf.llmx.test_editlink.TestEditLink.write_all_sessions_sample_params()` is used to compute values for link-depth atttributes, and then the function `drugmechcf.exp.metrics.regression_on_mod_depth_params` is used to peform a regression test.

