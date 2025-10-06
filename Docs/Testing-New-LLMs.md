# Testing New Large Language Models

There are several considerations to address when testing new models, including LLMs not from OpenAI. These are discussed below.


## Accessing the LLM

### OpenAI models

These should be accessible by supplying the appropriate model name to the class `drugmechcf.llm.openai.OpenAICompletionClient`, which is a wrapper around OpenAI's API.

### Public models

The easiest option is to use [vllm as an OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) to serve the LLM hosted at your own server.

Once you have [installed vllm](https://docs.vllm.ai/en/latest/getting_started/quickstart.html), you can run it as follows (example showing it hosting the model `Qwen/Qwen2.5-3B-Instruct`, which gets automatically downloaded from HuggingFace):

```
$ vllm serve Qwen/Qwen2.5-3B-Instruct --dtype auto --max-num-batched-tokens 32768
```

This LLM can now be accessed by specifying the `base_url`, e.g.  `base_url = "http://localhost:8000/v1"` argument.

### Other proprietary models

This will require modifying or replacing the class `drugmechcf.llm.openai.OpenAICompletionClient`.


## Running Counterfactual Queries

### Prompt tuning

LLMs typically require prompts tuned for that model in order to provide the most accurate response. The prompts for *Add-Link* queries are defined in `src/drugmechcf/llmx/prompts_addlink.py`, and for *Delete-Link* and *Invert-Link* queries in `src/drugmechcf/llmx/prompts_editlink.py`.

Tuning the prompt also includes tuning the response instructions. LLMs differ in their ability to follow instructions for providing a correctly formatted response.

### Response parsing

All the counterfactual query testers use the `drugmechcf.text.optsmatcher.OptionsMatcher` class for parsing responses to multiple-choice queries.

### Invoking the Counterfactual Query Testers

For *Add-Link* queries, the easiest way is to call the convenience function `drugmechcf.llmx.test_addlink.test_addlink_batch()`. Similarly for the *Delete-Link* and *Invert-Link* queries, the corresponding function is `drugmechcf.llmx.test_editlink.test_editlink_batch()`.

As an example, for vllm-served models, supply the following additional arguments to these functions:

* `model_key`: e.g. `"Qwen/Qwen2.5-3B-Instruct"`
* `api_key`: e.g. `"EMPTY"`
* `base_url`: e.g. `"http://localhost:8000/v1"`
* `n_worker_threads`: e.g. 1 (vllm most likely will not support multiple threads)
* `timeout_secs`: e.g. 300 (in seconds), depending on the model and the server


## Running Factual Queries

The code is written to test a default LLM model on the factual queries. For our accompanying paper, we tested the OpenAI 4o model. For testing other models, specific parameters need to be passed through the main entry points to the following function, or this function needs to be modified:

```
drugmechcf.llm.test_dmdb.get_default_llm_client()
```

The two entry points are:

* `drugmechcf.llm.test_dmdb.test_positives_batch()` for testing positve factual queries, and
* `drugmechcf.llm.test_dmdb.test_negatives_batch()` for testing negative factual queries.

Passing values to the following additional parameters can alter which model (supported via the OpenAI client) gets invoked:

* `model_key`: e.g. `"o3"`, or `"Qwen/Qwen2.5-3B-Instruct"` for the model served using `vllm` as described above.
* `api_key`: e.g. `"EMPTY"` for the vllm-served LLM example.
* `base_url`: e.g. `"http://localhost:8000/v1"` for the vllm-served LLM example.
* `timeout_secs`: e.g. 300 (in seconds), depending on the model and the server.



