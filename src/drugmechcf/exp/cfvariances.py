"""
Computes mean and variance of accuracy metrics across multiple runs of Counterfactual experiments:
    `multirun_stats_addlink()` and `multirun_stats_editlink()`,
and grouped metrics (using bootstrap on groups of single runs):
    `summarize_counterfactual_metrics_bs()`,
as described in the FLLM-25 paper.
See `MAIN` section at bottom of file for invoking from shell.
"""

import json
from collections import defaultdict, namedtuple
import glob
import os
import time
from typing import Any, Callable, Mapping, Union

import numpy as np
import pandas as pd
import scipy

from openai import BadRequestError

from tenacity import (
    retry,
    retry_if_exception_type,
    wait_random,
    stop_after_attempt,
)  # for retrying on `BadRequestError`

from drugmechcf.llm.openai import OpenAICompletionClient
from drugmechcf.llm.prompt_types import QueryType

from drugmechcf.llmx.test_addlink import TestAddLink, test_addlink_batch
from drugmechcf.llmx.test_editlink import TestEditLink, test_editlink_batch

from drugmechcf.utils.misc import (NpEncoder, pp_underlined_hdg, suppressed_stdout,
                                   pp_dict, pp_funcargs, reset_df_index, check_output_file_dir)

from .bootstrap import stratified_bootstrap

from drugmechcf.utils.projconfig import get_project_local_temp_dir


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

ADDLINK_INCLUDE_EXAMPLES_ONLY_IN_4O = False
"""Whether examples are included in Add-Link prompts only for the 4o model."""

DEBUG = False

BootstrapStats = namedtuple("BootstrapStats",
                            ["mean", "low", "high", "standard_error",
                             "n_samples_seq"
                             ])


ENABLE_UPSAMPLING_IN_BOOTSTRAP = False

USE_STRATIFIED_BOOTSTRAP = True

DEFAULT_N_RESAMPLES = 9999

TEMPDIR = get_project_local_temp_dir()
"""Location of temporary files. Where output files from each run are temporarily stored."""


# -----------------------------------------------------------------------------
#   Functions: Misc
# -----------------------------------------------------------------------------


def get_temp_suffix() -> str:
    return f"{os.getpid()}_{time.clock_gettime_ns(time.CLOCK_MONOTONIC) // 1000 % 1000}"


def get_temp_json(root_prefix: str = None) -> str:
    """
    Get path to temporary output JSON file.
    """

    if not os.path.exists(TEMPDIR):
        os.mkdir(TEMPDIR)

    if root_prefix is None:
        root_prefix = "tmp"

    tempf_path = f"{TEMPDIR}/{root_prefix}_{get_temp_suffix()}.json"
    return tempf_path


def pp_cum_metrics_stats(output_file: str,
                         sdict: dict[str, Any],
                         task: Union[str, list, tuple],
                         ) -> dict[str, Any]:

    if not isinstance(task, str):
        task = "/".join([str(t) for t in task])

    cum_metrics = sdict["tasks"][task]["cum_metrics"]

    if DEBUG:
        print("\n-----------")
        pp_dict(cum_metrics, "cum_metrics")
        print("-----------\n")

    # Convert to np.array to convert None to np.nan
    cum_metrics = dict((k, np.asarray(v)) for k, v in cum_metrics.items())

    # Save summary stats as list: mean, sd, min, max
    cum_metrics_stats = dict((k, [np.nanmean(v), np.nanstd(v), np.nanmin(v), np.nanmax(v)])
                             for k, v in cum_metrics.items())

    sdict["tasks"][task]["cum_metrics_stats"] = cum_metrics_stats

    # Save the sdict with the summary stats
    with open(output_file, "w") as f:
        json.dump(sdict, f, indent=4, cls=NpEncoder)    # type: ignore

    # Print to stdout
    df = pd.DataFrame.from_records([(k, *v) for k, v in cum_metrics_stats.items()],
                                   columns=["Metric", "Mean", "StdDev", "Min", "Max"],
                                   index="Metric")
    print(df.to_markdown(intfmt=',d', floatfmt=',.3f'))
    print()
    tot_llm = sum(cum_metrics['LLM Calls'])
    print(f"Total nbr LLM calls = {tot_llm:,d}")
    print()

    return sdict


def add_task_run(output_file: str,
                 sdict: dict[str, Any],
                 task: Union[str, list, tuple],
                 run_metrics: dict[str, Any],
                 accumulate_metrics_fn: Callable,
                 ) -> dict[str, Any]:
    """
    Add new completed task run to the output JSON file.
    """

    if not isinstance(task, str):
        task = "/".join([str(t) for t in task])

    if sdict.get("tasks") is None:
        sdict["tasks"] = dict()

    if sdict["tasks"].get(task) is None:
        sdict["tasks"][task] = dict()

    sdict["tasks"][task]["cum_metrics"] = accumulate_metrics_fn(run_metrics,
                                                                sdict["tasks"][task].get("cum_metrics"))

    with open(output_file, "w") as f:
        json.dump(sdict, f, indent=4, cls=NpEncoder)    # type: ignore

    return sdict


def check_task_nbr_runs(sdict: dict[str, Any], task: Union[str, list, tuple], max_nruns: int = 5) -> int:
    """
    How many more runs needed for this task?
    """

    if not isinstance(task, str):
        task = "/".join([str(t) for t in task])

    if not sdict.get("tasks") or not sdict["tasks"].get(task):
        return 0

    if not sdict["tasks"][task].get("cum_metrics"):

        if sdict["tasks"][task].get("cum_metrics_stats"):
            return max_nruns

        return 0

    return len(sdict["tasks"][task]["cum_metrics"].get('LLM Calls', []))


# -----------------------------------------------------------------------------
#   Functions: edit_link
# -----------------------------------------------------------------------------


def run_editlink(samples_data_file: str,
                 model_key: str = None,
                 model_params: dict = None,
                 insert_known_moas=False,
                 run_id: str = None):
    """
    Returns metrics dict,
        OR str(exception) if exception encountered.
    """

    # Latest prompt in `llmx.prompts_editlink.PROMPT_TEMPLATE`
    prompt_version = 2

    print(f"-- Running `test_batch_editlink({samples_data_file}, {prompt_version=}, {model_key=})`",
          f"run {run_id} ...",
          flush=True)

    # JSON file placed under ../Temp
    tmp_json = get_temp_json("editlink" + (f"_{run_id}" if run_id else ""))

    print(f" ... {tmp_json = }", flush=True)

    try:
        with suppressed_stdout():
            call_test_editlink_batch(samples_data_file,
                                     tmp_json,
                                     model_key=model_key,
                                     model_params=model_params,
                                     prompt_version=prompt_version,
                                     insert_known_moas=insert_known_moas,
                                     )
    except Exception as e:
        print("Exception:", e, flush=True)
        return str(e)

    tmp_metrics = TestEditLink.extract_main_metrics(tmp_json)

    tmp_metrics["LLM Calls"] = OpenAICompletionClient.get_nbr_calls()

    # Delete the tmp JSON file. Comment out if you want to keep the output files.
    os.remove(tmp_json)

    return tmp_metrics


@retry(retry=retry_if_exception_type(BadRequestError),
       stop=stop_after_attempt(3))
def call_test_editlink_batch(samples_data_file, tmp_json, model_key, prompt_version, insert_known_moas,
                             model_params=None):

    if model_params is None:
        model_params = dict()

    test_editlink_batch(samples_data_file,
                        tmp_json,
                        model_key=model_key,
                        prompt_version=prompt_version,
                        insert_known_moas=insert_known_moas,
                        **model_params
                        )
    return


def multirun_stats_editlink(input_file: str, output_file: str):
    """
    Run `nruns` runs of Change/Delete link query set to compile metrics variances (Change = Invert-Link).

    :param input_file: JSON file of options.
        File format:
            {
                "args": {
                    "insert_known_moas": true | false,  ... whether to insert known MoAs (i.e. closed world)
                    "model_keys": list[str],            ... keys for models to test
                    "model_params": dict[str, dict],    ... Optional additional params per model, to pass to
                                                            `drugmechcf.llmx.test_editlink.test_editlink_batch()`
                    "nruns": int,                       ... Nbr runs to comute variances from
                    "data_key": "change" | "delete"     ... Type of query, also indicates which query samples files
                }
            }

        File example (as used for results in the paper):
            {
                "args": {
                    "insert_known_moas": true,
                    "model_keys": [
                        "4o",
                        "o3",
                        "o3-mini",
                        "o4-mini"
                    ],
                    "nruns": 5,
                    "data_key": "change"
                }
            }

        A `model_key` can be one of the convenience abbreviated model names as defined in
        `drugmechcf.llm.openai.MODEL_KEYS`, or a model-name.

        `model_params` allows additional args to be passed to `drugmechcf.llmx.test_editlink.test_editlink_batch()`
        for a particular model, when needed. The format is:
            MODEL_KEY: dict[ ARG_NAME(str): VALUE ]
        Example use of "model_params" (filled only for models needing additional params):
                    "model_params": {
                        "Qwen/Qwen2.5-3B-Instruct": {
                            "api_key": "EMPTY",
                            "base_url": "http://localhost:8000/v1",
                            "n_worker_threads": 1,
                            "timeout_secs": 300
                        }
                    }

    :param output_file: JSON file where runs and accumulated metrics are stored
    """

    pp_funcargs(multirun_stats_editlink)

    assert input_file != output_file, "Input and output files are same!"

    if output_file is not None:
        check_output_file_dir(output_file)

    with open(input_file) as f:
        sdict = json.load(f)

    data_key = sdict["args"]["data_key"]

    model_keys = sdict["args"].get("model_keys")
    all_model_params = sdict["args"].get("model_params", dict())

    insert_known_moas = sdict["args"].get("insert_known_moas", False)
    nruns = sdict["args"].get("nruns", 5)

    assert data_key in ["change", "delete"], f'{data_key=} invalid. Must be one of ["change", "delete"].'

    if model_keys is None:
        model_keys = ["4o", "o3", "o3-mini", "o4-mini"]
    elif isinstance(model_keys, str):
        model_keys = [model_keys]

    data_files = sorted(glob.glob(f'../Data/Counterfactuals/{data_key}*.json'))

    print()
    print("Models =", ', '.join(model_keys))
    print("nbr Data files =", len(data_files), end="\n    ")
    print(*data_files, sep="\n    ")
    print()

    for samples_data_file in data_files:

        data_file_name = os.path.basename(samples_data_file)

        for mdl_k in model_keys:

            task = (data_file_name, mdl_k, insert_known_moas)

            model_params = all_model_params.get(mdl_k)

            remaining_runs = nruns - check_task_nbr_runs(sdict, task, max_nruns=nruns)
            if remaining_runs <= 0:
                print("Skipping completed task:", task)
                continue

            print()
            print("Task:", task, "-- remaining runs =", remaining_runs)
            print()

            for n in range(remaining_runs):

                tmp_metrics = run_editlink(samples_data_file,
                                           model_key=mdl_k,
                                           model_params=model_params,
                                           insert_known_moas=insert_known_moas,
                                           run_id=f"{n + 1}")

                if isinstance(tmp_metrics, str):
                    print("***** Exception encounterd! *****")
                    print(tmp_metrics)
                    print("***** ----- Exitting! ----- *****")
                    print()
                    return

                sdict = add_task_run(output_file, sdict, task, tmp_metrics, accumulate_editlink_metrics)

            print("\n")
            pp_underlined_hdg(f"EditLink Summary stats for: {', '.join([str(t) for t in task])}")
            print("Nbr runs =", nruns)
            print()
            sdict = pp_cum_metrics_stats(output_file, sdict, task)
            print(flush=True)

    print()
    print(f"Total nbr LLM calls = {OpenAICompletionClient.get_nbr_calls():,d}")
    print()
    print("Multi-run data written to:", output_file)

    return


def accumulate_editlink_metrics(sesn_metrics: dict[str, Any], cum_metrics=None) -> dict[str, list[int | float]]:
    """
    Accumulate metrics returned by `llmx.test_editlink.TestEditLink.extract_main_metrics()`
     into a single-level dict[list]

    :return: `cum_metrics`
    """

    if cum_metrics is None:
        cum_metrics = defaultdict(list)

    for k, v in sesn_metrics.items():
        if isinstance(v, Mapping):
            for kk, vv in v.items():
                cum_metrics[f"{k}/{kk}"].append(vv)
        else:
            cum_metrics[k].append(v)

    return cum_metrics


# -----------------------------------------------------------------------------
#   Functions: add_link
# -----------------------------------------------------------------------------


def run_addlink(samples_data_file: str,
                model_key: str = None,
                model_params: dict = None,
                insert_known_moas=False,
                run_id: str = None):
    """
    Returns metrics dict,
        OR str(exception) if exception encountered.
    """

    if ADDLINK_INCLUDE_EXAMPLES_ONLY_IN_4O:
        # Do not include examples for the reasoning models, as it may raise `BadRequestError`
        include_examples = model_key == "4o"
    else:
        include_examples = True

    print(f"-- `test_batch_addlink({os.path.basename(samples_data_file)}, {model_key=}, {include_examples=}",
          f", {insert_known_moas=})`  run {run_id} ...",
          flush=True)

    # JSON file placed under ../Temp
    tmp_json = get_temp_json("addlink" + (f"_{run_id}" if run_id else ""))

    print(f" ... {tmp_json = }", flush=True)

    try:
        with suppressed_stdout():
            call_test_addlink_batch(samples_data_file,
                                    tmp_json,
                                    model_key=model_key,
                                    model_params=model_params,
                                    insert_known_moas=insert_known_moas,
                                    include_examples=include_examples,
                                    )
    except Exception as e:
        print("Exception:", e, flush=True)
        return str(e)

    tmp_metrics = TestAddLink.extract_main_metrics(tmp_json)

    tmp_metrics["LLM Calls"] = OpenAICompletionClient.get_nbr_calls()

    # Delete the tmp JSON file. Comment out if you want to keep the output files.
    os.remove(tmp_json)

    return tmp_metrics


@retry(retry=retry_if_exception_type(BadRequestError),
       wait=wait_random(2, 10),
       stop=stop_after_attempt(7))
def call_test_addlink_batch(samples_data_file, tmp_json, model_key, insert_known_moas, include_examples,
                            model_params=None):
    test_addlink_batch(samples_data_file,
                       tmp_json,
                       model_key=model_key,
                       insert_known_moas=insert_known_moas,
                       include_examples=include_examples,
                       **model_params
                       )
    return


def multirun_stats_addlink(input_file: str, output_file: str):
    """
    Run `nruns` runs of AddLink query set to compile metrics variances.

    :param input_file: JSON file of options.
        File format:
            {
                "args": {
                    "insert_known_moas": true | false,  ... whether to insert known MoAs (i.e. closed world)
                    "model_keys": list[str],            ... keys for models to test
                    "model_params": dict[str, dict],    ... Optional additional params per model, to pass to
                                                            `drugmechcf.llmx.test_editlink.test_editlink_batch()`
                    "nruns": int,                       ... Nbr runs to comute variances from
                    "data_key": "AddLink"               ... Type of query, also indicates which query samples files
                }
            }

        File example (as used for results in the paper):
            {
                "args": {
                    "insert_known_moas": false,
                    "model_keys": [
                        "4o",
                        "o3",
                        "o3-mini",
                        "o4-mini"
                    ],
                    "nruns": 5,
                    "data_key": "AddLink"
                }
            }

        A `model_key` can be one of the convenience abbreviated model names as defined in
        `drugmechcf.llm.openai.MODEL_KEYS`, or a model-name.

        `model_params` allows additional args to be passed to `drugmechcf.llmx.test_editlink.test_editlink_batch()`
        for a particular model, when needed. The format is:
            MODEL_KEY: dict[ ARG_NAME(str): VALUE ]
        Example use of "model_params" (filled only for models needing additional params):
                    "model_params": {
                        "Qwen/Qwen2.5-3B-Instruct": {
                            "api_key": "EMPTY",
                            "base_url": "http://localhost:8000/v1",
                            "n_worker_threads": 1,
                            "timeout_secs": 300
                        }
                    }

    :param output_file: JSON file where runs and accumulated metrics are stored
    """

    pp_funcargs(multirun_stats_addlink)

    assert input_file != output_file, "Input and output files are same!"

    if output_file is not None:
        check_output_file_dir(output_file)

    with open(input_file) as f:
        sdict = json.load(f)

    data_key = sdict["args"]["data_key"]

    model_keys = sdict["args"].get("model_keys")
    all_model_params = sdict["args"].get("model_params", dict())

    insert_known_moas = sdict["args"].get("insert_known_moas", False)
    nruns = sdict["args"].get("nruns", 5)

    assert data_key == "AddLink", f'{data_key=} invalid. Must be "AddLink".'

    if model_keys is None:
        model_keys = ["4o", "o3", "o3-mini", "o4-mini"]
    elif isinstance(model_keys, str):
        model_keys = [model_keys]

    data_files = sorted(glob.glob(f'../Data/Counterfactuals/{data_key}*.json'))

    print()
    print("Models =", ', '.join(model_keys))
    print("nbr Data files =", len(data_files), end="\n    ")
    print(*data_files, sep="\n    ")
    print()

    for samples_data_file in data_files:

        data_file_name = os.path.basename(samples_data_file)

        for mdl_k in model_keys:

            task = (data_file_name, mdl_k, insert_known_moas)

            model_params = all_model_params.get(mdl_k)

            remaining_runs = nruns - check_task_nbr_runs(sdict, task, max_nruns=nruns)
            if remaining_runs <= 0:
                print("Skipping completed task:", task)
                continue

            print()
            print("Task:", task, "-- remaining runs =", remaining_runs)
            print()

            for n in range(remaining_runs):

                tmp_metrics = run_addlink(samples_data_file,
                                          model_key=mdl_k,
                                          model_params=model_params,
                                          insert_known_moas=insert_known_moas,
                                          run_id=f"{n + 1}")

                if isinstance(tmp_metrics, str):
                    print("***** Exception encounterd! *****")
                    print(tmp_metrics)
                    print("***** ----- Exitting! ----- *****")
                    print()
                    return

                sdict = add_task_run(output_file, sdict, task, tmp_metrics, accumulate_editlink_metrics)

            print("\n")
            pp_underlined_hdg(f"AddLink Summary stats for: {', '.join([str(t) for t in task])}")
            print("Nbr runs =", nruns)
            print()
            sdict = pp_cum_metrics_stats(output_file, sdict, task)
            print(flush=True)

    print()
    print(f"Total nbr LLM calls = {OpenAICompletionClient.get_nbr_calls():,d}")
    print()
    print("Multi-run data written to:", output_file)

    return


# -----------------------------------------------------------------------------
#   Functions: compile metrics
# -----------------------------------------------------------------------------


def compile_metrics(run_files: list[str], to_csv=False):

    print("Files processed:")
    print(*run_files, sep="\n")
    print()

    task_mstats = []

    metric_keys = ["binary-strict/accuracy", "binary-relaxed/accuracy", "accuracy"]

    sd_vals = []

    for rfile in sorted(run_files):
        with open(rfile) as jf:
            sdict = json.load(jf)

            rfdict = defaultdict(dict)
            task_mstats.append(rfdict)

            for task in sdict["tasks"]:
                cum_metrics_stats = sdict["tasks"][task]["cum_metrics_stats"]

                dataset, model, is_closed = task.split("/")

                is_pos = "pos" if "_pos_" in dataset else "neg"
                is_surface = "Yes" if "_dpi_" in dataset else "no"
                is_closed_label = "Yes" if is_closed == "True" else "no"
                qtype = dataset.split("_")[0]

                if not to_csv:
                    dataset = dataset.replace("_", "\\_")

                for mkey in metric_keys:
                    if v := cum_metrics_stats.get(mkey):
                        rfdict[(qtype, is_surface, is_closed_label, is_pos, dataset, mkey, "mean")][model] = v[0]
                        rfdict[(qtype, is_surface, is_closed_label, is_pos, dataset, mkey, "s.d.")][model] = v[1]
                        sd_vals.append(v[1])

    models = ["4o", "o3", "o3-mini", "o4-mini"]

    print("Strict Accuracies:\n")

    df = pd.DataFrame.from_records([(*k, *[v.get(m, np.nan) for m in models])
                                    for rfdict in task_mstats
                                    for k, v in rfdict.items()
                                    if "relaxed" not in k[5]
                                    ],
                                   columns=["QueryType", "is_Surface", "is_Closed", "is_Positive",
                                            "Dataset", "metric", "stat", *models]
                                   )
    df = df.sort_values(["QueryType", "is_Positive", "is_Closed", "is_Surface", "metric", "stat"],
                        ascending=[True, False, False, True, True, True],
                        ignore_index=True)

    if to_csv:
        print(df.to_csv())
    else:
        print(df.to_markdown(floatfmt='.3f'))
    print("\n")

    print("Relaxed Accuracies:\n")

    df = pd.DataFrame.from_records([(*k, *[v.get(m, np.nan) for m in models])
                                    for rfdict in task_mstats
                                    for k, v in rfdict.items()
                                    if "relaxed" in k[5]
                                    ],
                                   columns=["QueryType", "is_Surface", "is_Closed", "is_Positive",
                                            "Dataset", "metric", "stat", *models]
                                   )
    df = df.sort_values(["QueryType", "is_Positive", "is_Closed", "is_Surface", "metric", "stat"],
                        ascending=[True, False, True, True, True, True],
                        ignore_index=True)

    if to_csv:
        print(df.to_csv())
    else:
        print(df.to_markdown(floatfmt='.3f'))
    print()

    print()
    print(f"Range of s.d. = mean:{np.nanmean(sd_vals):.3f}, s.d.:{np.nanstd(sd_vals):.3f},",
          f"range:{[np.nanmin(sd_vals), np.nanmax(sd_vals)]}")

    return


# -----------------------------------------------------------------------------
#   Functions: Bootstrapped stats for group metrics
# -----------------------------------------------------------------------------


def bootstrapped_stats(session_files: list[str],
                       n_resamples: int = DEFAULT_N_RESAMPLES,
                       confidence_level: float = 0.9,
                       upsample_250 = False,
                       ) -> BootstrapStats:
    """
    As for `stratified_bootstrapped_stats()`, except bootstrapping is not stratified, which is not
    appropriate for mean across a group of categories.
    """

    all_is_correct_preds = []
    n_samples_seq = []

    for sesn_file in session_files:
        with open(sesn_file) as f:
            sdict = json.load(f)

        query_type = sdict["args"]["query_type"]
        assert query_type in QueryType.__members__

        if query_type == QueryType.ADD_LINK.name:
            is_correct_pred_fn = TestAddLink.is_sample_response_correct
        else:
            is_correct_pred_fn = TestEditLink.is_sample_response_correct

        sesn_correct_preds = [is_correct_pred_fn(s) for s in sdict["session"]]

        if ENABLE_UPSAMPLING_IN_BOOTSTRAP and upsample_250 and len(sesn_correct_preds) == 250:
            sesn_correct_preds = sesn_correct_preds * 4

        all_is_correct_preds.extend(sesn_correct_preds)
        n_samples_seq.append(len(sesn_correct_preds))

    bs = scipy.stats.bootstrap([all_is_correct_preds], np.mean,
                               n_resamples=n_resamples,
                               confidence_level=confidence_level)

    bs_stats = BootstrapStats(low=bs.confidence_interval.low, high=bs.confidence_interval.high,
                              mean=np.mean(bs.bootstrap_distribution),
                              standard_error=bs.standard_error,
                              n_samples_seq=n_samples_seq)

    return bs_stats


def stratified_bootstrapped_stats(session_files: list[str],
                                  n_resamples: int = DEFAULT_N_RESAMPLES,
                                  confidence_level: float = 0.9,
                                  upsample_250 = False,
                                  ) -> BootstrapStats:
    """
    Uses stratified bootstrap to compute confidence interval around mean response accuracy across a group of
    query sessions.

    :param session_files: List of session files to read.
    :param n_resamples: for bootstrap
    :param confidence_level:
    :param upsample_250: IF AddLink and delete/change qtypes, then upsample delete/change
        So n_samples = 250 gets repeated 4x.

    :return: `BootstrapStats` for response accuracy.
    """

    all_is_correct_preds = []
    data_groups = []
    n_samples_seq = []

    for i, sesn_file in enumerate(session_files):
        with open(sesn_file) as f:
            sdict = json.load(f)

        query_type = sdict["args"]["query_type"]
        assert query_type in QueryType.__members__

        if query_type == QueryType.ADD_LINK.name:
            is_correct_pred_fn = TestAddLink.is_sample_response_correct
        else:
            is_correct_pred_fn = TestEditLink.is_sample_response_correct

        sesn_correct_preds = [is_correct_pred_fn(s) for s in sdict["session"]]

        if upsample_250 and len(sesn_correct_preds) == 250:
            # Upsample the Edit-Link queries so their size matches data for Add-Link.
            # Edit-Link data size is 250, Add-Link data size is 1,000.
            sesn_correct_preds = sesn_correct_preds * 4

        n_samples = len(sesn_correct_preds)

        all_is_correct_preds.extend(sesn_correct_preds)
        data_groups.extend([i] * n_samples)
        n_samples_seq.append(n_samples)

    bs = stratified_bootstrap([all_is_correct_preds], [data_groups], np.mean,
                               n_resamples=n_resamples,
                               confidence_level=confidence_level)

    bs_stats = BootstrapStats(low=bs.confidence_interval.low, high=bs.confidence_interval.high,
                              mean=np.mean(bs.bootstrap_distribution),
                              standard_error=bs.standard_error,
                              n_samples_seq=n_samples_seq)

    return bs_stats


def read_all_session_files(bdir = "../Data/Sessions/Models") -> pd.DataFrame:
    """
    Compile a df with annotated session files.
    :param bdir: Parent dir, under which there is a subdir for each model, and within that dir are the
        JSON session files for each query type. The name of the session file is the same as that of the
        input samples file (from ".../Data/Counterfactuals/"), except it ends in "-k.json" if the query
        mode was "Closed World" (see `get_is_closed_world()` below).
    :return: the dataframe
    """

    # Parsers for the session file name, to extract various attributes.
    # ----
    def get_query_type(sfile_):
        qt = os.path.basename(sfile_).split("_")[0]
        return qt

    def get_pos_neg(sfile_):
        return "pos" if "_pos_" in sfile_ else "neg"

    def get_is_closed_world(sfile_):
        return "Yes" if sfile_.endswith("-k.json") else "no"

    def get_is_surface_cf(sfile_):
        return "Yes" if "_dpi_" in sfile_ else "no"

    def get_model(sfile_):
        return sfile_.split("/")[-2]
    # ---

    # Path format: {bdir}/{model}/*.json
    all_files = glob.glob(f"{bdir}/*/*.json")

    # Column names as expected in `summarize_counterfactual_metrics_bs`
    df = pd.DataFrame.from_records([(get_query_type(sfile), get_is_surface_cf(sfile),
                                     get_pos_neg(sfile), get_is_closed_world(sfile),
                                     get_model(sfile),
                                     sfile)
                                    for sfile in all_files],
                                   columns=["QueryType", "is_Surface",
                                            "is_Positive", "is_Closed",
                                            "Model",
                                            "File"]
                                   )

    return df


def summarize_counterfactual_metrics_bs(bdir: str = "../Data/Sessions/Models"):
    """
    Compile grouped metrics, as described in the paper in
        "Table VI: GROUPED ACCURACIES FOR COUNTERFACTUALS, SHOWING TRENDS."
    Use bootstrapping to compute mean and low/high error bars from a single saved run of a model on the cf sample files.
    """

    print("# Bootstrapped Metrics for Groups")
    print()
    print("Means +/- range with 90% confidence interval, calculated using stratified bootstrap on the",
          "combined results (of one run) of the samples from the cases in the corresponding groups.")
    print()
    print(f"{USE_STRATIFIED_BOOTSTRAP = }")
    print(f"{DEFAULT_N_RESAMPLES = :,d}")
    print()

    print("**Note**. In the tables below, 'low' and 'high' define a margin around the mean:",
          "'low' is units below mean, and 'high' is units above mean.  ")
    print("So the range is (mean + low) .. (mean + high).")
    print()

    df = read_all_session_files(bdir)

    if df is None:
        return

    print("Headings =", ", ".join(df.columns.values))
    print()

    pp_bs_stats("All Counterfactuals", df)

    # Open, Closed

    pp_bs_stats("All samples, Open world",
                       df[df["is_Closed"] == "no"])

    pp_bs_stats("All samples, Closed world",
                       df[df["is_Closed"] == "Yes"])

    # Pos, Neg

    pp_bs_stats("All Positive samples",
                       df[df["is_Positive"] == "pos"])

    pp_bs_stats("All Negative samples",
                       df[df["is_Positive"] == "neg"])

    # Pos - Neg, Open - Closed

    pp_bs_stats("Positive samples, Open world",
                       df[(df["is_Positive"] == "pos") & (df["is_Closed"] == "no")])

    pp_bs_stats("Negative samples, Open world",
                       df[(df["is_Positive"] == "neg") & (df["is_Closed"] == "no")])

    pp_bs_stats("Positive samples, Closed world",
                       df[(df["is_Positive"] == "pos") & (df["is_Closed"] == "Yes")])

    pp_bs_stats("Negative samples, Closed world",
                       df[(df["is_Positive"] == "neg") & (df["is_Closed"] == "Yes")])

    # Add-Link Pos, Neg; Open World

    pp_bs_stats("Add-Link Positive samples; Open World",
                df[(df["is_Positive"] == "pos") & (df["QueryType"] == "AddLink") & (df["is_Closed"] == "no")])

    pp_bs_stats("Add-Link Negative samples; Open World",
                df[(df["is_Positive"] == "neg") & (df["QueryType"] == "AddLink") & (df["is_Closed"] == "no")])

    # DPI - PPI, Open - Closed

    pp_bs_stats("Source is Drug (Surface cf.), Open world",
                       df[(df["is_Surface"] == "Yes") & (df["is_Closed"] == "no")])

    pp_bs_stats("Source is Not Drug (Deep cf.), Open world",
                       df[(df["is_Surface"] == "no") & (df["is_Closed"] == "no")])

    pp_bs_stats("Source is Drug (Surface cf.), Closed world",
                       df[(df["is_Surface"] == "Yes") & (df["is_Closed"] == "Yes")])

    pp_bs_stats("Source is Not Drug (Deep cf.), Closed world",
                       df[(df["is_Surface"] == "no") & (df["is_Closed"] == "Yes")])

    # Change + Delete, not Drug, Open World

    pp_bs_stats("Change + Delete, not Drug (Deep cf.), Open World - Positives",
                       df[ ((df["QueryType"] == "change") | (df["QueryType"] == "delete")) &
                           (df["is_Surface"] == "no") & (df["is_Closed"] == "no") & (df["is_Positive"] == "pos")])

    pp_bs_stats("Change + Delete, not Drug (Deep cf.), Open World - Negatives",
                       df[ ((df["QueryType"] == "change") | (df["QueryType"] == "delete")) &
                           (df["is_Surface"] == "no") & (df["is_Closed"] == "no") & (df["is_Positive"] == "neg")])

    return


def pp_bs_stats(hdg: str, df: pd.DataFrame):

    print("## " + hdg)
    print()

    uniq_qtypes = df["QueryType"].unique()
    upsample_250 = len(uniq_qtypes) > 1 and "AddLink" in uniq_qtypes

    models_seq = []
    bs_stats_seq = []

    for model, mdf in df.groupby("Model"):
        if USE_STRATIFIED_BOOTSTRAP:
            bs_stats = stratified_bootstrapped_stats(mdf["File"].values, upsample_250=upsample_250)
        else:
            bs_stats = bootstrapped_stats(mdf["File"].values, upsample_250=upsample_250)

        models_seq.append(model)
        bs_stats_seq.append(bs_stats)

    smdf = pd.DataFrame.from_records([["mean"] + [bs.mean for bs in bs_stats_seq],
                                      ["low"] + [bs.low - bs.mean for bs in bs_stats_seq],
                                      ["high"] + [bs.high - bs.mean for bs in bs_stats_seq],
                                      ["stderr"] + [bs.standard_error for bs in bs_stats_seq],
                                      ["nSamples"] + [sum(bs.n_samples_seq) for bs in bs_stats_seq]
                                      ],
                                     columns=["Metric"] + models_seq)

    print(reset_df_index(smdf).to_markdown(floatfmt=",.3f", index=False))
    print("\n")
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
#
# [src]$ python -m drugmechcf.exp.cfvariances {add_link | edit_link | ...}  [-m MODEL]
#
# e.g. (as executed from `$PROJDIR/src/`):
# [src]$ python -m drugmechcf.exp.cfvariances edit_link ../Data/Sessions/Variances/opt_change_open.json    \
#               ../Data/Sessions/Variances/runs_change_open.json  \
#               2>&1 | tee ../Data/Sessions/Variances/log_change_open.txt
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Compute summary stats from multiple runs.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... add_link
    _sub_cmd_parser = _subparsers.add_parser('add_link',
                                             help="Test Add-Link.")
    _sub_cmd_parser.add_argument('input_file', type=str,
                                 help="Input JSON file with args and possible incomplete multi-run session.")
    _sub_cmd_parser.add_argument('output_file', type=str,
                                 help="Output JSON file where multi-run session will be saved.")

    # ... edit_link
    _sub_cmd_parser = _subparsers.add_parser('edit_link',
                                             help="Test all Edit-Link.")
    _sub_cmd_parser.add_argument('input_file', type=str,
                                 help="Input JSON file with args and possible incomplete multi-run session.")
    _sub_cmd_parser.add_argument('output_file', type=str,
                                 help="Output JSON file where multi-run session will be saved.")

    # ... compile
    _sub_cmd_parser = _subparsers.add_parser('compile',
                                             help="Gather accuracy metrics from multiple run files.")
    _sub_cmd_parser.add_argument("--csv", action="store_true",
                                 help="Output table to CSV format")
    _sub_cmd_parser.add_argument('input_files', type=str, nargs="+",
                                 help="Input JSON files containing multi-run session.")

    # ... summarize
    _sub_cmd_parser = _subparsers.add_parser('summarize',
                                             help="Summarize accuracy metric Groups from Excel sheet.")
    _sub_cmd_parser.add_argument('base_dir', type=str, nargs="?",
                                 default="../Data/Sessions/Models",
                                 help="Input JSON files containing multi-run session.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'add_link':

        multirun_stats_addlink(_args.input_file, _args.output_file)

    elif _args.subcmd == 'edit_link':

        multirun_stats_editlink(_args.input_file, _args.output_file)

    elif _args.subcmd == 'compile':

        compile_metrics(_args.input_files, to_csv=_args.csv)

    elif _args.subcmd == 'summarize':

        summarize_counterfactual_metrics_bs(_args.base_dir)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
