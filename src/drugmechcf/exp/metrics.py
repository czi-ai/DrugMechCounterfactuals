"""
Analyze and summarize metrics
"""

from collections import Counter, defaultdict
import glob
import json
import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from drugmechcf.llm.openai import MODEL_KEYS


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def compile_session_accuracy_metrics(run_files: list[str], to_csv=False):
    """
    Compile accuracy metrics from a list of single-session JSON output files.
    Output to stdout.

    :param run_files: List of paths to JSON output files from single-session runs across models and query-types
    :param to_csv: Output to CSV? Default output is markdown.
    """

    print("Files processed:")
    print(*[f if to_csv else f.replace("_", "\\_") for f in run_files], sep="\n")
    print()

    model_to_mkey = dict((v, k) for k, v in MODEL_KEYS.items())

    task_mstats = defaultdict(lambda: defaultdict(dict))

    metric_keys = ["binary-strict/accuracy", "binary-relaxed/accuracy", "accuracy"]

    # ---
    def get_metric(metrics, m_key):
        sub_key = None
        if "/" in m_key:
            key, sub_key = m_key.split("/")
        else:
            key = m_key

        val = metrics.get(key)
        if val is None:
            return
        if isinstance(val, dict):
            return val.get(sub_key)
        else:
            return val
    # ---

    models = set()

    for rfile in sorted(run_files):
        with open(rfile) as jf:
            sdict = json.load(jf)

            qtype = sdict["args"]["query_type"]
            dataset = os.path.basename(sdict["args"]["samples_data_file"])

            is_pos = "pos" if not sdict["args"]["samples_are_negative"] else "neg"
            is_surface = "surface" if sdict["args"]["source_node_is_drug"] else "deep"
            is_closed = "closed" if sdict["args"]["insert_known_moas"] else "open"

            model = sdict["args"]["OpenAICompletionOpts"]["model"]
            model = model_to_mkey.get(model, model)

            models.add(model)

            if not to_csv:
                dataset = dataset.replace("_", "\\_")

            for mkey in metric_keys:
                if v := get_metric(sdict["metrics"], mkey):
                    mdict = task_mstats[(qtype, is_closed, is_pos, is_surface, dataset, mkey)]
                    mdict[model] = v

    models = sorted(models)

    print()
    print("## Strict Accuracies:\n")

    df = pd.DataFrame.from_records([(*k, *[mdict.get(m, np.nan) for m in models])
                                    for k, mdict in task_mstats.items()
                                    if "relaxed" not in k[5]
                                    ],
                                   columns=["QueryType", "is_Closed", "is_Positive", "is_Surface",
                                            "Dataset", "metric", *models]
                                   )
    df = df.sort_values(["QueryType", "is_Closed", "is_Positive", "is_Surface"],
                        ascending=[True, False, False, False],
                        ignore_index=True)

    if to_csv:
        print(df.to_csv())
    else:
        print(df.to_markdown(floatfmt='.3f'))
    print("\n")

    print("## Relaxed Accuracies:\n")

    df = pd.DataFrame.from_records([(*k, *[mdict.get(m, np.nan) for m in models])
                                    for k, mdict in task_mstats.items()
                                    if "relaxed" in k[5]
                                    ],
                                   columns=["QueryType", "is_Closed", "is_Positive", "is_Surface",
                                            "Dataset", "metric", *models]
                                   )
    df = df.sort_values(["QueryType", "is_Closed", "is_Positive", "is_Surface"],
                        ascending=[True, False, False, False],
                        ignore_index=True)

    if to_csv:
        print(df.to_csv())
    else:
        print(df.to_markdown(floatfmt='.3f'))
    print()

    return


# noinspection PyPep8Naming
def regression_on_mod_depth_params(data_file="./Data/Sessions/attrib_params_pos_ppi.txt",
                                   n_model_runs=100,
                                   test_size=0.25):
    """
    Regression on link-depth attributes to see if they effect accuracy of LLM response.

    :param data_file: as produced by `drugmechcf.llmx.test_editlink.TestEditLink.write_all_sessions_sample_params`
    :param n_model_runs:
    :param test_size:
    """
    df = pd.read_csv(data_file)
    X = df.drop('response_is_correct', axis=1)
    y = df['response_is_correct']

    n_correct = df['response_is_correct'].sum()
    n_samples = df.shape[0]
    expected_accuracy_min = n_correct / n_samples
    if expected_accuracy_min < 0.5:
        expected_accuracy_min = 1.0 - expected_accuracy_min

    print("Columns:", ", ".join(df.columns.values.tolist()))
    print(f"nbr Samples read = {n_samples:,d}")
    print(f"nbr Correct response = {n_correct}, {n_correct/n_samples:.1%}")
    print()
    print(f"Desired min Accuracy of LR = {expected_accuracy_min:.3f}")
    print()

    model_runs_data = []
    for _ in range(n_model_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        run_data = [accuracy_score(y_test, y_pred), *model.coef_[0].tolist(), model.intercept_[0]]
        model_runs_data.append(run_data)

    model_runs_data = np.asarray(model_runs_data)

    var_names = ['accuracy'] + df.columns.values.tolist()[1:] + ['intercept']
    for i, vnm in enumerate(var_names):
        data = model_runs_data[:, i]
        print(f'{vnm}: mean = {np.mean(data):.5f}, s.d. = {np.std(data, ddof=0):.5f}')
        print()

    return


def get_llm_opt_freq(files_patt: str):
    files = glob.glob(files_patt)
    print()
    print("nbr Files =", len(files))

    opt_cnt = Counter()
    for file in files:
        with open(file) as jf:
            jdict = json.load(jf)
            opt_cnt.update([s['opt_match_metrics']["option_key"] for s in jdict['session']])

    n_samples = sum(opt_cnt.values())
    print("nbr samples read =", n_samples)
    print()

    maxw = max(len(k) for k in opt_cnt)

    for k, v in opt_cnt.most_common():
        print(f"{k:{maxw}s} = {v:5,d} ... {v/n_samples:6.1%}")

    print()
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
#
# [Python]$ python -m drugmechcf.exp.metrics {compile | ...} ...
#
# e.g.
# [Python]$ python -m drugmechcf.exp.metrics compile ../Data/Sessions/Models/*/*.json > cfmetrics.md
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

    # ... compile
    _sub_cmd_parser = _subparsers.add_parser('compile',
                                             help="Gather accuracy metrics from multiple run files.")
    _sub_cmd_parser.add_argument("--csv", action="store_true",
                                 help="Output table to CSV format. Default output is Markdown.")
    _sub_cmd_parser.add_argument('input_files', type=str, nargs="+",
                                 help="Input JSON files, each containing single-run session.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'compile':

        compile_session_accuracy_metrics(_args.input_files, to_csv=_args.csv)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
