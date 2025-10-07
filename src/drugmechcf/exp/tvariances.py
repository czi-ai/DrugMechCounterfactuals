"""
Compute mean, sd for metrics across runs, for testing Factual knowledge of MoAs
"""

from collections import defaultdict
import os
from typing import List, Union

import numpy as np
import pandas as pd

from drugmechcf.llm.test_common import extract_main_moa_metrics

from drugmechcf.llm.prompt_types import PromptStyle
from drugmechcf.llm.test_dmdb import (test_positives_batch as test_positives_batch_known_moa,
                                      test_negatives_batch as test_negatives_batch_known_moa
                                      )

from drugmechcf.utils.misc import pp_underlined_hdg, suppressed_stdout, pp_funcargs

from drugmechcf.exp.cfvariances import get_temp_json

from drugmechcf.utils.projconfig import get_project_config


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#   Functions: Misc
# -----------------------------------------------------------------------------


def pp_cum_metrics_stats(cum_metrics: dict[str, List[Union[int, float]]], keys: List[str] = None):

    cum_metrics = dict((k, np.asarray(v)) for k, v in cum_metrics.items() if keys is None or k in keys)

    df = pd.DataFrame.from_records([(k, np.nanmean(v), np.nanstd(v), np.nanmin(v), np.nanmax(v))
                                    for k, v in cum_metrics.items()
                                    ],
                                   columns=["Metric", "Mean", "StdDev", "Min", "Max"],
                                   index="Metric")
    print(df.to_markdown(intfmt=',d', floatfmt=',.3f'))
    print()
    tot_llm = sum(cum_metrics['LLM Calls'])
    print(f"Total nbr LLM calls = {tot_llm:,d}")
    print()

    return tot_llm


# -----------------------------------------------------------------------------
#   Functions: known_moa
# -----------------------------------------------------------------------------


def run_known_moa(positives: bool, prompt_style, run_nbr: int = 0, samples_file: str = None):

    print(f"-- Running `run_known_moa({positives=}, {prompt_style=})`  run {run_nbr} ...", flush=True)

    tmp_json = get_temp_json(f"known_moa_{positives}_{prompt_style}")

    count = 1000
    prompt_version = 2

    with suppressed_stdout():
        if positives:
            test_positives_batch_known_moa(count=count,
                                           prompt_version=prompt_version,
                                           prompt_style=prompt_style,
                                           randomize=True,
                                           seed=42,
                                           json_file=tmp_json,
                                           pos_samples_file=samples_file,
                                           )
        else:
            test_negatives_batch_known_moa(count=count,
                                           prompt_version=prompt_version,
                                           prompt_style=prompt_style,
                                           seed=42,
                                           json_file=tmp_json,
                                           neg_samples_file=samples_file,
                                           )

    tmp_metrics = extract_main_moa_metrics(tmp_json)

    # Delete the tmp JSON file
    os.remove(tmp_json)

    return tmp_metrics


def multirun_stats_known_moa(nruns: int = 5):
    """
    Run `nruns` sessions for factual queries (known_moa) to compute response metric variances.
    Output is on stdout.

    :param nruns:
    """

    pp_funcargs(multirun_stats_known_moa)

    samples_dir = get_project_config().get_input_data_dir()

    for positives in [True, False]:

        samples_file = os.path.join(samples_dir, f"factuals_{'pos' if positives else 'neg'}_r1k.json")

        for prompt_style in [x.name for x in PromptStyle]:

            cum_metrics = defaultdict(list)

            for n in range(nruns):

                tmp_metrics = run_known_moa(positives, prompt_style,
                                            run_nbr=n + 1,
                                            samples_file=samples_file,
                                            )

                for k, v in tmp_metrics.items():
                    cum_metrics[k].append(v)

            print("\n")
            pp_underlined_hdg(f"Known MoA: Summary stats for {positives=}, {prompt_style=}")
            print("Nbr runs =", nruns)
            print()
            pp_cum_metrics_stats(cum_metrics)
            print(flush=True)

    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
#
# [Python]$ python -m exp.tvariances {known_moa | ...}
#
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

    # ... known_moa
    _sub_cmd_parser = _subparsers.add_parser('known_moa',
                                             help="Test known_moa.")
    _sub_cmd_parser.add_argument('count', nargs="?", type=int, default=5,
                                 help="Nbr of runs to summarize across.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'known_moa':

        multirun_stats_known_moa(_args.count)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
