"""
Stats on Drug Mechanisms Counterfactuals data
"""

from collections import Counter, defaultdict
import glob
import json
from typing import Any

import pandas as pd

from drugmechcf.utils.misc import ppmd_counts_df


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def analyze_one_data_file(data_json_file: str):

    with open(data_json_file) as jf:
        sdata = json.load(jf)

    assert isinstance(sdata, list), f"Expected to see a list. Unexpected type {type(sdata)}."

    n_samples = len(sdata)
    query_type = sdata[0]["query_type"]
    is_negative_sample = sdata[0]["is_negative_sample"]

    src_node_type_cnts = Counter(s["edit_link_info"]["source_node_type"] for s in sdata)
    tgt_node_type_cnts = Counter(s["edit_link_info"]["target_node_type"] for s in sdata)

    is_surface_level = set(src_node_type_cnts.keys()) <= {"Drug", "ChemicalSubstance"}

    cfdict = dict(data_file=data_json_file,
                  n_samples=n_samples,
                  query_type=query_type,
                  is_negative_sample=is_negative_sample,
                  is_surface_level=is_surface_level,
                  src_node_type_cnts=src_node_type_cnts,
                  tgt_node_type_cnts=tgt_node_type_cnts,
                  )

    return cfdict


def pp_counter(hdg: str, cntr: Counter, key_col: str, counts_col: str):
    print(hdg)
    print()

    df = pd.DataFrame.from_records(sorted(cntr.items()), columns=[key_col, counts_col])

    ppmd_counts_df(df, counts_col, add_pct_total=True)
    return


def pp_data_file_stats(cfdict: dict[str, Any], hdg_prefx: str = None):
    if hdg_prefx is None:
        hdg_prefx = ""
    elif not hdg_prefx.endswith(" "):
        hdg_prefx += " "

    print(f"{hdg_prefx}File:", cfdict["data_file"])
    print(f"    query_type = {cfdict['query_type']}")
    print(f"    is_surface_level = {cfdict['is_surface_level']}")
    print(f"    is_negative_sample = {cfdict['is_negative_sample']}")
    print(f"    n_samples = {cfdict['n_samples']:,d}")
    print()

    print()
    pp_counter("Source Node type counts:", cfdict['src_node_type_cnts'],
               "EntityType", "count")

    pp_counter("Target Node type counts:", cfdict['tgt_node_type_cnts'],
               "EntityType", "count")

    return


def analyze_data_files(data_dir: str = "../Data/Counterfactuals"):
    data_files = [f for f in sorted(glob.glob(f"{data_dir}/*.json")) if not f.startswith("factuals")]

    print()
    print("nbr Data files found =", len(data_files))
    print()

    cfdicts = []
    smry_counts = defaultdict(lambda: [0, 0])

    for i, data_file in enumerate(data_files, start=1):
        cfdict = analyze_one_data_file(data_file)
        cfdicts.append(cfdict)
        print()
        pp_data_file_stats(cfdict, hdg_prefx=f"[{i}]")
        print("----------------")

        scounts = smry_counts[(cfdict['query_type'], cfdict['is_surface_level'])]
        scounts[cfdict['is_negative_sample']] = cfdict['n_samples']

    # Summary table

    df = pd.DataFrame.from_records([[*k, *v] for k, v in sorted(smry_counts.items())],
                                   columns=["Query Type", "is Surface Level?", "Positives", "Negatives"])

    print()
    print("Dataset Summary:")
    print()
    print(df.to_markdown(index=False, intfmt=","))
    print()

    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
#
# [Python]$ python -m drugmechcf.analyze.cfstats stats
#
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Analyze Drug Mechanisms Counterfactuals dataset.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... stats
    _sub_cmd_parser = _subparsers.add_parser('stats',
                                             help="Summary stats for dataset.")
    _sub_cmd_parser.add_argument('datadir', type=str, nargs="?",
                                 default="../Data/Counterfactuals",
                                 help="Path to dir containing JSON data files.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'stats':

        analyze_data_files(_args.datadir)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
