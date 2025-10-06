"""
Reading Counterfactuals data.
See also: For usage in evaluating LLMs,
    `drugmechcf.llmx.test_addlink.TestAddLink.create_sample_task()`
    `drugmechcf.llmx.test_editlink.TestEditLink.create_sample_task()`
"""

from collections import Counter
import json
import os
from typing import Any

from drugmechcf.data.drugmechdb import DrugMechDB, load_drugmechdb
from drugmechcf.data.primekg import PrimeKG, RELATION_CONTRA_INDICATION, load_primekg

from drugmechcf.kgproc.addlink import create_new_moa_add_link

from drugmechcf.llm.prompt_types import EditLinkInfo
from drugmechcf.llmx.prompts_common import (get_drugmechdb_moa_for_prompt)
from drugmechcf.llmx.test_addlink import AddLinkTask
from drugmechcf.llmx.test_editlink import EditLinkTask

from drugmechcf.utils.misc import pp_underlined_hdg


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def create_sample_task(sample_data: dict[str, Any], drugmechdb: DrugMechDB) -> AddLinkTask | EditLinkTask:
    sample_data = sample_data.copy()
    edit_link_info = EditLinkInfo(**sample_data["edit_link_info"])
    sample_data["edit_link_info"] = edit_link_info

    assert sample_data["query_type"] in ["ADD_LINK", "CHANGE_LINK", "DELETE_LINK"]

    if sample_data["query_type"] == "ADD_LINK":
        task = AddLinkTask(**sample_data)

        if not task.is_negative_sample:
            # Get the source and target MoA
            source_moa = drugmechdb.get_indication_graph_with_id(edit_link_info.source_moa_id)
            target_moa = drugmechdb.get_indication_graph_with_id(edit_link_info.target_moa_id)
            # Add synthesized MoA resulting from adding new link
            task.moa = create_new_moa_add_link(drugmechdb,
                                               source_moa, edit_link_info.source_node,
                                               edit_link_info.new_relation,
                                               target_moa, edit_link_info.target_node,
                                               source_moa_drug_node=task.drug_id,
                                               target_moa_disease_node=task.disease_id
                                               )

    else:
        task = EditLinkTask(**sample_data)

        # Add the MoA ... source and target are the same
        task.moa = drugmechdb.get_indication_graph_with_id(edit_link_info.source_moa_id)

    return task


def pp_indented_moa(moa, indent="    "):
    moastr = get_drugmechdb_moa_for_prompt(moa)
    moalines = [indent + line for line in moastr.splitlines()]
    print(*moalines, sep="\n")
    return


def pp_sample_factual(sdict: dict[str, Any], drugmechdb: DrugMechDB, pkg: PrimeKG):

    print(f"Drug: {sdict['drug_name']} ({sdict['drug_id']})")
    print(f"Disease: {sdict['disease_name']} ({sdict['disease_id']})")

    if sdict["is_negative_sample"]:

        print("Negative sample")
        if pkg.has_edge(sdict['drug_id'], sdict['disease_id'], RELATION_CONTRA_INDICATION):
            print("  - Contra-indicated")
        else:
            print("  - No relation between Drug and Disease")

    else:

        moa = drugmechdb.get_indication_graph_with_id(sdict["moa_id"])
        print("Expected MoA:")
        pp_indented_moa(moa)

    return


def pp_sample_addlink(task: AddLinkTask):

    print(f"Drug: {task.drug_name} ({task.drug_id})")
    print(f"Disease: {task.disease_name} ({task.disease_id})")
    print("New Link:")
    eli = task.edit_link_info
    print(f"    {eli.source_node_type}: {eli.source_node_name} ({eli.source_node})",
          f" -- {eli.new_relation} -> ",
          f"{eli.target_node_type}: {eli.target_node_name} ({eli.target_node})")

    if task.is_negative_sample:
        print("Negative sample from PrimeKG")
    else:
        print("Expected (synthesized) MoA:")
        pp_indented_moa(task.moa)

    return


def pp_sample_editlink(task: EditLinkTask):
    print(f"Drug: {task.drug_name} ({task.drug_id})")
    print(f"Disease: {task.disease_name} ({task.disease_id})")
    print("Basis MoA:")
    pp_indented_moa(task.moa)
    print("Inverted" if task.query_type == "CHANGE_LINK" else "Deleted",
          "Link", "(negative sample):" if task.is_negative_sample else "(positive sample):")
    eli = task.edit_link_info
    print(f"    {eli.source_node_type}: {eli.source_node_name} ({eli.source_node})",
          f" -- {eli.new_relation} -> " if task.query_type == "CHANGE_LINK" else " ---> ",
          f"{eli.target_node_type}: {eli.target_node_name} ({eli.target_node})")

    return


def pp_samples(cf_samples_file: str, count=5, start=0):
    """
    Read a counterfactuals samples file and pprint some samples.
    :param cf_samples_file: Path to counterfactual samples JSON file
    :param count: How many samples to pprint
    :param start: Index from where to start printing samples
    """

    print("Loading data ...", flush=True)

    drugmechdb = load_drugmechdb()
    pkg = None

    with open(cf_samples_file) as jf:
        sdata = json.load(jf)

    assert isinstance(sdata, list), f"Expected to see a list. Unexpected type {type(sdata)}."

    query_type = sdata[0]["query_type"]
    is_negative_sample = sdata[0]["is_negative_sample"]

    is_factual = query_type == "known_moa"

    if is_factual and is_negative_sample:
        pkg = load_primekg()

    print()
    print("Counterfactuals File:", os.path.basename(cf_samples_file))
    print(f"    query_type = {query_type}")

    if not is_factual:
        src_node_type_cnts = Counter(s["edit_link_info"]["source_node_type"] for s in sdata)
        is_surface_level = set(src_node_type_cnts.keys()) <= {"Drug", "ChemicalSubstance"}
        print(f"    is_surface_level = {is_surface_level}")

    print(f"    is_negative_sample = {is_negative_sample}")
    print(f"    n_samples = {len(sdata):,d}")
    print()

    print()
    for i, sample_data in enumerate(sdata[start : start + count], start=1):
        pp_underlined_hdg(f"Sample {i}:")

        if is_factual:
            pp_sample_factual(sample_data, drugmechdb, pkg)
        else:
            task = create_sample_task(sample_data, drugmechdb)
            if task.query_type == "ADD_LINK":
                pp_sample_addlink(task)
            else:
                pp_sample_editlink(task)

        print("=======\n")

    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.data.cfdata examples PATH_TO_CF_SAMPLES_FILE

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Counterfactuals data',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... examples
    _sub_cmd_parser = _subparsers.add_parser('examples',
                                             help="Print summary and some samples from a CF Samples file.")
    _sub_cmd_parser.add_argument('-s', '--start', type=int, default=0,
                                 help="Index of first example.")
    _sub_cmd_parser.add_argument('-c', '--count', type=int, default=5,
                                 help="Max nbr of examples.")
    _sub_cmd_parser.add_argument('cf_samples_file',
                                 help="Path to query samples JSON file")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'examples':

        pp_samples(_args.cf_samples_file, _args.count, start=_args.start)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
