"""
Test DrugMechDB MoA against LLM
"""

import dataclasses
import json
from typing import List, Tuple

import numpy as np

from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.data.primekg import (RELATION_INDICATION, RELATION_CONTRA_INDICATION, RELATION_OFF_LABEL,
                                     DRUG_DISEASE_RELATIONS, PrimeKG)

from drugmechcf.llm.prompt_types import DrugDiseasePromptInfo, QueryType, PromptStyle
from drugmechcf.llm.drugmechdb_prompt_builder import DrugMechPromptBuilder
from drugmechcf.llm.primekg_prompt_builder import PrimeKGPromptBuilder

from drugmechcf.llm.openai import CompletionOutput, get_openai_client, OpenAICompletionClient
from drugmechcf.graphmatch.graphmatcher import GraphMatchScore, BasicGraphMatcher
from drugmechcf.llm.test_common import ENTITY_TYPE_EQUIVALENCES, test_moa_match, pprint_accumulated_metrics

from drugmechcf.utils.misc import NpEncoder, pp_funcargs, pp_underlined_hdg, check_output_file_dir


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

DEBUG_NO_LLM = False


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def get_default_llm_client(**llmparams) -> OpenAICompletionClient:
    """
    Return the `OpenAICompletionClient` that will be tested on Factual knowledge of MoAs.
    The default model (i.e. with `llmparams`=None) is a gpt-4o model.
    Provide args (see `drugmechcf.llm.openai.get_openai_client()`) or modify code to test other models.
    """
    llm_client = get_openai_client(**llmparams)
    return llm_client


def test_positives_batch(count: int = 10,
                         randomize: bool = False,
                         seed: int = 42,
                         match_entity_types=True,
                         use_etype_equivs=True,
                         prompt_style: str = "ANONYMIZED_DISEASE",
                         prompt_version: int = 0,
                         single_llm_session: bool = False,
                         debug_no_llm: bool = False,
                         json_file: str = None,
                         pos_samples_file: str = None,
                         **llmparams
                         ):
    """
    Compares `count` MoA's from DrugMechDB against ChatGPT.

    :param count: Nbr MoA's to test
    :param randomize: IF True THEN MoAs are selected at random, ELSE First `count` MoAs selected.

    :param match_entity_types: IF True THEN Node-matcher must match entity-types of interior nodes.
    :param use_etype_equivs: IF True and `match_entity_types`
        THEN use `ENTITY_TYPE_EQUIVALENCES` as equivalent Entity-Types

    :param prompt_version: Which version of prompts to use

    :param prompt_style: Integer rep of PromptStyle

    :param json_file: [opt] Output data to this JSON file

    :param debug_no_llm: [opt] Don't call LLM

    :param pos_samples_file: [opt] File containing positive samples.
        Otherwise positive samples obtained from DrugMechDB.

    :param **llmparams: [opt] args passed on to `drugmechcf.llm.test_dmdb.get_default_llm_client()`
        for testing specific LLM models.
    """

    pp_funcargs(test_positives_batch)

    if json_file is not None:
        check_output_file_dir(json_file)

    # Convert to enum value
    prompt_style = PromptStyle[prompt_style.upper()]

    jdict = None
    if json_file is not None:
        jdict = dict(args=dict(count=count,
                               random=randomize,
                               seed=seed,
                               match_entity_types=match_entity_types,
                               use_etype_equivs=use_etype_equivs,
                               prompt_version=prompt_version,
                               prompt_style=prompt_style.name,
                               single_llm_session=single_llm_session,
                               ),
                     session=[])

    prompt_builder = DrugMechPromptBuilder()
    drugmechdb = prompt_builder.drugmechdb

    if single_llm_session and not debug_no_llm:
        llm = get_default_llm_client(**llmparams)
    else:
        llm = None

    if pos_samples_file is None:
        if randomize:
            rng = np.random.default_rng(seed)
            dmdb_moa_indices = rng.permutation(drugmechdb.nbr_indications())
        else:
            dmdb_moa_indices = np.arange(drugmechdb.nbr_indications())

    else:
        dmdb_moa_indices = get_dmdb_indices_from_samples_file(pos_samples_file, drugmechdb)

    prompt_examples = prompt_builder.get_llm_prompt_example_nodes(QueryType.KNOWN_MOA, prompt_version)

    cum_metrics = None
    n_tested = 0
    n_valid_llm_response = 0

    n_llm_moas_skipped = 0
    n_llm_moas_skipped_llm_response_inc = 0

    for idx in dmdb_moa_indices:

        if n_tested >= count:
            break

        dmdb_moa = drugmechdb.indication_graphs[idx]

        # We describe the disease using one or more 'interior' nodes
        # Skip MoAs that do not have any more nodes
        if dmdb_moa.number_of_nodes() <= 3:
            n_llm_moas_skipped += 1
            continue

        drug_id = drugmechdb.get_moa_source_drug_node(dmdb_moa)
        disease_id = drugmechdb.get_moa_target_disease_node(dmdb_moa)

        # Skip because this (drug_id, disease_id) has multiple MoA's,
        # making comparison with LLM response potentially 'unfair' (LLM may respond with different MoA).
        if len(drugmechdb.get_indication_graphs(drug_id, disease_id)) > 1:
            n_llm_moas_skipped += 1
            continue

        # Skip if it is one of the examples in the prompt
        if (drug_id, disease_id) in prompt_examples:
            n_llm_moas_skipped += 1
            continue

        grmatcher = BasicGraphMatcher(verbosity=1)
        if use_etype_equivs:
            grmatcher.set_entity_type_equivalences(ENTITY_TYPE_EQUIVALENCES)

        prompt_info, llm_response, metrics = test_moa_llm_match(drug_id, disease_id, dmdb_moa=dmdb_moa,
                                                                prompt_builder=prompt_builder,
                                                                llm=llm,
                                                                grmatcher=grmatcher,
                                                                match_entity_types=match_entity_types,
                                                                use_etype_equivs=use_etype_equivs,
                                                                prompt_style=prompt_style,       # type: Ignore
                                                                prompt_version=prompt_version,
                                                                n_tested=n_tested + 1,
                                                                debug_no_llm=debug_no_llm,
                                                                verbose=True,
                                                                **llmparams
                                                                )

        if prompt_info is None:
            n_llm_moas_skipped += 1
            continue

        n_tested += 1

        if debug_no_llm or not llm_response.is_complete_response():

            if not debug_no_llm:
                print("--- Skipping because of incomplete response from LLM ---")
                n_llm_moas_skipped_llm_response_inc += 1

            if jdict is not None:
                j_moa_data = dict(prompt_info=prompt_info.to_serialized(),
                                  llm_response_error=llm_response.finish_reason if llm_response else None
                                  )
                jdict["session"].append(j_moa_data)

            continue

        n_valid_llm_response += 1

        j_moa_data = None
        if jdict is not None:
            j_moa_data = dict(prompt_info=prompt_info.to_serialized(),
                              llm_response=llm_response.message,
                              metrics=dataclasses.asdict(metrics) if metrics is not None else None   # type: Ignore
                              )
            jdict["session"].append(j_moa_data)

        if metrics is not None:
            cum_metrics = metrics.accumulate_metrics(cum_metrics)
            if j_moa_data:
                j_moa_data["node_match_scores"] = grmatcher.get_last_node_match_scores_simpledict()

    # /

    print()
    print("==========================================================")
    print()

    llm_metrics = pprint_accumulated_metrics(cum_metrics, n_tested, n_valid_llm_response,
                                             n_llm_moas_skipped, n_llm_moas_skipped_llm_response_inc)

    if json_file is not None:
        jdict["metrics"] = llm_metrics

        with open(json_file, "w") as f:
            json.dump(jdict, f, indent=4, cls=NpEncoder)   # type: Ignore

        print()
        print("Session data written to:", json_file)
        print()

    return


def get_dmdb_indices_from_samples_file(pos_samples_file: str, drugmechdb) -> list[int]:
    with open(pos_samples_file) as f:
        samples = json.load(f)

    assert samples[0]["query_type"] == "known_moa", "Samples file is not for Factuals!"
    assert not samples[0]["is_negative_sample"], "Samples are not Positive!"

    dmdb_moa_indices = [drugmechdb.moa_id_to_gidx[s["moa_id"]] for s in samples]

    return dmdb_moa_indices


def test_moa_llm_match(drug_id: str, disease_id: str,
                       dmdb_moa: MoaGraph = None,
                       prompt_builder: DrugMechPromptBuilder = None,
                       llm: OpenAICompletionClient = None,
                       grmatcher: BasicGraphMatcher = None,
                       match_entity_types=True,
                       use_etype_equivs=False,
                       prompt_style: PromptStyle = PromptStyle.ANONYMIZED_DISEASE,
                       prompt_version: int = 0,
                       n_tested: int = 0,
                       verbose: bool = True,
                       verbose_llm: bool = False,
                       main_heading: str = None,
                       debug_no_llm: bool = False,
                       **llmparams
                       )\
        -> Tuple[DrugDiseasePromptInfo | None, CompletionOutput | None, GraphMatchScore | None]:
    """
    Generate LLM Prompt, Query LLM, Match response against MoA.

    A GraphMatchScore is returned even when the LLM response says "No MoA".

    :return:
        - DrugDiseasePromptInfo | None if no LLM prompt possible
        - CompletionOutput (LLM response) | None if no LLM prompt possible OR debug_no_llm
        - GraphMatchScore | None if: no LLM prompt OR incomplete LLM response OR debug_no_llm
    """

    if prompt_builder is None:
        prompt_builder = DrugMechPromptBuilder()

    prompt_info = prompt_builder.get_full_drug_disease_prompt_from_moa(dmdb_moa, drug_id, disease_id,
                                                                       query_type=QueryType.KNOWN_MOA,
                                                                       prompt_style=prompt_style,
                                                                       prompt_version=prompt_version,
                                                                       verbose=verbose)

    if not prompt_info:
        return None, None, None

    if verbose:
        print()
        if main_heading:
            pp_underlined_hdg(main_heading)
        else:
            pp_underlined_hdg(f"[{n_tested}] {dmdb_moa.get_qualified_node_name(drug_id)} on "
                              + dmdb_moa.get_qualified_node_name(disease_id),
                              overline=True, linechar='=')

        print("LLM sub-Prompt:", prompt_info.drug_disease_subprompt, sep="\n  ")
        print()
        pp_underlined_hdg("DrugMechDB MoA")
        dmdb_moa.pprint(with_summary=True)
        print("---\n")

    if debug_no_llm:
        return prompt_info, None, None

    if llm is None:
        llm = get_default_llm_client(**llmparams)

    llm_response = llm(user_prompt=prompt_info.full_prompt, verbose=verbose_llm)

    if verbose:
        pp_underlined_hdg("Response from LLM - ChatGPT-4o")
        if llm_response.is_complete_response():
            print(llm_response.message)
        else:
            print("LLM stop reason =", llm_response.finish_reason)
        print("---\n")

    if not llm_response.is_complete_response():
        return prompt_info, llm_response, None

    dis_name_in_llm_response = prompt_builder.get_dis_name_in_llm_response(prompt_style, prompt_info.disease_name)
    llm_graph_name = f"ChatGPT-4o.MoA: {prompt_info.drug_name} treats {prompt_info.disease_name}"

    if grmatcher is None:
        grmatcher = BasicGraphMatcher(verbosity=1)
        if use_etype_equivs:
            grmatcher.set_entity_type_equivalences(ENTITY_TYPE_EQUIVALENCES)

    match_score = test_moa_match(grmatcher, dmdb_moa, llm_response.message,
                                 prompt_info.drug_name, dis_name_in_llm_response,
                                 prompt_builder.formal_to_kg_entity_type_names(),
                                 llm_graph_name,
                                 match_entity_types=match_entity_types,
                                 ref_nodes_in_prompt=prompt_info.disease_prompt_nodes,
                                 verbose=verbose
                                 )

    return prompt_info, llm_response, match_score


# -----------------------------------------------------------------------------
#   Functions: Negative Samples
# -----------------------------------------------------------------------------


def test_negatives_batch(count: int = 10,
                         include_contra_indications: bool = True,
                         prompt_style: str = "ANONYMIZED_DISEASE",
                         prompt_version: int = 1,
                         seed: int = None,
                         json_file: str = None,
                         debug_no_llm: bool = False,
                         neg_samples_file: str = None,
                         **llmparams
                         ):
    """
    Compares `count` MoA's from DrugMechDB against ChatGPT.

    :param count: Nbr -ive Drug-Disease pairs to test

    :param include_contra_indications: IF True THEN half the (Drug, Disease) will be from contra-indications.

    :param prompt_version: Which version of prompts to use

    :param prompt_style: Integer rep of PromptStyle

    :param seed: Seed to use in np random generator

    :param json_file: [opt] Output data to this JSON file

    :param debug_no_llm: [opt] Don't call LLM

    :param neg_samples_file: [opt] File containing negative samples.
        Otherwise negative samples obtained from PrimeKG.

    :param **llmparams: [opt] args passed on to `drugmechcf.llm.test_dmdb.get_default_llm_client()`
        for testing specific LLM models.
    """

    pp_funcargs(test_negatives_batch)

    if json_file is not None:
        check_output_file_dir(json_file)

    # Convert to enum value
    prompt_style = PromptStyle[prompt_style.upper()]

    prompt_builder = PrimeKGPromptBuilder()
    primekg = prompt_builder.primekg

    prompt_builder_dmdb = DrugMechPromptBuilder()
    prompt_examples = prompt_builder_dmdb.get_llm_prompt_example_nodes(QueryType.KNOWN_MOA, prompt_version)

    # ---[1]--- Get the -ive samples

    n_pairs_from_contra = 0
    # Include upto ~ half the pairs from Contra-indication
    if include_contra_indications:
        n_pairs_from_contra = count // 2

    n_pairs_from_no_reln = count - n_pairs_from_contra

    print()
    if neg_samples_file is None:
        print("From `get_negative_samples()`:")
        # Get extra pairs to account for skips
        drug_disease_pairs_contra, drug_disease_pairs_no_reln = \
            get_negative_samples(primekg, int(2.2 * n_pairs_from_contra), int(2.2 * n_pairs_from_no_reln), seed)

    else:
        print("From `get_negative_drug_disease_pairs_from_file()`:")
        drug_disease_pairs_contra, drug_disease_pairs_no_reln = \
            get_negative_drug_disease_pairs_from_file(neg_samples_file, n_pairs_from_contra, n_pairs_from_no_reln)

    print(f"    {len(drug_disease_pairs_contra) = }")
    print(f"    {len(drug_disease_pairs_no_reln) = }")
    print()

    # ---[2]--- Test each sample

    jdict = None
    if json_file is not None:
        jdict = dict(args=dict(negative_samples=True,
                               count=count,
                               seed=seed,
                               include_contra_indications=include_contra_indications,
                               prompt_version=prompt_version,
                               prompt_style=prompt_style.name
                               ),
                     samples_tested=dict(),
                     session=[])

    cum_metrics = None
    n_tested = 0    # start at 1, so it can be used in `test_moa_llm_match`
    n_valid_llm_response = 0

    n_samples_skipped = 0
    n_skipped_llm_response_inc = 0

    # Try to match targeted count from each group: drug_disease_pairs_contra, drug_disease_pairs_no_reln

    n_pairs_tested = []

    # We don't want to repeat any Drug-Disease pairs (some have multiple MoAs)
    tested_samples = set()

    for target_count, drug_disease_pairs in zip([n_pairs_from_contra, n_pairs_from_no_reln],
                                                [drug_disease_pairs_contra, drug_disease_pairs_no_reln]):

        if target_count == 0:
            n_pairs_tested.append(0)
            continue

        n_tested_this_target = 0

        for drug_id, disease_id in drug_disease_pairs:

            if n_tested_this_target >= target_count or n_tested >= count:
                break

            # Skip if it is one of the examples in the prompt
            if (drug_id, disease_id) in prompt_examples:
                print(f"... skipping {drug_id=}, {disease_id=} as they are in prompt examples.")
                n_samples_skipped += 1
                continue

            # Skip if already tested
            if (drug_id, disease_id) in tested_samples:
                continue
            else:
                tested_samples.add((drug_id, disease_id))

            prompt_info, llm_response, metrics = test_negative_sample(drug_id, disease_id,
                                                                      primekg_prompt_builder=prompt_builder,
                                                                      drugmechdb_prompt_builder=prompt_builder_dmdb,
                                                                      llm=None,
                                                                      prompt_style=prompt_style,       # type: Ignore
                                                                      prompt_version=prompt_version,
                                                                      n_tested=n_tested + 1,
                                                                      debug_no_llm=debug_no_llm,
                                                                      verbose=True,
                                                                      **llmparams
                                                                      )

            if prompt_info is None:
                n_samples_skipped += 1
                continue

            n_tested += 1
            n_tested_this_target += 1

            if debug_no_llm or not llm_response.is_complete_response():

                if not debug_no_llm:
                    print("--- Skipping because of incomplete response from LLM ---")
                    n_skipped_llm_response_inc += 1

                if jdict is not None:
                    j_moa_data = dict(prompt_info=prompt_info.to_serialized(),
                                      llm_response_error=llm_response.finish_reason if llm_response else None
                                      )
                    jdict["session"].append(j_moa_data)

                continue

            n_valid_llm_response += 1

            if jdict is not None:
                j_moa_data = dict(prompt_info=prompt_info.to_serialized(),
                                  llm_response=llm_response.message,
                                  metrics=dataclasses.asdict(metrics) if metrics is not None else None   # type: Ignore
                                  )
                jdict["session"].append(j_moa_data)

            if metrics is not None:
                cum_metrics = metrics.accumulate_metrics(cum_metrics)

        # /

        # Record how many were tested
        n_pairs_tested.append(n_tested_this_target)

    # /End for

    print()
    print("==========================================================")
    print()

    llm_metrics = pprint_accumulated_metrics(cum_metrics, n_tested, n_valid_llm_response,
                                             n_samples_skipped, n_skipped_llm_response_inc)

    if json_file is not None:
        jdict["samples_tested"]["samples_from_contra"] = n_pairs_tested[0]
        jdict["samples_tested"]["samples_no_relation"] = n_pairs_tested[1]
        jdict["metrics"] = llm_metrics

        with open(json_file, "w") as f:
            json.dump(jdict, f, indent=4, cls=NpEncoder)   # type: Ignore

        print()
        print("Session data written to:", json_file)
        print()

    return


def get_negative_drug_disease_pairs_from_file(neg_samples_file: str, n_pairs_from_contra, n_pairs_from_no_reln):
    with open(neg_samples_file) as f:
        samples = json.load(f)

    assert samples[0]["query_type"] == "known_moa", "Samples file is not for Factuals!"
    assert samples[0]["is_negative_sample"], "Samples are not Negative!"

    drug_disease_pairs = [(s["drug_id"], s["disease_id"]) for s in samples]

    # First half the samples in the file are from contra and the remaining from no_reln
    n_samples_half = len(drug_disease_pairs) // 2
    n_pairs_from_contra = min(n_pairs_from_contra, n_samples_half)
    n_pairs_from_no_reln = min(n_pairs_from_no_reln, n_samples_half)

    return (drug_disease_pairs[:n_pairs_from_contra],
            drug_disease_pairs[n_samples_half:n_samples_half + n_pairs_from_no_reln])


def get_negative_samples(primekg: PrimeKG,
                         n_pairs_from_contra: int,
                         n_pairs_from_no_reln: int,
                         seed: int = None
                         )\
        -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Get -ive samples of (head-node, tail-node) from PrimeKG.

    :return:
        - drug_disease_pairs_contra: List[(u, v), ...]
            Edges with Contra-indication relation, but no Indication / Off-label relation
        - drug_disease_pairs_no_reln: List[(u, v), ...]
            Edges with no Drug-Disease relation
    """

    drug_disease_pairs_contra: List[Tuple[str, str]] = []

    if n_pairs_from_contra > 0:
        # Remove Indication and Off-label edges from all Contra-indication edges, as some have both
        all_contra = [(u, v) for u, v in primekg.get_all_relation_edges(RELATION_CONTRA_INDICATION)
                      if not (primekg.has_edge(u, v, key=RELATION_INDICATION) or
                              primekg.has_edge(u, v, key=RELATION_OFF_LABEL))]

        if n_pairs_from_contra < len(all_contra):
            # noinspection PyTypeChecker
            drug_disease_pairs_contra = np.random.default_rng(seed).choice(all_contra, size=n_pairs_from_contra,
                                                                           replace=False).tolist()
        else:
            # Make a random permutaion of all the pairs
            # noinspection PyTypeChecker
            drug_disease_pairs_contra = np.random.default_rng(seed).permutation(all_contra).tolist()

    drug_disease_pairs_no_reln = primekg.get_negative_drug_disease_samples(n_pairs_from_no_reln, seed=seed)

    return drug_disease_pairs_contra, drug_disease_pairs_no_reln


def test_negative_sample(drug_id: str, disease_id: str,
                         primekg_prompt_builder: PrimeKGPromptBuilder,
                         drugmechdb_prompt_builder: DrugMechPromptBuilder,
                         llm: OpenAICompletionClient = None,
                         grmatcher: BasicGraphMatcher = None,
                         prompt_style: PromptStyle = PromptStyle.ANONYMIZED_DISEASE,
                         prompt_version: int = 0,
                         n_tested: int = 0,
                         debug_no_llm: bool = False,
                         verbose: bool = True,
                         **llmparams
                         ) \
        -> Tuple[DrugDiseasePromptInfo | None, CompletionOutput | None, GraphMatchScore | None]:
    """
    For a -ive sample extracted from PrimeKG,
    Generate LLM Prompt, Query LLM, Match response against MoA.

    A GraphMatchScore is returned even when the LLM response says "No MoA".

    :return:
        - DrugDiseasePromptInfo | None if no LLM prompt possible
        - CompletionOutput (LLM response) | None if no LLM prompt possible
        - GraphMatchScore | None if: no LLM prompt or incomplete LLM response
    """

    # ---[1]--- Get Disease-sub-prompt from PrimeKG

    prompt_info = primekg_prompt_builder.build_prompt_info(drug_id, disease_id, prompt_style, prompt_version,
                                                           is_negative_sample=True)

    disease_subprompt, prompt_nodes = primekg_prompt_builder.get_disease_prompt_data(prompt_info, verbose=False)

    if not disease_subprompt:
        if verbose:
            print(f"... skipping {drug_id=}, {disease_id=} due to empty prompt.")
        return None, None, None

    prompt_info.disease_subprompt = disease_subprompt
    prompt_info.disease_prompt_nodes = prompt_nodes

    # ---[2]--- Get the full prompt from DrugMechDB's prompt-builder

    llm_full_prompt, drug_disease_prompt = drugmechdb_prompt_builder.build_full_prompt(prompt_info)

    prompt_info.full_prompt = llm_full_prompt
    prompt_info.drug_disease_subprompt = drug_disease_prompt

    if debug_no_llm:
        return prompt_info, None, None

    # ---[3]--- Query LLM

    if llm is None:
        llm = get_default_llm_client(**llmparams)

    if grmatcher is None:
        grmatcher = BasicGraphMatcher()

    llm_response = llm(user_prompt=prompt_info.full_prompt)

    if verbose:
        primekg = primekg_prompt_builder.primekg
        print()
        pp_underlined_hdg(f"[{n_tested}] Neg Sample: {primekg.get_qualified_node_name(drug_id)} on "
                          + primekg.get_qualified_node_name(disease_id),
                          overline=True, linechar='=')

        print("LLM sub-Prompt:", prompt_info.drug_disease_subprompt, sep="\n  ")
        print()
        pp_underlined_hdg("PrimeKG Drug-Disease Relations:")
        n_relns = 0
        for reln in DRUG_DISEASE_RELATIONS:
            if primekg.has_edge(drug_id, disease_id, key=reln):
                print("    +", reln)
                n_relns += 1
        if n_relns == 0:
            print(" ... None.")
        print("---\n")

        pp_underlined_hdg("Response from LLM - ChatGPT-4o")
        if llm_response.is_complete_response():
            print(llm_response.message)
        else:
            print("LLM stop reason =", llm_response.finish_reason)
        print("---\n")

    if not llm_response.is_complete_response():
        return prompt_info, llm_response, None

    # ---[4]--- Metrics

    dis_name_in_llm_response = drugmechdb_prompt_builder.get_dis_name_in_llm_response(prompt_style,
                                                                                      prompt_info.disease_name)
    llm_graph_name = f"ChatGPT-4o.MoA: {prompt_info.drug_name} treats {prompt_info.disease_name}"

    match_score = test_moa_match(grmatcher, None, llm_response.message,
                                 prompt_info.drug_name, dis_name_in_llm_response,
                                 drugmechdb_prompt_builder.formal_to_kg_entity_type_names(),
                                 llm_graph_name,
                                 ref_nodes_in_prompt=prompt_info.disease_prompt_nodes,
                                 verbose=verbose
                                 )

    return prompt_info, llm_response, match_score


# -----------------------------------------------------------------------------
#   Functions: Misc
# -----------------------------------------------------------------------------


def test_one_dmdb(drug_id: str, disease_id: str,
                  prompt_style: str = "ANONYMIZED_DISEASE",
                  prompt_version: int = 0,
                  use_etype_equivs=True,
                  verbose=True
                  ):

    pp_funcargs(test_one_dmdb)

    # Convert to enum value
    prompt_style = PromptStyle[prompt_style.upper()]

    prompt_builder = DrugMechPromptBuilder()
    drugmechdb = prompt_builder.drugmechdb

    indication_graphs = drugmechdb.get_indication_graphs(drug_id, disease_id)
    if len(indication_graphs) != 1:
        print(f"Unexpected nbr indications graphs ({len(indication_graphs)}) found for "
                      f"{drug_id=}, {disease_id=}!")
        print()

        if len(indication_graphs) == 0:
            return

    moa = indication_graphs[0]

    print("Using MoA id =", drugmechdb.get_moa_id(moa))
    print()

    prompt_info, llm_response, metrics = \
        test_moa_llm_match(drug_id, disease_id, moa,
                           prompt_builder=prompt_builder,
                           prompt_style=prompt_style,       # type: Ignore
                           prompt_version=prompt_version,
                           use_etype_equivs=use_etype_equivs,
                           verbose=verbose)

    if verbose:
        pp_underlined_hdg("LLM full prompt:")
        print(prompt_info.full_prompt)
        print("---\n")

    if prompt_info is None:
        print("No LLM prompt possible.")
        return

    if metrics and not verbose:
        print("Match metrics:")
        metrics.pprint(indent=4)

    return


def get_session_samples_data(session_json_file: str) -> List[Tuple[bool, str, int, str, str, str, str, str, str]]:
    """
    Extract session samples data from session JSON file.

    :return:
        - Samples: List[ (is_negative_sample,
                          prompt_style, prompt_version, source_kg,
                          moa_id,
                          drug_id, disease_id,
                          drug_name, disease_name),
                        ... ]
    """

    with open(session_json_file) as f:
        jdict = json.load(f)

    samples = [(spi.get("is_negative_sample", False),
                spi["prompt_style"], spi["prompt_version"], spi["source_kg"],
                spi["moa_id"],
                spi["drug_id"], spi["disease_id"],
                spi["drug_name"], spi["disease_name"]
                )
               for spi in map(lambda s: s["prompt_info"],
                              jdict["session"])]

    return samples


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
#
# [Python]$ python -m drugmechcf.llm.test_dmdb {test | positives | ...}
#
# --- e.g. Testing +ive samples (executed from `$PROJDIR/src/`):
#
# $ python -m drugmechcf.llm.test_dmdb positives -c 1000 -p 2 \
#           -f ../Data/Counterfactuals/factuals_pos_r1k.json \
#           -j ../Data/Sessions/Factuals/pos_p2_f1k.json \
#           2>&1 | tee ../Data/Sessions/Factuals/pos_p2_f1k.txt
#
# $ python -m drugmechcf.llm.test_dmdb positives -r -c 100 -p 2 -j ../Data/Sessions/Factuals/pos_p2_r100.json 2>&1 \
#           | tee ../Data/Sessions/Factuals/pos_p2_r100.txt
# $ python -m drugmechcf.llm.test_dmdb positives -r -c 100 -p 2 -s NAMED_DISEASE \
#              -j ../Data/Sessions/Factuals/pos_p2_r100_named.json 2>&1 \
#           | tee ../Data/Sessions/Factuals/pos_p2_r100_named.txt
# $ python -m drugmechcf.llm.test_dmdb positives -r -c 100 -p 2 -s NAMED_DISEASE_WITH_ALL_ASSOCIATIONS \
#              -j ../Data/Sessions/Factuals/pos_p2_r100_named_assoc.json 2>&1 \
#           | tee ../Data/Sessions/Factuals/pos_p2_r100_named_assoc.txt
#
# --- e.g. Testing -ive samples (executed from `$PROJDIR/src/`):
#
# $ python -m drugmechcf.llm.test_dmdb negs -c 1000 -p 2 --contras \
#              -f ../Data/Counterfactuals/factuals_neg_r1k.json \
#              -j ../Data/Sessions/Factuals/neg_p2_f1k_contras.json \
#           2>&1 | tee ../Data/Sessions/Factuals/neg_p2_f1k_contras.txt
#
# $ python -m drugmechcf.llm.test_dmdb negs -c 100 --seed 42 -p 2 --contras \
#              -j ../Data/Sessions/Factuals/neg_p2_100_contras.json \
#           2>&1 | tee ../Data/Sessions/Factuals/neg_p2_100_contras.txt
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Matching LLM response MoA to DrugMechDB.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... test
    _sub_cmd_parser = _subparsers.add_parser('test',
                                             help="Test on specific Drug and Disease.")
    _sub_cmd_parser.add_argument('-s', '--prompt_style', type=str,
                                 choices=[x.name for x in PromptStyle],
                                 default="ANONYMIZED_DISEASE",
                                 help="Prompt style. Default is ANONYMIZED_DISEASE.")
    _sub_cmd_parser.add_argument('-p', '--prompt_version', type=int, default=0,
                                 help="Prompt version to use. Default is ver 0.")
    _sub_cmd_parser.add_argument('-v', '--verbose', action='store_true',
                                 help="Verbose mode.")
    _sub_cmd_parser.add_argument('drug_id', type=str,
                                 help="Drug-ID.")
    _sub_cmd_parser.add_argument('disease_id', type=str,
                                 help="Disease-ID.")

    # ... positives
    _sub_cmd_parser = _subparsers.add_parser('positives',
                                        help="Test ChatGPT-4o response match on factual queries from DrugMechDB.")
    _sub_cmd_parser.add_argument('-j', '--json', type=str,
                                 help="Also write session data to this JSON file.")
    _sub_cmd_parser.add_argument('-c', '--count', type=int, default=1,
                                 help="Nbr of DrugMechDB MoA's to test.")
    _sub_cmd_parser.add_argument('-s', '--prompt_style', type=str,
                                 choices=[x.name for x in PromptStyle],
                                 default="ANONYMIZED_DISEASE",
                                 help="Prompt style. Default is ANONYMIZED_DISEASE.")
    _sub_cmd_parser.add_argument('-p', '--prompt_version', type=int, default=1,
                                 help="Prompt version to use. Default is ver 1.")
    _sub_cmd_parser.add_argument('-r', '--random', action='store_true',
                                 help="Test a random selection of MoA's. Otherwise it is the first `n`.")
    _sub_cmd_parser.add_argument('--dont_match_entity_types', action='store_true',
                                 help="Dont require EntityType to match when matching nodes. Otherwise it is required.")
    _sub_cmd_parser.add_argument('-f', '--samples_file', type=str,
                                 help="JSON file containing positive samples.")

    # ... negs
    _sub_cmd_parser = _subparsers.add_parser('negs',
                                             help="Test ChatGPT-4o response match on -ive samples from PrimeKG.")
    _sub_cmd_parser.add_argument('-j', '--json', type=str,
                                 help="Also write session data to this JSON file.")
    _sub_cmd_parser.add_argument('-c', '--count', type=int, default=1,
                                 help="Nbr of DrugMechDB MoA's to test.")
    _sub_cmd_parser.add_argument('-s', '--prompt_style', type=str,
                                 choices=[x.name for x in PromptStyle],
                                 default="ANONYMIZED_DISEASE",
                                 help="Prompt style. Default is ANONYMIZED_DISEASE.")
    _sub_cmd_parser.add_argument('-p', '--prompt_version', type=int, default=1,
                                 help="Prompt version to use. Default is ver 0.")
    _sub_cmd_parser.add_argument('--seed', type=int, default=None,
                                 help="Seed for random nbr generator, for replicability.")
    _sub_cmd_parser.add_argument('--contras', action='store_true',
                                 help="Include contra-indications in -ive samples?")
    _sub_cmd_parser.add_argument('-f', '--samples_file', type=str,
                                 help="JSON file containing negative samples.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'test':

        test_one_dmdb(_args.drug_id,
                      _args.disease_id,
                      prompt_version=_args.prompt_version,
                      prompt_style=_args.prompt_style,
                      use_etype_equivs=True,
                      # verbose=_args.verbose,
                      )

    elif _args.subcmd == 'positives':

        test_positives_batch(_args.count,
                             randomize=_args.random,
                             match_entity_types=not _args.dont_match_entity_types,
                             use_etype_equivs=True,
                             prompt_style=_args.prompt_style,
                             prompt_version=_args.prompt_version,
                             json_file=_args.json,
                             pos_samples_file=_args.samples_file,
                             )

    elif _args.subcmd == 'negs':

        test_negatives_batch(_args.count,
                             include_contra_indications=_args.contras,
                             prompt_style=_args.prompt_style,
                             prompt_version=_args.prompt_version,
                             seed=_args.seed,
                             json_file=_args.json,
                             neg_samples_file=_args.samples_file,
                             )

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
