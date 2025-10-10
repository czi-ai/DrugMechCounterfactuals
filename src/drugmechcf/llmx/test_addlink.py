"""
Test the AddLink use cases
"""


from collections import Counter, defaultdict
import concurrent.futures
import dataclasses
import json
import re
import sys
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.data.drugmechdb import load_drugmechdb, DRUG_ENTITY_TYPE
from drugmechcf.llm.openai import MODEL_KEYS, OpenAICompletionOpts, OpenAICompletionClient, CompletionOutput
from drugmechcf.llm.prompt_types import QueryType, DrugDiseasePromptInfo, EditLinkInfo
from drugmechcf.llm.test_common import (YES_NO_PATT, test_moa_match,
                                        ENTITY_TYPE_EQUIVALENCES, pprint_accumulated_metrics)
from drugmechcf.graphmatch.graphmatcher import BasicGraphMatcher, GraphMatchScore

from drugmechcf.text.optsmatcher import OptionsMatcher

from drugmechcf.llmx.prompts_addlink import PromptBuilder

from drugmechcf.kgproc.addlink import create_new_moa_add_link

from drugmechcf.utils.misc import (NpEncoder, buffered_stdout, pp_funcargs, pp_underlined_hdg,
                                   pp_dict, check_output_file_dir)


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

# Set to +ive int to limit nbr samples tested.
MAX_TEST_SAMPLES = 0

DEBUG_NO_LLM = False


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class AddLinkTask:
    """Includes all the fields in a data sample."""

    # The Drug - Disease that the MoA query is about, as nodes in DrugMechDB, and their names.

    drug_id: str
    drug_name: str

    disease_id: str
    disease_name: str

    # Edit-link details

    edit_link_info: EditLinkInfo

    # Additional info

    is_negative_sample: bool

    moa: MoaGraph = None
    """For +ive samples, the MoA synthesized by adding a link."""

    query_type: str = "ADD_LINK"
# /


@dataclasses.dataclass
class NegSampleMetrics:

    no_moa_found: bool

    response_match_score_for_no: float

    response_match_score_for_yes: float

    def to_serialized(self) -> Dict[str, Any]:
        """
        Make a serializable dict repr
        """
        # noinspection PyTypeChecker
        d = dataclasses.asdict(self)
        return d

    def pprint(self, stream=None, float_precision: int = 3, indent: int = 0):
        pfx = " " * indent
        # noinspection PyTypeChecker
        for fld in dataclasses.fields(self):
            fval = getattr(self, fld.name)
            if fval is not None and fld.type == float:
                fval = format(fval, f'.{float_precision}f')
            print(f"{pfx}{fld.name} = {fval}", file=stream)
        print(file=stream)

    def accumulate_metrics(self, cum_metrics: Dict[str, List[Union[bool, int, float]]] = None) \
            -> Dict[str, List[Union[bool, int, float]]]:

        if cum_metrics is None:
            cum_metrics = defaultdict(list)

        # noinspection PyTypeChecker
        for fld in dataclasses.fields(self):
            fval = getattr(self, fld.name)
            cum_metrics[fld.name].append(0 if fval is None else fval)

        return cum_metrics

# /


class TestAddLink:

    def __init__(self,
                 # Params for OpenAI client
                 model_key: str = "o3-mini",
                 reasoning_effort: str = "medium",
                 seed: int = 42,
                 temperature: float = 1.0,
                 extra_body: dict = None,
                 # ... for vllm-served LLMs
                 api_key: str = None,
                 base_url: str = None,
                 #
                 timeout_secs: int = 60,
                 use_flex_service_tier: bool = True,
                 #
                 # Params for Multi-threading
                 n_worker_threads: int = 32,
                 #
                 # Params for PromptBuilder
                 prompt_version: int = 0,
                 include_examples: bool = True,
                 insert_known_moas: bool = False,
                 ):

        self.llm_opts = OpenAICompletionOpts(model=MODEL_KEYS.get(model_key, model_key),
                                             reasoning_effort=reasoning_effort,
                                             seed=seed,
                                             temperature=temperature,
                                             extra_body=extra_body)

        self.timeout_secs = timeout_secs

        self.use_flex_service_tier = use_flex_service_tier

        self.api_key = api_key
        self.base_url = base_url

        self.n_worker_threads = n_worker_threads

        self.drugmechdb = load_drugmechdb()

        self.prompt_builder = PromptBuilder(self.drugmechdb, prompt_version, include_examples, insert_known_moas)

        self.show_full_prompt = False
        self.show_response = False

        self.neg_opts_matcher = None
        self._build_neg_response_matcher()

        return

    def _build_neg_response_matcher(self):
        """
        For matching LLM response to expected response for -ive samples.
        Opt keys are 'YES', 'NO'.
        Last call in init
        """
        response_opts = dict(YES="YES.", NO=self.prompt_builder.get_negative_response_text())
        self.neg_opts_matcher = OptionsMatcher(response_opts)
        return

    def create_llm_client(self) -> OpenAICompletionClient:
        llm_client = OpenAICompletionClient(self.llm_opts,
                                            api_key=self.api_key,
                                            base_url=self.base_url,
                                            timeout_secs=self.timeout_secs,
                                            use_flex_service_tier=self.use_flex_service_tier)

        return llm_client

    def create_sample_task(self, sample_data: dict[str, Any]) -> AddLinkTask:
        sample_data = sample_data.copy()
        edit_link_info = EditLinkInfo(**sample_data["edit_link_info"])
        sample_data["edit_link_info"] = edit_link_info
        task = AddLinkTask(**sample_data)

        if not task.is_negative_sample:
            # Get the source and target MoA
            source_moa = self.drugmechdb.get_indication_graph_with_id(edit_link_info.source_moa_id)
            target_moa = self.drugmechdb.get_indication_graph_with_id(edit_link_info.target_moa_id)
            # Add synthesized MoA resulting from adding new link
            task.moa = create_new_moa_add_link(self.drugmechdb,
                                               source_moa, edit_link_info.source_node,
                                               edit_link_info.new_relation,
                                               target_moa, edit_link_info.target_node,
                                               source_moa_drug_node=task.drug_id,
                                               target_moa_disease_node=task.disease_id
                                               )

        return task

    def test_batch(self,
                   samples_data_file: str,
                   output_json_file: str = None,
                   show_full_prompt=False,
                   show_response=False,
                   ):

        pp_underlined_hdg(f"AddLink Batch Test", linechar='~', overline=True)
        pp_funcargs(self.test_batch)

        if output_json_file is not None:
            check_output_file_dir(output_json_file)

        self.show_full_prompt = show_full_prompt
        self.show_response = show_response

        with open(samples_data_file) as f:
            test_samples = json.load(f)

        print(f"Nbr test samples read    = {len(test_samples):,d}")

        neg_sample_counts = Counter([s["is_negative_sample"] for s in test_samples])
        if len(neg_sample_counts.keys()) > 1:
            print(f"WARNING: Both +ive and -ive samples encountered in file!")
            print(f"... +ive = {neg_sample_counts[False]}, -ive = {neg_sample_counts[True]}")
            print("Exitting!")
            return

        samples_are_negative = test_samples[0]["is_negative_sample"]

        query_type = QueryType[test_samples[0]["query_type"]]

        if query_type is not QueryType.ADD_LINK:
            print(f"WARNING: {query_type=} not supported in {self.__class__.__name__}")
            print("Exitting!")
            return

        print("Samples data:")
        print(f"    nbr Samples = {len(test_samples)}")
        print(f"    {samples_are_negative = }")
        print(f"    query_type = {query_type.name}")
        print()

        source_node_type_counts = Counter([s["edit_link_info"]["source_node_type"] for s in test_samples])
        print("Counts by source node type:\n    ", end="")
        print(*[f"{snt}={cnt}" for snt, cnt in sorted(source_node_type_counts.items())], sep=", ")
        print()

        addlink_tasks = [self.create_sample_task(sample) for sample in test_samples]

        if len(addlink_tasks) > MAX_TEST_SAMPLES > 0:
            addlink_tasks = addlink_tasks[:MAX_TEST_SAMPLES]

        n_tested = len(addlink_tasks)

        print(f"Nbr samples being tested = {n_tested:,d}")
        print()

        # Execute the batch

        task_results_seq = self.test_samples_mt(addlink_tasks, samples_are_negative)

        # Accumulate metrics and compile session record

        source_node_is_drug_count = source_node_type_counts[DRUG_ENTITY_TYPE]

        jdict = None
        if output_json_file is not None:
            jdict = dict(args=dict(OpenAICompletionOpts=self.llm_opts.as_params_dict(),
                                   timeout_secs=self.timeout_secs,
                                   use_flex_service_tier=self.use_flex_service_tier,
                                   n_worker_threads=self.n_worker_threads,
                                   #
                                   samples_data_file=samples_data_file,
                                   query_type=QueryType.ADD_LINK.name,
                                   samples_are_negative=samples_are_negative,
                                   #
                                   MAX_TEST_SAMPLES=MAX_TEST_SAMPLES,
                                   source_node_is_drug=source_node_is_drug_count > 0,
                                   source_node_is_not_drug=source_node_is_drug_count == 0,
                                   source_node_type_counts=source_node_type_counts,
                                   #
                                   prompt_version=self.prompt_builder.prompt_version,
                                   include_examples=self.prompt_builder.include_examples,
                                   insert_known_moas=self.prompt_builder.insert_known_moas,
                                   ),
                         session=[])

        n_valid_llm_response = 0
        cum_metrics = None

        for task_result in task_results_seq:
            node_match_scores = None
            if samples_are_negative:
                prompt_info, llm_response, match_metrics = task_result
            else:
                prompt_info, llm_response, match_metrics, node_match_scores = task_result

            if jdict is not None:
                sdict = dict(prompt_info=prompt_info.to_serialized(),
                             llm_response=llm_response.message if llm_response else None,
                             llm_finish_reason=llm_response.finish_reason if llm_response else None,
                             match_metrics=match_metrics.to_serialized() if match_metrics is not None else None,
                             node_match_scores=node_match_scores,
                             )

                jdict["session"].append(sdict)

            if llm_response is not None and llm_response.is_complete_response():
                n_valid_llm_response += 1

            if match_metrics is not None:
                cum_metrics = match_metrics.accumulate_metrics(cum_metrics)

        print()
        print("==========================================================")
        print()

        if samples_are_negative:
            llm_metrics = pprint_neg_sample_metrics(cum_metrics, n_tested, n_valid_llm_response)
        else:
            llm_metrics = pprint_accumulated_metrics(cum_metrics, n_tested, n_valid_llm_response,
                                                     n_samples_skipped=0,
                                                     n_skipped_llm_response_inc=n_tested - n_valid_llm_response)

        if output_json_file is not None:
            jdict["metrics"] = llm_metrics

            with open(output_json_file, "w") as f:
                json.dump(jdict, f, indent=4, cls=NpEncoder)  # type: Ignore

            print()
            print("Session data written to:", output_json_file)
            print()

        return

    def test_samples_mt(self, addlink_tasks: list[AddLinkTask], samples_are_negative: bool = False):
        task_results_seq = []
        ntasks = len(addlink_tasks)

        print(f"* test_samples_mt: [{'-ive' if samples_are_negative else '+ive'} samples]",
              f"Executing {ntasks} tasks on {self.n_worker_threads} threads.")
        print(flush=True)

        if samples_are_negative:
            test_sample_fn = self.test_negative_sample
        else:
            test_sample_fn = self.test_positive_sample

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_worker_threads) as executor:

            futures_to_task_n = {executor.submit(test_sample_fn, al_task, f"{n}/{ntasks}"):
                                     (al_task, n)
                                 for n, al_task in enumerate(addlink_tasks, start=1)}

            for future in concurrent.futures.as_completed(futures_to_task_n):
                al_task, n = futures_to_task_n[future]
                task_results = future.result()

                task_results_seq.append(task_results)

                # Print task info
                with buffered_stdout():
                    if samples_are_negative:
                        neg_sample_metrics = task_results[2]
                        print(f" -- [{n}/{ntasks}] neg_sample_metrics returned = {neg_sample_metrics is not None}")
                        if neg_sample_metrics is not None:
                            print(f" -- [{n}/{ntasks}] no_moa_found = {neg_sample_metrics.no_moa_found},",
                                  f"response_match_score_for_no = {neg_sample_metrics.response_match_score_for_no:.3f}")
                    else:
                        graph_match_metrics = task_results[2]
                        print(f" -- [{n}/{ntasks}] GraphMatchScore returned = {graph_match_metrics is not None}")
                        if graph_match_metrics is not None:
                            print(f"        {graph_match_metrics.interior_node_match_score=},",
                                  f"{graph_match_metrics.edge_match_score_reduced=}")
                    print("----")

        print("* test_samples_mt: Completed!")
        print(flush=True)

        return task_results_seq

    def test_positive_sample(self, addlink_task: AddLinkTask,
                             taskn: str = None,
                             ) -> tuple[DrugDiseasePromptInfo,
                                        CompletionOutput | None,
                                        GraphMatchScore | None,
                                        dict[str, float] | None]:
        """
        Test one positive AddLink sample.
        :param addlink_task:
        :param taskn: A string label of the format "{task_n} / {n_tasks}"
        :return:
            - DrugDiseasePromptInfo: showing the prompt and related info
            - CompletionOutput: LLM response,
                or None if response was incomplete, or DEBUG_NO_LLM
            - GraphMatchScore: Matching returned graph to reference,
                or None if no LLM response
            - node_match_scores as a dict (see `BasicGraphMatcher.get_last_node_match_scores_simpledict`),
                or None if no node matches
        """

        prompt_info = self.prompt_builder.get_full_prompt(addlink_task.drug_id, addlink_task.drug_name,
                                                          addlink_task.disease_id, addlink_task.disease_name,
                                                          edit_link_info=addlink_task.edit_link_info,
                                                          is_negative_sample=addlink_task.is_negative_sample,
                                                          moa=addlink_task.moa
                                                          )

        if taskn is None:
            taskn = ""
        else:
            taskn = f"[{taskn}] "

        query_name = f"{taskn}AddLink +ive: {prompt_info.drug_name} for {prompt_info.disease_name}"

        with buffered_stdout():
            pp_underlined_hdg(query_name)

        if self.show_full_prompt or DEBUG_NO_LLM:
            with buffered_stdout():
                print(f"... Prompt for {query_name}:", flush=True)
                print(prompt_info.full_prompt)
                print("---")
                print(flush=True)

        if DEBUG_NO_LLM:
            return prompt_info, None, None, None

        llm_client = self.create_llm_client()

        llm_response = llm_client(user_prompt=prompt_info.full_prompt)

        if self.show_response:
            with buffered_stdout():
                print(f"LLM response for {query_name}:")
                if not llm_response.is_complete_response():
                    print(f" ... Incomplete. finish_reason = '{llm_response.finish_reason}")
                else:
                    print(llm_response.message)

                print("----\n")

        if not llm_response.is_complete_response():
            return prompt_info, llm_response, None, None

        llm_graph_name = f"LLM.MoA: {prompt_info.drug_name} treats {prompt_info.disease_name}"

        grmatcher = BasicGraphMatcher(verbosity=1)
        grmatcher.set_entity_type_equivalences(ENTITY_TYPE_EQUIVALENCES)

        print(f"Computing metrics for {query_name} ...\n")
        match_metrics = test_moa_match(grmatcher, addlink_task.moa, llm_response.message,
                                       prompt_info.drug_name, prompt_info.disease_name,
                                       self.prompt_builder.formal_to_kg_entity_type_names(),
                                       llm_graph_name,
                                       match_entity_types=True,
                                       ref_nodes_in_prompt=prompt_info.disease_prompt_nodes,
                                       verbose=self.show_response
                                       )

        node_match_scores = grmatcher.get_last_node_match_scores_simpledict()

        return prompt_info, llm_response, match_metrics, node_match_scores

    def test_negative_sample(self, addlink_task: AddLinkTask,
                             taskn: str = None,
                             ) \
            -> tuple[DrugDiseasePromptInfo, CompletionOutput | None, NegSampleMetrics | None]:
        """
        Test one NEGATIVE AddLink sample.
        :param addlink_task:
        :param taskn: A string label of the format "{task_n} / {n_tasks}"
        :return:
            - DrugDiseasePromptInfo: showing the prompt and related info
            - CompletionOutput: LLM response,
                or None if response was incomplete, or DEBUG_NO_LLM
            - NegSampleMetrics: Metrics on match to expected response,
                or None if no LLM response
        """

        prompt_info = self.prompt_builder.get_full_prompt(addlink_task.drug_id, addlink_task.drug_name,
                                                          addlink_task.disease_id, addlink_task.disease_name,
                                                          edit_link_info=addlink_task.edit_link_info,
                                                          is_negative_sample=addlink_task.is_negative_sample,
                                                          moa=addlink_task.moa
                                                          )

        if taskn is None:
            taskn = ""
        else:
            taskn = f"[{taskn}] "

        query_name = f"{taskn}AddLink NEG-ive: {prompt_info.drug_name} for {prompt_info.disease_name}"

        with buffered_stdout():
            pp_underlined_hdg(query_name)

        if self.show_full_prompt or DEBUG_NO_LLM:
            with buffered_stdout():
                print(f"... Prompt for {query_name}:", flush=True)
                print(prompt_info.full_prompt)
                print("---")
                print(flush=True)

        if DEBUG_NO_LLM:
            return prompt_info, None, None

        llm_client = self.create_llm_client()

        llm_response = llm_client(user_prompt=prompt_info.full_prompt)

        if not llm_response.is_complete_response():
            if self.show_response:
                with buffered_stdout():
                    print(f"LLM response for {query_name}:")
                    print(f" ... Incomplete. finish_reason = '{llm_response.finish_reason}")
                    print("----\n")

            return prompt_info, llm_response, None

        match_metrics = self.check_negative_sample_response(llm_response.message)

        try:
            with buffered_stdout():
                if self.show_response:
                    print(f"LLM response for {query_name}:")
                    print(llm_response.message)
                    print("----\n")

                print(f"Computing metrics for {query_name} ...\n")

                if match_metrics is None:
                    print("    match_metrics NOT returned.")
                else:
                    print(f"    no_moa_found = {match_metrics.no_moa_found},",
                          f"response_match_score_for_no = {match_metrics.response_match_score_for_no:.3f}")
                    print("----\n")

        except Exception as e:
            print(f"*** ERROR: in query: {query_name} ...", file=sys.stderr)
            raise e

        return prompt_info, llm_response, match_metrics

    def check_negative_sample_response(self, llm_response_msg: str):

        # Heuristic: look for lines starting with "NO"
        best_scores = dict(YES=0, NO=0)

        if llm_response_msg:
            lines = llm_response_msg.strip().splitlines()
            for line_ in lines[::-1]:
                line_ = line_.strip()
                if re.match(YES_NO_PATT, line_):
                    best_matched_opt, best_match_score = self.neg_opts_matcher.get_best_matching_option(line_)
                    if best_matched_opt:
                        if best_match_score > best_scores[best_matched_opt]:
                            best_scores[best_matched_opt] = best_match_score

        no_moa_retd = best_scores["NO"] > best_scores["YES"]

        neg_metrics = NegSampleMetrics(no_moa_found=no_moa_retd,
                                       response_match_score_for_no=best_scores["NO"],
                                       response_match_score_for_yes=best_scores["YES"]
                                       )

        return neg_metrics

    @staticmethod
    def is_sample_response_correct(sample: dict[str, Any]) -> bool:
        """
        Is the LLM response for this recorded sample a Correct response (i.e. matches expected response)?
        For +ive samples, the expected response is a MoA.
        For -ive samples, the expected response is No MoA.

        :param sample: One sample, as stored in the session file
        """
        is_negative_sample = sample["prompt_info"]["is_negative_sample"]

        match_metrics = sample["match_metrics"]

        if is_negative_sample:
            return match_metrics["no_moa_found"]
        else:
            return match_metrics["has_target_graph"]

    @staticmethod
    def extract_main_metrics(session_json_file: str) -> dict[str, Any]:
        """
        Extracts the main metrics from JSON session file, used e.g. to compute variances.
        """
        with open(session_json_file) as f:
            jdict = json.load(f)

        metrics = jdict["metrics"]

        metric_keys = ["n_gt", "n_gf",
                       "recall",
                       "n_true_pos", "n_true_neg", "n_false_pos", "n_false_neg",
                       "accuracy_gt", "accuracy_gf",
                       ]

        if not jdict['args']['samples_are_negative']:
            metric_keys += ["prop_very_diff_moas",
                            "interior_node_match_score:mean",
                            "edge_match_score_reduced:mean"
                            ]

        main_metrics = dict((k, metrics.get(k, np.NaN)) for k in metric_keys)

        return main_metrics

# /


# -----------------------------------------------------------------------------
#   Functions: Utils
# -----------------------------------------------------------------------------


def pprint_neg_sample_metrics(cum_metrics: Dict[str, List[bool | float]],
                              n_tested: int,
                              n_valid_llm_response: int,
                              ) \
        -> Dict[str, Union[int, float, List[float]]] | None:

    if cum_metrics is None:
        print()
        print("No metrics to accumulate!")
        print()
        return None

    n_samples = len(cum_metrics["no_moa_found"])

    # Metrics on Yes/No response
    y_true = np.ones(n_samples, dtype=bool)
    y_pred = np.asarray(cum_metrics["no_moa_found"], dtype=bool)
    n_gt = y_true.sum()
    n_gf = n_samples - n_gt

    print("Negative samples test stats:")
    print(f"    Nbr samples tested = {n_tested:,d}")
    print(f"    Nbr samples found  = {n_samples:,d}")
    print(f"    Found == Tested?     {n_tested == n_samples}")
    print()
    print(f"    Nbr valid LLM response  = {n_samples:5,d} ... Counts match? {n_samples == n_valid_llm_response}")
    print(f"    Nbr incomplete response = {n_tested - n_samples:5,d}")
    print()
    print(f"    Nbr positive samples    = {n_gt:5,d}")
    print(f"    Nbr negative samples    = {n_gf:5,d}")
    print()

    # Metrics on Yes/No response
    # noinspection PyUnboundLocalVariable
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0.0)

    n_true_pos = np.logical_and(y_true, y_pred).sum()
    n_true_neg = np.logical_and(~y_true, ~y_pred).sum()
    n_false_neg = np.logical_and(y_true, ~y_pred).sum()
    n_false_pos = np.logical_and(~y_true, y_pred).sum()

    metrics = dict(n_gt=n_gt, n_gf=n_gf,
                   n_true_pos=n_true_pos, n_true_neg=n_true_neg,
                   n_false_neg=n_false_neg, n_false_pos=n_false_pos,
                   accuracy = accuracy_score(y_true, y_pred, normalize=True),
                   accuracy_gt = accuracy_score(y_true[y_true], y_pred[y_true], normalize=True) if n_gt else 0,
                   accuracy_gf = accuracy_score(y_true[~y_true], y_pred[~y_true], normalize=True) if n_gf else 0,
                   precision=prec, recall=recall, f1=f1,
                   )

    pp_dict(metrics, "Binary metrics")

    # Compute score seq stats
    for k in ["response_match_score_for_no", "response_match_score_for_yes"]:
        vals = np.asarray(cum_metrics[k])

        v_mean = np.nanmean(vals)
        q1, q2, q3 = np.nanpercentile(vals, [25, 50, 75])

        print(f"    {k}:", end="  ")
        print(f"mean = {v_mean:.3f}")
        print(f"\trange = [{np.nanmin(vals):.3f}, {np.nanmax(vals):.3f}], ",
              f"quartiles = [{q1:.3f}, {q2:.3f}, {q3:.3f}]")

        metrics[f"{k}:mean"] = v_mean
        metrics[f"{k}:quartiles"] = [q1, q2, q3]

    print()
    return metrics


# -----------------------------------------------------------------------------
#   Functions: Commands
# -----------------------------------------------------------------------------


def test_addlink_batch(samples_data_file: str,
                       output_json_file: str,
                       *,
                       model_key: str = "o3-mini",
                       # reasoning_effort: str = "medium",
                       # temperature: float = 1.0,
                       #
                       api_key: str = None,
                       base_url: str = None,
                       #
                       n_worker_threads: int = 32,
                       timeout_secs: int = 60,
                       #
                       include_examples: bool = True,
                       insert_known_moas: bool = False,
                       #
                       show_full_prompt=False,
                       show_response=False,
                       #
                       max_samples=None,
                       ):

    global DEBUG_NO_LLM, MAX_TEST_SAMPLES

    if max_samples:
        MAX_TEST_SAMPLES = max_samples

    tester = TestAddLink(model_key=model_key,
                         api_key=api_key,
                         base_url=base_url,
                         include_examples=include_examples,
                         insert_known_moas=insert_known_moas,
                         n_worker_threads=n_worker_threads,
                         timeout_secs=timeout_secs,
                         )
    tester.test_batch(samples_data_file,
                      output_json_file,
                      show_full_prompt=show_full_prompt,
                      show_response=show_response,
                      )

    show_globals()
    return


def show_globals():
    print()
    print("Globals:")
    print(f"    {MAX_TEST_SAMPLES = }")
    print(f"    {DEBUG_NO_LLM = }")
    print()
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
#
# [Python]$ python -m llmx.test_addlink batch ...
#
# --- e.g. Testing +ive samples (executed from `$PROJDIR/src/`):
#
# $ python -m drugmechcf.llmx.test_addlink batch ../Data/Counterfactuals/AddLink_pos_dpi_r1k.json  \
#           ../Data/Sessions/Models/o3-mini/addlink_pos_dpi.json 2>&1  \
#           | tee ../Data/Sessions/Models/o3-mini/addlink_pos_dpi_log.txt
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Test Add-Link MoAs with LLM.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... batch
    _sub_cmd_parser = _subparsers.add_parser('batch',
                                             help="Test ChatGPT on batch of Add-Link samples from DrugMechDB.")
    _sub_cmd_parser.add_argument('-m', '--model_key', type=str,
                                 choices=list(MODEL_KEYS.keys()), default="o3-mini",
                                 help="LLM model type or id.")
    #
    _sub_cmd_parser.add_argument('-x', '--dont_include_examples', action='store_true',
                                 help="Do NOT include example queries in the prompt.")
    _sub_cmd_parser.add_argument('-k', '--insert_known_moas', action='store_true',
                                 help="Insert Known MoAs in the prompt (Closed-world setting).")
    #
    _sub_cmd_parser.add_argument('-f', '--show_full_prompt', action='store_true',
                                 help="Show the full LLM prompt.")
    _sub_cmd_parser.add_argument('-r', '--show_response', action='store_true',
                                 help="Show LLM response.")
    # args
    _sub_cmd_parser.add_argument('samples_data_file', type=str,
                                 help="Samples data file.")
    _sub_cmd_parser.add_argument('output_json_file', nargs="?", type=str, default=None,
                                 help="Output session file.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'batch':

        test_addlink_batch(_args.samples_data_file,
                           _args.output_json_file,
                           model_key=_args.model_key,
                           #
                           include_examples=not _args.dont_include_examples,
                           insert_known_moas=_args.insert_known_moas,
                           #
                           show_full_prompt=_args.show_full_prompt,
                           show_response=_args.show_response,
                           )

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
