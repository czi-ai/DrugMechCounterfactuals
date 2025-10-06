"""
Testing DELETE / INVERT - LINK
"""

from collections import defaultdict, Counter
import concurrent.futures
import dataclasses
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from drugmechcf.data.drugmechdb import load_drugmechdb, DRUG_ENTITY_TYPE
from drugmechcf.data.moagraph import MoaGraph

from drugmechcf.llm.prompt_types import QueryType, DrugDiseasePromptInfo, EditLinkInfo

from drugmechcf.llm.openai import MODEL_KEYS, OpenAICompletionOpts, OpenAICompletionClient, CompletionOutput

from drugmechcf.llmx.prompts_editlink import (PromptBuilder,
                                   POSITIVE_RESPONSE_OPTIONS_STRICT, POSITIVE_RESPONSE_OPTIONS_RELAXED,
                                   NEGATIVE_RESPONSE_OPTIONS)

from drugmechcf.text.optsmatcher import OptionsMatcher

from drugmechcf.utils.misc import NpEncoder, pp_funcargs, pp_underlined_hdg, pp_dict, buffered_stdout


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

# Set to +ive int to limit nbr samples tested.
MAX_TEST_SAMPLES = 0

DEBUG_NO_LLM = False

DEBUG = False


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class EditLinkTask:
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

    query_type: str

    moa: MoaGraph = None
    """The MoA in which the link is being edited."""
# /


@dataclasses.dataclass
class OptionsMatchScore:

    is_positive_sample: bool

    query_type: QueryType

    llm_option_value: str | None
    """The option text extracted from LLM response."""

    option_key: str | None
    """Best matching option's Key."""

    match_score: float
    """Match-score for `option_key`."""

    def to_serialized(self) -> Dict[str, Any]:
        """
        Make a serializable dict repr
        """
        # noinspection PyTypeChecker
        d = dataclasses.asdict(self)
        d["query_type"] = self.query_type.name
        return d

    def accumulate_metrics(self, cum_metrics: Dict[str, List[Any]] = None) \
            -> Dict[str, List[Any]]:
        if cum_metrics is None:
            cum_metrics = defaultdict(list)

        # noinspection PyTypeChecker
        for fld in dataclasses.fields(self):
            if fld == "llm_option_value":
                continue
            fval = getattr(self, fld.name)
            cum_metrics[fld.name].append(fval)

        return cum_metrics

# /


@dataclasses.dataclass
class SampleParams:
    query_type: QueryType
    is_positive_sample: bool
    response_is_correct: bool
    drug_node: str
    disease_node: str

    edit_link_info: EditLinkInfo

    link_dist_from_root: int
    link_dist_from_sink: int

    @staticmethod
    def get_analytic_fieldnames():
        return ["response_is_correct", "link_dist_from_root", "link_dist_from_sink", "total_dist"]

    def to_analytic_fields(self):
        return [self.response_is_correct,
                self.link_dist_from_root,
                self.link_dist_from_sink,
                self.link_dist_from_root + self.link_dist_from_sink,
                ]
# /


class TestEditLink:

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
                 insert_known_moas: bool = False,
                 add_unknown_response_opt: bool = False,
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

        self.prompt_version = prompt_version
        self.insert_known_moas = insert_known_moas
        self.add_unknown_response_opt = add_unknown_response_opt

        self.show_full_prompt = False
        self.show_response = False

        # Built in the batch tester
        self.prompt_builder: Optional[PromptBuilder] = None

        return

    def create_llm_client(self) -> OpenAICompletionClient:
        llm_client = OpenAICompletionClient(self.llm_opts,
                                            api_key=self.api_key,
                                            base_url=self.base_url,
                                            timeout_secs=self.timeout_secs,
                                            use_flex_service_tier=self.use_flex_service_tier)

        return llm_client

    def create_prompt_builder(self, query_type: QueryType) -> PromptBuilder:
        prompt_builder = PromptBuilder(self.drugmechdb,
                                       query_type,
                                       prompt_version=self.prompt_version,
                                       insert_known_moas=self.insert_known_moas,
                                       add_unknown_response_opt=self.add_unknown_response_opt,
                                       )
        return prompt_builder

    def check_samples_get_query_type(self, test_samples)\
            -> tuple[bool | None, dict[str, int] | None, QueryType | None]:
        """
        :return: 3 items:
            - samples_are_negative | None,
            - source_node_type_counts | None,
            - QueryType | None
        """

        print(f"Nbr test samples read    = {len(test_samples):,d}")

        neg_sample_counts = Counter([s["is_negative_sample"] for s in test_samples])
        if len(neg_sample_counts.keys()) > 1:
            print(f"WARNING: Both +ive and -ive samples encountered in file!")
            print(f"... +ive = {neg_sample_counts[False]}, -ive = {neg_sample_counts[True]}")
            return None, None, None

        samples_are_negative = test_samples[0]["is_negative_sample"]

        source_node_type_counts = Counter([s["edit_link_info"]["source_node_type"] for s in test_samples])
        print("Counts by source node type:\n    ", end="")
        print(*[f"{snt}={cnt}" for snt, cnt in sorted(source_node_type_counts.items())], sep=", ")
        print()

        query_type_counts = Counter([s["query_type"] for s in test_samples])
        if len(query_type_counts.keys()) > 1:
            print("WARNING: Multiple query_type's encountered!\n ...", end="")
            print(*[f"{qtype}={cnt}" for qtype, cnt in sorted(query_type_counts.items())], sep=", ")
            return samples_are_negative, source_node_type_counts, None

        query_type = QueryType[test_samples[0]["query_type"]]

        if query_type not in [QueryType.CHANGE_LINK, QueryType.DELETE_LINK]:
            print(f"WARNING: {query_type=} not supported in {self.__class__.__name__}")
            return samples_are_negative, source_node_type_counts, None

        # noinspection PyTypeChecker
        return samples_are_negative, source_node_type_counts, query_type

    def create_sample_task(self, sample_data: dict[str, Any]) -> EditLinkTask:
        sample_data = sample_data.copy()
        edit_link_info = EditLinkInfo(**sample_data["edit_link_info"])
        sample_data["edit_link_info"] = edit_link_info
        task = EditLinkTask(**sample_data)

        # Add the MoA ... source and target are the same
        task.moa = self.drugmechdb.get_indication_graph_with_id(edit_link_info.source_moa_id)

        return task

    def test_batch(self,
                   samples_data_file: str,
                   output_json_file: str = None,
                   show_full_prompt=False,
                   show_response=False,
                   ):

        pp_underlined_hdg(f"Edit-Link Batch Test", linechar='~', overline=True)
        pp_funcargs(self.test_batch)

        self.show_full_prompt = show_full_prompt
        self.show_response = show_response

        with open(samples_data_file) as f:
            test_samples = json.load(f)

        samples_are_negative, source_node_type_counts, query_type = self.check_samples_get_query_type(test_samples)
        if query_type is None:
            print("Exitting!")
            return

        print("Samples data:")
        print(f"    nbr Samples = {len(test_samples)}")
        print(f"    {samples_are_negative = }")
        print(f"    query_type = {query_type.name}")
        print()

        editlink_tasks = [self.create_sample_task(sample) for sample in test_samples]

        if len(editlink_tasks) > MAX_TEST_SAMPLES > 0:
            editlink_tasks = editlink_tasks[:MAX_TEST_SAMPLES]

        n_tested = len(editlink_tasks)

        print(f"Nbr samples being tested = {n_tested:,d}")
        print()

        # Execute the batch

        self.prompt_builder = self.create_prompt_builder(query_type)

        task_results_seq = self.test_samples_mt(editlink_tasks, samples_are_negative)

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
                                   query_type=query_type.name,
                                   samples_are_negative=samples_are_negative,
                                   #
                                   MAX_TEST_SAMPLES=MAX_TEST_SAMPLES,
                                   source_node_is_drug=source_node_is_drug_count > 0,
                                   source_node_is_not_drug=source_node_is_drug_count == 0,
                                   source_node_type_counts=source_node_type_counts,
                                   #
                                   prompt_version=self.prompt_version,
                                   insert_known_moas=self.insert_known_moas,
                                   add_unknown_response_opt=self.add_unknown_response_opt,
                                   ),
                         session=[])

        n_valid_llm_response = 0
        cum_metrics = None

        for task_result in task_results_seq:
            prompt_info, llm_response, opt_match_metrics = task_result

            if jdict is not None:
                sdict = dict(prompt_info=prompt_info.to_serialized(),
                             llm_response=llm_response.message if llm_response else None,
                             llm_finish_reason=llm_response.finish_reason if llm_response else None,
                             opt_match_metrics=opt_match_metrics.to_serialized() if opt_match_metrics is not None
                                               else None,
                             )

                jdict["session"].append(sdict)

            if llm_response is not None and llm_response.is_complete_response():
                n_valid_llm_response += 1

            if opt_match_metrics is not None:
                cum_metrics = opt_match_metrics.accumulate_metrics(cum_metrics)

        print()
        print("==========================================================")
        print()

        llm_metrics = pprint_accumulated_metrics(cum_metrics, self.prompt_builder,
                                                 n_tested, n_valid_llm_response,
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

    def test_samples_mt(self, editlink_tasks: list[EditLinkTask], samples_are_negative: bool = False):
        task_results_seq = []
        ntasks = len(editlink_tasks)

        print(f"* test_samples_mt: [{'-ive' if samples_are_negative else '+ive'} samples]",
              f"Executing {ntasks} tasks on {self.n_worker_threads} threads.")
        print(flush=True)

        test_sample_fn = self.test_one_sample

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_worker_threads) as executor:

            futures_to_task_n = {executor.submit(test_sample_fn, al_task, f"{n}/{ntasks}"):
                                     (al_task, n)
                                 for n, al_task in enumerate(editlink_tasks, start=1)}

            for future in concurrent.futures.as_completed(futures_to_task_n):
                al_task, n = futures_to_task_n[future]
                task_results = future.result()

                task_results_seq.append(task_results)

                # Print task info
                with buffered_stdout():
                    opt_match_score: OptionsMatchScore = task_results[2]
                    if opt_match_score is not None:
                        print(f" -- [{n}/{ntasks}] selected option = {opt_match_score.option_key},",
                              f"score = {opt_match_score.match_score:.2f}")
                    else:
                        print(f" -- [{n}/{ntasks}] OptionsMatchScore = None")

        print("* test_samples_mt: Completed!")
        print(flush=True)

        return task_results_seq

    def test_one_sample(self, editlink_task: EditLinkTask,
                        taskn: str = None,
                        )\
            -> Tuple[DrugDiseasePromptInfo | None, CompletionOutput | None, OptionsMatchScore | None]:

        prompt_info = self.prompt_builder.get_full_prompt(editlink_task.drug_id, editlink_task.drug_name,
                                                          editlink_task.disease_id, editlink_task.disease_name,
                                                          edit_link_info=editlink_task.edit_link_info,
                                                          is_negative_sample=editlink_task.is_negative_sample,
                                                          moa=editlink_task.moa
                                                          )

        if taskn is None:
            taskn = ""
        else:
            taskn = f"[{taskn}] "

        query_type = prompt_info.query_type

        pos_neg = "-ive" if prompt_info.is_negative_sample else "+ive"

        query_name = f"{taskn}{query_type.name} {pos_neg}: {prompt_info.drug_name} for {prompt_info.disease_name}"

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

        if self.show_response:
            with buffered_stdout():
                print(f"LLM response for {query_name}:")
                if not llm_response.is_complete_response():
                    print(f" ... Incomplete. finish_reason = '{llm_response.finish_reason}")
                else:
                    print(llm_response.message)

                print("----\n")

        if not llm_response.is_complete_response():
            return prompt_info, llm_response, None

        resp_sumry, llm_response_opt, match_score = self.parse_llm_response_summary(llm_response.message)

        if DEBUG:
            with buffered_stdout():
                pp_underlined_hdg(f"{query_name} -- LLM Response Option Match:")
                print(f"Extracted response option text:\n  [{resp_sumry}]")
                print("Best matching option =", llm_response_opt)
                print(f"Match score = {match_score:.3f}",
                      "... below match threshold!" if llm_response_opt is None else "")
                print()

        metrics = OptionsMatchScore(query_type=query_type,
                                    is_positive_sample=not prompt_info.is_negative_sample,
                                    llm_option_value=resp_sumry,
                                    option_key=llm_response_opt,
                                    match_score=match_score
                                    )

        return prompt_info, llm_response, metrics

    def parse_llm_response_summary(self, response: str, min_score=0.4, default_opt=None)\
            -> Tuple[str, str | None, float | None]:

        opts_matcher = OptionsMatcher(self.prompt_builder.get_response_options_dict())

        resp_sumry = opts_matcher.get_option_para(response,
                                                  heading=self.prompt_builder.get_summary_heading(),
                                                  strip_markup=True)

        selected_opt, match_score = opts_matcher.get_best_matching_option(resp_sumry,
                                                                          min_score=min_score,
                                                                          warn_on_low_score=min_score)

        if selected_opt is None and default_opt is not None:
            selected_opt = default_opt

        return resp_sumry, selected_opt, match_score

    def analyze_response_attribution_multiple_sessions(self, session_json_file_seq: list[str]):

        pp_underlined_hdg("Analysing Multiple Session Files", linechar='=', overline=True)

        print("Nbr session files =", len(session_json_file_seq))
        print("All session files:")
        print(*[f"    {'/'.join(pathlib.PurePath(f).parts[-2:])}" for f in session_json_file_seq], sep="\n")
        print()

        cum_metrics = None
        for sesn_file in session_json_file_seq:
            print()
            cum_metrics = self.analyze_response_attribution_one_session(sesn_file, cum_metrics)

        print()
        pp_underlined_hdg("Total session results")

        n_samples = cum_metrics['total_counts']['n_samples']
        n_correct = cum_metrics['total_counts']['n_correct']
        n_incorrect = cum_metrics['total_counts']['n_incorrect']

        print("Total sessions data:")
        print(f"    query_types =", ", ".join(sorted(set(cum_metrics["cum_metrics"]))))
        print(f"    n_samples = {n_samples:,d}")
        print(f"    Correct responses  = {n_correct:6,d}  ...  {n_correct/n_samples:.3f}")
        print(f"    Inorrect responses = {n_incorrect:6,d}  ...  {n_incorrect/n_samples:.3f}")
        print()

        self.pp_contrastive_metrics(cum_metrics, "LinkDist.from.root",
                                    "correct_by_link_dist_from_root",
                                    'incorrect_by_link_dist_from_root')

        self.pp_contrastive_metrics(cum_metrics, "LinkDist.from.sink",
                                    "correct_by_link_dist_from_sink",
                                    'incorrect_by_link_dist_from_sink')

        self.pp_contrastive_metrics(cum_metrics, "LinkDist.Min",
                                    "correct_by_min_link_dist",
                                    'incorrect_by_min_link_dist')

        self.pp_contrastive_metrics(cum_metrics, "LinkDist.Total",
                                    "correct_by_total_link_dist",
                                    'incorrect_by_total_link_dist')

        self.pp_contrastive_metrics(cum_metrics, "SourceNode.type",
                                    "correct_by_source_node_type",
                                    'incorrect_by_source_node_type')

        return

    def analyze_response_attribution_one_session(self, session_json_file: str, cum_metrics=None, display=True):
        """
        Compares incorrect response to 'depth' of the internal link
        :param session_json_file: File output by `test_batch()`
        :param cum_metrics:
        :param display:
        """

        if display:
            f_parts = pathlib.PurePath(session_json_file).parts
            f_name = "/".join(f_parts[-2:])
            pp_underlined_hdg(f"Analyzing session: {f_name}")

        with open(session_json_file) as f:
            jdict = json.load(f)

        query_type = QueryType[jdict["args"]["query_type"]]
        samples_are_negative = jdict["args"]["samples_are_negative"]
        source_node_is_drug = jdict["args"]["source_node_is_drug"]

        assert not source_node_is_drug, "This fn only for when `source_node_is_not_drug`."

        incorrect_by_source_node_type = Counter()
        incorrect_by_link_dist_from_root = Counter()
        incorrect_by_link_dist_from_sink = Counter()
        incorrect_by_min_link_dist = Counter()
        incorrect_by_total_link_dist = Counter()

        correct_by_source_node_type = Counter()
        correct_by_link_dist_from_root = Counter()
        correct_by_link_dist_from_sink = Counter()
        correct_by_min_link_dist = Counter()
        correct_by_total_link_dist = Counter()

        n_samples = 0
        max_dist = 0
        n_correct, n_incorrect = 0, 0
        for sample in jdict["session"]:
            n_samples += 1

            sparams = self.get_sample_analytic_params(sample)

            eli = sparams.edit_link_info

            # Is correct response?
            if sparams.response_is_correct:
                n_correct += 1
            else:
                n_incorrect += 1

            link_dist_from_root = sparams.link_dist_from_root
            link_dist_from_sink = sparams.link_dist_from_sink

            if max_dist < max(link_dist_from_root, link_dist_from_sink):
                max_dist = max(link_dist_from_root, link_dist_from_sink)

            if sparams.response_is_correct:
                correct_by_link_dist_from_root[link_dist_from_root] += 1
                correct_by_link_dist_from_sink[link_dist_from_sink] += 1
                correct_by_min_link_dist[min(link_dist_from_root, link_dist_from_sink)] += 1
                correct_by_total_link_dist[link_dist_from_root + link_dist_from_sink] += 1
                correct_by_source_node_type[eli.source_node_type] += 1
            else:
                incorrect_by_link_dist_from_root[link_dist_from_root] += 1
                incorrect_by_link_dist_from_sink[link_dist_from_sink] += 1
                incorrect_by_min_link_dist[min(link_dist_from_root, link_dist_from_sink)] += 1
                incorrect_by_total_link_dist[link_dist_from_root + link_dist_from_sink] += 1
                incorrect_by_source_node_type[eli.source_node_type] += 1

        session_metrics = dict(total_counts=dict(n_samples=n_samples, n_correct=n_correct, n_incorrect=n_incorrect),
                               correct_by_link_dist_from_root=correct_by_link_dist_from_root,
                               correct_by_link_dist_from_sink=correct_by_link_dist_from_sink,
                               correct_by_min_link_dist=correct_by_min_link_dist,
                               correct_by_total_link_dist=correct_by_total_link_dist,
                               correct_by_source_node_type=correct_by_source_node_type,
                               incorrect_by_link_dist_from_root=incorrect_by_link_dist_from_root,
                               incorrect_by_link_dist_from_sink=incorrect_by_link_dist_from_sink,
                               incorrect_by_min_link_dist=incorrect_by_min_link_dist,
                               incorrect_by_total_link_dist=incorrect_by_total_link_dist,
                               incorrect_by_source_node_type=incorrect_by_source_node_type,
                               )

        # Report
        if display:
            print("Session data:")
            print(f"    query_type = {query_type.name}")
            print(f"    {samples_are_negative = }")
            print(f"    {source_node_is_drug = }")
            print(f"    {n_samples = :,d}")
            print(f"    nbr Incorrect responses = {n_samples - n_correct}")
            print(f"    Correct response pct = {n_correct/n_samples:.3f}")
            print()

            self.pp_contrastive_metrics(session_metrics, "LinkDist.from.root",
                                        "correct_by_link_dist_from_root",
                                        'incorrect_by_link_dist_from_root')

            self.pp_contrastive_metrics(session_metrics, "LinkDist.from.sink",
                                        "correct_by_link_dist_from_sink",
                                        'incorrect_by_link_dist_from_sink')

            self.pp_contrastive_metrics(session_metrics, "SourceNode.type",
                                        "correct_by_source_node_type",
                                        'incorrect_by_source_node_type')

        if cum_metrics is None:
            cum_metrics = defaultdict(Counter)
            # noinspection PyTypeChecker
            cum_metrics["query_type"] = []

        for k, v in session_metrics.items():
            cum_metrics[k].update(v)

        # noinspection PyUnresolvedReferences
        cum_metrics["query_type"].append(query_type.name)

        return cum_metrics

    def get_sample_analytic_params(self, sample: dict[str, Any]) -> SampleParams:

        opt_match_metrics = sample['opt_match_metrics']
        query_type = QueryType[opt_match_metrics['query_type']]

        llm_opt = opt_match_metrics["option_key"]

        # Is correct response?

        is_positive_sample = opt_match_metrics["is_positive_sample"]
        response_is_correct = False

        if is_positive_sample:
            if llm_opt in POSITIVE_RESPONSE_OPTIONS_STRICT[query_type]:
                response_is_correct = True
        else:
            if llm_opt in NEGATIVE_RESPONSE_OPTIONS[query_type]:
                response_is_correct = True

        drug_node = sample["prompt_info"]["drug_id"]
        disease_node = sample["prompt_info"]["disease_id"]

        eli = EditLinkInfo(**sample["prompt_info"]["edit_link_info"])

        moa = self.drugmechdb.get_indication_graph_with_id(eli.source_moa_id)

        link_dist_from_root = nx.shortest_path_length(moa, drug_node, eli.source_node)
        link_dist_from_sink = nx.shortest_path_length(moa, eli.source_node, disease_node)

        params = SampleParams(query_type=query_type,    # type: ignore
                              is_positive_sample=is_positive_sample,
                              response_is_correct=response_is_correct,
                              drug_node=drug_node,
                              disease_node=disease_node,
                              edit_link_info=eli,
                              link_dist_from_root=link_dist_from_root,
                              link_dist_from_sink=link_dist_from_sink
                              )
        return params

    def write_all_sessions_sample_params(self, session_json_file_seq: list[str], csv_file: str):
        print("Nbr session files =", len(session_json_file_seq))
        print("All session files:")
        print(*[f"    {'/'.join(pathlib.PurePath(f).parts[-2:])}" for f in session_json_file_seq], sep="\n")
        print()

        data = []
        for sesn_file in session_json_file_seq:
            with open(sesn_file) as f:
                jdict = json.load(f)

            for sample in jdict["session"]:
                sparams = self.get_sample_analytic_params(sample)
                data.append(sparams.to_analytic_fields())

        print(f"Total nbr samples read = {len(data):,d}")
        print()

        with open(csv_file, "w") as csvf:
            print(*SampleParams.get_analytic_fieldnames(), sep=",", file=csvf)  # type: ignore
            for row in data:
                print(*row, sep=",", file=csvf)  # type: ignore

        print("Data written to:", csv_file)
        return

    @staticmethod
    def pp_contrastive_metrics(attribution_metrics, key: str, corr_fld: str, incorr_fld: str):

        cnts_corr = attribution_metrics[corr_fld]
        cnts_incorr = attribution_metrics[incorr_fld]

        key_values = sorted(set(cnts_corr.keys()) | set(cnts_incorr.keys()))

        n_corr = attribution_metrics["total_counts"]["n_correct"]
        n_incorr = attribution_metrics["total_counts"]["n_incorrect"]

        df = pd.DataFrame.from_records([(k,
                                         cnts_corr[k], cnts_incorr[k],
                                         cnts_corr[k]/n_corr, cnts_incorr[k]/n_incorr,
                                         cnts_corr[k]/(cnts_corr[k] + cnts_incorr[k]),
                                         cnts_incorr[k]/(cnts_corr[k] + cnts_incorr[k]),
                                         )
                                        for k in key_values],
                                       columns=[key,
                                                "Correct.cnt", "Incorrect.cnt",
                                                "Correct.dist", "Incorrect.dist",
                                                "Correct.prop", "Incorrect.prop",
                                                ])
        df.loc['Total'] = df.sum()
        df.loc[df.index[-1], key] = ''
        # Fix *.prop
        df.loc[df.index[-1], "Correct.prop"] = n_corr/(n_corr + n_incorr)
        df.loc[df.index[-1], "Incorrect.prop"] = n_incorr/(n_corr + n_incorr)

        print(f"Correct and Incorrect responses by '{key}':\n")
        print(df.to_markdown(floatfmt=[',.0f', ',.0f', ',.0f', ',.0f', ',.3f', ',.3f', ',.3f', ',.3f']))
        print()

        return

    @staticmethod
    def is_sample_response_correct(sample: dict[str, Any], is_relaxed: bool = False) -> bool:
        """
        Is the LLM response for this recorded sample a Correct response (i.e. matches expected response)?

        :param sample: One sample, as stored in the session file
        :param is_relaxed: Whether to use a relaxed setting ("Partially Blocked" allowed for +ives)
        """
        qtype: QueryType = QueryType[sample["prompt_info"]["query_type"]]   # type: ignore
        is_negative_sample = sample["prompt_info"]["is_negative_sample"]
        llm_opt = sample["opt_match_metrics"]["option_key"]

        res = model_pred_is_pos(qtype, llm_opt,
                                 is_positive_sample=not is_negative_sample,
                                 is_relaxed=is_relaxed)

        return not res if is_negative_sample else res

    @staticmethod
    def extract_main_metrics(session_json_file: str) -> dict[str, Any]:
        """
        Extracts the main metrics from JSON session file, used e.g. to compute variances.
        """
        with open(session_json_file) as f:
            jdict = json.load(f)

        metrics = jdict["metrics"]

        opt_keys = [k for k in metrics.keys() if k.startswith("opt_cnt")] + \
                   [k for k in metrics.keys() if k.startswith("opt_pct")]

        # Default value for absent metric is np.NaN
        main_metrics = dict((k, metrics.get(k, np.NaN)) for k in opt_keys)

        for bk in ["binary-strict", "binary-relaxed"]:
            mbnry = metrics.get(bk)

            if not mbnry:
                continue

            main_metrics[bk] = dict((k, mbnry.get(k, np.NaN))
                                    for k in ["n_gt", "n_gf",
                                              "recall",
                                              "n_true_pos", "n_true_neg", "n_false_pos", "n_false_neg",
                                              "accuracy", "accuracy_gt", "accuracy_gf",
                                              ]
                                    )

        return main_metrics

# /


# -----------------------------------------------------------------------------
#   Functions: metrics
# -----------------------------------------------------------------------------


def model_pred_is_pos(qtype: QueryType, llm_opt: str, is_positive_sample: bool, is_relaxed: bool) -> bool:
    """
    Is the LLM response correct (did the LLM return the expected option) for +ive sample,
        and Not Correct for -ive sample?
    Expected responses are defined in the globals exported in `llmx.prompts_editlink` (see code below).
    For computing `y_pred` for `precision_recall_fscore_support`.
    """

    if is_positive_sample:
        if is_relaxed:
            return llm_opt in POSITIVE_RESPONSE_OPTIONS_RELAXED[qtype]
        else:
            return llm_opt in POSITIVE_RESPONSE_OPTIONS_STRICT[qtype]
    else:
        return llm_opt not in NEGATIVE_RESPONSE_OPTIONS[qtype]


def pprint_accumulated_metrics(cum_metrics: Dict[str, List[Any]],
                               prompt_builder: PromptBuilder,
                               n_tested: int,
                               n_valid_llm_response: int,
                               n_samples_skipped: int,
                               n_skipped_llm_response_inc: int
                               ) \
        -> Dict[str, Union[int, float, List[float]]] | None:
    """
    Prints stats.
    Prints summary stats on OptionsMatchScore.
    Returns computed metrics.

    Since True-Pos for these queries have No proposed MoA, things are a bit different from ADD_LINK.

    :param cum_metrics: Accumulated OptionsMatchScore
    :param prompt_builder:
    :param n_tested:
    :param n_valid_llm_response:
    :param n_samples_skipped:
    :param n_skipped_llm_response_inc:
    """

    print("EditLink batch test stats:")

    if cum_metrics is not None:
        # Check the cum_metrics is correct type
        assert "option_key" in cum_metrics, f"`cum_metrics` is not accumulation of `OptionsMatchScore`?"

        n_samples = len(cum_metrics["is_positive_sample"])
        y_true = np.asarray(cum_metrics["is_positive_sample"], dtype=bool)
        n_gt = y_true.sum()
        n_gf = y_true.shape[0] - n_gt

        assert n_gf == 0 or n_gt == 0, \
            ("This function assumes all the samples are either +ive or -ive. " +
             f"Instead it received {n_gt=}, {n_gf=}.")

        tested_qtypes = set(cum_metrics["query_type"])
        assert len(tested_qtypes) == 1, f"Multiple ({len(tested_qtypes)}) query-types found!"

        # Binary predictions
        y_pred_strict = np.asarray([model_pred_is_pos(qtype, opt, is_positive, is_relaxed=False)
                                    for qtype, opt, is_positive in
                                    zip(cum_metrics["query_type"],
                                        cum_metrics["option_key"],
                                        cum_metrics["is_positive_sample"])]
                                   )
        y_pred_rlxd   = np.asarray([model_pred_is_pos(qtype, opt, is_positive, is_relaxed=True)
                                    for qtype, opt, is_positive in
                                    zip(cum_metrics["query_type"],
                                        cum_metrics["option_key"],
                                        cum_metrics["is_positive_sample"])]
                                   )
    else:
        n_samples = n_gt = n_gf = y_pred_strict = y_pred_rlxd = y_true = 0

    pp_common_metrics(n_gf, n_gt, n_samples, n_samples_skipped, n_skipped_llm_response_inc, n_tested,
                      n_valid_llm_response)

    if cum_metrics is None:
        print()
        print("No metrics to accumulate!")
        print()
        return None

    # Table of Option-Key counts

    response_opts_dict = prompt_builder.get_response_options_dict()

    all_opt_keys = list(response_opts_dict.keys()) + [None]

    opts = np.asarray(cum_metrics["option_key"])

    # Print Table of opt counts

    opt_counts: Dict[str, List[Any]] = dict()
    floatfmt = ['']     # format for the index col

    qtype = cum_metrics["query_type"][0]

    counts = Counter(opts)
    opt_counts[f"{qtype} Counts"] = [counts[ok] for ok in all_opt_keys]
    opt_counts[f"{qtype} pct"] = [cnt / n_samples for cnt in opt_counts[f"{qtype} Counts"]]
    floatfmt.extend(['.0f', '.1%'])

    df = pd.DataFrame(opt_counts, index=[str(x) for x in all_opt_keys])
    df.loc['Totals'] = df.sum(numeric_only=True)
    print(df.to_markdown(floatfmt=floatfmt))
    print()

    # Opt metrics for saving to session

    metrics = dict()

    tot = 0
    for ok in all_opt_keys:
        cnt = counts[ok]
        tot += cnt
        metrics["opt_cnt:{ok}".format(ok=ok)] = cnt
        metrics["opt_pct:{ok}".format(ok=ok)] = cnt / n_samples

    metrics["opt_pct:{ok}".format(ok="TOTAL")] = tot

    # Binary prediction metrics

    metrics["binary-strict"] = pp_binary_metrics(n_gf, n_gt, y_pred_strict, y_true, msg="Strict")
    metrics["binary-relaxed"] = pp_binary_metrics(n_gf, n_gt, y_pred_rlxd, y_true, msg="Relaxed")

    print("Summary of Response Option match scores:")

    all_opt_match_scores = np.asarray(cum_metrics["match_score"])

    # Set score for no-match to None, so summary stats are only for +ive match
    # noinspection PyComparisonWithNone
    opt_is_none = (opts == None)
    # noinspection PyTypeChecker
    opt_match_scores = np.where(opt_is_none, None, all_opt_match_scores)

    v_mean = np.nanmean(opt_match_scores)
    q1, q2, q3 = np.nanpercentile(opt_match_scores, [25, 50, 75])

    print(f"    match_score:  mean = {v_mean:.3f}")
    print(f"\trange = [{np.nanmin(opt_match_scores):.3f}, {np.nanmax(opt_match_scores):.3f}], ",
          f"quartiles = [{q1:.3f}, {q2:.3f}, {q3:.3f}]")

    n_opt_is_none = np.sum(opt_is_none)
    print(f"    nbr of No-matches =", n_opt_is_none)

    metrics["match_score:is None"] = n_opt_is_none

    metrics["match_score:mean"] = v_mean
    metrics["match_score:quartiles"] = [q1, q2, q3]

    print()
    return metrics


def pp_binary_metrics(n_gf, n_gt, y_pred, y_true, msg: str = None):
    """

    :param n_gf: nbr Ground False samples
    :param n_gt: nbr Ground True samples
    :param y_pred: Prediction on: Whether each sample is a +ive sample
    :param y_true: Ground-truth: Whether each sample is a +ive sample
    :param msg:

    :return: Binary `metrics` dict
    """

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
                   accuracy=accuracy_score(y_true, y_pred, normalize=True),
                   accuracy_gt=accuracy_score(y_true[y_true], y_pred[y_true], normalize=True) if n_gt else 0,
                   accuracy_gf=accuracy_score(y_true[~y_true], y_pred[~y_true], normalize=True) if n_gf else 0,
                   precision=prec, recall=recall, f1=f1,
                   )

    if msg:
        msg = f" - {msg}"
    else:
        msg = ""

    pp_dict(metrics, f"Binary metrics{msg}")

    print()
    return metrics


def pp_common_metrics(n_gf, n_gt, n_samples, n_samples_skipped, n_skipped_llm_response_inc, n_tested,
                      n_valid_llm_response):
    print(f"    Nbr samples skipped       = {n_samples_skipped:5,d}   ... (this should be 0 if no examples)")
    print(f"    Nbr samples tested on LLM = {n_tested:5,d}")
    print(f"    Nbr valid LLM response    = {n_samples:5,d} ... Counts match? {n_samples == n_valid_llm_response}")
    print(f"    Nbr incomplete response   = {n_skipped_llm_response_inc:5,d}")
    print()
    print(f"    Nbr positive samples    = {n_gt:5,d}")
    print(f"    Nbr negative samples    = {n_gf:5,d}")
    print()
    return


def extract_combined_metrics(source_drug_jsession: str, source_protein_jsession: str):
    """
    Prints a ccomparative side-by-side table of counts for each LLM-selected option
    :param source_drug_jsession:
    :param source_protein_jsession:
    """

    from drugmechcf.llmx.prompts_editlink import RESPONSE_OPTIONS

    print("Reading:", source_drug_jsession)
    with open(source_drug_jsession) as f:
        jdict_source_drug = json.load(f)

    print("Reading:", source_protein_jsession)
    with open(source_protein_jsession) as f:
        jdict_source_protein = json.load(f)

    print("Processing ...", flush=True)

    assert jdict_source_drug["args"]["source_node_is_drug"], "Arg 1 must be for `source_node_is_drug`"
    assert not jdict_source_protein["args"]["source_node_is_drug"], "Arg 2 must be for NOT `source_node_is_drug`"

    source_drug_opts = [session["metrics"]["option_key"] for session in jdict_source_drug["session"]]
    source_protein_opts = [session["metrics"]["option_key"] for session in jdict_source_protein["session"]]

    source_drug_opts_counts = Counter(source_drug_opts)
    source_protein_opts_counts = Counter(source_protein_opts)

    source_drug_total = len(jdict_source_drug["session"])
    source_protein_total = len(jdict_source_protein["session"])

    print()

    if (qt1 := jdict_source_drug["args"]["query_type"]) == (qt2 := jdict_source_protein["args"]["query_type"]):
        print(jdict_source_protein["args"]["query_type"])
        qt1 = qt2 = ""
    else:
        qt1 += ": "
        qt2 += ": "

    print("", f'{qt1}Source-is-Drug', "",
          f'{qt2}Source-is-not-Drug',
          sep="\t")
    print("Option", "Count", "pct", "Count", "pct", sep="\t")

    for opt in list(RESPONSE_OPTIONS.keys()) + [None, "Unknown"]:
        print(opt,
              source_drug_opts_counts[opt], f"{source_drug_opts_counts[opt]/source_drug_total:.1%}",
              source_protein_opts_counts[opt], f"{source_protein_opts_counts[opt]/source_protein_total:.1%}",
              sep="\t"
              )

    print("TOTAL", source_drug_total, "", source_protein_total, "", sep="\t")

    print("\n")
    return


# -----------------------------------------------------------------------------
#   Functions: Commands
# -----------------------------------------------------------------------------


def test_editlink_batch(samples_data_file: str,
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
                        prompt_version: int = 0,
                        insert_known_moas: bool = False,
                        #
                        show_full_prompt=False,
                        show_response=False,
                        add_unknown_response_opt: bool = False,
                        #
                        no_llm=False,
                        #
                        max_samples=None,
                        ):

    pp_funcargs(test_editlink_batch)

    global DEBUG_NO_LLM, MAX_TEST_SAMPLES

    DEBUG_NO_LLM = no_llm

    if max_samples:
        MAX_TEST_SAMPLES = max_samples

    if model_key is None:
        model_key = "o3-mini"

    tester = TestEditLink(model_key=model_key,
                          api_key=api_key,
                          base_url=base_url,
                          prompt_version=prompt_version,
                          insert_known_moas=insert_known_moas,
                          add_unknown_response_opt=add_unknown_response_opt,
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
    print("Global flags:")
    print(f"    {MAX_TEST_SAMPLES = }")
    print(f"    {DEBUG_NO_LLM = }")
    print(f"    {DEBUG = }")
    print()
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
#
# [Python]$ python -m llmx.test_editlink {batch | ...}
#
# Examples (executed from `$PROJDIR/src/`):
#
# [Python]$ python -m drugmechcf.llmx.test_editlink batch ../Data/Counterfactuals/change_pos_dpi_r250.json  \
#                       ../Data/Sessions/o3-mini/change_pos_dpi_r250.json        \
#                       2>&1 | tee ../Data/Sessions/o3-mini/change_pos_dpi_r250_log.txt
#
# [Python]$ python -m drugmechcf.llmx.test_editlink attrib_metrics ../Data/Sessions/*/*/*_pos_ppi_r250.json \
#                       2>&1 | tee ../Data/Sessions/attrib_report_pos_ppi.txt
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Test Edit-Link MoAs with LLM.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... batch
    _sub_cmd_parser = _subparsers.add_parser('batch',
                                             help="Test ChatGPT on batch of Edit-Link samples.")
    _sub_cmd_parser.add_argument('-p', '--prompt_version', type=int, default=2,
                                 help="Prompt version to use. Default is ver 2.")
    _sub_cmd_parser.add_argument('-m', '--model_key', type=str,
                                 choices=list(MODEL_KEYS.keys()), default="o3-mini",
                                 help="LLM model type or id.")
    #
    _sub_cmd_parser.add_argument('-u', '--add_unknown_response_opt', action='store_true',
                                 help="Add 'Unknown' to the response options.")
    _sub_cmd_parser.add_argument('-k', '--insert_known_moas', action='store_true',
                                 help="Insert Known MoAs in the prompt (Closed-world setting).")
    #
    _sub_cmd_parser.add_argument('-f', '--show_full_prompt', action='store_true',
                                 help="Show the full LLM prompt.")
    _sub_cmd_parser.add_argument('-r', '--show_response', action='store_true',
                                 help="Show LLM response.")
    #
    _sub_cmd_parser.add_argument('-n', '--no_llm', action='store_true',
                                 help="Don't call LLM.")
    # args
    _sub_cmd_parser.add_argument('samples_data_file', type=str,
                                 help="Samples data file.")
    _sub_cmd_parser.add_argument('output_json_file', nargs="?", type=str, default=None,
                                 help="Output session file.")

    # ... attrib_metrics
    _sub_cmd_parser = _subparsers.add_parser('attrib_metrics',
                                             help="Summarize combined Attribution metrics.")
    _sub_cmd_parser.add_argument('jsession_files', type=str, nargs="+",
                                 help="One or more JSON session file(s) for Source is NOT Drug.")

    # ... anal_params
    _sub_cmd_parser = _subparsers.add_parser('anal_params',
                                             help="Get analytic params across sessions.")
    _sub_cmd_parser.add_argument('csv_file', type=str,
                                 help="Output CSV file path")
    _sub_cmd_parser.add_argument('jsession_files', type=str, nargs="+",
                                 help="One or more JSON session file(s) for Source is NOT Drug.")

    # ... combo_metrics
    _sub_cmd_parser = _subparsers.add_parser('combo_metrics',
                                             help="Summarize combined metrics.")
    _sub_cmd_parser.add_argument('jsession_source_drug', type=str,
                                 help="JSON session file for Source is Drug.")
    _sub_cmd_parser.add_argument('jsession_source_not_drug', type=str,
                                 help="JSON session file for Source is NOT Drug.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'batch':

        test_editlink_batch(_args.samples_data_file,
                            _args.output_json_file,
                            model_key=_args.model_key,
                            #
                            prompt_version=_args.prompt_version,
                            insert_known_moas=_args.insert_known_moas,
                            #
                            show_full_prompt=_args.show_full_prompt,
                            show_response=_args.show_response,
                            add_unknown_response_opt=_args.add_unknown_response_opt,
                            #
                            no_llm=_args.no_llm,
                            )

    elif _args.subcmd == 'attrib_metrics':

        tester_ = TestEditLink()
        tester_.analyze_response_attribution_multiple_sessions(_args.jsession_files)

    elif _args.subcmd == 'anal_params':

        tester_ = TestEditLink()
        tester_.write_all_sessions_sample_params(_args.jsession_files, _args.csv_file)

    elif _args.subcmd == 'combo_metrics':

        extract_combined_metrics(_args.jsession_source_drug, _args.jsession_source_not_drug)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
