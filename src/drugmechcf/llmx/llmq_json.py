"""
Classes to execute query and gather JSON output
"""

import dataclasses
import json
from typing import Any

from openai import APITimeoutError

from drugmechcf.llm.openai import OpenAICompletionOpts, OpenAICompletionClient, CompletionOutput
from drugmechcf.utils.misc import buffered_stdout


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class SQListResponseInfo:
    llm_responses: list[CompletionOutput]

    final_response_code: str

    response_list: list[dict[str, Any]]

    def is_complete_response(self) -> bool:
        return self.final_response_code == "stop"

    def asdict(self) -> dict[str, Any]:
        # noinspection PyTypeChecker
        ridict = dataclasses.asdict(self)

        # Keep only the reponse texts
        ridict["llm_responses"] = [resp.message for resp in self.llm_responses]

        return ridict
# /


class SingleQueryToJsonList:

    JSON_DELIMITER = "+++++"

    def __init__(self, *,
                 llm_client: OpenAICompletionClient = None,
                 llm_opts: OpenAICompletionOpts = None,
                 timeout_secs: int = None
                 ):

        assert llm_client is not None or llm_opts is not None, \
            "At least one of `llm_client` or `llm_opts` must be specified."

        self.llm_client = llm_client
        self.llm_opts = llm_opts
        self.timeout_secs = timeout_secs

        self.show_full_prompt = False
        self.show_response = False

        return

    def get_json_delimiter_prompt_instruction(self) -> str:
        """
        Get instruction to put in prompt that tells LLM how to delimit the JSON output.
        This instruction is added to the end of the prompt.
        """
        pj_instr = (f'Please start the JSON output after a line containing "{self.JSON_DELIMITER}",\n'
                    f'and after the JSON output ends, output another line containing "{self.JSON_DELIMITER}".')
        return pj_instr

    def invoke(self, prompt: str,
               followup_prompt: str = None,
               max_followups: int = 0,
               query_name: str = None,
               verbosity=0,
               ) -> SQListResponseInfo:
        """
        Calls LLM with `prompt` that asks for JSON outpput. Result is List of items.
        IF `max_followups` > 0 THEN this many follow up queries issued to LLM, whose output extends the list.
        All follow-up queries are done using the same LLM client.

        :param prompt: The main or first prompt

        :param followup_prompt: IF provided THEN this prompt used for follow up queries.
        :param max_followups: How many follow-up queries after the first query.

        :param query_name: Used in output trace, if `self.show_full_prompt = True`.

        :param verbosity:

        :return: List of LLM responses, List of items
        """

        if self.llm_client is not None:
            llm_client = self.llm_client
        else:
            llm_client = OpenAICompletionClient(self.llm_opts, timeout_secs=self.timeout_secs)

        if query_name:
            query_name = f" for '{query_name}'"
        else:
            query_name = ""

        qprompt = prompt + "\n" + self.get_json_delimiter_prompt_instruction()

        resp_info = SQListResponseInfo(llm_responses=[], response_list=[], final_response_code="None")

        for pnbr in range(max_followups + 1):

            if self.show_full_prompt or verbosity > 0:
                with buffered_stdout():
                    print(f"... Prompt {pnbr + 1}{query_name}:", flush=True)
                    if self.show_full_prompt:
                        print(qprompt)
                        print("---")
                        print(flush=True)

            try:
                llm_response = llm_client(user_prompt=qprompt, verbose=verbosity > 2)

            except APITimeoutError:
                resp_info.final_response_code = "APITimeoutError"
                if self.show_response:
                    with buffered_stdout():
                        print(f"... LLM response{query_name}:")
                        print("    Request timed out")
                        print(flush=True)
                break

            resp_info.llm_responses.append(llm_response)

            # noinspection PyBroadException
            try:
                json_list = extract_json_list(llm_response.message, self.JSON_DELIMITER)
            except Exception:
                json_list = []

            resp_info.response_list.extend(json_list)
            resp_info.final_response_code = llm_response.finish_reason

            if self.show_full_prompt or self.show_response or verbosity > 0:
                print(f"    LLM response{query_name} complete =", llm_response.is_complete_response(), flush=True)

            if self.show_response:
                with buffered_stdout():
                    print(f"... LLM response{query_name}:")
                    print(llm_response.message)
                    print("---")
                    print(flush=True)

            if not llm_response.is_complete_response():
                break

            if followup_prompt is not None and pnbr == 0:
                qprompt = followup_prompt + "\n" + self.get_json_delimiter_prompt_instruction()

        return resp_info
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def extract_json_list(txt: str, delim: str):

    obj = None
    je = len(txt)

    if (js := txt.find(delim)) < 0:
        js = 0

    try:
        js += txt[js:].index("[")

        if (je := txt[js:].rfind(delim)) < 0:
            je = len(txt)
        else:
            je += js

        je = js + txt[js:je].rindex("]") + 1

        obj = json.loads(txt[js : je])

    except Exception as e:
        with buffered_stdout():
            print()
            print("*** WARNING: (extract_json_list) Error in extracting JSON list:")
            print("Exception =", e)
            print(f" ... {js=}, {je=}")
            print(f"sub-txt = ||{txt[js : je]}||")
            print()
            print(f"full-txt = ||{txt}||")
            print()
            raise e

    return obj
