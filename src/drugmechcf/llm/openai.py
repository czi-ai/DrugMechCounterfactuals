"""
Interface to OpenAI LLMs
"""

import dataclasses
# import logging
import os
import pprint
import sys
import textwrap
import threading
from typing import Union

# noinspection PyUnresolvedReferences
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    wait_random,
)  # for exponential backoff

from openai import OpenAI, RateLimitError, BadRequestError
from openai.types.chat.chat_completion import ChatCompletion

from drugmechcf.utils.misc import ValidatedDataclass


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

INCLUDE_PROJECT_ID = False


MODEL_KEYS = {
    "o3": "o3-2025-04-16",              # Reasoning. Cost: $2 / 1M input, $8 / 1M output
    "o3-mini": "o3-mini-2025-01-31",    # Reasoning. Cost: $1.10, $4.40
    "o4-mini": "o4-mini-2025-04-16",    # Reasoning. Cost: $1.10, $4.40
    "4o": "gpt-4o-2024-08-06",          # General model: $2.50, $10
    "o1": "o1-2024-12-17",              # Prev. full reasoning model. Most expensive: $15, $60
}


# Tenacity doc recommends the following for automatic logging of retries.
# However, with logging setup, OpenAI API starts logging lots of messages.
#
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
#     @retry(retry=retry_if_exception_type(RateLimitError),
#            after=after_log(logger, logging.DEBUG),                            # ... Log Retries
#            wait=wait_random_exponential(min=5, max=90) + wait_random(0, 10),
#            stop=stop_after_attempt(6))
#     def _call_chat_completion_with_backoff(self, messages):
#           ...
#


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class OpenAICompletionOpts(ValidatedDataclass):
    """
    Options that can be set in a `client.chat.completions.create(...)`.
    Defaults are the default values for OpenAI completion client, unless otherwise specified.
    """

    model: str = "gpt-4o-2024-08-06"
    """
    Which LLM model to use.
    Alt: 'gpt-4o-2024-11-20'
    Training data is up to Oct 2023.
    Ref: https://platform.openai.com/docs/models#gpt-4o
    """

    max_completion_tokens: int = None
    """
    An upper bound for the number of tokens that can be generated for a completion, including visible output tokens
    and reasoning tokens.
    
    The doc on reasoning models recommends using at least 25,000 for this param.
    ref: https://platform.openai.com/docs/guides/reasoning#allocating-space-for-reasoning
    """

    reasoning_effort: str = "medium"
    """
    Possible values: "low", "medium", "high".
    Used only for models: o1 and o3-mini
    """

    presence_penalty: float = 0.0
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on whether 
    they appear in the text so far, increasing the model's likelihood to talk about new topics.
    """

    seed: int = 42
    """
    From OpenAI: This feature is in Beta. If specified, our system will make a best effort to sample deterministically,
    such that repeated requests with the same seed and parameters should return the same result. 
    Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter 
    to monitor changes in the backend.
    """

    temperature: float = 0.2
    """
    From OpenAI: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output 
    more random, while lower values like 0.2 will make it more focused and deterministic.
    Default: OpenAI API default is 1.0. Using a lower default to make it more deterministic.
    Alternative parameter is 'top_p'.
    
    NOTE: Reasoning models (o1, o3-mini) do not currently allow any value other than 1.0.
    """

    # top_p: float = 1.0
    """
    An alternative to sampling with temperature, called nucleus sampling, where the model considers the results
    of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass
    are considered. We generally recommend altering this or temperature but not both.
    Default: 1.0
    Alternative parameter is 'temperature'.
    """

    extra_body: dict = None
    """
    For passing additional args, e.g. as needed for LLMs served using `vllm`.
    See: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

    def as_params_dict(self):
        skip_flds = ["optsdir"]
        # 'reasoning_effort' is not a valid param for 4o models.
        if self.model.startswith("gpt-4o"):
            skip_flds.append("reasoning_effort")

        # noinspection PyTypeChecker
        d = dataclasses.asdict(self)

        for fld in skip_flds:
            del d[fld]

        return d
# /


class CompletionOutput:

    def __init__(self):
        self.llm_output = None
        return

    @property
    def message(self) -> str:
        return self.llm_output.choices[0].message.content

    @property
    def finish_reason(self) -> str:
        return self.llm_output.choices[0].finish_reason

    def is_complete_response(self) -> bool:
        return self.finish_reason == "stop"

    @staticmethod
    def from_openai_client(response: ChatCompletion):
        output = CompletionOutput()
        output.llm_output = response
        return output

    def pprint(self, linewidth=90, stream=None):
        pprint.pprint(self.llm_output, stream=stream, width=linewidth)
        return
# /


class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1

    def value(self):
        with self._lock:
            return self._value
# /


class OpenAICompletionClient:
    """
    Convenience wrapper around the openai client.

    Notes:
        1. `OpenAI()` automatically checks the env variables 'OPENAI_API_KEY' and 'OPENAI_PROJECT_ID'.
        2. For project-specific API keys, you do not need to set the project (or env var OPENAI_PROJECT_ID).
           If you do, then make sure it is the Project ID (which looks something like 'Proj_ABCxyz...') and
           *not* the Project Name.
        3. For querying LLMs served by `vllm`, provide args:
                - `api_key` (or set env var OPENAI_API_KEY appropriately)
                - `base_url`, e.g. "http://localhost:8000/v1"
    """

    # Static member, counts nbr LLM calls across threads, in `self._execute_llm()`.
    # See: OpenAICompletionClient.get_nbr_calls()
    CallCounter = ThreadSafeCounter()

    def __init__(self, opts: Union[str, OpenAICompletionOpts] = None,
                 *,
                 api_key: str = None,
                 timeout_secs: int = None,
                 use_flex_service_tier: bool = False,
                 base_url: str = None,
                 ):
        """

        :param opts:  OpenAICompletionOpts

        :param api_key:
            If left unspecified, this will be retrieved from the env var OPENAI_API_KEY.
            If specified, the env var is not checked. Can use this arg for VLLM-served models,
                e.g. api_key = "EMPTY".

        :param timeout_secs: Timeout in seconds. Default is 10 minutes.
            IF a call to the OpenAI client takes longer than this time
            THEN it will terminate with the exception: `openai.APITimeoutError`

        :param use_flex_service_tier:
            IF True THEN
                "provides significantly lower costs for Chat Completions or Responses requests
                 in exchange for slower response times and occasional resource unavailability.
                 It is ideal for non-production or lower-priority tasks such as model evaluations,
                 data enrichment, or asynchronous workloads."
            ref: https://platform.openai.com/docs/guides/flex-processing?api-mode=chat
            NOTE - currently (June 10, 2025) only for models: o3, o4-mini

        :param base_url:
            For OpenAI models, leave unspecified.
            For VLLM-served models, example URL is "http://localhost:8000/v1".
        """

        if opts is None:
            opts = OpenAICompletionOpts()
        elif isinstance(opts, str):
            opts = OpenAICompletionOpts.from_json_file(opts)

        self.opts: OpenAICompletionOpts = opts

        self.use_flex_service_tier = use_flex_service_tier
        if self.use_flex_service_tier:
            if not self.can_use_flex_service_tier(self.opts):
                self.use_flex_service_tier = False

        if api_key is None:
            try:
                api_key = os.environ["OPENAI_API_KEY"]
            except KeyError:
                raise KeyError("Environment variable 'OPENAI_API_KEY' not found."
                               "Please put your OpenAI API Key in this environment variable.")

        if INCLUDE_PROJECT_ID:
            try:
                project_id = os.environ["OPENAI_PROJECT_ID"]
            except KeyError:
                raise KeyError("Environment variable 'OPENAI_PROJECT_ID' not found."
                               "Please put your OpenAI Project-ID in this environment variable.")

            self.client = OpenAI(api_key=api_key, project=project_id, timeout=timeout_secs, base_url=base_url)

        else:
            self.client = OpenAI(api_key=api_key, timeout=timeout_secs, base_url=base_url)

        # Where client full response is cached
        self.last_response = None
        return

    @staticmethod
    def can_use_flex_service_tier(opts: OpenAICompletionOpts, show_message=True) -> bool:
        """
        flex_service_tier currently (as of June 10, 2025) only valid for models: o3, o4-mini
        """
        can_use = ((opts.model.startswith("o3-") and "mini" not in opts.model) or
                    opts.model.startswith("o4-mini-"))

        if not can_use and show_message:
            print("****************************************************************")
            print(f"*** flex_service_tier not available for model {opts.model}")
            print(f"*** Only available for: o3, o4-mini")
            print(f"*** Overriding to False")
            print("****************************************************************")
            print()

        return can_use

    def _execute_llm(self, system_prompt: str = None, user_prompt: str = None, verbose=False) -> CompletionOutput:

        assert system_prompt or user_prompt, "At least one of `system_prompt`, `user_prompt` must be provided."

        self.CallCounter.increment()

        messages = []
        if system_prompt:
            messages.append(dict(role="system", content=system_prompt))
        if user_prompt:
            messages.append(dict(role="user", content=user_prompt))

        if verbose:
            print("---------------", "Request:", sep="\n")
            pprint.pprint(messages)

        try:
            self._call_chat_completion_with_backoff(messages)
        except BadRequestError as e:
            print("*** BadRequestError in the following prompt:", file=sys.stderr)
            print(messages, file=sys.stderr)
            print("***", file=sys.stderr)
            raise e

        if verbose:
            print("---------------", "Response:", sep="\n")
            pprint.pprint(self.last_response.dict(), width=80, compact=False)
            print(flush=True)

        return CompletionOutput.from_openai_client(self.last_response)

    @retry(retry=retry_if_exception_type(RateLimitError),
           # after=after_log(logger, logging.DEBUG),
           wait=wait_random_exponential(min=5, max=90) + wait_random(0, 10),
           stop=stop_after_attempt(6))
    def _call_chat_completion_with_backoff(self, messages):
        """
        IF RateLimitError is raised THEN
            This will keep retrying up to 6 times.
            IF it still does not succeed, then the exception can be accessed as follows:

            try:
                self._call_chat_completion_with_backoff(...)
            except tenacity.RetryError as e:
                causing_exception = e.last_attempt.exception()

        -- logging suppressed: Retries won't get logged to `logger`
        """
        return self._call_chat_completion(messages)

    def _call_chat_completion(self, messages):
        if self.use_flex_service_tier:
            self.last_response = self.client.chat.completions.create(
                messages = messages,
                service_tier = "flex",
                **self.opts.as_params_dict()
            )
        else:
            self.last_response = self.client.chat.completions.create(
                messages = messages,
                **self.opts.as_params_dict()
            )

        return self.last_response

    def __call__(self, system_prompt: str = None, user_prompt: str = None, verbose=False):
        """
        :exception: openai.APITimeoutError
            if timeout exceeded.
        """
        return self._execute_llm(system_prompt, user_prompt, verbose=verbose)

    @staticmethod
    def get_nbr_calls() -> int:
        return OpenAICompletionClient.CallCounter.value()
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def get_exp_client(temperature: float = 1.0, model = "o1-preview-2024-09-12") -> OpenAICompletionClient:

    opts = OpenAICompletionOpts(model = model,
                                temperature=temperature)

    client = OpenAICompletionClient(opts)

    return client


def get_openai_client(*,
                      model_key: str = None,
                      reasoning_effort: str = "medium",
                      use_flex_service_tier: bool = False,
                      timeout_secs: int = 600,
                      temperature: float = 1.0,
                      # ... for vllm-served LLMs
                      api_key: str = None,
                      base_url: str = None,
                      ) -> OpenAICompletionClient:
    """
    Allows easy configuration of params relevant to reasoning models.

    :param model_key: A key from `MODEL_KEYS` or a specific model "o3-mini-2025-01-31" or "o1-2024-12-17"
    :param reasoning_effort:
    :param use_flex_service_tier: Default is False
        IF True THEN
            "provides significantly lower costs for Chat Completions or Responses requests
             in exchange for slower response times and occasional resource unavailability.
    :param timeout_secs: OpenAI's Default value is 10 minutes.
    :param temperature: OpenAI's Default is 1.0

    :param api_key:
    :param base_url:
        These two params are passed on to `OpenAICompletionClient`. Used for connecting to vllm-hosted LLMs.
    """

    assert reasoning_effort in ["low", "medium", "high"]

    if model_key is None:
        # This is the default model used in many experiments
        model = "gpt-4o-2024-08-06"
    else:
        model = MODEL_KEYS.get(model_key, model_key)

    # Higher timeout for o1
    if model_key == "o1" and timeout_secs == 60:
        timeout_secs = 240

    if model[:2] in ["o3", "o1"] and temperature != 1.0:
        print(f"*** WARNING (get_openai_client): setting temperature to 1.0 for reasoning model {model_key=},"
              f" {model=}.")
        temperature = 1.0

    opts = OpenAICompletionOpts(model=model,
                                reasoning_effort=reasoning_effort,
                                temperature=temperature)

    client = OpenAICompletionClient(opts, use_flex_service_tier=use_flex_service_tier, timeout_secs=timeout_secs,
                                    api_key=api_key, base_url=base_url)

    return client


def pprint_message(msg: str, *, linewidth: int = 80, file=None):
    """
    Pretty-print LLM response message `msg` to `file`.
    Paragraphs are separated by an empty line.

    :param msg:
    :param linewidth: Used to wrap text.
    :param file: Output stream. Default is STDOUT.
    """
    for para in msg.split("\n\n"):
        print(*textwrap.wrap(para, width=linewidth), sep="\n", file=file)
        print(file=file)
    print("\n", file=file)
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.llm.openai
#

if __name__ == "__main__":

    from drugmechcf.utils.misc import pp_underlined_hdg

    pp_underlined_hdg("Testing OpenAI Client", linechar="=")
    print()

    sys_prompt = "You are a helpful assistant."
    user_prompts = ["Who has the most Olympic medals?", "Why is it colder at higher altitutes?"]

    llm = OpenAICompletionClient()
    for i, uq in enumerate(user_prompts, start=1):
        pp_underlined_hdg(f"Qn {i}: {uq}")
        reply = llm(system_prompt=sys_prompt, user_prompt=uq, verbose=True)
        print()
        pp_underlined_hdg("Answer:")
        pprint_message(reply.message, linewidth=90)

    print("Done.")
