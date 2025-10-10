"""
Test Qwen models with VLLM-hosted LLM
"""

import os.path

from drugmechcf.llmx.test_addlink import test_addlink_batch
from drugmechcf.llmx.test_editlink import test_editlink_batch


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

DEFAULT_QWEN_MODEL = "Qwen3-4B-Thinking-2507-FP8"

# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def test_qwen(samples_data_file: str,
              output_json_file: str | None,
              *,
              model: str = None,
              insert_known_moas: bool = False,
              max_samples: int = 0,
              show_response=False):
    """
    Run queries from `samples_data_file` against a Qwen model hosted by vllm.

    :param samples_data_file: One of the Counterfactuals query files
    :param output_json_file: Session output written here
    :param model: Name of the Qwen model
    :param insert_known_moas: Whether to construct a Closed-world prompt
    :param max_samples: Max nbr query samples to run. Default is all.
    :param show_response: Whether to show LLM response in output.
    """

    if model is None:
        model = DEFAULT_QWEN_MODEL

    if "/" not in model:
        model = "Qwen/" + model

    # adjust for your environment
    timeout_secs = 600
    n_worker_threads = 1

    samples_filenm = os.path.basename(samples_data_file)

    if samples_filenm.startswith("AddLink"):

        test_addlink_batch(samples_data_file,
                           output_json_file,
                           model_key=model,
                           api_key="EMPTY",
                           base_url="http://localhost:8000/v1",
                           insert_known_moas=insert_known_moas,
                           n_worker_threads=n_worker_threads,
                           timeout_secs=timeout_secs,
                           max_samples=max_samples,
                           show_response=show_response,
                           )

    elif samples_filenm.startswith("change_") or samples_filenm.startswith("delete_"):

        # Latest prompt in `drugmechcf.llmx.prompts_editlink.PROMPT_TEMPLATE`
        prompt_version = 2

        test_editlink_batch(samples_data_file, output_json_file,
                            model_key=model,
                            api_key="EMPTY",
                            base_url="http://localhost:8000/v1",
                            prompt_version=prompt_version,
                            insert_known_moas=insert_known_moas,
                            n_worker_threads=n_worker_threads,
                            timeout_secs=timeout_secs,
                            max_samples=max_samples,
                            show_response=show_response,
                            )

    else:
        raise NotImplementedError("Unrecognized samples file: " + samples_data_file)

    return


# ======================================================================================================
#   Main
# ======================================================================================================


if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Test vllm-hosted LLM.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... batch
    _sub_cmd_parser = _subparsers.add_parser('batch',
                                             help="Test a Qwen model on batch of Edit-Link samples.")
    _sub_cmd_parser.add_argument('-m', '--model', type=str, default=None,
                                 help=f"The name of the Qwen model. Default is '{DEFAULT_QWEN_MODEL}'.")
    _sub_cmd_parser.add_argument('-k', '--insert_known_moas', action='store_true',
                                 help="Insert Known MoAs in the prompt (Closed-world setting).")
    _sub_cmd_parser.add_argument('-n', '--nbr_samples', type=int, default=None,
                                 help="Number of samples to run. Default is all.")
    # args
    _sub_cmd_parser.add_argument('samples_data_file', type=str,
                                 help="Samples data file.")
    _sub_cmd_parser.add_argument('output_json_file', nargs="?", type=str, default=None,
                                 help="Output session file.")

    # ... test
    _sub_cmd_parser = _subparsers.add_parser('test',
                                             help="Test a Qwen model on batch of Edit-Link samples.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'batch':

        test_qwen(_args.samples_data_file,
                  _args.output_json_file,
                  model=_args.model,
                  insert_known_moas=_args.insert_known_moas,
                  max_samples=_args.nbr_samples
                  )

    elif _args.subcmd == 'test':

        test_qwen("../Data/Counterfactuals/change_neg_dpi_r250.json",
                  None,
                  model=DEFAULT_QWEN_MODEL,
                  max_samples=5,
                  show_response=True,
                  )

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
